from typing import Tuple, Sequence
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import distrax
import optax

from .flow import Flow


class Encoder(nn.Module):
    latent_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, X, training):
        for channel in self.hidden_dims:
            X = nn.Conv(channel, (3, 3), strides=2, padding=1)(X)
            X = nn.BatchNorm(use_running_average=not training)(X)
            X = jax.nn.relu(X)

        X = X.reshape((-1, np.prod(X.shape[-3:])))
        mu = nn.Dense(self.latent_dim)(X)
        logvar = nn.Dense(self.latent_dim)(X)

        return mu, logvar


class Decoder(nn.Module):
    output_dim: Tuple[int, int, int]
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, X, training):
        H, W, C = self.output_dim

        # TODO: relax this restriction
        factor = 2 ** len(self.hidden_dims)
        assert (
            H % factor == W % factor == 0
        ), f"output_dim must be a multiple of {factor}"
        H, W = H // factor, W // factor

        X = nn.Dense(H * W * self.hidden_dims[-1])(X)
        X = jax.nn.relu(X)
        X = X.reshape((-1, H, W, self.hidden_dims[-1]))

        for hidden_channel in reversed(self.hidden_dims[:-1]):
            X = nn.ConvTranspose(
                hidden_channel, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2))
            )(X)
            X = nn.BatchNorm(use_running_average=not training)(X)
            X = jax.nn.relu(X)

        X = nn.ConvTranspose(C, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2)))(X)
        X = jax.nn.sigmoid(X)

        return X


def reparameterize(key, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, logvar.shape)
    return mean + eps * std


class VAE(nn.Module):
    # Flow prior
    #
    #   X --encoder--> Z --decoder--> recon <=> X
    #                  ^
    #                  |
    #                flow
    #                  |
    #               epsilon
    #
    # Flow posterior
    #
    #   X --encoder--> Z --flow--> Z_k --decoder--> recon <=> X
    #                               ^
    #                               |
    #                            epsilon

    beta: float
    latent_dim: int
    hidden_dims: Sequence[int]
    output_dim: Tuple[int, int, int]

    flow_prior: bool
    flow_num_coupling_layers: int
    flow_num_bins: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.hidden_dims)
        self.decoder = Decoder(self.output_dim, self.hidden_dims)
        self.epsilon = distrax.MultivariateNormalDiag(
            loc=jnp.zeros((self.latent_dim,)), scale_diag=jnp.ones((self.latent_dim,))
        )
        self.flow = Flow(
            self.latent_dim,
            self.hidden_dims,
            self.flow_num_coupling_layers,
            self.flow_num_bins,
        )
        self.flow.setup()  # FIXME: why is this needed?
        if self.flow_prior:
            self.prior = distrax.Transformed(self.epsilon, self.flow)
        else:
            self.prior = self.epsilon

    def __call__(self, key, X, training):
        mean, logvar = self.encoder(X, training)
        posterior = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(0.5 * logvar)
        )
        Z = reparameterize(key, mean, logvar)

        if not self.flow_prior:
            posterior = distrax.Transformed(posterior, self.flow)
            Z = self.flow.forward(Z)

        recon = self.decoder(Z, training)

        likelihood = self.beta * jnp.mean((recon - X) ** 2)
        kl_divergence = -(self.prior.log_prob(Z) - posterior.log_prob(Z))

        return likelihood - kl_divergence

    def decode(self, epsilon, training):
        if self.flow_prior:
            Z = self.flow.forward(epsilon)
        else:
            Z = epsilon

        return self.decoder(Z, training)


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]


def create_train_state(
    key,
    beta: float,
    latent_dim: int,
    hidden_dims: Sequence[int],
    specimen: jnp.ndarray,
    flow_prior: bool,
    flow_num_coupling_layers: int,
    flow_num_bins: int,
    learning_rate: float,
):
    vae = VAE(
        beta,
        latent_dim,
        hidden_dims,
        specimen.shape,
        flow_prior,
        flow_num_coupling_layers,
        flow_num_bins,
    )
    key_dummy = jax.random.PRNGKey(42)
    variables = vae.init(key, key_dummy, specimen, True)
    tx = optax.adam(learning_rate)
    state = TrainState.create(
        apply_fn=vae.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )

    return state


@jax.jit
def train_step(state, key, image):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        elbo, new_model_state = state.apply_fn(
            variables, key, image, True, mutable=["batch_stats"]
        )
        return -elbo.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)

    state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )

    return state, loss


@jax.jit
def decode(state, Z):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    decoded = state.apply_fn(variables, Z, False, method=VAE.decode)

    return decoded


def mnist_demo():
    from torch import Generator
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from torchvision.datasets import MNIST

    beta = 1
    latent_dim = 20
    hidden_dims = (32, 64, 128, 256, 512)
    specimen = jnp.empty((1, 32, 32, 1))
    flow_prior = True
    flow_num_coupling_layers = 8
    flow_num_bins = 4

    target_epoch = 2
    batch_size = 256
    lr = 1e-3

    transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    mnist_train = MNIST(
        "/tmp/torchvision", train=True, download=True, transform=transform
    )
    generator = Generator().manual_seed(42)
    loader = DataLoader(mnist_train, batch_size, shuffle=True, generator=generator)

    key = jax.random.PRNGKey(42)
    state = create_train_state(
        key,
        beta,
        latent_dim,
        hidden_dims,
        specimen,
        flow_prior,
        flow_num_coupling_layers,
        flow_num_bins,
        lr,
    )

    for epoch in range(target_epoch):
        loss_train = 0
        for X, _ in loader:
            image = jnp.array(X).reshape((-1, *specimen.shape))
            key, key_Z = jax.random.split(key)
            state, loss = train_step(state, key_Z, image)
            loss_train += loss

        print(f"Epoch {epoch + 1}: train loss {loss_train}")


if __name__ == "__main__":
    mnist_demo()
