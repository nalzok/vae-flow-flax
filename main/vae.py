from typing import Tuple, Sequence, Optional
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import distrax
import optax
from torch import Generator
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

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
        *_, H, W, C = self.output_dim

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


class VAE(nn.Module):
    beta: float
    latent_dim: int
    hidden_dims: Sequence[int]
    output_dim: Tuple[int, int, int]

    flow_location: Optional[str]
    flow_num_coupling_layers: int
    flow_hidden_dims: Sequence[int]
    flow_num_bins: int

    def setup(self):
        if self.flow_location not in (None, "posterior", "prior"):
            raise ValueError(f"Unknown flow location {self.flow_location}")

        self.encoder = Encoder(self.latent_dim, self.hidden_dims)
        self.decoder = Decoder(self.output_dim, self.hidden_dims)
        self.epsilon = distrax.MultivariateNormalDiag(
            loc=jnp.zeros((self.latent_dim,)), scale_diag=jnp.ones((self.latent_dim,))
        )
        self.flow = Flow(
            self.latent_dim,
            self.flow_hidden_dims,
            self.flow_num_coupling_layers,
            self.flow_num_bins,
        )
        if self.flow_location == "prior":
            self.prior = distrax.Transformed(self.epsilon, self.flow.bijector)
        else:
            self.prior = self.epsilon

    def __call__(self, key, X, training):
        mean, logvar = self.encoder(X, training)
        posterior = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(0.5 * logvar)
        )

        if self.flow_location == "posterior":
            posterior = distrax.Transformed(posterior, self.flow.bijector)

        Z, posterior_log_prob = posterior.sample_and_log_prob(seed=key, sample_shape=())

        recon = self.decoder(Z, training)

        log_prob = -self.beta * jnp.sum((recon - X) ** 2)
        kl_divergence = -(self.prior.log_prob(Z) - posterior_log_prob)
        elbo = log_prob - kl_divergence

        return elbo, recon

    def decode(self, epsilon, training):
        if self.flow_location == "prior":
            Z = self.flow(epsilon)
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
    flow_location: Optional[str],
    flow_num_coupling_layers: int,
    flow_hidden_dims: Sequence[int],
    flow_num_bins: int,
    learning_rate: float,
):
    vae = VAE(
        beta,
        latent_dim,
        hidden_dims,
        specimen.shape,
        flow_location,
        flow_num_coupling_layers,
        flow_hidden_dims,
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


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def train_step(state, key, image):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        (elbo, recon), new_model_state = state.apply_fn(
            variables, key, image, True, mutable=["batch_stats"]
        )
        return -elbo.sum(), (new_model_state, recon)

    (loss, (new_model_state, recon)), grads = loss_fn(state.params)
    loss = jax.lax.psum(loss, axis_name="batch")
    grads = jax.lax.psum(grads, axis_name="batch")

    state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )

    return state, loss, recon


@jax.pmap
def decode(state, Z):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    decoded = state.apply_fn(variables, Z, False, method=VAE.decode)

    return decoded


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def cross_replica_mean(batch_stats):
    return jax.lax.pmean(batch_stats, "batch")


def save(
    images: np.ndarray,
    flow_location: Optional[str],
    plot_every: int,
    title: str,
    identifier: str,
):
    npy_name = Path(f"results/npy/{identifier}_{flow_location}.npy")
    npy_name.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_name, images)

    pdf_name = Path(f"results/pdf/{identifier}_{flow_location}.pdf")
    png_name = Path(f"results/png/{identifier}_{flow_location}.png")
    pdf_name.parent.mkdir(parents=True, exist_ok=True)
    png_name.parent.mkdir(parents=True, exist_ok=True)

    nrows, ncols, *_ = images.shape
    plt.box(False)
    fig, axes = plt.subplots(
        nrows, ncols, constrained_layout=True, figsize=plt.figaspect(1)
    )
    for row, images_row in enumerate(images):
        for col, image in enumerate(images_row):
            ax = axes[row, col]
            ax.imshow(image.reshape(32, 32), cmap="gray")
            if col == 0:
                ax.set_ylabel(
                    f"{(row + 1) * plot_every - 1}",
                    rotation="horizontal",
                    ha="right",
                    va="center",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{title} ({flow_location})", fontsize="xx-large")
    plt.savefig(pdf_name)
    plt.savefig(png_name)
    plt.close()


def fit_vae(
    beta: float,
    latent_dim: int,
    hidden_dims: Sequence[int],
    specimen: jnp.ndarray,
    flow_location: Optional[str],
    flow_num_coupling_layers: int,
    flow_hidden_dims: Sequence[int],
    flow_num_bins: int,
    target_epoch: int,
    batch_size: int,
    learning_rate: float,
    device_count: int,
    loader: DataLoader,
):
    key = jax.random.PRNGKey(42)
    state = create_train_state(
        key,
        beta,
        latent_dim,
        hidden_dims,
        specimen,
        flow_location,
        flow_num_coupling_layers,
        flow_hidden_dims,
        flow_num_bins,
        learning_rate,
    )
    state = flax.jax_utils.replicate(state)

    images_shape = (device_count, device_count, *specimen.shape[1:-1])
    orig_images = np.zeros(images_shape)
    recon_images = np.zeros(images_shape)
    generated_images = np.zeros(images_shape)

    plot_every = target_epoch // device_count
    for epoch in range(target_epoch):
        elbo_epoch = 0
        for X, _ in loader:
            image = jnp.array(X).reshape((device_count, -1, *specimen.shape[1:]))
            key, *key_Z = jax.random.split(key, device_count + 1)
            key_Z = jnp.array(key_Z)
            state, loss, recon = train_step(state, key_Z, image)
            elbo_epoch += -flax.jax_utils.unreplicate(loss)

            if (epoch + 1) % plot_every == 0:
                orig_images[epoch // plot_every] = image[:, -1, ..., 0]
                recon_images[epoch // plot_every] = recon[:, -1, ..., 0]

        # Sync the batch statistics across replicas so that evaluation is deterministic.
        state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))

        print(f"Epoch {epoch + 1}: ELBO {elbo_epoch / batch_size}")

        key, key_Z = jax.random.split(key)
        Z = jax.random.normal(key_Z, (device_count, latent_dim))
        generated_image = decode(state, Z)
        generated_images[epoch // plot_every] = generated_image.reshape(
            images_shape[1:]
        )

    save(orig_images, flow_location, plot_every, "Original", "orig")
    save(recon_images, flow_location, plot_every, "Reconstructed", "recon")
    save(generated_images, flow_location, plot_every, "Generated", "gen")


if __name__ == "__main__":
    beta = 1
    latent_dim = 20
    hidden_dims = (32, 64, 128, 256, 512)
    specimen = jnp.empty((1, 32, 32, 1))
    flow_num_coupling_layers = 32
    flow_hidden_dims = (1024, 1024, 1024)
    flow_num_bins = 16

    target_epoch = 512
    batch_size = 256
    learning_rate = 1e-4

    device_count = jax.local_device_count()
    if target_epoch % device_count != 0:
        raise ValueError(f"target_epoch should be divisible by {device_count}")
    if batch_size % device_count != 0:
        raise ValueError(f"batch_size should be divisible by {device_count}")

    transform = T.Compose([T.Resize(specimen.shape[1:-1]), T.ToTensor()])
    mnist_train = MNIST(
        "/tmp/torchvision", train=True, download=True, transform=transform
    )
    generator = Generator().manual_seed(42)
    loader = DataLoader(mnist_train, batch_size, shuffle=True, generator=generator)

    for flow_location in (None, "prior", "posterior"):
        print(f"{flow_location = }")
        fit_vae(
            beta,
            latent_dim,
            hidden_dims,
            specimen,
            flow_location,
            flow_num_coupling_layers,
            flow_hidden_dims,
            flow_num_bins,
            target_epoch,
            batch_size,
            learning_rate,
            device_count,
            loader,
        )
