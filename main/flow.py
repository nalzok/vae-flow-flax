"""A simple example of a flow model trained on MNIST."""

from typing import Sequence, List, Callable, Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax


def make_conditioner(
    event_shape: Sequence[int], hidden_sizes: Sequence[int], num_bijector_params: int
) -> nn.Module:
    """Creates an MLP conditioner for each layer of the flow."""
    layers: List[Callable[..., Any]] = [
        lambda x: x.reshape((-1, *x.shape[-len(event_shape) :]))
    ]

    for hidden_size in hidden_sizes:
        layers.append(nn.Dense(hidden_size))
        layers.append(nn.relu)

    # We initialize this linear layer to zero so that the flow is initialized
    # to the identity function.
    layers.append(
        nn.Dense(
            np.prod(event_shape) * num_bijector_params,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )
    )
    layers.append(
        lambda x: x.reshape(
            (
                *x.shape[:-3],
                *event_shape,
                num_bijector_params,
            )
        )
    )

    return nn.Sequential(layers)


class Flow(nn.Module):
    latent_dim: int
    hidden_dims: Sequence[int]
    num_coupling_layers: int
    num_bins: int

    def setup(self):
        """Creates the flow model."""
        # Alternating binary mask.
        event_shape = (self.latent_dim,)
        mask = jnp.arange(0, np.prod(event_shape)) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)

        def bijector_fn(params: jnp.ndarray):
            return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters.
        num_bijector_params = 3 * self.num_bins + 1

        layers = []
        for _ in range(self.num_coupling_layers):
            layer = distrax.MaskedCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=make_conditioner(
                    event_shape, self.hidden_dims, num_bijector_params
                ),
            )
            layers.append(layer)
            # Flip the mask after each layer.
            mask = jnp.logical_not(mask)

        self.flow: distrax.Bijector = distrax.Chain(layers)

    def __call__(self, X):
        if len(X.shape) == 0:
            X = X[jnp.newaxis, ...]
        return self.flow.forward(X)


if __name__ == "__main__":
    latent_dim = 20
    hidden_dims = (32, 64, 128, 256, 512)
    flow_num_coupling_layers = 8
    flow_num_bins = 4

    key = jax.random.PRNGKey(42)
    vae = Flow(
        latent_dim,
        hidden_dims,
        flow_num_coupling_layers,
        flow_num_bins,
    )
    Z_dummy = jnp.empty((1, latent_dim))
    variables = vae.init(key, Z_dummy)
