import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Sequence, Callable

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP)

    Parameters
    - layer_sizes: list of integers
        Output sizes of each of the layers, including final
    - activations: list of functions
        To be applied at each layer including final

    Note:
        Input size is fixed at init from the #cols of dummy input x
        model = MLP(layer_sizes = [32, 3], activations = [nn.relu]*2)
        params = model.init(key, x)
    """
    layer_sizes: Sequence[int]
    activations: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]

    @nn.compact
    def __call__(self, x:jnp.ndarray):
        diff_in_lens = len(self.layer_sizes) - len(self.activations)
        if diff_in_lens > 0:
            activations = self.activations + (jax.nn.identity,)*diff_in_lens
        else:
            activations = self.activations

        for act, lsize in zip(activations, self.layer_sizes):
            x = act(nn.Dense(features=lsize)(x))

        return x
