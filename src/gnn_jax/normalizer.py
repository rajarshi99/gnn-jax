import flax.linen as nn
import jax
import jax.numpy as jnp

class Normalizer(nn.Module):
    """
    Online normalizer based on Welford's algo
    Accumulates statistics from DATA
    No trainable parameters
    """

    feature_dim: int
    eps: float = 1e-8
    max_accumulations: int = 1_000_000

    def setup(self):
        self.count = self.variable(
                "stats", "count",
                lambda: jnp.array(0, dtype=jnp.int64)
                )
        self.mean = self.variable(
                "stats", "mean",
                lambda: jnp.zeros(self.feature_dim, dtype=jnp.float64)
                )
        self.M2 = self.variable(
                "stats", "M2",
                lambda: jnp.zeros(self.feature_dim, dtype=jnp.float64)
                )
        self.std = self.variable(
                "stats", "std",
                lambda: jnp.ones(self.feature_dim, dtype=jnp.float64)
                )


    # ------------------------------------------------------------------
    # statistics update (DATA ONLY)
    # ------------------------------------------------------------------

    def accumulate(self, x: jnp.ndarray):
        """
        CHECK THE LOGIC ONCE AGAIN
        Accumulate statistics from data.

        x: [..., feature_dim]
        returns number of features accumulated
        """
        if x.shape[-1] != self.feature_dim:
            raise ValueError("Feature dimension mismatch")

        # flatten all leading dims
        x_flat = x.reshape(-1, self.feature_dim)

        n = x_flat.shape[0]

        # stop accumulating if max reached
        remaining = jnp.maximum(self.max_accumulations - self.count.value, 0)
        n_batch = jnp.minimum(n, remaining)
        # Not possible inside JIT x_batch = x_flat[:n_batch]
        mask = (jnp.arange(n) < n_batch)
        x_batch_padded = x_flat * mask[:,None]

        # Calculate new count
        count_new = self.count.value + n_batch
        # assert count_new > 0

        # Calculate new mean
        mean_batch = jnp.sum(x_batch_padded, axis=0) / n_batch
        delta = mean_batch - self.mean.value
        mean_new = self.mean.value + delta*n_batch/count_new

        # Calculate new M2
        diff_padded = x_batch_padded - mean_batch
        M2_batch = jnp.sum(diff_padded*diff_padded, axis=0)
        M2_new = self.M2.value + M2_batch + delta*delta * self.count.value*n_batch/count_new

        # Calculate new std
        var = M2_new/count_new
        std_new = jnp.sqrt(jnp.maximum(var,self.eps))

        self.count.value = count_new
        self.mean.value = mean_new
        self.M2.value = M2_new
        self.std.value = std_new

        return n_batch

    # ------------------------------------------------------------------
    # transforms
    # ------------------------------------------------------------------

    def normalize(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize using current statistics.
        """
        return (x - self.mean.value) / self.std.value

    def denormalize(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse transform back to physical space.
        """
        return z * self.std.value + self.mean.value
