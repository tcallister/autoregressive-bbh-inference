from numpyro.distributions.distribution import Distribution
import jax
import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
    validate_sample,
)

class TransformedUniform(Distribution):

    arg_constraints = {"low": constraints.real, "high": constraints.real}
    support = constraints.real
    reparameterized_params = ["low","high"]

    def __init__(self, low=0.0, high=1.0, *, validate_args=None):

        self.low, self.high = promote_shapes(low, high)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)

        # Sample on the reals from a normal distribution
        logit_sample_unscaled = jax.random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
            )
        logit_sample = 2.5*logit_sample_unscaled

        # Transform to variable bounded on (low,high)
        exp_logit = jnp.exp(logit_sample)
        x = (exp_logit*self.high + self.low)/(1.+exp_logit)

        return x
        
    @validate_sample
    def log_prob(self,value):

        # Jacobian from unbounded logit variable to bounded sample
        dlogit_dx = 1./(value-self.low) + 1./(self.high-value)

        # Subtract Gaussian log-prob and apply Jacobian
        logit_value = jnp.log((value-self.low)/(self.high-value))
        return logit_value**2/(2.*2.5**2)-jnp.log(dlogit_dx)

