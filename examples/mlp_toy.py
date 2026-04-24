"""
Toy example: Train a simple MLP on y = sin(x)

This script is demonstrates:
    - MLP import from gnn_jax package
    - Training procedure in JAX ecosystem
"""
import jax
import jax.numpy as jnp

from flax import linen as nn
import optax

from gnn_jax.mlp import MLP

key = jax.random.key(0)

N = 256 # No. of data points
key, subkey = jax.random.split(key)

# Toy training data
x_data = jax.random.uniform(subkey, (N, 1), minval=-jnp.pi, maxval=jnp.pi)
y_data = jnp.sin(x_data)

# Toy validation data
x_valid = jnp.linspace(-jnp.pi, jnp.pi).reshape(-1,1)
y_valid = jnp.sin(x_valid)

# Initilised model and its parameters
model = MLP(layer_sizes = [32]*3 + [1], activations = [nn.relu]*3 + [lambda x:x])
key, subkey = jax.random.split(key)
params = model.init(subkey, x_data)

# Initilised optimizer with learning rate
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# More mathematical but less useful in our context
# def loss_fn(y_pred):
#     return jnp.mean((y_pred - y_data)**2)

# Training and Validation loss on toy data
def train_loss_fn(params):
    y_pred = model.apply(params, x_data)
    return jnp.mean((y_pred - y_data)**2)
def valid_loss_fn(params):
    y_pred = model.apply(params, x_valid)
    return jnp.mean((y_pred - y_valid)**2)


@jax.jit
def train_step(params, opt_state):
    loss, grads = jax.value_and_grad(train_loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

# Training process
for step in range(2000):
    params, opt_state, loss = train_step(params, opt_state)
    if step % 100 == 0:
        train_loss = train_loss_fn(params)
        valid_loss = valid_loss_fn(params)
        print(f"train step {step:4d} | loss = {train_loss:.6f} | validation_loss = {valid_loss:.6f}")

# Printing out the shapes of the params
print("Final loss = {loss:.6f}")
print("The params have the following labels and shapes")
print(jax.tree_util.tree_map(lambda x: x.shape, params))

# Some fun with ood points
x_ood = jnp.array([f*jnp.pi for f in [1, 1.5, 2]]).reshape(-1,1)
y_ood = model.apply(params, x_ood)
print(f"For out-of-distribution points: {x_ood}")
print(f"Model predictions: {y_ood}")
