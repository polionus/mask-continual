import jax 
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from rng import make_key_gen
import optax

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = jnp.array(y)

# One-hot encode for MLP
y_onehot = jax.nn.one_hot(y, 3)
X = jnp.array(X, dtype=jnp.float32)


WIDTH = 64
DEPTH = 10


### Let's train a mask over a network and investigate:
key = jax.random.key(100)
keygen = make_key_gen(key)


def straight_through_mask(logits: jax.Array):

    soft = jax.nn.sigmoid(logits)
    hard = (logits > 0.5).astype(jnp.float32)

    return soft + jax.lax.stop_gradient(hard - soft)

class Model(eqx.Module):
    
    layers: list[eqx.nn.Linear]
    output_layer: eqx.nn.Linear



    def __init__(self, input_size: int, hidden_size: int):
        self.layers = [eqx.nn.Linear(input_size, WIDTH, key=next(keygen))]
        self.layers += [eqx.nn.Linear(WIDTH, WIDTH, key = next(keygen)) for _ in range(DEPTH - 2)]
        
        self.output_layer = eqx.nn.Linear(WIDTH, hidden_size, key = next(keygen))
        # self.mask_logits = [jnp.ones(shape = self.layers[index].out_features) 
        #                     for index in range(len(self.layers))]

    def __call__(self, x: jax.Array):
        
        for index in range(len(self.layers)):
            x = jax.nn.relu((self.layers[index](x)))
            
        
        output = self.output_layer(x)
        return output
           

### Let's create a dataset: 

def loss_fn(model, X, y):
    y_hat = eqx.filter_vmap(model)(X)
    return jnp.mean((y - y_hat)**2)


model = Model(4, 3)
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

#### Let's create the mask for our model: 

##Mask everything: 

@eqx.filter_jit
def make_step(model, x, y, opt_state):
    @eqx.filter_value_and_grad
    def loss_fn(model, x, y):
        # model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x)
        return jnp.mean((y - pred_y) ** 2)

    # diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
     
    
def train(model: eqx.Module, X: jax.Array, y: jax.Array, optimizer, opt_state, num_epochs: int): 
    # mask = jnp.ones((500, y.shape[1]))

    for epoch in range(num_epochs):
        model, opt_state, loss = make_step(model, X, y_onehot, opt_state)

        print(f"Loss:{loss}")

    print(f"Final Loss:{loss}")
    return model


model = train(model, X, y, optimizer= optimizer, opt_state= opt_state, num_epochs=5000)
# print((jax.nn.sigmoid(model.mask_logits[0]) > 0.5).astype(jnp.float32))