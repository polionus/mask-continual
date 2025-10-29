import jax 
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from rng import make_key_gen
import optax



from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log", mode="a")]
)



X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = jnp.array(y)

# One-hot encode for MLP
y_onehot = jax.nn.one_hot(y, 3)
X = jnp.array(X, dtype=jnp.float32)


WIDTH = 1024
DEPTH = 10


### Let's train a mask over a network and investigate:
key = jax.random.key(100)
keygen = make_key_gen(key)


def straight_through_mask(logits: jax.Array):

    soft = jax.nn.sigmoid(logits)
    hard = (soft > 0.5).astype(jnp.float32)

    return soft + jax.lax.stop_gradient(hard - soft)

class Model(eqx.Module):
    
    mask_logits: list[jax.Array]
    layers: list[eqx.nn.Linear]
    output_layer: eqx.nn.Linear



    def __init__(self, input_size: int, hidden_size: int):
        self.layers = [eqx.nn.Linear(input_size, WIDTH, key=next(keygen))]
        self.layers += [eqx.nn.Linear(WIDTH, WIDTH, key = next(keygen)) for _ in range(DEPTH - 2)]
        
        self.output_layer = eqx.nn.Linear(WIDTH, hidden_size, key = next(keygen))
        self.mask_logits = [jax.random.normal(key = next(keygen), shape = self.layers[index].out_features) 
                            for index in range(len(self.layers))]

    def __call__(self, x: jax.Array):
        
        for index in range(len(self.layers)):
            h = self.layers[index](x) 
            mask = straight_through_mask(self.mask_logits[index])
            x = jax.nn.tanh(h * mask)
           
        
        output = self.output_layer(x)
       
        return output
           


model = Model(4, 3)
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

##Mask everything: 
filter_spec = jtu.tree_map(lambda _: False, model)

filter_spec = eqx.tree_at(
    lambda tree: tree.mask_logits,                     # select the list of mask arrays
    filter_spec,
    replace=[True] * len(model.mask_logits),     # one boolean per mask array
)


@eqx.filter_jit
def make_step(model, x, y, opt_state):
    @eqx.filter_value_and_grad
    def loss_fn(diff_model, static_model, x, y):
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, y_onehot))

    diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = loss_fn(diff_model, static_model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
     
    
def train(model: eqx.Module, X: jax.Array, y: jax.Array, optimizer, opt_state, num_epochs: int): 
    # mask = jnp.ones((500, y.shape[1]))

    for epoch in range(num_epochs):
        model, opt_state, loss = make_step(model, X, y_onehot, opt_state)

        if epoch % 100 == 0:    
            logging.debug(f"Epoch: {epoch}, Loss: {loss} ")

    logging.debug(f"Final Loss: {loss} ")
    return model


logging.debug("Starting training...")
model = train(model, X, y, optimizer= optimizer, opt_state= opt_state, num_epochs=50000)
logging.debug("Finished Training")