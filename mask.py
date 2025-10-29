import jax 
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from rng import make_key_gen
import optax
import tyro


### What would be the output of the masks if we did trained for just a little bit?
### Would there be a lot of bang for our buck?


### Should we mask weights or activations?
### It is definitely possible to have lower loss with more parameters in the randomly initialized network.


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log", mode="a")]
)


def main(width = 1024, depth = 10, seed = 0, train_weights = False, num_spochs = 3):
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    y = jnp.array(y)

    # One-hot encode for MLP
    y_onehot = jax.nn.one_hot(y, 3)
    X = jnp.array(X, dtype=jnp.float32)


    ### Let's train a mask over a network and investigate:
    key = jax.random.key(seed)
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
            self.layers = [eqx.nn.Linear(input_size, width, key=next(keygen))]
            self.layers += [eqx.nn.Linear(width, width, key = next(keygen)) for _ in range(depth - 2)]
            
            self.output_layer = eqx.nn.Linear(width, hidden_size, key = next(keygen))
            self.mask_logits = [jax.random.normal(key = next(keygen), shape = self.layers[index].out_features) 
                                for index in range(len(self.layers))]

        ### Use jax.lax.scan to make it faster.
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
    filter_spec = jtu.tree_map(lambda _: train_weights, model)

    filter_spec = eqx.tree_at(
        lambda tree: tree.mask_logits,                     # select the list of mask arrays
        filter_spec,
        replace=[not train_weights] * len(model.mask_logits),     # one boolean per mask array
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
                logging.info(f"Epoch: {epoch}, Loss: {loss} ")

        logging.info(f"Final Loss: {loss} ")
        return model


    logging.info("Starting training...")
    model = train(model, X, y, optimizer= optimizer, opt_state= opt_state, num_epochs=num_epochs)
    logging.info("Finished Training")

    ### Save model:
    eqx.tree_serialise_leaves("model.eqx", model)

tyro.cli(main)
