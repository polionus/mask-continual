import jax


def make_key_gen(key: int):

    while True: 
        key, subkey = jax.random.split(key)
        yield subkey
