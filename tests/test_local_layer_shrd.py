import jax

import catfish.lvd.models.dist_layers as dl
import catfish.lvd.models.dist_utils as du


def main():
    mesh_shape = (4,2,1)
    dist_manager = du.DistManager(mesh_shape)
    prng_key = dist_manager.get_key(42)

    shrd_linear = dl.ShrdLinear(dist_manager, prng_key, 64, 64)
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key,(64,))
    y = shrd_linear(x)
    print(y)


if __name__ == "__main__":
    main()
