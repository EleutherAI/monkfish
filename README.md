# Monkfish: Distributed latent video model training on TPUs (and other stuff maybe)

This is the training code for a 2 stage autoregressive video model. Code works for training latents. 2nd stage is a WIP.

# Running on a single TPU
```
 python -m monkfish.main.main config.json local [args...]
```

# Running distributed from a head node:
```
 ray start --head --num-cpus=1 --port=6379
 PROJECT_SOURCE=path/to/monkfish python -m monkfish.main.main config.json distributed [args...]
 ray stop
```


# References For Developers

Parameter scaling: 
 - [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813)
 - [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872) (NTK init with global LR is used for most experiments)

Jax sharding: 
 - [Using JAX in multi-host and multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html)

Data loader Design: 
 - [Distributed data loading in a multi-host/multi-process environment](https://jax.readthedocs.io/en/latest/distributed_data_loading.html)

