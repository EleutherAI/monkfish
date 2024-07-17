# Monkfish: Distributed latent video model training on TPUs (and other stuff maybe)

This is the training code for a 2 stage autoregressive video model.

# TODO:
- [ ] Chunked scatter/gather/init functions
- [ ] Parallel model save/load
- [x] Dtype conversions at scatter/gather/init functions
- [x] Distributed data loading 
- [ ] Distributed model training
- [x] Multi-platform file backend via PyFilesystem2
- [ ] GPU Support
- [ ] SLURM Support
- [ ] Kubernetes Support
- [ ] Text conditional diffusion Transformer
- 6-D parallelism
   - [ ] FSDP
   - [ ] Ring attention
   - [ ] Pipeline parallelism
   - [ ] Async swarm
- [ ] Llama 3 support
- [ ] Sophisticated logging (Logfire/SQL database)

# References For Developers

Parameter scaling: 
 - [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813)
 - [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872) (NTK init with global LR is used for most experiments)

Jax sharding: 
 - [Using JAX in multi-host and multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html)

Data loader Design: 
 - [Distributed data loading in a multi-host/multi-process environment](https://jax.readthedocs.io/en/latest/distributed_data_loading.html)
