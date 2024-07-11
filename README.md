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
- [ ] Text conditional diffusion Transformer
- 6-D parallelism
   - [ ] FSDP
   - [ ] Ring attention
   - [ ] Pipeline parallelism
   - [ ] Async swarm
- [ ] Llama 3 support
- [ ] Sophisticated logging (Logfire/SQL database)


