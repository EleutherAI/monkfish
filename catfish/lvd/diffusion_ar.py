def sample_datapoint(data, key):
    prompt_key, completion_key = jax.random.split(key)
    
    prompt_dist, completion_dist = data
    prompt_sample = lvd.vae.sample_gaussian(prompt_dist, prompt_key)
    completion_sample = lvd.vae.sample_gaussian(prompt_dist,completion_key)
    return prompt_sample,completion_sample

def sample(args, cfg):
    n_samples = cfg["dt"]["sample"]["n_sample"]
    n_steps = cfg["dt"]["sample"]["n_steps"]
    n_latent = cfg["lvm"]["n_latent"]
    l_x = cfg["dt"]["l_x"]
    l_y = cfg["dt"]["l_y"]

    with jax.default_device(jax.devices("cpu")[0]):

        vae_state = lvd.utils.load_checkpoint(args.vae_checkpoint)
        trained_vae = vae_state[0]
        m_encoder, m_decoder = map(lambda x: jax.vmap(jax.vmap(x)), trained_vae)

        dt_state = lvd.utils.load_checkpoint(args.diffusion_checkpoint)
        trained_dt = dt_state[0]


        key = jax.random.PRNGKey(cfg["seed"])

        data_key, dt_sample_key, encode_sample_key, decode_sample_key = jax.random.split(key, 4)

        with lvd.latent_dataset.LatentDataset(data_directory=args.data_dir, 
            batch_size=n_samples, prompt_length=l_x, completion_length=l_y) as ld:
            prompt_samples, completion_samples = sample_datapoint(next(ld), data_key)

        latent_continuations = sample_diffusion(prompt_samples, trained_dt, f_neg_gamma, dt_sample_key, n_steps, completion_samples.shape[1:])

        continuation_frames = lvd.vae.sample_gaussian(m_decoder(latent_continuations), decode_sample_key)
        print(continuation_frames.shape)
        
        for sample in continuation_frames:
            print(sample.shape)
            lvd.utils.show_samples(sample)

def train(args, cfg):
    key = jax.random.PRNGKey(cfg["seed"])
    ckpt_dir = cfg["dt"]["train"]["ckpt_dir"]
    lr = cfg["dt"]["train"]["lr"]
    ckpt_interval = cfg["dt"]["train"]["ckpt_interval"]
    latent_paths = cfg["dt"]["train"]["data_dir"]
    batch_size = cfg["dt"]["train"]["bs"]
    clip_norm = cfg["dt"]["train"]["clip_norm"]
    metrics_path = cfg["dt"]["train"]["metrics_path"]

    n_layers = cfg["dt"]["n_layers"]
    d_io = cfg["lvm"]["n_latent"]
    d_l = cfg["dt"]["d_l"]
    d_mlp = cfg["dt"]["d_mlp"]
    n_q = cfg["dt"]["n_q"]
    d_qk = cfg["dt"]["d_qk"]
    d_dv = cfg["dt"]["d_dv"]
    l_x = cfg["dt"]["l_x"]
    l_y = cfg["dt"]["l_y"]

    adam_optimizer = optax.adam(lr)
    optimizer = optax.chain(adam_optimizer, optax.zero_nans(), optax.clip_by_global_norm(clip_norm))
    loss_fn = lambda a, b, c: diffusion_loss(a, b, f_neg_gamma, c)
    
    if args.checkpoint is None:
        key = jax.random.PRNGKey(cfg["seed"])
        init_key, state_key = jax.random.split(key)
        model = diffusion_transformer.LatentVideoTransformer(init_key, n_layers, d_io, d_l, d_mlp, n_q, d_qk, d_dv)
        opt_state = optimizer.init(model)
        i = 0
        state = model, opt_state, state_key, i
    else:
        checkpoint_path = args.checkpoint
        state = lvd.utils.load_checkpoint(checkpoint_path)
    
    with open(metrics_path,"w") as f:
        #TODO: Fix LatentDataset RNG
        with lvd.latent_dataset.LatentDataset(data_directory=args.data_dir, 
            batch_size=batch_size, prompt_length=l_x, completion_length=l_y) as ld:
            for _ in lvd.utils.tqdm_inf():
                data = sample_datapoint(next(ld),state[2])
                loss, state = lvd.utils.update_state(state, data, optimizer, loss_fn)
                f.write(f"{loss}\n")
                f.flush()
                iteration = state[3]
                if (iteration % ckpt_interval) == (ckpt_interval - 1):
                    ckpt_path = lvd.utils.ckpt_path(ckpt_dir, iteration+1, "dt")
                    lvd.utils.save_checkpoint(state, ckpt_path)