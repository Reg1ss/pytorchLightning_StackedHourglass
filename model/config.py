__config__ = {
    'nstack': 1,
    'inp_dim': 128, #256
    'oup_dim': 16,
    'num_parts': 16,
    'num_eval': 2958, ## number of val examples used. entire set is 2958
    'batch_size': 8,
    'input_res': 128,   #256
    'output_res': 32,   #64
    'lr': 1e-5,
    'decay_iters': 100000,
    'decay_lr': 2e-4,
    'threshold':0.5,
    'checkpoint_path':'./checkpoint/test_save_02/'
}