__config__ = {
    'nstack': 8,
    'inp_dim': 256,
    'oup_dim': 16,
    'num_parts': 16,
    'num_eval': 2958, ## number of val examples used. entire set is 2958
    'train_num_eval': 300, ## number of train examples tested at test time
    'batch_size': 16,
    'input_res': 256,
    'output_res': 64,
    'lr': 1e-3,
    'decay_iters': 100000,
    'decay_lr': 2e-4,

}