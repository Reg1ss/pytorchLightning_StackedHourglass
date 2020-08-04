from pytorch_lightning import Trainer
from model import poseNet, config

this_config = config.__config__
nstack = this_config['nstack']
inp_dim = this_config['inp_dim']
oup_dim = this_config['oup_dim']
batch_size = this_config['batch_size']
net = poseNet.poseNet(nstack, inp_dim, oup_dim, batch_size)


trainer = Trainer(gpus=1,default_root_dir='./checkpoint/test_hg1_01')
trainer.fit(net)


