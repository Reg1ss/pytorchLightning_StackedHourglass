from pytorch_lightning import Trainer
from model import poseNet, config
from pytorch_lightning.callbacks import EarlyStopping

this_config = config.__config__
inp_dim = this_config['inp_dim']
oup_dim = this_config['oup_dim']
net = poseNet.poseNet(inp_dim, oup_dim)

early_stopping = EarlyStopping('val_loss')
trainer = Trainer(gpus=1, default_root_dir='./checkpoint/test_hg1_02', early_stop_callback=early_stopping, log_gpu_memory=True)
trainer.test(ckpt_path='/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass/checkpoint/test_hg1_02/lightning_logs/version_3/checkpoints/epoch=10.ckpt')