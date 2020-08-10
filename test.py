from pytorch_lightning import Trainer
from model import poseNet
from pytorch_lightning.callbacks import EarlyStopping


model = poseNet.poseNet()
net = model.load_from_checkpoint('/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass'
                                                     '/checkpoint/test_hg1_02/lightning_logs/version_3/checkpoints/epoch=10.ckpt')

early_stopping = EarlyStopping('val_loss')
trainer = Trainer(gpus=1, default_root_dir='./checkpoint/test_hg1_02', early_stop_callback=early_stopping, log_gpu_memory=True)
trainer.test(net)
#trainer.test(net)