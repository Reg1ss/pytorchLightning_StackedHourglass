from pytorch_lightning import Trainer
from model import poseNet
from pytorch_lightning.callbacks import EarlyStopping


model = poseNet.poseNet()
net = model.load_from_checkpoint('/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass'
                                                     '/checkpoint/test_hg1_03/lightning_logs/version_0/checkpoints/epoch=2.ckpt',
                                 hparams_file='/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass'
                                                     '/checkpoint/test_hg1_03/lightning_logs/version_0/hparams.yaml',)

trainer = Trainer(gpus=1)
trainer.test(net)