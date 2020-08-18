from pytorch_lightning import Trainer
from model import poseNet
from pytorch_lightning.callbacks import EarlyStopping


model = poseNet.poseNet()
net = model.load_from_checkpoint('/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass'
                                                     '/checkpoint/test_hg1_04/lightning_logs/version_30/checkpoints/epoch=24.ckpt',
                                 hparams_file='/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass'
                                                     '/checkpoint/test_hg1_04/lightning_logs/version_30/hparams.yaml',)

trainer = Trainer(gpus=1)
trainer.test(net)