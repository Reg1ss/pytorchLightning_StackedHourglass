from pytorch_lightning import Trainer
from model import poseNet
from pytorch_lightning.callbacks import EarlyStopping


net = poseNet.poseNet()

early_stopping = EarlyStopping('val_loss')
trainer = Trainer(gpus=1, default_root_dir='./checkpoint/test_hg1_01',
                  resume_from_checkpoint='/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass/checkpoint/test_hg1_02/lightning_logs/version_3/checkpoints/epoch=10.ckpt',
                  early_stop_callback=early_stopping, log_gpu_memory=True)
trainer.fit(net)    #tensorboard --logdir lightning_logs/


