from pytorch_lightning import Trainer
from model import poseNet
from pytorch_lightning.callbacks import EarlyStopping


net = poseNet.poseNet()

early_stopping = EarlyStopping('val_loss',patience=3)
trainer = Trainer(gpus=1, default_root_dir='./checkpoint/run_hg1_04',
                  early_stop_callback=early_stopping, log_gpu_memory=True)
trainer.fit(net)    #tensorboard --logdir lightning_logs/


