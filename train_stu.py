import torch
from pytorch_lightning import Trainer
from model import poseNet_tch,poseNet_stu
from pytorch_lightning.callbacks import EarlyStopping

'''
#load tch net
print("=> loading teacher network")
checkpoint = torch.load('./checkpoint/run_tch/checkpoint.pt')
net_tch = poseNet_tch.poseNet()
#deal with key error
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[13:]   #delete 'model.module.'
    new_state_dict[name] = v
net_tch.load_state_dict(new_state_dict)
'''

#train stu net
#net_stu = poseNet_stu.poseNet()
net_stu = poseNet_stu.poseNet()
early_stopping = EarlyStopping('val_loss',patience=5)
trainer = Trainer( gpus=1, default_root_dir='./checkpoint/run_KD_0.7_01',
                  early_stop_callback=early_stopping, log_gpu_memory=True)
trainer.fit(net_stu)    #tensorboard --logdir lightning_logs/


