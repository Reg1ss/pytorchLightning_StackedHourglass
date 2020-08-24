import sys
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data
from data import MPII as mpii
from data import MPII_dataLoader

from model import poseNet_tch
from model import config_stu,config_tch

#  load tch net
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
mpii.init()
pred_tch = []
print("=> loading teacher network")
checkpoint = torch.load('/home/reg1s/PycharmProjects/pytorchLightning_StackedHourglass/checkpoint/run_tch/checkpoint.pt')
net_tch = poseNet_tch.poseNet().cuda()
this_config_stu = config_stu.__config__
this_config_tch = config_tch.__config__
# deal with key error
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[13:]  # delete 'model.module.'
    new_state_dict[name] = v
net_tch.load_state_dict(new_state_dict)
# init train set and validation set
mpii_train, mpii_validation = MPII_dataLoader.init(this_config_stu)

# train dataloader
mpii_which_dataloader = DataLoader(mpii_train, batch_size=this_config_tch['batch_size'], num_workers=8,
                                   shuffle=False)
print('(train set)Getting predictions from tch net')

# get training set prediction from teacher network
mpii_which_dataloader_iter = iter(mpii_which_dataloader)

#for i in (range(2)):
for i in (range(len(mpii_which_dataloader_iter))):
    if i % 100 == 0:
        print('have predicted ', i, ' batches')
    with torch.no_grad():
        input, heatmap = mpii_which_dataloader_iter.next()

        input_cuda = input.cuda()
        # get predictions
        combined_heatmap_preds_tch = net_tch(input_cuda)
        for j in range(combined_heatmap_preds_tch.shape[0]):
            # self.old_idxs[idx] = old_idx
            pred_tch.append(combined_heatmap_preds_tch[j, this_config_tch['nstack'] - 1].cpu().numpy().astype(np.float32))

        # validation dataloader
mpii_which_dataloader = DataLoader(mpii_validation, batch_size=this_config_tch['batch_size'], num_workers=8,
                                   shuffle=False)
print('(validation set)Getting predictions from tch net')
# get validation set prediction from teacher network
mpii_which_dataloader_iter = iter(mpii_which_dataloader)
for i in (range(len(mpii_which_dataloader_iter))):
#for i in (range(2)):
    if i % 20 == 0:
        print('have predicted ', i, ' batches')
    with torch.no_grad():
        input, heatmap = mpii_which_dataloader_iter.next()
        input_cuda = input.cuda()
        # get predictions
        combined_heatmap_preds_tch = net_tch(input_cuda)
        for j in range(combined_heatmap_preds_tch.shape[0]):
            # self.old_idxs[idx] = old_idx
            pred_tch.append(combined_heatmap_preds_tch[j, this_config_tch['nstack'] - 1].cpu().numpy().astype(np.float32))
pred_tch_array = np.array(pred_tch)
print('ll ', pred_tch_array.shape)
np.save('pred_tch.npy', arr=pred_tch_array)
