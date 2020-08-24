import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model.layers import Conv, Hourglass, Pool, Residual
from loss.calc_loss import Calc_loss
from data import MPII_dataLoader, MPII_test_dataLoader
from model import config_tch
from utils.inference import do_inference, mpii_eval

from model import test_save_model


class poseNet(pl.LightningModule):  # not pl.core.LightningModule!!!

    def __init__(self, bn=False, **kwargs):
        super(poseNet, self).__init__()

        self.hparams.num_workers = 8  # workers number
        self.this_config = config_tch.__config__
        inp_dim = self.this_config['inp_dim']
        oup_dim = self.this_config['oup_dim']
        self.nstack = self.this_config['nstack']
        self.threshold = self.this_config['threshold']
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),  # res: 128*128 if input_res=256
            Residual(64, 128),
            Pool(2, 2),  # res:64*64 the max res in the network
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        # it has to use ModuleList, because different module has different parameters
        self.hgs = nn.ModuleList([
                                     nn.Sequential(
                                         Hourglass(4, inp_dim, bn)
                                     ) for num in range(self.nstack)
                                     ])

        '''
        self.afterHg = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            )for num in range(self.nstack)
        ])

        self.pred_heatmaps = nn.ModuleList([
            Conv(inp_dim, oup_dim, 1, bn=False, relu=False) for num in range(self.nstack)
        ])

        self.module_outs = nn.ModuleList([
            Conv(inp_dim, inp_dim, 1, bn=False, relu=False) for num in range(self.nstack-1)
        ])

        self.merge_pred_heatmaps = nn.ModuleList([
            Conv(oup_dim, inp_dim, 1, bn=False, relu=False) for num in range(self.nstack-1)
        ])
        '''

        self.features = nn.ModuleList([
                                          nn.Sequential(
                                              Residual(inp_dim, inp_dim),
                                              Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
                                          ) for i in range(self.nstack)])
        from model.layers import Merge
        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(self.nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(self.nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(self.nstack - 1)])

        self.calc_loss = Calc_loss(self.nstack)

        # for test phase
        self.all_gt_kps = []
        self.all_preds = []
        self.all_norm = []

    def forward(self, imgs):
        input = imgs.permute(0, 3, 1, 2)  ##swap order [batch_size,channel,size,size]
        input = self.pre(input)
        '''
        all_pred_heatmaps = []  # nstack*16*size*size
        for i in range(self.nstack):
            hg = self.hgs[i](input)
            afterHg = self.afterHg[i](hg)
            pred_heatmaps = self.pred_heatmaps[i](afterHg) # 16*size*size
            all_pred_heatmaps.append(pred_heatmaps)
            if i < self.nstack - 1:
                input += self.module_outs[i](afterHg) + self.merge_pred_heatmaps[i](pred_heatmaps)

        return torch.stack(all_pred_heatmaps, dim=1) #lists are stacked to become a tensor; dim=0:所有个tensor直接拼起来；dim=1，所有tensor第一个维度拼起来作为第一个维度...
        '''
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](input)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)  # [batch_size, njoints, out_size, out_size]
            # print('pred:',preds.shape)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                input = input + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)


