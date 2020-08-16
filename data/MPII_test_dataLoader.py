import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
from . import imgProcessing
from . import MPII as mpii
from .MPII_dataLoader import  GenerateHeatmap

class MPIIDataLoader(torch.utils.data.Dataset):
    def __init__(self, config, mpii, index):
        self.input_res = config['input_res']
        self.output_res = config['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['num_parts'])
        self.mpii = mpii
        self.index = index

    #must override for a torch.utils.data.Dataset
    def __len__(self):
        return len(self.index)

    #must override for a torch.utils.data.Dataset
    def __getitem__(self, idx):
        return self.loadImageAndGt(self.index[idx % len(self.index)])    #in case index exceeds

    #return orginal img & ground truth heatmaps
    def loadImageAndGt(self, idx):
        mpii = self.mpii

        ## load + crop
        orig_img = mpii.get_img(idx)
        path = mpii.get_path(idx)
        c = mpii.get_center(idx)
        s = mpii.get_scale(idx)
        normalize = mpii.get_normalized(idx)

        inp = imgProcessing.crop(orig_img, c, s, (self.input_res, self.input_res))
        '''
        for i in range(np.shape(keypoints)[1]):
            if keypoints[0, i, 0] > 0:
                keypoints[0, i, :2] = imgProcessing.transform(keypoints[0, i, :2], c, s,
                                                               (self.input_res, self.input_res))
        '''


        # get kps
        kps = mpii.get_kps(idx)
        #get center
        center = mpii.get_center(idx)
        #get scale
        scale = mpii.get_scale(idx)
        #get normalization
        norm = mpii.get_normalized(idx)
        #generate heatmaps
        heatmaps = self.generateHeatmap(kps)
        # check
        #print('kp', kps)
        #print('htm ', np.sum(heatmaps))
        return inp.astype(np.float32), kps.astype(np.float32), heatmaps.astype(np.float32), \
               center.astype(np.float32), scale.astype(np.float32), norm.astype(np.float32)


def init(config):
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    mpii.init()

    _, test_index = mpii.setup_val_split()
    #dataset = {key: MPIIDataLoader(config, mpii, index) for key, index in zip(['train', 'valid'], [train_index, valid_index])}

    testset = MPIIDataLoader(config, mpii, test_index)

    return testset