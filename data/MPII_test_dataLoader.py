import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
from . import imgProcessing
from . import MPII as mpii

class MPIIDataLoader(torch.utils.data.Dataset):
    def __init__(self, config, mpii, index):
        self.input_res = config['input_res']
        self.output_res = config['output_res']
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
        orig_keypoints = mpii.get_kps(idx)
        kptmp = orig_keypoints.copy()
        c = mpii.get_center(idx)
        s = mpii.get_scale(idx)
        normalize = mpii.get_normalized(idx)

        cropped = imgProcessing.crop(orig_img, c, s, (self.input_res, self.input_res))
        for i in range(np.shape(orig_keypoints)[1]):
            if orig_keypoints[0, i, 0] > 0:
                orig_keypoints[0, i, :2] = imgProcessing.transform(orig_keypoints[0, i, :2], c, s,
                                                               (self.input_res, self.input_res))
        keypoints = np.copy(orig_keypoints)

        ## augmentation -- to be done to cropped image
        height, width = cropped.shape[0:2]
        center = np.array((width / 2, height / 2))
        scale = max(height, width) / 200

        aug_rot = 0

        aug_rot = (np.random.random() * 2 - 1) * 30.
        aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        scale *= aug_scale

        mat_mask = imgProcessing.get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]  #Generate transformation matrix

        mat = imgProcessing.get_transform(center, scale, (self.input_res, self.input_res), aug_rot)[:2]
        inp = cv2.warpAffine(cropped, mat, (self.input_res, self.input_res)).astype(np.float32) / 255 #cv2.WarpAffine: applying affine transformation
        keypoints[:, :, 0:2] = imgProcessing.kpt_affine(keypoints[:, :, 0:2], mat_mask) #keypoints affine to output_res
        if np.random.randint(2) == 0:
            inp = self.preprocess(inp)
            inp = inp[:, ::-1]
            keypoints = keypoints[:, mpii.flipped_parts['mpii']]
            keypoints[:, :, 0] = self.output_res - keypoints[:, :, 0]
            orig_keypoints = orig_keypoints[:, mpii.flipped_parts['mpii']]
            orig_keypoints[:, :, 0] = self.input_res - orig_keypoints[:, :, 0]

        ## set keypoints to 0 when were not visible initially (so heatmap all 0s)
        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0, i, 0] == 0 and kptmp[0, i, 1] == 0:
                keypoints[0, i, 0] = 0
                keypoints[0, i, 1] = 0
                orig_keypoints[0, i, 0] = 0
                orig_keypoints[0, i, 1] = 0
        heatmaps = self.generateHeatmap(keypoints)

        # get kps
        kps = mpii.get_kps(idx)
        #get center
        center = mpii.get_center(idx)
        #get scale
        scale = mpii.get_scale(idx)
        #get normalization
        norm = mpii.get_normalized(idx)
        return inp.astype(np.float32), kps.astype(np.float32), heatmaps.astype(np.float32), \
               center.astype(np.float32), scale.astype(np.float32), norm.astype(np.float32)

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:, :, 0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:, :, 1] = np.maximum(np.minimum(data[:, :, 1], 1), 0)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data


def init(config):
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    mpii.init()

    _, test_index = mpii.setup_val_split()
    #dataset = {key: MPIIDataLoader(config, mpii, index) for key, index in zip(['train', 'valid'], [train_index, valid_index])}

    testset = MPIIDataLoader(config, mpii, test_index)

    return testset