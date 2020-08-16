import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
from . import imgProcessing
from . import MPII as mpii

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        heatmaps = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue
                    ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                    br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    heatmaps[idx, aa:bb, cc:dd] = np.maximum(heatmaps[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return heatmaps


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
        return self.loadImageAndGt(self.index[idx % len(self.index)])    #get the real idx for train or validation set

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

        #generate heatmaps
        heatmaps = self.generateHeatmap(keypoints)
        # check
        # print('kp', kps)
        # print('htm ', np.sum(heatmaps))
        return inp.astype(np.float32), heatmaps.astype(np.float32)

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

    train_index, valid_index = mpii.setup_val_split()
    #dataset = {key: MPIIDataLoader(config, mpii, index) for key, index in zip(['train', 'valid'], [train_index, valid_index])}

    trainset = MPIIDataLoader(config, mpii, train_index)
    validset = MPIIDataLoader(config, mpii, valid_index)

    return  trainset, validset
    #use_data_loader = config['train']['use_data_loader']

    # loaders = {}
    # for key in dataset:
    #     loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True,
    #                                                num_workers=config['train']['num_workers'], pin_memory=False)

    # def gen(phase):
    #     #batchsize = config['train']['batchsize']
    #     batchnum = config['train']['{}_iters'.format(phase)]
    #     loader = loaders[phase].__iter__()  #== enumerate(loaders[phase], 0):
    #     for i in range(batchnum):
    #         imgs, heatmaps = next(loader)
    #         yield {  # yield keyword make a function become a generator
    #             'imgs': imgs,  # cropped and augmented
    #             'heatmaps': heatmaps,  # based on keypoints. 0 if not in img for joint
    #         }
    #
    # return lambda key: gen(key)
