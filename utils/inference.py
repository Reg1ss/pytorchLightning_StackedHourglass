import cv2
import torch
import tqdm
import os
import numpy as np
import h5py
import copy

from utils.getCoordinates import HeatmapParser
import data.imgProcessing
import data.MPII as ds

parser = HeatmapParser()

def post_process(det, mat_, trainval, c=None, s=None, resolution=None):
    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
    cropped_preds = parser.parse(np.float32([det]))[0]
    #print('before ', cropped_preds)
    if len(cropped_preds) > 0:
        cropped_preds[:, :, :2] = data.imgProcessing.kpt_affine(cropped_preds[:, :, :2] * 4, mat)
    #print('after', cropped_preds)
    preds = np.copy(cropped_preds)
    # for inverting predictions from input res on cropped to original image
    if trainval != 'cropped':
        for j in range(preds.shape[1]):
            preds[0, j, :2] = data.imgProcessing.transform(preds[0, j, :2], c, s, resolution, invert=1)
    return preds


def inference(img, model, config, c, s):
    """
    forward pass at test time
    calls post_process to post process results
    """
    height, width = img.shape[1:3]
    center = (width / 2, height / 2)
    scale = max(height, width) / 200
    res = (config['input_res'], config['input_res'])

    mat_t = data.imgProcessing.get_transform(center, scale, res)[:2]
    inp = img / 255

    def array2dict(tmp):
        return {
            'det': tmp,  # [batch_size, nstack, n_joints, size, size]
        }

    tmp1 = array2dict(model(inp))  # inp: [batch_size, channel(3), size, size]
    tmp2 = array2dict(model(torch.flip(inp,[-2])))
    #print('inp', inp)
    #print('inpi', torch.flip(inp, [-2]))
    # tmp1 = array2dict(tmp1)
    # tmp2 = array2dict(tmp2)

    tmp = {}
    tmp1['det'] = tmp1['det'].cpu().numpy()
    tmp2['det'] = tmp2['det'].cpu().numpy()
    for ii in tmp1:
        tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]), axis=0)
    det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][ds.flipped_parts['mpii']]   #second -1 is the last prediction of the network
    if det is None:
        return [], []
    det = det / 2
    det = np.minimum(det, 1)

    return post_process(det, mat_t, 'valid', c, s, res)


def do_inference(img, model, config, c, s):
    ans = inference(img, model, config, c, s)
    if len(ans) > 0:
        ans = ans[:, :, :3]
    return [{'keypoints': ans}]


def mpii_eval(preds, gt_kps, normalizing, bound=0.5):
    """
    Use PCK with threshold of .5 of normalized distance (presumably head size)
    """

    correct = {'all': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                       'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
                       'shoulder': 0},
               'visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                           'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
                           'shoulder': 0},
               'not visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                               'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0,
                               'shoulder': 0}}
    count = copy.deepcopy(correct)
    correct_train = copy.deepcopy(correct)
    count_train = copy.deepcopy(correct)
    idx = 0
    for p, g, norm in zip(preds, gt_kps, normalizing):
        for j in range(g.shape[1]):
            vis = 'visible'
            if g[0, j, 0] == 0:  ## not in picture!
                continue
            if g[0, j, 2] == 0:
                vis = 'not visible'
            joint = 'ankle'
            if j == 1 or j == 4:
                joint = 'knee'
            elif j == 2 or j == 3:
                joint = 'hip'
            elif j == 6:
                joint = 'pelvis'
            elif j == 7:
                joint = 'thorax'
            elif j == 8:
                joint = 'neck'
            elif j == 9:
                joint = 'head'
            elif j == 10 or j == 15:
                joint = 'wrist'
            elif j == 11 or j == 14:
                joint = 'elbow'
            elif j == 12 or j == 13:
                joint = 'shoulder'


            count['all']['total'] += 1
            count['all'][joint] += 1
            count[vis]['total'] += 1
            count[vis][joint] += 1

            #compute distance
            error = np.linalg.norm(p[0]['keypoints'][0, j, :2] - g[0, j, :2]) / norm
            #print('p ', p[0])
            #print('g ', g)

            if bound > error:
                correct['all']['total'] += 1
                correct['all'][joint] += 1
                correct[vis]['total'] += 1
                correct[vis][joint] += 1
        idx += 1

    ## breakdown by validation set / training set
    for k in correct:
        print(k, ':')
        for key in correct[k]:
            print('Val PCK @,', bound, ',', key, ':', round(correct[k][key] / max(count[k][key], 1), 4), ', count:',
                  count[k][key])
            #print('Tra PCK @,', bound, ',', key, ':', round(correct_train[k][key] / max(count_train[k][key], 1), 3),
            #     ', count:', count_train[k][key])
        print('\n')

