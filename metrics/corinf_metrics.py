import numpy as np
from skimage import morphology
from skimage import graph
from PIL import Image
from random import randint
import torch

import numpy as np
from skimage import morphology
from skimage.graph import route_through_array
from random import randint

class CorInf(object):
    def __init__(self, thresh=0.5, n_paths=1000):
        self.thresh = thresh
        self.n_paths = n_paths

    def set_device_number_type(self, img):
        if isinstance(img, torch.Tensor):
            if img.is_cuda:
                img = img.cpu()
            img = img.detach().numpy()
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise ValueError("The type of image is wrong!")
        img = np.where(img > self.thresh, 1, 0)
        if img.dtype == np.uint8:
            pass
        else:
            img = img.astype(np.uint8)

        return img
    def compute_single(self, gt, pred):
        # 0, 1 and 2 mean, respectively, that path is infeasible, shorter/larger and correct
        result = []
        gt, pred = self.set_device_number_type(gt), self.set_device_number_type(pred)
        if np.sum(gt) == 0 or np.sum(pred) == 0:
            return self.n_paths/self.n_paths, 0, 0
        else:
            # binarize pred according to thresh
            pred_cc = morphology.label(pred)

            # get centerlines of gt and pred
            gt_cent = morphology.skeletonize(gt)
            gt_cent_cc = morphology.label(gt_cent)
            pred_cent = morphology.skeletonize(pred)
            pred_cent_cc = morphology.label(pred_cent)

            # costs matrices
            gt_cost = np.ones(gt_cent.shape)
            gt_cost[gt_cent == 0] = 10000
            pred_cost = np.ones(pred_cent.shape)
            pred_cost[pred_cent == 0] = 10000

            # build graph and find shortest paths
            for i in range(self.n_paths):
                # pick randomly a first point in the centerline
                R_gt_cent, C_gt_cent = np.where(gt_cent == 1)
                idx1 = randint(0, len(R_gt_cent) - 1)
                label = gt_cent_cc[R_gt_cent[idx1], C_gt_cent[idx1]]
                ptx1 = (R_gt_cent[idx1], C_gt_cent[idx1])

                # pick a second point that is connected to the first one
                R_gt_cent_label, C_gt_cent_label = np.where(gt_cent_cc == label)
                idx2 = randint(0, len(R_gt_cent_label) - 1)
                ptx2 = (R_gt_cent_label[idx2], C_gt_cent_label[idx2])

                # if points have different labels in pred image, no path is feasible
                if (pred_cc[ptx1] != pred_cc[ptx2]) or pred_cc[ptx1] == 0:
                    result.append(0)

                else:
                    # find corresponding centerline points in pred centerlines
                    R_pred_cent, C_pred_cent = np.where(pred_cent == 1)
                    poss_corr = np.zeros((len(R_pred_cent), 2))
                    poss_corr[:, 0] = R_pred_cent
                    poss_corr[:, 1] = C_pred_cent
                    poss_corr = np.transpose(np.asarray([R_pred_cent, C_pred_cent]))
                    dist2_ptx1 = np.sum((poss_corr - np.asarray(ptx1)) ** 2, axis=1)
                    dist2_ptx2 = np.sum((poss_corr - np.asarray(ptx2)) ** 2, axis=1)
                    corr1 = poss_corr[np.argmin(dist2_ptx1)]
                    corr2 = poss_corr[np.argmin(dist2_ptx2)]

                    # find shortest path in gt and pred
                    gt_path, cost1 = route_through_array(gt_cost, ptx1, ptx2)
                    gt_path = np.asarray(gt_path)

                    pred_path, cost2 = route_through_array(pred_cost, corr1, corr2)
                    pred_path = np.asarray(pred_path)

                    # compare paths length
                    path_gt_length = np.sum(np.sqrt(np.sum(np.diff(gt_path, axis=0) ** 2, axis=1)))
                    path_pred_length = np.sum(np.sqrt(np.sum(np.diff(pred_path, axis=0) ** 2, axis=1)))
                    if pred_path.shape[0] < 2:
                        result.append(2)
                    else:
                        if ((path_gt_length / path_pred_length) < 0.9) or ((path_gt_length / path_pred_length) > 1.1):
                            result.append(1)
                        else:
                            result.append(2)

            return result.count(0)/self.n_paths, result.count(1)/self.n_paths, result.count(2)/self.n_paths

    def compute_batch(self, gt, pred):
        n_infeasible = []
        n_shorter = []
        n_correct = []

        for i in range(gt.shape[0]):

            infeasible, shorter, correct = self.compute_single(gt[i], pred[i])
            n_infeasible.append(infeasible)
            n_shorter.append(shorter)
            n_correct.append(correct)
        return np.mean(n_infeasible), np.mean(n_shorter), np.mean(n_correct)

if __name__ == '__main__':
    # load gt and pred
    SrcVessels = np.array(Image.open("/data/xiaoyi/Fundus/STARE/train/mask_512_png/im0001.png"))
    RefVessels = SrcVessels.copy()
    SrcVessels = np.repeat(SrcVessels[np.newaxis, :, :], 3, axis=0)
    RefVessels = np.repeat(RefVessels[np.newaxis, :, :], 3, axis=0)

    # get metrics
    n_paths = 100
    thresh = 0.5
    topo_metric = CorInf(thresh, n_paths)
    n_infeasible, n_shorter, n_correct = topo_metric.compute_batch(SrcVessels, RefVessels)
    print(f'Number of infeasible paths: {n_infeasible}')
    print(f'Number of shorter paths: {n_shorter}')
    print(f'Number of correct paths: {n_correct}')