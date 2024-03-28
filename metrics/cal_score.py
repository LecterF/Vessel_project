import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


class CalScore(object):
    def __init__(self, threshold=0.5, alpha=2, radius=2):
        self.threshold = threshold
        self.alpha = alpha
        self.radius = radius

    def count_connect_component(self, img):
        ret, binary = cv2.threshold(img, self.threshold, 1, cv2.THRESH_BINARY)
        if np.all(binary) ==0 or np.all(binary) == 1:
            return 0
        else:
            num_labels, labels = cv2.connectedComponents(binary)
            return num_labels - 1

    def set_device_number_type(self, img):
        if isinstance(img, torch.Tensor):
            if img.is_cuda:
                img = img.cpu()
            img = img.detach().numpy()
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise ValueError("The type of image is wrong!")
        img = np.where(img > self.threshold, 1, 0)
        if img.dtype == np.uint8:
            pass
        else:
            img = img.astype(np.uint8)

        return img

    def connectivity(self, preds, labels):
        connectnum_preds = self.count_connect_component(preds)
        connectnum_labels = self.count_connect_component(labels)
        pixel_num = np.sum(labels == 1)
        connect = 1 - min(1, abs(connectnum_preds - connectnum_labels) / pixel_num)
        return connect

    def area(self, preds, labels):
        preds_dilated = cv2.dilate(preds, np.ones((self.alpha, self.alpha), np.uint8))
        labels_dilated = cv2.dilate(labels, np.ones((self.alpha, self.alpha), np.uint8))
        intersection1 = np.logical_and(preds, labels_dilated)
        intersection2 = np.logical_and(preds_dilated, labels)
        intersection_union = np.sum(intersection1 + intersection2)
        union = np.sum(np.logical_or(preds, labels))
        area_score = intersection_union / union
        return area_score

    def length(self, preds, labels):
        preds_skeleton = skeletonize(preds)
        labels_skeleton = skeletonize(labels)
        preds_dilated = cv2.dilate(preds.astype(np.uint8), np.ones((self.radius, self.radius), np.uint8))
        labels_dilated = cv2.dilate(labels.astype(np.uint8), np.ones((self.radius, self.radius), np.uint8))
        intersection1 = np.logical_and(preds_skeleton, labels_dilated)
        intersection2 = np.logical_and(preds_dilated, labels_skeleton)
        intersection_union = np.sum(intersection1 + intersection2)
        union = np.sum(np.logical_or(preds_skeleton, labels_skeleton))
        length_score = intersection_union / union
        return length_score

    def cal_metric(self, preds, labels):
        preds = self.set_device_number_type(preds)
        labels = self.set_device_number_type(labels)
        connect = self.connectivity(preds, labels)
        area_score = self.area(preds, labels)
        length_score = self.length(preds, labels)
        return connect, area_score, length_score

    def cal_metric_batch(self, preds, labels):
        assert preds.shape == labels.shape
        B, H, W = preds.shape
        Connects = []
        Area_scores = []
        Length_scores = []
        batch_cal = []
        for i in range(B):
            connect, area_score, length_score = self.cal_metric(preds[i], labels[i])
            Connects.append(connect)
            Area_scores.append(area_score)
            Length_scores.append(length_score)
            cal = connect * area_score * length_score
            batch_cal.append(cal)
        Connects = np.array(Connects).mean()
        Area_scores = np.array(Area_scores).mean()
        Length_scores = np.array(Length_scores).mean()
        batch_cal = np.array(batch_cal).mean()

        return Connects, Area_scores, Length_scores, batch_cal


if __name__ == "__main__":
    cal = CalScore()
    pred_path = '/data/xiaoyi/Fundus/AFIO/all/mask_512_png/IM000001.png'
    label_path = '/data/xiaoyi/Fundus/HRF/all/mask_512_png/01_dr.png'
    pred = cv2.imread(pred_path, 0)
    label = cv2.imread(label_path, 0)
    pred = np.repeat(pred[np.newaxis, :, :], 3, axis=0)
    label = np.repeat(label[np.newaxis, :, :], 3, axis=0)
    cal = cal.cal_metric_batch(pred, label)
