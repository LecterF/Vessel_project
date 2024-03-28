import cv2
import numpy as np
import os
import torch

class Connectivity(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def count_connect_componen(self, img):
        """
        :param img: array
        :return:
        """
        # img = cv2.imread(img_path, 0)
        ret, binary = cv2.threshold(img, self.threshold, 1, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary)
        return num_labels-1

    def set_channel(self, img):
        """
        :param img: array
        :return:
        """
        if len(img.shape) == 4:
            img = np.squeeze(img, axis=1)
        elif len(img.shape) == 3:
            pass
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            raise ValueError("The shape of image is wrong!")
        return img
    def set_number(self, img):
        """
        :param img: array (0-255)
        :return: img: array(0,1)
        """
        img = np.where(img > 0.5, 1, 0)
        return img
    def set_type(self, img):
        """
        :param img: array
        :return:
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise ValueError("The type of image is wrong!")

        if img.dtype == np.uint8:
            pass
        else:
            img = img.astype(np.uint8)

        return img
    def connectivity_matrix(self, preds, labels):
        """
        :param preds: classification results after softmax (N,H,W)  (0,1)
        :param labels: pixel-level label without dilated (N,H,W)  (0,1)
        :return: the number of recalled calcification and FP
        """
        preds, labels = self.set_channel(preds), self.set_channel(labels)
        preds, labels = self.set_number(preds), self.set_number(labels)
        preds, labels = self.set_type(preds), self.set_type(labels)
        assert preds.shape == labels.shape
        B, H, W = preds.shape
        Connect = 0.0

        for i in range(preds.shape[0]):
            connectnum_pred = self.count_connect_componen(preds[i, :, :])
            connectnum_label = self.count_connect_componen(labels[i, :, :])

            pixel_num = np.sum(labels==1)
            connect_per_image = 1 - min(1, abs(connectnum_pred - connectnum_label) / pixel_num)
            Connect += connect_per_image
        Connect = Connect / B
        return Connect







if __name__ == '__main__':
    pred_path = '/data/xiaoyi/Fundus/AFIO/all/mask_512_png/IM000001.png'
    label_path = '/data/xiaoyi/Fundus/HRF/all/mask_512_png/01_dr.png'
    pred = cv2.imread(pred_path, 0)
    label = cv2.imread(label_path, 0)
    pred = np.repeat(pred[np.newaxis, :, :], 3, axis=0)
    label = np.repeat(label[np.newaxis, :, :], 3, axis=0)
    connect = Connectivity()
    print(connect.connectivity_matrix(pred, label))