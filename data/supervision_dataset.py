import os
import glob
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
import albumentations.pytorch as AT

class SupervisionDataset(data.Dataset):
    def __init__(self, train=True, data_dir=r'data/ref', no_augment=True, aug_prob=0.5, img_size=(224, 224),
                 image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]):
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        self._set_filesystem()
        self._set_augmentations()

    def _set_augmentations(self):
        if self.aug:
            augmentation = [
                A.HorizontalFlip(p=self.aug_prob),
                A.VerticalFlip(p=self.aug_prob),
                # 添加更多的增强操作...
            ]
        else:
            augmentation = []

        self.img_transform = A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(mean=self.image_mean, std=self.image_std),
            AT.ToTensorV2()
        ])

        self.mask_transform = A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            AT.ToTensorV2()
        ])

        self.augmentation = A.Compose(augmentation)

    def _set_filesystem(self):
        self.ext = ('.tif', '.gif', '.png', '.jpg', '.jpeg')
        self.image_list = glob.glob(os.path.join(self.data_dir, '*'))

    def label2mask(self, label):
        """
        将label转换为mask
        :param label: (0-255)
        :return: [0,1,2,3,....]
        """
        label = label.astype(np.uint8)
        unique_values = np.unique(label)
        mask = np.searchsorted(unique_values, label)

        return mask.astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = img_path.replace('image', 'mask')
        filename = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample = self.augmentation(image=img,mask=label)
        img = sample['image']
        label = self.label2mask(sample['mask'])
        img = self.img_transform(image=img)['image']
        label = self.mask_transform(image=label)['image']

        return img, label, filename

    def __len__(self):
        return len(self.image_list)




