import numpy as np
import torch
import random
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import os
import cv2
from scipy import ndimage
from typing import Callable, Optional
import Config as config
import warnings
import logging
from sklearn.model_selection import train_test_split
import shutil


logger = logging.getLogger("Load_Dataset")
if not logger.handlers:
    logger.setLevel(logging.INFO)


def correct_dims(image: np.ndarray, mask: np.ndarray):

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)
    if image.shape[2] != 3:

        if image.shape[2] > 3:
            image = image[:, :, :3]
        else:
            image = np.repeat(image[:, :, :1], 3, axis=2)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    image = np.transpose(image, (2, 0, 1))


    if mask.ndim == 3:
        mask = np.squeeze(mask)
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, 0)

    return image, mask


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_h, self.output_w = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']


        p = random.random()
        if p < 0.5:
            image, label = random_rot_flip(image, label)
        elif p < 0.8:
            image, label = random_rotate(image, label)


        if image.shape[0] != self.output_h or image.shape[1] != self.output_w:
            image = cv2.resize(image, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)
        if label.shape[0] != self.output_h or label.shape[1] != self.output_h:
            label = cv2.resize(label, (self.output_w, self.output_h), interpolation=cv2.INTER_NEAREST)


        image, label = correct_dims(image, label)

        if getattr(config, "debug_aug", False):

            print(f"[Aug] image: {tuple(image.shape)}, label: {tuple(label.shape)}, "
                  f"label.max: {label.max():.1f}")

        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).float()}


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_h, self.output_w = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if image.shape[0] != self.output_h or image.shape[1] != self.output_w:
            image = cv2.resize(image, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)
        if label.shape[0] != self.output_h or label.shape[1] != self.output_h:
            label = cv2.resize(label, (self.output_w, self.output_h), interpolation=cv2.INTER_NEAREST)
        image, label = correct_dims(image, label)
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).float()}


class ImageToImage2D(Dataset):
    def __init__(self,
                 dataset_path: str,
                 joint_transform: Optional[Callable] = None,
                 one_hot_mask: int = 0,
                 image_size: int = 256,
                 n_labels: int = 1,
                 val_size: float = 0.2) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.image_size = image_size
        self.n_labels = n_labels
        self.one_hot_mask = one_hot_mask
        self.joint_transform = joint_transform
        self.val_size = val_size

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Image path does not exist: {self.input_path}")
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Label path does not exist: {self.output_path}")

        self.images_list = [f for f in os.listdir(self.input_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

        if len(self.images_list) == 0:
            raise ValueError(f"No valid image files found in {self.input_path}")

        logger.info(f"Found {len(self.images_list)} image files")
        print(f"Found {len(self.images_list)} image files")


        val_img_path = os.path.join(os.path.dirname(dataset_path), 'Val_Folder', 'img')


        if os.path.basename(dataset_path) == 'Train_Folder':
            if not os.path.exists(val_img_path) or len(os.listdir(val_img_path)) == 0:
                print("Validation set is missing or empty; sampling a validation split from the training set")

                train_images, val_images = train_test_split(self.images_list, test_size=self.val_size)

                val_img_dir = os.path.join(os.path.dirname(dataset_path), 'Val_Folder', 'img')
                val_label_dir = os.path.join(os.path.dirname(dataset_path), 'Val_Folder', 'labelcol')
                os.makedirs(val_img_dir, exist_ok=True)
                os.makedirs(val_label_dir, exist_ok=True)
                for img in val_images:
                    shutil.copy(os.path.join(self.input_path, img), val_img_dir)
                    mask = self._read_mask(os.path.splitext(img)[0])
                    if mask is not None:
                        cv2.imwrite(os.path.join(val_label_dir, os.path.splitext(img)[0] + '.png'), mask)
                self.images_list = train_images
                print(f"Copied {len(val_images)} images to the validation directory")
            else:
                print(f"Validation directory already exists and contains valid images:{val_img_path}")
        else:

            if len(self.images_list) == 0:
                print(f"Warning: {dataset_path} dataset path is empty or contains no files")
            else:
                print(f"Dataset directory already exists and contains valid images:{self.input_path}")

    def __len__(self):
        return len(self.images_list)

    def _read_mask(self, base_name: str):

        for ext in ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            p = os.path.join(self.output_path, base_name + ext)
            if os.path.exists(p):
                return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return None

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image_path = os.path.join(self.input_path, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base = os.path.splitext(image_filename)[0]
        mask = self._read_mask(base)
        if mask is None:
            raise ValueError(f"Mask not found for image {image_filename}")


        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        sample = {'image': image, 'label': mask}
        if self.joint_transform is not None:
            sample = self.joint_transform(sample)
        else:

            to_tensor = T.ToTensor()
            image_t = to_tensor(image)
            mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
            sample = {'image': image_t, 'label': mask_t}
        return sample
