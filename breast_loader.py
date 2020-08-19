"""Pytorch Dataset object that loads 32x32 patches that contain single cells."""

import random

from skimage import io, color
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

import utils_augemntation
from skimage.util import view_as_blocks


class BreastCancerBagsCross(data_utils.Dataset):
    def __init__(self, path, train_val_idxs=None, test_idxs=None, train=True, shuffle_bag=False, data_augmentation=False, loc_info=False):
        self.path = path
        self.train_val_idxs = train_val_idxs
        self.test_idxs = test_idxs
        self.train = train
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.location_info = loc_info

        self.data_augmentation_img_transform = transforms.Compose([utils_augemntation.RandomHEStain(),
                                                                   utils_augemntation.HistoNormalize(),
                                                                   utils_augemntation.RandomRotate(),
                                                                   utils_augemntation.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                                                        (0.5, 0.5, 0.5))])

        self.normalize_to_tensor_transform = transforms.Compose([utils_augemntation.HistoNormalize(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])

        self.dir_list_train, self.dir_list_test = self.split_dir_list(self.path, self.train_val_idxs, self.test_idxs)
        if self.train:
            self.bag_list_train, self.labels_list_train = self.create_bags(self.dir_list_train)
        else:
            self.bag_list_test, self.labels_list_test, self.img_test = self.create_bags(self.dir_list_test)

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):
        import glob
        dirs = glob.glob(path + '*.tif')
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]

        return dir_list_train, dir_list_test

    def create_bags(self, dir_list):
        bag_list = []
        labels_list = []
        img_list = []
        for dir in dir_list:
            img = io.imread(dir)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            bag = view_as_blocks(img, block_shape=(32, 32, 3)).reshape(-1, 32, 32, 3)

            # store single cell labels
            label = 1 if 'malignant' in dir else 0

            # shuffle
            if self.shuffle_bag:
                random.shuffle(bag)

            bag_list.append(bag)
            labels_list.append(label)
            img_list.append(img)
        if self.train:
            return bag_list, labels_list
        else:
            return bag_list, labels_list, img_list

    def transform_and_data_augmentation(self, bag):
        if self.data_augmentation:
            img_transform = self.data_augmentation_img_transform
        else:
            img_transform = self.normalize_to_tensor_transform

        bag_tensors = []
        for img in bag:
            if self.location_info:
                bag_tensors.append(torch.cat(
                    (img_transform(img[:, :, :3]),
                    torch.from_numpy(img[:, :, 3:].astype(float).transpose((2, 0, 1))).float(),
)))
            else:
                bag_tensors.append(img_transform(img))
        return torch.stack(bag_tensors)

    def __len__(self):
        if self.train:
            return len(self.labels_list_train)
        else:
            return len(self.labels_list_test)

    def __getitem__(self, index):
        if self.train:
            bag = self.bag_list_train[index]
            label = self.labels_list_train[index]
            return self.transform_and_data_augmentation(bag), label
        else:
            bag = self.bag_list_test[index]
            label = self.labels_list_test[index]
            img = self.img_test[index]
            return self.transform_and_data_augmentation(bag), label, img

        return self.transform_and_data_augmentation(bag), label

