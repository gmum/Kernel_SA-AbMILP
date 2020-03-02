"""Pytorch Dataset object that loads 32x32 patches that contain single cells."""

import os
import random

import scipy.io
import numpy as np

from PIL import Image
from skimage import io, color
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

import utils_augemntation


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
            self.bag_list_test, self.labels_list_test = self.create_bags(self.dir_list_test)

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):
        dirs = [x[0] for x in os.walk(path)]
        dirs.pop(0)
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]

        return dir_list_train, dir_list_test

    def create_bags(self, dir_list):
        bag_list = []
        labels_list = []
        for dir in dir_list:
            # Get image name
            img_name = dir.split('/')[-1]

            # bmp to pillow
            img_dir = dir + '/' + img_name + '.bmp'
            img = io.imread(img_dir)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            if self.location_info:
                xs = np.arange(0, 500)
                xs = np.asarray([xs for i in range(500)])
                ys = xs.transpose()
                img = np.dstack((img, xs, ys))

            # crop malignant cells
            dir_epithelial = dir + '/' + img_name + '_epithelial.mat'
            with open(dir_epithelial, 'rb') as f:
                mat_epithelial = scipy.io.loadmat(f)

            cropped_cells_epithelial = []
            for (x,y) in mat_epithelial['detection']:
                x = np.round(x)
                y = np.round(y)

                if self.data_augmentation:
                    x = x + np.round(np.random.normal(0, 3, 1))
                    y = y + np.round(np.random.normal(0, 3, 1))

                if x < 13:
                    x_start = 0
                    x_end = 27
                elif x > 500 - 14:
                    x_start = 500 - 27
                    x_end = 500
                else:
                    x_start = x - 13
                    x_end = x + 14

                if y < 13:
                    y_start = 0
                    y_end = 27
                elif y > 500 - 14:
                    y_start = 500 - 27
                    y_end = 500
                else:
                    y_start = y - 13
                    y_end = y + 14

                cropped_cells_epithelial.append(img[int(y_start):int(y_end), int(x_start):int(x_end)])

            # crop all other cells
            dir_inflammatory = dir + '/' + img_name + '_inflammatory.mat'
            dir_fibroblast = dir + '/' + img_name + '_fibroblast.mat'
            dir_others = dir + '/' + img_name + '_others.mat'

            with open(dir_inflammatory, 'rb') as f:
                mat_inflammatory = scipy.io.loadmat(f)
            with open(dir_fibroblast, 'rb') as f:
                mat_fibroblast = scipy.io.loadmat(f)
            with open(dir_others, 'rb') as f:
                mat_others = scipy.io.loadmat(f)

            all_coordinates = np.concatenate((mat_inflammatory['detection'], mat_fibroblast['detection'], mat_others['detection']), axis=0)

            cropped_cells_others = []
            for (x, y) in all_coordinates:
                x = np.round(x)
                y = np.round(y)

                if self.data_augmentation:
                    x = x + np.round(np.random.normal(0, 3, 1))
                    y = y + np.round(np.random.normal(0, 3, 1))

                if x < 13:
                    x_start = 0
                    x_end = 27
                elif x > 500 - 14:
                    x_start = 500 - 27
                    x_end = 500
                else:
                    x_start = x - 13
                    x_end = x + 14

                if y < 13:
                    y_start = 0
                    y_end = 27
                elif y > 500 - 14:
                    y_start = 500 - 27
                    y_end = 500
                else:
                    y_start = y - 13
                    y_end = y + 14

                cropped_cells_others.append(img[int(y_start):int(y_end), int(x_start):int(x_end)])

            # generate bag
            bag = cropped_cells_epithelial + cropped_cells_others

            # store single cell labels
            labels = np.concatenate((np.ones(len(cropped_cells_epithelial)), np.zeros(len(cropped_cells_others))), axis=0)

            # shuffle
            if self.shuffle_bag:
                zip_bag_labels = list(zip(bag, labels))
                random.shuffle(zip_bag_labels)
                bag, labels = zip(*zip_bag_labels)

            # append every bag two times if training
            if self.train:
                for _ in [0,1]:
                    bag_list.append(bag)
                    labels_list.append(labels)
            else:
                bag_list.append(bag)
                labels_list.append(labels)

            # bag_list.append(bag)
            # labels_list.append(labels)

        return bag_list, labels_list

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
            label = [max(self.labels_list_train[index]), self.labels_list_train[index]]
        else:
            bag = self.bag_list_test[index]
            label = [max(self.labels_list_test[index]), self.labels_list_test[index]]

        return self.transform_and_data_augmentation(bag), label


class ColonCancerWhole(data_utils.Dataset):
    def __init__(self, path, train_val_idxs=None, test_idxs=None, train=True):
        self.path = path
        self.train_val_idxs = train_val_idxs
        self.test_idxs = test_idxs
        self.train = train

        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])

        self.dir_list_train, self.dir_list_test = self.split_dir_list(self.path, self.train_val_idxs, self.test_idxs)
        if self.train:
            self.img_list_train, self.labels_list_train, self.coordinates_train = self.create_bags(self.dir_list_train)
        else:
            self.img_list_test, self.labels_list_test, self.coordinates_test = self.create_bags(self.dir_list_test)

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):
        dirs = [x[0] for x in os.walk(path)]
        dirs.pop(0)
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]

        return dir_list_train, dir_list_test

    @staticmethod
    def create_bags(dir_list):
        img_list = []
        labels_list = []
        coordinate_list = []

        for dir in dir_list:
            # Get image name
            img_name = dir.split('/')[-1]

            # bmp to pillow
            img_dir = dir + '/' + img_name + '.bmp'
            with open(img_dir, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')

            # crop malignant cells
            dir_epithelial = dir + '/' + img_name + '_epithelial.mat'
            with open(dir_epithelial, 'rb') as f:
                mat_epithelial = scipy.io.loadmat(f)

            # crop all other cells
            dir_inflammatory = dir + '/' + img_name + '_inflammatory.mat'
            dir_fibroblast = dir + '/' + img_name + '_fibroblast.mat'
            dir_others = dir + '/' + img_name + '_others.mat'

            with open(dir_inflammatory, 'rb') as f:
                mat_inflammatory = scipy.io.loadmat(f)
            with open(dir_fibroblast, 'rb') as f:
                mat_fibroblast = scipy.io.loadmat(f)
            with open(dir_others, 'rb') as f:
                mat_others = scipy.io.loadmat(f)

            benign_coordinates = np.concatenate((mat_inflammatory['detection'].astype(float), mat_fibroblast['detection'].astype(float), mat_others['detection'].astype(float)), axis=0)
            all_coordinates = np.concatenate((mat_epithelial['detection'].astype(float), mat_inflammatory['detection'].astype(float), mat_fibroblast['detection'].astype(float), mat_others['detection'].astype(float)), axis=0)

            # store single cell labels
            labels = np.concatenate((np.ones(len(mat_epithelial['detection'])), np.zeros(len(benign_coordinates))), axis=0)

            img_list.append(img)
            labels_list.append(labels)
            coordinate_list.append(all_coordinates)

        return img_list, labels_list, coordinate_list

    def __len__(self):
        if self.train:
            return len(self.labels_list_train)
        else:
            return len(self.labels_list_test)

    def __getitem__(self, index):
        if self.train:
            img = self.to_tensor_transform(self.img_list_train[index])
            label = [max(self.labels_list_train[index]), self.labels_list_train[index]]
            coordinates = self.coordinates_train[index]
        else:
            img = self.to_tensor_transform(self.img_list_test[index])
            label = [max(self.labels_list_test[index]), self.labels_list_test[index]]
            coordinates = self.coordinates_test[index]

        return img, label, coordinates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from kfold_indices_warwick import kfold_indices_warwick

    to_pil = transforms.Compose([transforms.ToPILImage()])

    kwargs = {}
    batch_size = 1

    train_folds, test_folds = kfold_indices_warwick(100, 10, seed=2)
    train_fold, val_fold = kfold_indices_warwick(len(train_folds[0]), 10, seed=2)
    train_fold = [train_folds[0][i] for i in train_fold]
    val_fold = [train_folds[0][i] for i in val_fold]

    path = './Classification/'

    train_loader = data_utils.DataLoader(ColonCancerBagsCross(path,
                                                              train_val_idxs=train_fold[0],
                                                              test_idxs=test_folds[0],
                                                              train=True,
                                                              shuffle_bag=True,
                                                              data_augmentation=True),
                                         batch_size=1,
                                         shuffle=True,
                                         **kwargs)

    cancer_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        print(label[0][0])
        if label[0][0] == 1.0:
            cancer_bags_train += 1
        if batch_idx in [0, 1]:
            plot_data = bag.squeeze(0)
            for i in range(100):
                # print(label[1][i])
                plt.subplot(10, 10, i + 1)
                plt.imshow(to_pil(plot_data[i, :, :, :]))
            plt.show()

    print('Number of malignant bags in train set {}/{}'.format(cancer_bags_train, len(train_loader)))

    val_loader = data_utils.DataLoader(ColonCancerBagsCross(path,
                                                              train_val_idxs=val_fold[0],
                                                              test_idxs=test_folds[0],
                                                              train=True,
                                                              shuffle_bag=True,
                                                              data_augmentation=True),
                                         batch_size=1,
                                         shuffle=True,
                                         **kwargs)

    cancer_bags_val = 0
    for batch_idx, (bag, label) in enumerate(val_loader):
        print(label[0][0])
        if label[0][0] == 1.0:
            cancer_bags_val += 1
            # if batch_idx in [0, 1]:
            #     plot_data = bag.squeeze(0)
            #     for i in range(20):
            #         # print(label[1][i])
            #         plt.subplot(5, 6, i + 1)
            #         plt.imshow(to_pil(plot_data[i, :, :, :]))
            #     plt.show()

    print('Number of malignant bags in val set {}/{}'.format(cancer_bags_val, len(val_loader)))

    test_loader = data_utils.DataLoader(ColonCancerBagsCross(path,
                                                            train_val_idxs=val_fold[0],
                                                            test_idxs=test_folds[0],
                                                            train=False,
                                                            shuffle_bag=False,
                                                            data_augmentation=False),
                                       batch_size=1,
                                       shuffle=False,
                                       **kwargs)

    cancer_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        print(label[0][0])
        if label[0][0] == 1.0:
            cancer_bags_test += 1
            # if batch_idx in [0, 1]:
            #     plot_data = bag.squeeze(0)
            #     for i in range(20):
            #         # print(label[1][i])
            #         plt.subplot(5, 6, i + 1)
            #         plt.imshow(to_pil(plot_data[i, :, :, :]))
            #     plt.show()

    print('Number of malignant bags in test set {}/{}'.format(cancer_bags_test, len(test_loader)))
