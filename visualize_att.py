"""Loads the test set two times: for testing and for visualization. For every image z's are computed. All z's are
scaled. Each patch is multiplied with it's correcponding z."""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

from colon_loader import ColonCancerWhole
from colon_loader import ColonCancerBagsCross
from utils import kfold_indices_warwick

from torch.autograd import Variable

import copy

to_pil = transforms.Compose([transforms.ToPILImage()])

path = './Classification/'

#snapshot_path = './snapshots/att_no_kernel_2020-02-16 01:10:44/'  #'./snapshots/att_inv_q_spatial_2020-02-15 18:31:46/'
#snapshot_path = './snapshots/att_inv_q_spatial_2020-02-15 18:31:46/'
snapshot_path = './snapshots/att_gauss_spatial_2020-02-15 20:12:26/'
#snapshot_path = './snapshots/CNN_2020-02-01 12:22:58/'
#model_name = 'att_no_kernel'
#model_name = 'att_inv_q_spatial'
model_name = 'att_gauss_spatial'
model_name = 'CNN'
args = torch.load(snapshot_path + model_name + '_fold1' + '.config')
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
self_att = True

# EVAL FOR EVERY FOLD===================================================================================================
train_folds, test_folds = kfold_indices_warwick(args.dataset_size, args.kfold_test, args.seed)

for current_fold in range(4, 5):
    print('folds:' + str(current_fold))

    # Load datasets
    test_set = ColonCancerBagsCross(
        path,
        train_val_idxs=train_folds[current_fold - 1],
        test_idxs=test_folds[current_fold - 1],
        train=False,
        shuffle_bag=False,
        data_augmentation=False,
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    whole_img_set = ColonCancerWhole(
        path,
        train_val_idxs=train_folds[current_fold - 1],
        test_idxs=test_folds[current_fold - 1],
        train=False,
    )

    whole_img_set_loader = torch.utils.data.DataLoader(whole_img_set, batch_size=1, shuffle=False, **kwargs)

    # Get all the data
    whole_imgs = []
    whole_labels = []
    whole_coordinates = []
    for batch_idx, (data_whole, labels_whole, coordinates_whole) in enumerate(whole_img_set_loader):
        data_whole = data_whole[0].squeeze(0)
        data_whole = data_whole.numpy()
        data_whole = np.rollaxis(data_whole, 0, 3)
        whole_imgs.append(data_whole)
        whole_labels.append(labels_whole)
        whole_coordinates.append(coordinates_whole.squeeze(0).numpy())


    # Load model and set to eval mode
    trained_model_path = snapshot_path + model_name + '_fold' + str(current_fold)
    # args = torch.load(trained_model_path + '.config')
    model = torch.load(trained_model_path + '.models')
    model.eval()

    # Predict
    for batch_idx, (data, target) in enumerate(test_loader):
        ### PREDICT
        target = target[0]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        y_prob, y_hat, z, A, _, __ = model.forward(data)

        if self_att:
            A, self_attention = A

        print(target.cpu().data.numpy()[0], y_hat.cpu().data.numpy()[0])

        z_numpy = z.cpu().data.numpy() 
        z_norm = (z_numpy - np.min(z_numpy)) / (np.max(z_numpy) - np.min(z_numpy))

        self_attention_numpy = self_attention.cpu().data.numpy()
        self_attention_norm = \
            (self_attention_numpy - np.min(self_attention_numpy)) / (np.max(self_attention_numpy) - np.min(self_attention_numpy))

        ### PLOT
        img = copy.copy(whole_imgs[batch_idx])

        # All cells create and apply mask
        mask_all = np.zeros((500, 500, 3))

        for i, (x, y) in enumerate(whole_coordinates[batch_idx]):
            x = np.round(x)
            y = np.round(y)

            if x < 13:
                x_start = 0
                x_end = 27
            elif x > 500 - 13:
                x_start = 500 - 27
                x_end = 500
            else:
                x_start = x - 13
                x_end = x + 14

            if y < 13:
                y_start = 0
                y_end = 27
            elif y > 500 - 13:
                y_start = 500 - 27
                y_end = 500
            else:
                y_start = y - 13
                y_end = y + 14

            mask_all[int(y_start):int(y_end), int(x_start):int(x_end), :] = 1

        img_masked = img * mask_all

        # Epi cells create and apply mask
        mask_epi = np.zeros((500, 500, 3))

        epi_or_not = whole_labels[batch_idx][1].squeeze(0).numpy()
        masked_coordinates = whole_coordinates[batch_idx] * epi_or_not[:, None]

        for i, (x, y) in enumerate(masked_coordinates):
            if x:
                x = np.round(x)
                y = np.round(y)

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

                mask_epi[int(y_start):int(y_end), int(x_start):int(x_end), :] = 1

        img_epi = img * mask_epi

        # All cells create and apply mask
        mask_attention = np.zeros((500, 500, 3))
        mask_self_attention = np.zeros((500, 500, 3))
        mask_true = np.zeros((500, 500, 3))

        sorted_coordinates = whole_coordinates[batch_idx].tolist()
        z_norm = z_norm.tolist()
        self_attention_norm = self_attention_norm.tolist()

        #z_norm, sorted_coordinates = (list(t) for t in zip(*sorted(zip(z_norm, sorted_coordinates))))
        #self_attention_norm, _ = \
        #    (list(t) for t in zip(*sorted(zip(self_attention_norm, whole_coordinates[batch_idx].tolist()))))

        z_norm = np.array(z_norm)[0]
        print(z_norm.shape)
        order = np.argsort(z_norm)
        z_norm = z_norm[order]  

        self_attention_norm = np.array(self_attention_norm)[0]
        self_order = np.argsort(self_attention_norm[0])
        self_attention_norm = self_attention_norm[self_order]

        self_coordinates = np.array(sorted_coordinates)[self_order]
        sorted_coordinates = np.array(sorted_coordinates)[order]

        for i, (x, y) in enumerate(sorted_coordinates):
            x = np.round(x)
            y = np.round(y)

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

            mask_attention[int(y_start):int(y_end), int(x_start):int(x_end), :] = z_norm[i]
            mask_true[int(y_start):int(y_end), int(x_start):int(x_end), :] = 1
        
        '''for i, (x, y) in enumerate(self_coordinates):
            x = np.round(x)
            y = np.round(y)

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
            mask_self_attention[int(y_start):int(y_end), int(x_start):int(x_end), :] = self_attention_norm[i][i]
'''

        img_attention = rescale_intensity(img * mask_attention, out_range=(0, 255)).astype(np.uint8)
        #img_self_attention = rescale_intensity(img * mask_self_attention, out_range=(0, 255)).astype(np.uint8)
        img_true = rescale_intensity(img * mask_true, out_range=(0, 255)).astype(np.uint8)
        img_epi = rescale_intensity(img * mask_epi, out_range=(0, 255)).astype(np.uint8)

        plt.imsave(f'{snapshot_path}/MIL_attention_{batch_idx}.png', img_attention, format="png")
        #plt.imsave(f'{snapshot_path}/MIL_attention_{batch_idx}_self.png', img_self_attention, format="png")
        plt.imsave(f'{snapshot_path}/MIL_attention_{batch_idx}_true.png', img_true, format="png")
        plt.imsave(f'{snapshot_path}/MIL_attention_{batch_idx}_epi.png', img_epi, format="png")
        plt.imsave(f'{snapshot_path}/MIL_attention_{batch_idx}_whole.png', img, format="png")
