from __future__ import print_function

import argparse
import datetime
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from experiments import experiment
from utils import kfold_indices_warwick
from utils import load_warwick
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='VNOM')

parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='MOM', help='momentum (default: 0.9)')
parser.add_argument('--early_stopping_epochs', type=int, default=25, metavar='N', help='number of epochs for early stopping')
parser.add_argument('--reg', type=float, default=5*10e-4, metavar='r', help='weight decay')
parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=14, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model_name', type=str, default='CNN', metavar='N', help='models name: demon, disr, delse')
parser.add_argument('--activation', type=str, default=nn.ReLU(), metavar='N', help='activation function')
parser.add_argument('--classification_threshold', type=float, default=0.5, metavar='N', help='classification threshold')
parser.add_argument('--operator', type=str, default='max', metavar='op', help='Choose type of MIL pooling layer')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='N', help='use adam or other')
parser.add_argument('--L', type=int, default=512, metavar='N', help='parameter for attention (input hidden units, low-dim embedding)')
parser.add_argument('--D', type=int, default=128, metavar='N', help='parameter for attention (internal hidden units)')
parser.add_argument('--K', type=int, default=1, metavar='N', help='parameter for attention (number of attentions)')

# dataset
parser.add_argument('--dataset_name', type=str, default='warwick', metavar='N', help='name of the dataset: bags_mnist')
parser.add_argument('--dataset_size', type=int, default=100, metavar='N', help='number of images in dataset (default: 58)')
parser.add_argument('--kfold_test', type=int, default=5, metavar='k', help='number of folds (default: 10)')
parser.add_argument('--kfold_val', type=int, default=20, metavar='k', help='percentage of the train set that will be used for validation (default: 10)')
parser.add_argument('--loc_info', action='store_true', default=False, help='training with location')
parser.add_argument('--self_att', action='store_true', default=False, help='training with self attention')
parser.add_argument('--loc_gauss', action='store_true', default=False, help='training with location gauss kernel')
parser.add_argument('--loc_inv_q', action='store_true', default=False,
                    help='training with location inverse quadratic kernel')
parser.add_argument('--att_gauss_abnormal', action='store_true', default=False, help='training with attention gauss kernel')
parser.add_argument('--att_gauss_spatial', action='store_true', default=False,
                    help='training with attention gauss kernel')
parser.add_argument('--att_inv_q_abnormal', action='store_true', default=False, help='training with attention gauss kernel')
parser.add_argument('--att_inv_q_spatial', action='store_true', default=False, help='training with inverse quadratic kernel')
parser.add_argument('--att_module', action='store_true', default=False, help='training with module kernel')
parser.add_argument('--out_loc', action='store_true', default=False, help='outer location')
parser.add_argument('--loc_att', action='store_true', default=False, help='relative location')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('GPU is ON!')

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:19]

    model_name = '' + args.model_name

    if args.loc_gauss or args.loc_inv_q or args.loc_att:
        args.loc_info = True

    if args.att_gauss_abnormal or args.att_inv_q_abnormal or args.att_gauss_spatial or args.att_inv_q_spatial or \
            args.att_module:
        args.self_att = True

    print(args)

    with open('experiment_log_' + args.operator + '.txt', 'a') as f:
        print(args, file=f)

    # IMPORT MODEL======================================================================================================
    from models.CNN import CNN as Model

    # START KFOLDS======================================================================================================
    print('\nSTART KFOLDS CROSS VALIDATION\n')
    print(f'{args.kfold_test} Test-Train folds each has {args.epochs} epochs for a '
          f'{1.0/args.kfold_val}/{(args.kfold_val - 1.0)/args.kfold_val} Valid-Train split\n')

    train_folds, test_folds = kfold_indices_warwick(args.dataset_size, args.kfold_test, seed=args.seed)

    train_error_folds = []
    test_error_folds = []
    for current_fold in range(1, args.kfold_test + 1):


        print('#################### Train-Test fold: {}/{} ####################'.format(current_fold, args.kfold_test))

        # DIRECTORY FOR SAVING==========================================================================================
        snapshots_path = 'snapshots/'
        dir = snapshots_path + model_name + '_' + args.model_signature + '/'
        sw = SummaryWriter(f'tensorboard/{model_name}_{args.model_signature}_fold_{current_fold}')

        if not os.path.exists(dir):
            os.makedirs(dir)

        # LOAD DATA=====================================================================================================
        train_fold, val_fold = kfold_indices_warwick(len(train_folds[current_fold - 1]), args.kfold_val, seed=args.seed)
        train_fold = [train_folds[current_fold - 1][i] for i in train_fold]
        val_fold = [train_folds[current_fold - 1][i] for i in val_fold]
        loc = True if args.loc_info or args.out_loc else False
        train_set, val_set, test_set = load_warwick(train_fold[0], val_fold[0], test_folds[current_fold - 1], loc)

        # CREATE MODEL==================================================================================================
        print('\tcreate models')
        model = Model(args)
        if args.cuda:
            model.cuda()

        # INIT OPTIMIZER================================================================================================
        print('\tinit optimizer')
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg, momentum=0.9)
        else:
            raise Exception('Wrong name of the optimizer!')

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        # PERFORM EXPERIMENT============================================================================================
        print('\tperform experiment\n')

        train_error, test_error = experiment(
            args,
            kwargs,
            current_fold,
            train_set,
            val_set,
            test_set,
            model,
            optimizer,
            scheduler,
            dir,
            sw,
        )

        # APPEND FOLD RESULTS===========================================================================================
        train_error_folds.append(train_error)
        test_error_folds.append(test_error)

        with open('final_results_' + args.operator + '.txt', 'a') as f:
            print('RESULT FOR A SINGLE FOLD\n'
                  'SEED: {}\n'
                  'OPERATOR: {}\n'
                  'FOLD: {}\n'
                  'ERROR (TRAIN): {}\n'
                  'ERROR (TEST): {}\n\n'.format(args.seed, args.operator, current_fold, train_error, test_error),
                  file=f)
    # ======================================================================================================================
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    with open('experiment_log_' + args.operator + '.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

    return np.mean(train_error_folds), np.std(train_error_folds), np.mean(test_error_folds), np.std(test_error_folds)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    seeds = [71, 79, 53, 32, 98]
    operators = ['mean']

    for operator in operators:
        args.operator = operator

        train_mean_list = []
        test_mean_list = []

        for seed in seeds:
            args.seed = seed
            train_mean, train_std, test_mean, test_std = run(args, kwargs)

            with open('final_results_' + args.operator + '.txt', 'a') as f:
                print('RESULT FOR A SINGLE SEED, 5 FOLDS\n'
                      'SEED: {}\n'
                      'OPERATOR: {}\n'
                      'TRAIN MEAN {} AND STD {}\n'
                      'TEST MEAN {} AND STD {}\n\n'.format(seed, args.operator, train_mean, train_std, test_mean, test_std),
                      file=f)

            train_mean_list.append(train_mean)
            test_mean_list.append(test_mean)

        with open('final_results_' + args.operator + '.txt', 'a') as f:
            print('RESULT FOR 5 SEEDS, 5 FOLDS\n'
                  'OPERATOR: {}\n'
                  'TRAIN MEAN {} AND STD {}\n'
                  'TEST MEAN {} AND STD {}\n\n'.format(args.operator, np.mean(train_mean_list), np.std(train_mean_list), np.mean(test_mean_list), np.std(test_mean_list)),
                  file=f)

        with open('final_results_' + args.operator + '.txt', 'a') as f:
            print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
