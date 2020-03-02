import numpy as np
from torch.autograd import Variable

from colon_loader import ColonCancerBagsCross
import time

def train(args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0.
    train_error = 0.

    # set models in training mode
    model.train(True)

    # start training
    for batch_idx, (data, label) in enumerate(train_loader):
        label = label[0]
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, gamma, gamma_kernel = model.calculate_objective(data, label)
        train_loss += loss.data[0]
        train_error += model.calculate_classification_error(data, label)[0]
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

    # calculate final loss
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return model, train_loss, train_error, gamma, gamma_kernel


def evaluate(args, model, train_loader, data_loader, mode):
    # set model to evaluation mode
    model.eval()

    if mode == 'validation':
        # set loss to 0
        evaluate_loss = 0.
        evaluate_error = 0.
        # CALCULATE classification error and log-likelihood for VALIDATION SET
        for batch_idx, (data, label) in enumerate(data_loader):
            label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            # reset gradients# calculate loss and metrics
            evaluate_loss += model.calculate_objective(data, label)[0].data[0]
            evaluate_error += model.calculate_classification_error(data, label)[0]

        # calculate final loss
        evaluate_loss /= len(data_loader)
        evaluate_error /= len(data_loader)

    if mode == 'test':
        # set loss to 0
        train_error = 0.
        train_loss = 0.
        evaluate_error = 0.
        evaluate_loss = 0.
        # CALCULATE classification error and log-likelihood for TEST SET
        t_ll_s = time.time()
        for batch_idx, (data, label) in enumerate(data_loader):
            label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            evaluate_loss += model.calculate_objective(data, label)[0].data[0]
            evaluate_error += model.calculate_classification_error(data, label)[0]
        t_ll_e = time.time()
        evaluate_error /= len(data_loader)
        evaluate_loss /= len(data_loader)
        print('\tTEST classification error value (time): {:.4f} ({:.2f}s)'.format(evaluate_error, t_ll_e - t_ll_s))
        print('\tTEST log-likelihood value (time): {:.4f} ({:.2f}s)\n'.format(evaluate_loss, t_ll_e - t_ll_s))

        # CALCULATE classification error and log-likelihood for TRAINING SET
        t_ll_s = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            train_loss += model.calculate_objective(data, label)[0].data[0]
            train_error += model.calculate_classification_error(data, label)[0]
        t_ll_e = time.time()
        train_error /= len(train_loader)
        train_loss /= len(train_loader)
        print('\tTRAIN classification error value (time): {:.4f} ({:.2f}s)'.format(train_error, t_ll_e - t_ll_s))
        print('\tTRAIN log-likelihood value (time): {:.4f} ({:.2f}s)\n'.format(train_loss, t_ll_e - t_ll_s))

    if mode == 'test':
        return evaluate_loss, evaluate_error, train_loss, train_error
    else:
        return evaluate_loss, evaluate_error


def kfold_indices_warwick(N, k, seed=777):
    r = np.random.RandomState(seed)
    all_indices = np.arange(N, dtype=int)
    r.shuffle(all_indices)
    idx = [int(i) for i in np.floor(np.linspace(0, N, k + 1))]
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold + 1]]
        valid_folds.append(valid_indices)
        train_fold = np.setdiff1d(all_indices, valid_indices)
        r.shuffle(train_fold)
        train_folds.append(train_fold)
    return train_folds, valid_folds


def load_warwick(train_fold, val_fold, test_fold, loc_info):
    print('\t-> Loading the following dataset')
    train_set, val_set, test_set = load_warwick_cross(train_fold, val_fold, test_fold, loc_info)
    return train_set, val_set, test_set


def load_warwick_cross(train_fold, val_fold, test_fold, loc_info):

    train_set = ColonCancerBagsCross('./Classification/',
                                     train_val_idxs=train_fold,
                                     test_idxs=test_fold,
                                     train=True,
                                     shuffle_bag=True,
                                     data_augmentation=True,
                                     loc_info=loc_info)

    val_set = ColonCancerBagsCross('./Classification/',
                                   train_val_idxs=val_fold,
                                   test_idxs=test_fold,
                                   train=True,
                                   shuffle_bag=True,
                                   data_augmentation=True,
                                   loc_info=loc_info)

    test_set = ColonCancerBagsCross('./Classification/',
                                    train_val_idxs=train_fold,
                                    test_idxs=test_fold,
                                    train=False,
                                    shuffle_bag=False,
                                    data_augmentation=False,
                                    loc_info=loc_info)

    return train_set, val_set, test_set
