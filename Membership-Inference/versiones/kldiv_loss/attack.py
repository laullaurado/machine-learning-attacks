#!/usr/bin/python

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import model
from model import init_params as w_init, VisionTransformerModel
from train import train_model, train_attack_model, prepare_attack_data
from sklearn.metrics import classification_report
import argparse
import numpy as np
import os
import copy

# Set the seed for reproducibility
np.random.seed(1234)
# Flag to enable early stopping
need_earlystop = False

########################
# Model Hyperparameters
########################
# Number of filters for target and shadow models
target_filters = [128, 256, 256]
shadow_filters = [64, 128, 128]
# New FC layers size for pretrained model
n_fc = [256, 128]
# For CIFAR-10 and MNIST dataset
num_classes = 10
# No. of training epochs
num_epochs = 50
# How many samples per batch to load
batch_size = 128
# Learning rate
learning_rate = 0.0003
# Learning rate decay
lr_decay = 0.96
# Regularizer
reg = 1e-4
# Percentage of dataset to use for shadow model
shadow_split = 0.6
# Number of validation samples
n_validation = 1000
# Number of processes
num_workers = 2
# Hidden units for MNIST model
n_hidden_mnist = 32

################################
# Attack Model Hyperparameters
################################
NUM_EPOCHS = 50
BATCH_SIZE = 10
# Learning rate
LR_ATTACK = 0.0003
# L2 Regulariser
REG = 1e-7
# Weight decay
LR_DECAY = 0.96
# No of hidden units
n_hidden = 128
# Binary Classifier
out_classes = 2

def get_cmd_arguments():
    parser = argparse.ArgumentParser(prog="Membership Inference Attack")
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'MNIST'], help='Which dataset to use (CIFAR10 or MNIST)')
    parser.add_argument('--dataPath', default='./data', type=str, help='Path to store data')
    parser.add_argument('--modelPath', default='./model', type=str, help='Path to save or load model checkpoints')
    parser.add_argument('--trainTargetModel', action='store_true', help='Train a target model, if false then load an already trained model')
    parser.add_argument('--trainShadowModel', action='store_true', help='Train a shadow model, if false then load an already trained model')
    parser.add_argument('--need_augm', action='store_true', help='To use data augmentation on target and shadow training set or not')
    parser.add_argument('--need_topk', action='store_true', help='Flag to enable using Top 3 posteriors for attack data')
    parser.add_argument('--param_init', action='store_true', help='Flag to enable custom model params initialization')
    parser.add_argument('--verbose', action='store_true', help='Add Verbosity')
    return parser.parse_args()

def get_data_transforms(dataset, augm=False):
    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        if augm:
            train_transforms = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
    else:  # MNIST
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])

        if augm:
            train_transforms = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(28, padding=4),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize
            ])
    return train_transforms, test_transforms

def split_dataset(train_dataset):
    total_size = len(train_dataset)
    split1 = total_size // 4
    split2 = split1 * 2
    split3 = split1 * 3

    indices = list(range(total_size))
    np.random.shuffle(indices)

    s_train_idx = indices[:split1]
    s_test_idx = indices[split1:split2]
    t_train_idx = indices[split2:split3]
    t_test_idx = indices[split3:]

    return s_train_idx, s_test_idx, t_train_idx, t_test_idx

def get_data_loader(dataset, data_dir, batch, shadow_split=0.5, augm_required=False, num_workers=1):
    assert ((shadow_split >= 0) and (shadow_split <= 1)), "[!] shadow_split should be in the range [0, 1]."

    train_transforms, test_transforms = get_data_transforms(dataset, augm_required)

    if dataset == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transforms, download=True)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transforms)
    else:
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, transform=train_transforms, download=True)
        test_set = torchvision.datasets.MNIST(root=data_dir, train=False, transform=test_transforms)

    s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(train_set)

    s_train_sampler = SubsetRandomSampler(s_train_idx)
    s_out_sampler = SubsetRandomSampler(s_out_idx)
    t_train_sampler = SubsetRandomSampler(t_train_idx)
    t_out_sampler = SubsetRandomSampler(t_out_idx)

    if dataset == 'CIFAR10':
        target_val_idx = t_out_idx[:n_validation]
        shadow_val_idx = s_out_idx[:n_validation]
    else:
        target_val_idx = t_out_idx[:n_validation]
        shadow_val_idx = s_out_idx[:n_validation]

    t_val_sampler = SubsetRandomSampler(target_val_idx)
    s_val_sampler = SubsetRandomSampler(shadow_val_idx)

    if dataset == 'CIFAR10':
        t_train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=t_train_sampler, num_workers=num_workers)
        t_out_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=t_out_sampler, num_workers=num_workers)
        t_val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=t_val_sampler, num_workers=num_workers)
        s_train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=s_train_sampler, num_workers=num_workers)
        s_out_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=s_out_sampler, num_workers=num_workers)
        s_val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=s_val_sampler, num_workers=num_workers)
    else:
        t_train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=t_train_sampler, num_workers=num_workers)
        t_out_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=t_out_sampler, num_workers=num_workers)
        t_val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=t_val_sampler, num_workers=num_workers)
        s_train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=s_train_sampler, num_workers=num_workers)
        s_out_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=s_out_sampler, num_workers=num_workers)
        s_val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch, sampler=s_val_sampler, num_workers=num_workers)

    print('Total Test samples in {} dataset : {}'.format(dataset, len(test_set)))
    print('Total Train samples in {} dataset : {}'.format(dataset, len(train_set)))
    print('Number of Target train samples: {}'.format(len(t_train_sampler)))
    print('Number of Target valid samples: {}'.format(len(t_val_sampler)))
    print('Number of Target test samples: {}'.format(len(t_out_sampler)))
    print('Number of Shadow train samples: {}'.format(len(s_train_sampler)))
    print('Number of Shadow valid samples: {}'.format(len(s_val_sampler)))
    print('Number of Shadow test samples: {}'.format(len(s_out_sampler)))

    return t_train_loader, t_val_loader, t_out_loader, s_train_loader, s_val_loader, s_out_loader

def attack_inference(model, test_X, test_Y, device):
    print('----Attack Model Testing----')

    targetnames = ['Non-Member', 'Member']
    pred_y = []
    true_y = []

    X = torch.cat(test_X)
    Y = torch.cat(test_Y)

    inferdataset = TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset=inferdataset, batch_size=50, shuffle=False, num_workers=num_workers)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            logits = outputs[0]
            _, predictions = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            true_y.append(labels.cpu())
            pred_y.append(predictions.cpu())

    attack_acc = correct / total
    print('Attack Test Accuracy is  : {:.2f}%'.format(100 * attack_acc))

    true_y = torch.cat(true_y).numpy()
    pred_y = torch.cat(pred_y).numpy()

    print('---Detailed Results----')
    print(classification_report(true_y, pred_y, target_names=targetnames, zero_division=0))

def create_attack(dataset, dataPath, modelPath, trainTargetModel, trainShadowModel, need_augm, need_topk, param_init, verbose):
    dataset = dataset
    need_augm = need_augm
    verbose = verbose
    top_k = need_topk

    if dataset == 'CIFAR10':
        img_size = 32
        input_dim = 3
    else:
        img_size = 28
        input_dim = 1

    datasetDir = os.path.join(dataPath, dataset)
    modelDir = os.path.join(modelPath, dataset)

    if not os.path.exists(datasetDir):
        try:
            os.makedirs(datasetDir)
        except OSError:
            pass

    if not os.path.exists(modelDir):
        try:
            os.makedirs(modelDir)
        except OSError:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    t_train_loader, t_val_loader, t_test_loader, s_train_loader, s_val_loader, s_test_loader = get_data_loader(
        dataset, datasetDir, batch_size, shadow_split, need_augm, num_workers)

    if (trainTargetModel):
        if dataset == 'CIFAR10':
            target_model = VisionTransformerModel(num_classes=num_classes, pretrained=True).to(device)
        else:
            target_model = VisionTransformerModel(num_classes=num_classes, pretrained=False).to(device)

        if (param_init):
            target_model.apply(w_init)

        if verbose:
            print('----Target Model Architecture----')
            print(target_model)
            print('----Model Learnable Params----')
            for name, param in target_model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        loss = nn.KLDivLoss(reduction='batchmean')

        optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=reg)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

        targetX, targetY = train_model(target_model, t_train_loader, t_val_loader, t_test_loader, loss, optimizer,
                                       lr_scheduler, device, modelDir, verbose, num_epochs, top_k, need_earlystop,
                                       is_target=True)

    else:
        target_file = os.path.join(modelDir, 'best_target_model.ckpt')
        print('Use Target model at the path ====> [{}] '.format(modelDir))
        if dataset == 'CIFAR10':
            target_model = VisionTransformerModel(num_classes=num_classes, pretrained=True).to(device)
        else:
            target_model = VisionTransformerModel(num_classes=num_classes, pretrained=False).to(device)

        target_model.load_state_dict(torch.load(target_file))
        print('---Preparing Attack Training data---')
        t_trainX, t_trainY = prepare_attack_data(target_model, t_train_loader, device, top_k)
        t_testX, t_testY = prepare_attack_data(target_model, t_test_loader, device, top_k, test_dataset=True)
        targetX = t_trainX + t_testX
        targetY = t_trainY + t_testY

    if (trainShadowModel):
        if dataset == 'CIFAR10':
            shadow_model = VisionTransformerModel(num_classes=num_classes, pretrained=True).to(device)
        else:
            shadow_model = VisionTransformerModel(num_classes=num_classes, pretrained=False).to(device)

        if (param_init):
            shadow_model.apply(w_init)

        if verbose:
            print('----Shadow Model Architecture---')
            print(shadow_model)
            print('---Model Learnable Params----')
            for name, param in shadow_model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        shadow_loss = nn.KLDivLoss(reduction='batchmean')
        shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=learning_rate, weight_decay=reg)
        shadow_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(shadow_optimizer, gamma=lr_decay)

        shadowX, shadowY = train_model(shadow_model, s_train_loader, s_val_loader, s_test_loader, shadow_loss,
                                       shadow_optimizer, shadow_lr_scheduler, device, modelDir, verbose, num_epochs,
                                       top_k, need_earlystop, is_target=False)
    else:
        print('Using Shadow model at the path  ====> [{}] '.format(modelDir))
        shadow_file = os.path.join(modelDir, 'best_shadow_model.ckpt')
        assert os.path.isfile(shadow_file), 'Shadow Model Checkpoint not found, aborting load'
        if dataset == 'CIFAR10':
            shadow_model = VisionTransformerModel(num_classes=num_classes, pretrained=True).to(device)
        else:
            shadow_model = VisionTransformerModel(num_classes=num_classes, pretrained=False).to(device)

        shadow_model.load_state_dict(torch.load(shadow_file))
        print('----Preparing Attack training data---')
        trainX, trainY = prepare_attack_data(shadow_model, s_train_loader, device, top_k)
        testX, testY = prepare_attack_data(shadow_model, s_test_loader, device, top_k, test_dataset=True)
        shadowX = trainX + testX
        shadowY = trainY + testY

    input_size = shadowX[0].size(1)
    print('Input Feature dim for Attack Model : [{}]'.format(input_size))

    attack_model = model.AttackMLP(input_size, n_hidden, out_classes).to(device)

    if (param_init):
        attack_model.apply(w_init)

    attack_loss = nn.KLDivLoss(reduction='batchmean')
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=LR_ATTACK, weight_decay=REG)
    attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer, gamma=LR_DECAY)

    attackdataset = (shadowX, shadowY)

    attack_valacc = train_attack_model(attack_model, attackdataset, attack_loss, attack_optimizer, attack_lr_scheduler,
                                       device, modelDir, NUM_EPOCHS, BATCH_SIZE, num_workers, verbose)

    print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100 * attack_valacc))

    attack_path = os.path.join(modelDir, 'best_attack_model.ckpt')
    attack_model.load_state_dict(torch.load(attack_path))

    attack_inference(attack_model, targetX, targetY, device)

if __name__ == '__main__':
    args = get_cmd_arguments()
    print(args)
    create_attack(args.dataset, args.dataPath, args.modelPath, args.trainTargetModel, args.trainShadowModel, args.need_augm, args.need_topk, args.param_init, args.verbose)
