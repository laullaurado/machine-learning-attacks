import pandas as pd
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import PIL

from af_train import training, test, test_class
from af_models import ViTTargetModel, AttackModel
from af_datasets import UTKFace, AttackData

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
PATH = 'Attribute-Inference/UTKFace/'
TEST_SPLIT = 0.2
ATTACK_SPLIT = 0.5

samples = pd.read_pickle('Attribute-Inference/UTKFaceDF.pkl')
np.random.seed(SEED)

dataset_size = len(samples)
indices = list(range(dataset_size))
split = int(np.floor(TEST_SPLIT * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
attack_split = int(np.floor(ATTACK_SPLIT * len(train_indices)))
np.random.shuffle(train_indices)
attack_indices, train_indices = train_indices[attack_split:], train_indices[:attack_split]

train_samples = samples.iloc[train_indices]
test_samples = samples.iloc[test_indices]
attack_samples = samples.iloc[attack_indices]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_LEARNING_RATE = 0.0003
TARGET_BATCH_SIZE = 128

transform_vit = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

target_model = ViTTargetModel().to(DEVICE)
target_criterion = nn.CrossEntropyLoss()
# attack_criterion = nn.NLLLoss()
target_optimizer = torch.optim.Adam(target_model.parameters(), lr=TARGET_LEARNING_RATE)

target_train_loader = DataLoader(UTKFace(train_samples, 'gender', transform_vit), 
                                                    batch_size=TARGET_BATCH_SIZE)

target_test_loader = DataLoader(UTKFace(test_samples, 'gender', transform_vit), 
                                                    batch_size=TARGET_BATCH_SIZE)

ATTACK_LEARNING_RATE = 0.0003
ATTACK_BATCH_SIZE = 128

attack_model = AttackModel(64).to(DEVICE)
attack_criterion = nn.CrossEntropyLoss()
attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=ATTACK_LEARNING_RATE)

attack_train_loader = DataLoader(AttackData(attack_samples, target_model, transform_vit), 
                                                    batch_size=ATTACK_BATCH_SIZE)
attack_test_loader = DataLoader(AttackData(test_samples, target_model, transform_vit), 
                                                batch_size=ATTACK_BATCH_SIZE)

def perform_train_dummy(target_epochs, attack_epochs):
    target_model_path = 'Attribute-Inference/models/target_model_' + str(target_epochs) + '.pth'
    attack_model_path = 'Attribute-Inference/models/attack_model_' + str(attack_epochs) + '.pth' 

    print('Training Target Model for ' + str(target_epochs) + ' epochs...')
    training(target_epochs, target_train_loader, target_optimizer, target_criterion, target_model, target_model_path, True)
    target_model.to('cpu')

    print('Loading Target Model...')
    target_model.load_state_dict(torch.load(target_model_path))

    print('Testing Target Model...')
    test(target_test_loader, target_model, True)
    print('\n')

    print('Training Attack Model for ' + str(attack_epochs) + ' epochs...')
    training(attack_epochs, attack_train_loader, attack_optimizer, attack_criterion, attack_model, attack_model_path, False)
    attack_model.to('cpu')

    print('Loading Attack Model...')
    attack_model.load_state_dict(torch.load(attack_model_path))

    print('Testing Attack Model...')
    test(attack_test_loader, attack_model, False)
    test_class(attack_test_loader, attack_model, False)
