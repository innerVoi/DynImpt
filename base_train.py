import os
from random import shuffle
from munch import Munch
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np

from utils import set_seed, create_dir_for_file
from models import ResNet50
from torch.utils.data import Subset, Dataset, DataLoader
import scipy.stats as stats
from torch.optim.lr_scheduler import MultiStepLR


class CustomCIFAR100(Dataset):
    def __init__(self, train_data, selected_indices):
        self.cifar100 = train_data
        self.selected_indices = selected_indices

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        selected_idx = self.selected_indices[idx]
        image, target = self.cifar100[selected_idx]
        return image, target, selected_idx


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target,
                                     reduction='sum'
                                     ).item()
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # model.train()

    return 100. * correct / len(test_loader.dataset)


def train(config: Munch, dataset: str, s: dict):
    sigma, tau, gamma, momentum = s['sigma'], s['tau'], s['gamma'], s['momentum']
    print(f'sigma={sigma}, tau={tau}, gamma={gamma}, momentum={momentum}')
    # lower, upper = momentum / 2, 1 - momentum / 2
    # print(f'lower={lower}, upper={upper}')
    args = config.default
    print(f'seed:{args.seed}')
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment': config.wandb.experiment})

    use_cuda = args.use_cuda and torch.cuda.is_available()
    set_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_transform = eval(dataset_config.train_transform)
    test_transform = eval(dataset_config.test_transform)
    train_dataset = eval('datasets.' + dataset)(root=args.data_dir,
                                                train=True,
                                                download=True,
                                                transform=train_transform
                                                )
    test_dataset = eval('torchvision.datasets.' + dataset)(root=args.data_dir,
                                                           train=False,
                                                           transform=test_transform
                                                           )
    train_size = int(len(train_dataset) * (1-momentum))
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size], generator=torch.Generator().manual_seed(0))
    # train_size, test_size = 10000, 2000
    # train_dataset, _ = torch.utils.data.random_split(train_dataset,
    #                                                  [train_size, len(train_dataset) - train_size], generator=torch.Generator().manual_seed(0))
    #
    # # 抽取测试集
    # test_dataset, _ = torch.utils.data.random_split(test_dataset,
    #                                                 [test_size, len(test_dataset) - test_size], generator=torch.Generator().manual_seed(0))

    print(f'train_size: {len(train_dataset)}, test_size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True)

    model = eval(args.model)(num_classes=dataset_config.num_classes
                             ).to(device)
    optimiser = eval('optim.' + args.optim)(model.parameters(),
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay
                                            )
    scheduler = MultiStepLR(optimiser, milestones=[60, 120, 160], gamma=0.2)


    testacc_list = []
    for idx in range(args.epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            output = model(data)

            loss = F.cross_entropy(output, target, reduction='none')
            loss = torch.mean(loss)
            loss.backward()
            optimiser.step()

        test_acc = test(model, device, test_loader)
        testacc_list.append(test_acc)
        scheduler.step()

    test_acc = test(model, device, test_loader)
    testacc_list.append(test_acc)
    # run.log({experiment_name+"test acc": max(testacc_list)})
    # run.finish()
    max_acc = np.max(testacc_list)
    print(f'max_acc={max_acc}')

    # log_file_path = \
    #     os.path.join(args.logs_dir, dataset, str(args.seed), 'log.csv')
    # create_dir_for_file(log_file_path)
    # with open(log_file_path, 'w') as f:
    #     f.write('iter,test_acc\n')
    #     for idx in range(len(testacc_list)):
    #         f.write(f'{idx+1},{testacc_list[idx]}\n')

    return max(testacc_list)


if __name__ == '__main__':
    train()
