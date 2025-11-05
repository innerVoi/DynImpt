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
import copy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import json
from scipy.stats import percentileofscore
import pickle
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

class RawData(Dataset):
    def __init__(self, train_data):
        self.data = train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx]
        return idx, image, target

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

# 定义保存图像的函数
def save_image(img, filename):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imsave(filename, np.transpose(npimg, (1, 2, 0)))

def generate_noisy(train_dataset, noise_ratio=0.05, seed=2):
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 获取原始数据和标签
    train_data = train_dataset.data
    train_labels = np.array(train_dataset.targets)

    # 生成带噪声特征的数据
    num_noise = int(len(train_data) * noise_ratio)
    all_indices = np.arange(len(train_data))
    noise_indices_features = np.random.choice(all_indices, num_noise, replace=False)
    noisy_data = train_data.copy()
    for idx in noise_indices_features:
        noise = np.random.normal(0, 0.1, noisy_data[idx].shape) * 255
        noisy_data[idx] = np.clip(noisy_data[idx] + noise, 0, 255)

    # 从所有索引中删除已经用于特征噪声的索引
    remaining_indices = np.setdiff1d(all_indices, noise_indices_features)
    # 生成带噪声标签的数据
    noise_indices_labels = np.random.choice(remaining_indices, num_noise, replace=False)

    noisy_labels = train_labels.copy()
    for idx in noise_indices_labels:
        original_label = noisy_labels[idx]
        new_label = np.random.choice([label for label in range(100) if label != original_label])
        noisy_labels[idx] = new_label

    # 使用 deepcopy 创建新的训练数据集
    noisy_train_dataset = copy.deepcopy(train_dataset)
    noisy_train_dataset.data = noisy_data
    noisy_train_dataset.targets = noisy_labels.tolist()

    # 保存带噪声特征的图像
    print("Saving images with noisy features:")
    for i, idx in enumerate(noise_indices_features[:5]):  # 只保存前5个带噪声特征的图像
        img = torch.tensor(noisy_data[idx]).permute(2, 0, 1).float() / 255  # 预处理图像
        save_image(img, f'noisy_feature_image_{i}.png')
        print(f"Saved noisy feature image {i} with label: {noisy_labels[idx]} (Index: {idx})")

    # 保存带噪声标签的图像
    print("Saving images with noisy labels:")
    for i, idx in enumerate(noise_indices_labels[:5]):  # 只保存前5个带噪声标签的图像
        img = torch.tensor(noisy_data[idx]).permute(2, 0, 1).float() / 255  # 预处理图像
        save_image(img, f'noisy_label_image_{i}.png')
        print(f"Saved noisy label image {i} with original label: {train_labels[idx]}, noisy label: {noisy_labels[idx]} (Index: {idx})")

    return train_dataset, noise_indices_features, noise_indices_labels


def train(config:Munch, dataset:str, s:dict):
    sigma, tau, gamma, momentum = s['sigma'], s['tau'], s['gamma'], s['momentum']
    print(f'sigma={sigma}, tau={tau}, gamma={gamma}, momentum={momentum}')
    if sigma == 2.0:
        # 仅使用简单样本
        lower, upper = 0., 1. - momentum
    elif sigma == 3.0:
        # 仅使用困难样本
        lower, upper = momentum, 1.
    else:
        # 使用分数在中间的相对困难样本
        lower, upper = momentum / 2, 1 - momentum / 2

    # 使用分数在中间的相对困难样本
    #lower, upper = momentum / 2, 1 - momentum / 2
    # # 仅使用简单样本
    #lower, upper = 0., 1. - momentum
    # # 仅使用困难样本
    #lower, upper = momentum, 1.
    print(f'lower={lower}, upper={upper}')
    args = config.default
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment':config.wandb.experiment})

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
    train_dataset = eval('datasets.'+dataset)(root=args.data_dir,
                                         train=True,
                                         download=True,
                                         transform=train_transform
                                        )
    test_dataset = eval('torchvision.datasets.'+dataset)(root=args.data_dir,
                                         train=False,
                                         transform=test_transform
                                        )
    # train_size, test_size = 10000, 2000
    # train_dataset, _ = torch.utils.data.random_split(train_dataset,
    #                                                  [train_size, len(train_dataset) - train_size], generator=torch.Generator().manual_seed(0))
    #
    # # 抽取测试集
    # test_dataset, _ = torch.utils.data.random_split(test_dataset,
    #                                                 [test_size, len(test_dataset) - test_size], generator=torch.Generator().manual_seed(0))


    print(f'train_size: {len(train_dataset)}, test_size: {len(test_dataset)}')

    train_dataset, noise_indices_features, noise_indices_labels = generate_noisy(train_dataset)


    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True)

    model = eval(args.model)(num_classes=dataset_config.num_classes
                            ).to(device)
    optimiser = eval('optim.' + args.optim)(model.parameters(), 
                                            lr=args.lr, 
                                            momentum=args.momentum, 
                                            weight_decay=args.weight_decay
                                           )
    scheduler = MultiStepLR(optimiser, milestones=[60, 120, 160], gamma=0.2)

    loss_list = []  # 窗口内所有样本的损失
    testacc_list = []
    label_list = [999 for _ in range(len(train_dataset))]
    num_classes = 100
    raw_dataset = RawData(train_dataset)
    selected_noise_features, selected_noise_labels = [], []
    noise_percentiles_features, noise_percentiles_labels = [], []
    for idx in range(args.epochs):
        model.train()
        the_loss = [0. for _ in range(len(train_dataset))]
        if idx < args.window_size:
            print(f'epoch={idx},train_size={len(raw_dataset)}')
            train_loader = DataLoader(raw_dataset, **train_kwargs)
            for i, (origin_index, data, target) in enumerate(train_loader):
                if idx == 0:
                    # label_list.extend(target.tolist())
                    for data_i in range(len(data)):
                        label_list[origin_index[data_i]] = target[data_i].item()
                data, target = data.to(device), target.to(device)
                #print(f'data size={data.size()}')
                optimiser.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target, reduction='none')
                # the_loss.extend([loss[j].item() for j in range(len(data))])
                for data_i in range(len(data)):
                    the_loss[origin_index[data_i]] = loss[data_i].item()
                loss = torch.mean(loss)
                loss.backward()
                optimiser.step()
                # run.log({experiment_name+"loss": loss.item()})
            loss_list.append(the_loss)
            # np.save('scores/label.npy', label_list)
        else:
            #if (idx + 1) % (args.epochs + 10) == 0:
            if (idx + 1) % (args.epochs // 10) == 0:
                # the_loss.extend(loss_list[-1])
                the_loss = [loss_list[-1][i] for i in range(len(loss_list[-1]))]
                select_indices = [i for i in range(len(label_list))]
            else:
                # 宏观趋势得分
                first_loss, end_loss = np.mean(loss_list[0]), np.mean(loss_list[-1])
                loss_decay = first_loss - end_loss
                sample_loss_decay = [loss_list[0][i] - loss_list[-1][i] for i in range(len(loss_list[0]))]
                if loss_decay > 0:
                    macro_change = [-sample_loss_decay[i] / loss_decay for i in range(len(sample_loss_decay))]
                else:
                    macro_change = [sample_loss_decay[i] / loss_decay for i in range(len(sample_loss_decay))]
                max_contribution, min_contribution = np.max(macro_change), np.min(macro_change)
                macro_scores = [(macro_change[i] - min_contribution) / (max_contribution - min_contribution) for i in
                                range(len(macro_change))]

                # 当前分布得分
                the_loss = [loss_list[-1][i] for i in range(len(loss_list[-1]))]
                max_loss, min_loss = np.max(the_loss), np.min(the_loss)
                micro_scores = [(the_loss[i] - min_loss) / (max_loss - min_loss) for i in range(len(the_loss))]

                # 利用标准差做分数
                loss_std_list = np.std(loss_list, axis=0)
                max_std, min_std = np.max(loss_std_list), np.min(loss_std_list)
                uncertain_scores = [(loss_std_list[i] - min_std) / (max_std - min_std) for i in
                                    range(len(loss_std_list))]

                #final_scores = [micro_scores[i] + macro_scores[i] + uncertain_scores[i] for i in range(len(the_loss))]
                if tau == 0.1:
                    final_scores = [macro_scores[i] for i in range(len(the_loss))]
                elif tau == 0.2:
                    final_scores = [micro_scores[i] for i in range(len(the_loss))]
                elif tau == 0.3:
                    final_scores = [uncertain_scores[i] for i in range(len(the_loss))]
                elif tau == 0.4:
                    final_scores = [macro_scores[i] + micro_scores[i] for i in range(len(the_loss))]
                elif tau == 0.5:
                    final_scores = [macro_scores[i] + uncertain_scores[i] for i in range(len(the_loss))]
                elif tau == 0.6:
                    final_scores = [micro_scores[i] + uncertain_scores[i] for i in range(len(the_loss))]
                else:
                    final_scores = [micro_scores[i] + macro_scores[i] + uncertain_scores[i] for i in range(len(the_loss))]

                select_indices = []
                class_scores = [[] for _ in range(num_classes)]
                class_indices = [[] for _ in range(num_classes)]
                for i in range(len(label_list)):
                    class_scores[label_list[i]].append(final_scores[i])
                    class_indices[label_list[i]].append(i)
                for c in range(num_classes):
                    class_select_indices = []
                    lower_bound, upper_bound = np.quantile(class_scores[c], lower), np.quantile(class_scores[c], upper)
                    for i in range(len(class_scores[c])):
                        if (class_scores[c][i] >= lower_bound) and (class_scores[c][i] <= upper_bound):
                            class_select_indices.append(class_indices[c][i])
                        else:
                            continue
                    select_indices.extend(class_select_indices)


            if idx % 4 == 0:
                # 找出当前轮次选择到的噪声索引
                intersection_noise_features = list(set(select_indices) & set(noise_indices_features))
                selected_noise_features.append(intersection_noise_features)
                intersection_noise_labels = list(set(select_indices) & set(noise_indices_labels))
                selected_noise_labels.append(intersection_noise_labels)

                # 保存噪声样本在所有数据中的分位数
                noise_percentiles_features.append([percentileofscore(final_scores, final_scores[index], kind='rank') for index in noise_indices_features])
                noise_percentiles_labels.append([percentileofscore(final_scores, final_scores[index], kind='rank')  for index in noise_indices_labels])

                print(f'selected_noise_features={selected_noise_features},\n, intersection_noise_labels={intersection_noise_labels}')
                print(f'noise_percentiles_features={noise_percentiles_features},\n, noise_percentiles_labels={noise_percentiles_labels}')

            custom_dataset = CustomCIFAR100(train_dataset, select_indices)
            print(f'epoch={idx},train_size={len(custom_dataset)}')
            train_loader = DataLoader(custom_dataset, **train_kwargs)
            for i, (data, target, origin_index) in enumerate(train_loader):
                # print(origin_index)
                data, target = data.to(device), target.to(device)
                optimiser.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target, reduction='none')
                # 更新loss列表
                for data_i in range(len(data)):
                    the_loss[origin_index[data_i]] = loss[data_i].item()
                loss = torch.mean(loss)
                loss.backward()
                optimiser.step()
                # run.log({experiment_name+"loss": loss.item()})

            loss_list.append(the_loss)
            loss_list.pop(0)

        test_acc = test(model, device, test_loader)
        # run.log({experiment_name + "test acc": test_acc})
        testacc_list.append(test_acc)
        scheduler.step()


    with open(f'./noise_saved/selected_noise_features_{momentum}.pkl', 'wb') as f:
        pickle.dump(selected_noise_features, f)
    with open(f'./noise_saved/selected_noise_labels_{momentum}.pkl', 'wb') as f:
        pickle.dump(selected_noise_labels, f)

    with open(f'./noise_saved/noise_percentiles_features_{momentum}.pkl', 'wb') as f:
        pickle.dump(noise_percentiles_features, f)
    with open(f'./noise_saved/noise_percentiles_labels_{momentum}.pkl', 'wb') as f:
        pickle.dump(noise_percentiles_labels, f)



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
