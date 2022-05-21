from .dataset_util import get_datasets
from .dataset_transformation import GaussianNoise
import torchvision.transforms as transforms
from utility import load_config
from torch.utils import data


def load_dataset(dataset='cifar10', augtype='none', batch_size=32, data_loc='C:/Users/Michele/Desktop/FilePersonali/NAS_MachineLearningDeepLearning/files', sigma=0, pin_memory=True):
    train_dt, valid_dt, xshape, class_num = get_datasets(dataset, data_loc, cutout=0)
    if augtype == 'gaussnoise':
        train_dt.transform.transforms = train_dt.transform.transforms[2:]
        train_dt.transform.transforms.append(GaussianNoise(std=sigma))
    elif augtype == 'cutout':
        train_dt.transform.transforms = train_dt.transform.transforms[2:]
        train_dt.transform.transforms.append(transforms.RandomErasing(p=0.9, scale=(0.02, 0.04)))
    elif augtype == 'none':
        train_dt.transform.transforms = train_dt.transform.transforms[2:]

    if dataset == 'cifar10':
        acc_type = 'ori-test'
        val_acc_type = 'x-valid'
    else:
        acc_type = 'x-test'
        val_acc_type = 'x-valid'

    if 'cifar10' in dataset:
        cifar_split = load_config('files/cifar-split.txt', verbose=True)
        train_split, valid_split = cifar_split['train'], cifar_split['valid']
        train_loader = data.DataLoader(train_dt, batch_size=batch_size, num_workers=0, pin_memory=pin_memory,
                                       sampler=data.SubsetRandomSampler(train_split))
    else:
        train_loader = data.DataLoader(train_dt, batch_size=batch_size, num_workers=0, pin_memory=pin_memory)

    return train_loader
