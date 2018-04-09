import argparse
import tarfile
import json
import errno
import os

import requests
import scipy.io as sio
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def download_file(url, path):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
    return path


def download_oxford17(directory):
    data_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
    filename = data_url.split('/')[-1]
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        mkdir_p(directory)
        print('downloading...')
        download_file(data_url, path=path)

    data_extract_path = os.path.join(directory, '17flowers')
    if not os.path.exists(data_extract_path):
        print('extracting...')
        with tarfile.TarFile(path) as f:
            f.extractall(data_extract_path)

    data_path = data_extract_path

    splits_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat'
    filename = splits_url.split('/')[-1]
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        mkdir_p(directory)
        print('downloading splits...')
        download_file(splits_url, path=path)

    splits = sio.loadmat(path)
    one_indexed = 1
    train_split = splits['trn1'].reshape(-1) - one_indexed
    validation_split = splits['val1'].reshape(-1) - one_indexed
    test_split = splits['tst1'].reshape(-1) - one_indexed
    N = train_split.shape[0] + validation_split.shape[0] + test_split.shape[0]

    print('{} samples across train/validation/test'.format(N))

    classes = [
        'buttercup',
        'colts_foot',
        'daffodil',
        'daisy',
        'dandelion',
        'fritillary',
        'iris',
        'pansy',
        'sunflower',
        'windflower',
        'snowdrop',
        'lily_valley',
        'bluebell',
        'crocus',
        'tigerlily',
        'tulip',
        'cowslip',
        ]

    class_to_idx = {c: i for i, c in enumerate(classes)}

    sample_id_to_class = {i: classes[i // 80] for i in range(N)}

    transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        ])

    train_data = CustomImageFolder(data_path, train_split, classes, sample_id_to_class, class_to_idx, transform=transform)
    validation_data = CustomImageFolder(data_path, validation_split, classes, sample_id_to_class, class_to_idx, transform=transform)
    test_data = CustomImageFolder(data_path, test_split, classes, sample_id_to_class, class_to_idx, transform=transform)

    print('done')

    return train_data, validation_data, test_data


def make_dataset(dir, class_to_idx, extensions, sample_ids_to_use, sample_id_to_class):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    sample_id = int(fname.split('.')[0].split('_')[1])
                    if sample_id in sample_ids_to_use:
                        classname = sample_id_to_class[sample_id]
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[classname])
                        images.append(item)

    return images


class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, sample_ids_to_use, classes, sample_id_to_class, class_to_idx,
            transform=None, target_transform=None):
        extensions = torchvision.datasets.folder.IMG_EXTENSIONS
        samples = make_dataset(root, class_to_idx, extensions, sample_ids_to_use, sample_id_to_class)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.imgs = self.samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class DataManager(object):
    def __init__(self, dataset):
        super(DataManager, self).__init__()
        self.dataset = dataset
        
    def get_datasets(self):
        home = os.path.expanduser('~')
        if self.dataset is 'oxford17':
            directory = os.path.join(home, 'data', 'oxford17')
            train_data, validation_data, test_data = download_oxford17(directory)
        else:
            raise ValueError

        return train_data, validation_data, test_data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 50)
        self.fc2 = nn.Linear(50, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 4))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 4))
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class ConfusionMatrix():
    def __init__(self, size):
        self.matrix = np.zeros((size, size), dtype=np.int32)

    def update(self, x, y):
        np.add.at(self.matrix, (x, y), 1)

    def accuracy(self):
        return self.matrix.diagonal().sum() / self.matrix.sum()

    def reset(self):
        self.matrix.fill(0)


class NDConfusionMatrix():
    def __init__(self, size, ndims=1):
        self.ndims = ndims
        self.matrix = np.zeros((ndims, size, size), dtype=np.int32)

    def update(self, x, y, dim=0):
        for i in range(self.ndims):
            np.add.at(self.matrix, (i, x[:, i], y), 1)

    def accuracy(self, topk=-1):
        if topk < 0:
            topk = self.ndims
        correct = self.matrix[:topk, :, :].diagonal(axis1=1, axis2=2).sum()
        total = self.matrix[0, :, :].sum()
        return correct / total

    def reset(self):
        self.matrix.fill(0)


def main(options):
    data_manager = DataManager(options.dataset)
    train_dataset, validation_dataset, test_dataset = data_manager.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=options.batch_size, shuffle=True,
        num_workers=options.workers, pin_memory=True)

    nbatches = len(train_dataset.imgs) // options.batch_size

    model = Net()
    optimizer = optim.Adam(model.parameters())
    cm = NDConfusionMatrix(17, 3)

    for epoch in range(options.epochs):
        total_loss = 0
        for step, batch in tqdm(enumerate(train_loader), total=nbatches):
            data, target = batch
            outp = model(Variable(data))
            loss = nn.NLLLoss()(F.log_softmax(outp, dim=1), Variable(target))

            cm.update(outp.data.topk(3, dim=1)[1].numpy(), target.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
        print(total_loss / nbatches)
        print(cm.accuracy(topk=1))
        print(cm.accuracy(topk=3))
        print(cm.matrix[0])
        cm.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='oxford17', choices=['oxford17', 'oxford102'])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
    options = parser.parse_args()

    print(json.dumps(options.__dict__, sort_keys=True, indent=4))

    main(options)
