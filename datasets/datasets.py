import numpy as np
import os
from torch.utils.data import Dataset
import torch
from scipy.sparse import load_npz

bath_path = os.path.dirname(__file__)


def get_metadata(dataset_name):
    if dataset_name == 'VOC2007':
        meta = dict(
            num_classes=20,
        )
    else:
        raise ValueError('Dataset name Error!')
    return meta


class MLLFeatureDataset(Dataset):
    def __init__(self, dataset_name, path=os.path.join(bath_path, 'data'),
                 phase='train', feature_extractor='imagenet_resnet50'):
        self.dataset_name = dataset_name
        self.meta = get_metadata(dataset_name)
        self.labels = load_npz(os.path.join(path, f'{dataset_name}_{phase}_labels.npz')).toarray()
        self.features = np.load(
            os.path.join(path, f'{dataset_name}_{phase}_features_{feature_extractor}.npy'))
        self.phase = phase

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        sample = {}
        sample['image'] = torch.Tensor(self.features[index])
        sample['labels'] = torch.Tensor(self.labels[index])
        sample['idx'] = index
        return sample
