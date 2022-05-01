import torch.nn
from utils.utils import set_seed, Logger
from methods.flem import FLEM
from datasets.datasets import MLLFeatureDataset
from torch.utils.data import DataLoader
from utils.metrics import evaluation


def get_model():
    model = FLEM(num_classes=train_dataset.labels.shape[1],
                 loss_func=torch.nn.BCEWithLogitsLoss(),
                 num_feature=num_feature,
                 alpha=alpha,
                 beta=beta,
                 method=method,
                 max_epoch=max_epoch,
                 optimizer=optimizer,
                 lr=lr,
                 adjust_lr=False,
                 verbose=verbose,
                 device=device)
    return model


def get_dataset():
    feature_extractor = 'imagenet_resnet50'
    train_dataset = MLLFeatureDataset(dataset_name, phase='train', feature_extractor=feature_extractor)
    val_dataset = MLLFeatureDataset(dataset_name, phase='val', feature_extractor=feature_extractor)
    test_dataset = MLLFeatureDataset(dataset_name, phase='test', feature_extractor=feature_extractor)
    return train_dataset, val_dataset, test_dataset


optimizer = 'adamw'
lr = 0.001
batch_size = 64
num_workers = 0
max_epoch = 30
loss_type = 'bce'
device = 'cpu'
verbose = False
seed = 0
set_seed(seed)

dataset_name = 'VOC2007'
train_dataset, val_dataset, test_dataset = get_dataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
num_feature = train_dataset.features.shape[1]
alpha, beta, method = 0.001, 0.001, 'ld'

print('FLEM-D Start Testing!')
model = get_model()
model.load('.', epoch='FLEM-D VOC07')
y_pred, y_test = model.get_result(test_loader)
result = evaluation(y_test, y_pred)
for key in result.keys():
    print(f'{key}: {result[key]}')
