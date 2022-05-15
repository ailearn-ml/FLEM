import os
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


def _train():
    print('Start Training!')
    model = get_model()
    best_Hamming = 1
    for epoch in range(max_epoch):
        avg_loss = model.train_loop(epoch, train_loader, log=print, print_freq=10)
        result = model.test_loop(val_loader, return_loss=False)
        Hamming = result['Hamming']
        print('Epoch %d training done | Hamming: %.4f | Loss: %f | lr: %f' % (epoch, Hamming, avg_loss, model.lr_now))
        if Hamming < best_Hamming:
            best_Hamming = Hamming
            model.save(train_dir, epoch='best_Hamming')
        if epoch == max_epoch - 1:
            model.save(train_dir, epoch=epoch)


def _test():
    print('Start Testing!')
    model = get_model()
    model.load(train_dir, epoch='best_Hamming')
    y_pred, y_test = model.get_result(test_loader)
    result = evaluation(y_test, y_pred)
    for key in result.keys():
        print(f'{key}: {result[key]}')


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

algorithm = 'flem'
method = 'ld'
alpha = 0.001
beta = 0.001
dataset_name = 'VOC2007'

train_dataset, val_dataset, test_dataset = get_dataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
num_feature = train_dataset.features.shape[1]
train_dir = os.path.join('save', dataset_name, f'{algorithm}_{method}_{alpha}_{beta}',
                         f'{optimizer}_{lr}_{batch_size}_{max_epoch}_{loss_type}_{seed}',
                         'train')
log_dir = os.path.join('save', dataset_name, f'{algorithm}_{method}_{alpha}_{beta}',
                       f'{optimizer}_{lr}_{batch_size}_{max_epoch}_{loss_type}_{seed}', 'log')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print = Logger(f'{log_dir}/log.txt').logger.warning
print(
    f'{dataset_name}, {algorithm}, {method}, {alpha}, {beta}, {optimizer}, {lr}, {batch_size}, {max_epoch}, {seed}')

if not os.path.exists(os.path.join(train_dir, f'{max_epoch - 1}.tar')):
    _train()
_test()
