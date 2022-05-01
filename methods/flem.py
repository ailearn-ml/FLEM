from torch import nn
from methods.template import MLLTemplate
import torch
from utils.metrics import cosine_similarity


def kaiming_normal_init_net(net):
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


class LE(nn.Module):
    def __init__(self, num_feature, num_classes, hidden_dim=128):
        super(LE, self).__init__()
        self.fe1 = nn.Sequential(
            nn.Linear(num_feature, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.fe2 = nn.Linear(hidden_dim, hidden_dim)
        self.le1 = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.le2 = nn.Linear(hidden_dim, hidden_dim)
        self.de1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, num_classes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_classes),
        )
        self.de2 = nn.Linear(num_classes, num_classes)

    def forward(self, x, y):
        x = self.fe1(x) + self.fe2(self.fe1(x))
        y = self.le1(y) + self.le2(self.le1(y))
        d = torch.cat([x, y], dim=-1)
        d = self.de1(d) + self.de2(self.de1(d))
        return d


class FLEM(MLLTemplate):
    def __init__(self, num_classes, num_feature, loss_func, method='ld', alpha=0.01, beta=0.01,
                 max_epoch=None, optimizer='sgd', lr=0.1, momentum=0.9, weight_decay=1e-4,
                 threshold=0, adjust_lr=False, verbose=False, device='cuda:0'):
        super(FLEM, self).__init__(num_classes=num_classes, num_feature=num_feature,
                                   adjust_lr=adjust_lr, max_epoch=max_epoch, verbose=verbose, device=device)
        self.lr = lr
        self.lr_now = lr
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.loss_func = loss_func
        self.classifier = nn.Linear(self.num_feature, num_classes)
        self.le = LE(num_feature, num_classes)
        kaiming_normal_init_net(self.classifier)
        kaiming_normal_init_net(self.le)

        params_decay = (p for name, p in self.named_parameters() if 'bias' not in name)
        params_no_decay = (p for name, p in self.named_parameters() if 'bias' in name)
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                [{'params': params_decay, 'lr': lr, 'weight_decay': weight_decay},
                 {'params': params_no_decay, 'lr': lr}], amsgrad=True)
        if optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                [{'params': params_decay, 'lr': lr, 'weight_decay': weight_decay},
                 {'params': params_no_decay, 'lr': lr}], amsgrad=True)
        else:
            self.optimizer = torch.optim.SGD(
                [{'params': params_decay, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay},
                 {'params': params_no_decay, 'lr': lr, 'momentum': momentum}])
        self.to(self.device)

    def set_forward(self, x):
        x = self.classifier(x)
        return x

    def set_forward_loss(self, pred, le, x=None, y=None):
        assert y is not None
        loss_pred_cls = self.loss_func(pred, y)
        loss_pred_ld = nn.CrossEntropyLoss()(pred, torch.softmax(le.detach(), dim=1))
        loss_le_cls = self.loss_enhanced(le, pred, y)
        if self.method == 'ld':
            loss_le_spec = nn.CrossEntropyLoss()(le, torch.softmax(pred.detach(), dim=1))
        elif self.method == 'threshold':
            loss_le_spec = 0
            for i in range(pred.shape[0]):
                neg_index = y[i] == 0
                pos_index = y[i] == 1
                if torch.sum(pos_index) == 0: continue
                loss_le_spec += torch.maximum(le[i][neg_index][torch.argmax(le[i][neg_index])] - le[i][pos_index][
                    torch.argmin(le[i][pos_index])] + self.threshold, torch.tensor([0]).to(self.device))
        elif self.method == 'sim':
            with torch.no_grad():
                sim_x = cosine_similarity(x, x).detach()  # [n,n]
            sim_y = cosine_similarity(le, le)
            loss_le_spec = nn.MSELoss()(sim_y, sim_x)
        else:
            raise ValueError('Wrong E2ELE method!')
        loss_pred = self.alpha * loss_pred_ld + (1 - self.alpha) * loss_pred_cls
        loss_le = self.beta * loss_le_spec + (1 - self.beta) * loss_le_cls
        return loss_le + loss_pred

    def loss_enhanced(self, pred, teach, y):
        eps = 1e-7
        gamma1 = 0
        gamma2 = 1
        x_sigmoid = torch.sigmoid(pred)
        los_pos = y * torch.log(x_sigmoid.clamp(min=eps, max=1 - eps))
        los_neg = (1 - y) * torch.log((1 - x_sigmoid).clamp(min=eps, max=1 - eps))
        loss = los_pos + los_neg
        with torch.no_grad():
            teach_sigmoid = torch.sigmoid(teach)
            teach_pos = teach_sigmoid
            teach_neg = 1 - teach_sigmoid
            pt0 = teach_pos * y
            pt1 = teach_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = gamma1 * y + gamma2 * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        return -loss.mean()

    def train_loop(self, epoch, train_loader, log=None, print_freq=10):
        self.train()
        if not log:
            log = print
        self.epoch = epoch
        if self.adjust_lr:
            self.adjust_learning_rate()
        total_loss = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
            x = batch['image'].to(self.device)
            y = batch['labels'].to(self.device)
            pred = self.classifier(x)
            le = self.le(x.detach(), y)
            self.optimizer.zero_grad()
            loss = self.set_forward_loss(pred, le, x, y)
            loss.backward()
            self.clip_gradient()
            self.optimizer.step()
            total_loss += loss.item()
            num_samples += x.shape[0]
            if self.verbose and (i % print_freq) == 0:
                log('Epoch %d | Batch %d/%d | Loss %f' % (
                    epoch, i, len(train_loader), total_loss / num_samples))
        avg_loss = total_loss / num_samples
        return avg_loss
