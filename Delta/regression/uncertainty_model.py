import os
from time import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim
from torch.utils.data import DataLoader

from regression.featurize import SampleEntity
from regression.TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)
from regression.TreeConvolution.util import prepare_trees
import matplotlib.pyplot as plt

CUDA = torch.cuda.is_available()

T = 3

class MSEVAR(nn.Module):
    def __init__(self,var_weight=1):
        super(MSEVAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, pred, target,var):
        var_wei = (self.var_weight * var).reshape(-1,1)
        loss1 = torch.mul(torch.exp(-var_wei), (pred - target) ** 2)
        loss2 = var_wei
        loss3 = 0
        loss = (loss1 + loss2 + loss3)
        return loss.mean()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")

def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")

def collate_fn(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets

def collate(x):
    trees = []
    indexes = []
    for subx in x:
        tree, index = subx
        trees.append(tree)
        indexes.append(index)
    trees = torch.stack(trees, dim=0)
    indexes = torch.stack(indexes, dim=0)
    return (trees, indexes)

def transformer(x: SampleEntity):
    return x.get_feature()

def left_child(x: SampleEntity):
    return x.get_left()

def right_child(x: SampleEntity):
    return x.get_right()

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


class Net(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(Net, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None
        self.p = 0.1

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 256, self.p),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128, self.p),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64, self.p),
            TreeLayerNorm(),
            DynamicPooling(),
            
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(32, 1)
        )
        self.fc_v = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(32, 1)
        )

    def forward(self, trees):
        x = self.tree_conv(trees).float()
        y = self.fc(x).reshape(-1,)
        z = self.fc_v(x).reshape(-1,)
        return y, z

    def build_trees(self, feature, device):
        return prepare_trees(feature, transformer, left_child, right_child,  device=device)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.update = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
            self.update = True
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            # reset counter if validation loss improves
            self.counter = 0
            self.update = True
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            self.update = False
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True



class Model():
    def __init__(self, feature_generator, device) -> None:
        self._net = None
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self.device = device

    def load(self, path):
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        self._net = Net(self._input_feature_dim)
        if CUDA:
            self._net.load_state_dict(torch.load(_nn_path(path)))
        else:
            self._net.load_state_dict(torch.load(
                _nn_path(path), map_location=torch.device('cpu')))
        self._net.eval()

        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self._net.state_dict(), _nn_path(path))

        with open(_feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)
    
    def multi_forward(self, net, x, times):
        ys, zs = [], []
        for _ in range(times):
            y, z = net(x)
            ys.append(y)
            zs.append(z)
        y_avg = torch.stack(ys, dim=0).mean(dim=0)
        z_avg = torch.stack(zs, dim=0).mean(dim=0)
        return y_avg, z_avg
    
    def get_MC_samples(self, network, X, mc_times=64):
        network = network.to(self.device)
        network.eval()
        network.apply(apply_dropout)
        pred_v = []; a_u = []

        for t in range(mc_times):
            prediction, var = network(X)
            pred_v.append(prediction.cpu().detach().numpy())
            a_u.append(var.cpu().detach().numpy())

        pred_v = np.array(pred_v); a_u = np.array(a_u)
        a_u = np.sqrt(np.exp(np.mean(a_u, axis=0)))
        pred_mean = np.mean(pred_v, axis=0)
        e_u = np.sqrt(np.var(pred_v, axis=0))
        return pred_mean.squeeze(), a_u.squeeze(), e_u.squeeze()
    

    def fit(self, X, Y, test_x, test_y):
        if isinstance(Y, list):
            Y = np.array(Y)
            Y = Y.reshape(-1, 1)

        batch_size = 512
        stop_epoch = 100
        if self._net is None:
            input_feature_dim = len(X[0].get_feature())
            print("input_feature_dim:", input_feature_dim)
            self._net = Net(input_feature_dim)
            self._input_feature_dim = input_feature_dim
        pairs = []
        trees = self._net.build_trees(X, self.device)
        for i in range(len(Y)):
            pairs.append(((trees[0][i], trees[1][i]), Y[i]))
        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)
        
        optimizer = torch.optim.Adam(self._net.parameters(), lr=0.001)

        self._net = self._net.to(self.device)
        self._net.train()
        early_stopping = EarlyStopping(patience=stop_epoch)
        # loss_fn = torch.nn.MSELoss()
        loss_fn = MSEVAR()
        mse_fn = torch.nn.MSELoss()
        losses = []
        mses = []
        start_time = time()
        
        best_net = self._net
        for epoch in range(stop_epoch):
            loss_accum = 0
            mse_accum = 0
            for x, y in dataset:

                y = y.float().to(self.device).reshape(-1,)
                # tree = self._net.build_trees(x, self.device)
                tree = collate(x)
                y_pred, var = self.multi_forward(self._net, tree, T)
                # y_pred, var = self._net(tree)
                loss = loss_fn(y_pred, y, var)
                # loss = loss_fn(y_pred, y)
                mse = mse_fn(y_pred, y)
                # loss += 100 * mse
                # loss = mse_fn(y_pred, y)
                loss_accum += loss.item()
                mse_accum += mse.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            mse_accum /= len(dataset)
            print("Epoch", epoch, "training loss:", loss_accum, " mse:", mse_accum)
            losses.append(loss_accum)
            mses.append(mse_accum)
            # _, test_mse = self.test(test_x, test_y)
            early_stopping(loss_accum)
            if early_stopping.update:
                best_net = self._net
            if early_stopping.early_stop:
                stop_epoch = epoch+1
                break
        self._net = best_net
        print(f'INFO: Early stopping ! Epoch: {stop_epoch}')
        print("training time:", time() - start_time, "batch size:", batch_size)
        # print(f"[test] min mse: {min(test_mses)}")
        return mse_accum, stop_epoch

    def test(self, x, y):
        with torch.no_grad():
            # y = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(
            #             self.device, non_blocking=True).reshape(-1,)
            tree = self._net.build_trees(x, self.device)
            y_pred, a_var, e_var = self.get_MC_samples(self._net, tree, 1)
            # y_pred, var = self.multi_forward(self._net, tree, T)
            loss_fn = MSEVAR()
            mse_fn = torch.nn.MSELoss()
            # loss = loss_fn(y_pred, y, var)
            # mse = mse_fn(y_pred, y)
            mse = np.mean((y_pred-y)**2)
            # print("test mse: ", mse.item())
        return y_pred, np.sqrt(np.var(a_var))

    def predict(self, x):
        self._net = self._net.to(self.device)
        self._net.eval()

        if not isinstance(x, list):
            x = [x]
        tree = self._net.build_trees(x, self.device)
        pred = self._net(tree).cpu().detach().numpy()
        pred = self._feature_generator.normalizer.inverse_norm(pred, "Execution Time")
        return pred

        