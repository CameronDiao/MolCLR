import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from matplotlib.lines import Line2D

import torch
from torch import nn
import torch.nn.functional as F
#from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

import dgl

from dataset.dataset_test import MolTestDatasetWrapper
from dataset.dataset_clique import MolCliqueDatasetWrapper
#from dataset.dataset_clique import MolCliqueDataset
from utils import clique
from utils import vocab


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))

def _plot_grad_flow(named_parameters, epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            n_label = n.replace('self.', '').replace('.weight', '')
            layers.append(n_label)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'],
                loc='upper right')
    plt.savefig('plots/grad_flow_{}.png'.format(epoch), bbox_inches="tight")

def _gen_clique_to_mol(clique_list, mol_to_clique):
    clique_to_mol = defaultdict(list)
    for mol in mol_to_clique:
        for clique in mol_to_clique[mol]:
            clique_to_mol[clique_list.index(clique)].append(mol)
    return clique_to_mol

def _get_training_molecules(train_loader):
    train_mol = []
    for data in train_loader:
        for d in data.to_data_list():
            train_mol.append(d.mol_index.item())
    return train_mol

def _ortho_constraint(device, prompt):
    return torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))

#def _ortho_constraint(device, prompt):
#    p = prompt.detach().clone()
#    cols = prompt.shape[1]
#    rows = prompt.shape[0]
#    w1 = p.view(-1, cols)
#    wt = torch.transpose(w1, 0, 1)
#    m = torch.matmul(wt, w1)
#    identity = torch.eye(cols, device=device)
#
#    w_tmp = (m - identity)
#    height = w_tmp.size(0)
#    u = F.normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
#    v = F.normalize(torch.matmul(w_tmp.T, u), dim=0, eps=1e-12)
#    u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
#    sigma = torch.dot(u, torch.matmul(w_tmp, v))
#
#    return (torch.norm(sigma, 2))**2


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        #print("Running on:", device)

        return device

    def _step(self, model, data, n_iter, mol_idx, clique_idx, epoch=float('inf')):   
        # get the prediction
        __, pred = model(data, mol_idx, clique_idx)
        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
            loss += self.config['ortho_weight'] * _ortho_constraint(self.device, model.get_label_emb())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def _gen_cliques(self, smiles_data):
        mol_to_clique = {}
        clique_set = set()
        if self.config['vocab'] == 'mgssl':
            for i, m in enumerate(smiles_data):
                mol_to_clique[i] = {}
                mol = clique.get_mol(m)
                cliques, edges = clique.brics_decomp(mol)
                if len(edges) <= 1:
                    cliques, edges = clique.tree_decomp(mol)
                for c in cliques:
                    cmol = clique.get_clique_mol(mol, c)
                    cs = clique.get_smiles(cmol)
                    clique_set.add(cs)
                    if cs not in mol_to_clique[i]:
                        mol_to_clique[i][cs] = 1
                    else:
                        mol_to_clique[i][cs] += 1
        elif self.config['vocab'] == 'junction':
            for i, m in enumerate(smiles_data):
                mol_to_clique[i] = {}
                mol = vocab.get_mol(m)
                cliques, edges = vocab.tree_decomp(mol)
                for c in cliques:
                    cmol = vocab.get_clique_mol(mol, c)
                    cs = vocab.get_smiles(cmol)
                    clique_set.add(cs)
                    if cs not in mol_to_clique[i]:
                        mol_to_clique[i][cs] = 1
                    else:
                        mol_to_clique[i][cs] += 1
        return list(clique_set), mol_to_clique

    def _filter_cliques(self, threshold, train_loader, clique_list, mol_to_clique, clique_to_mol):
        train_mol = _get_training_molecules(train_loader)
        
        fil_clique_list = []
        for i, d in enumerate(clique_list):
            if sum(mol in train_mol for mol in clique_to_mol[i]) <= threshold:
                fil_clique_list.append(d)
        
        tmol_to_clique = deepcopy(mol_to_clique)
        for mol in mol_to_clique:
            for clique in mol_to_clique[mol].keys():
                if clique in fil_clique_list:
                    if self.config['init'] == 'uniform':
                        tmol_to_clique[mol]['EMP'] = 1
                    del tmol_to_clique[mol][clique]
        
        mol_to_clique = deepcopy(tmol_to_clique)
        emp_mol = []
        for mol in tmol_to_clique:
            if self.config['init'] == 'uniform':
                if all('EMP' in clique for clique in tmol_to_clique[mol].keys()):
                    emp_mol.append(mol)
            elif self.config['init'] == 'zeros':
                if len(tmol_to_clique[mol]) == 0:
                    mol_to_clique[mol]['EMP'] = 1
                    emp_mol.append(mol)

        clique_list = list(set(clique_list) - set(fil_clique_list))
        return emp_mol, clique_list, mol_to_clique

    def _extract_train_cliques(self, batch, mol_to_clique, clique_list):
        mol_idx = []
        clique_idx = []
        for i, d in enumerate(batch.to_data_list()):
            for clique in mol_to_clique[d.mol_index.item()].keys():
                mol_idx.append(i)
                clique_idx.append(clique_list.index(clique))

        mol_idx = torch.tensor(mol_idx).to(self.device)
        clique_idx = torch.tensor(clique_idx).to(self.device)

        #motif_samples = motif_embed(clique_idx).to(self.device)

        #return mol_idx, motif_samples
        return mol_idx, clique_idx

    def _extract_test_cliques(self, batch, mol_to_clique, clique_list):
        mol_idx = []
        clique_idx = []
        for i, d in enumerate(batch.to_data_list()):
            for clique in mol_to_clique[d.mol_index.item()].keys():
                mol_idx.append(i)
                clique_idx.append(clique_list.index(clique))

        mol_idx = torch.tensor(mol_idx).to(self.device)
        clique_idx = torch.tensor(clique_idx).to(self.device)

        #motif_samples = motif_emb_tensor.index_select(0, clique_idx).to(self.device)
        
        #return mol_idx, motif_samples
        return mol_idx, clique_idx

    def train(self):
        smiles_data, train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        #full_data_loader = self.dataset.get_full_data_loader()

        clique_list, mol_to_clique = self._gen_cliques(smiles_data)
        clique_to_mol = _gen_clique_to_mol(clique_list, mol_to_clique)
        emp_mol, clique_list, mol_to_clique = self._filter_cliques(self.config['threshold'], train_loader, clique_list, mol_to_clique, clique_to_mol)
        num_motifs = len(clique_list) + 1
        #num_motifs = len(clique_list)

        clique_dataset = MolCliqueDatasetWrapper(clique_list, self.config['batch_size'], self.config['dataset']['num_workers'])
        clique_loader = clique_dataset.get_data_loaders()

        self.normalizer = None
      
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            #print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_molclr import GINet
            model = GINet(feat_dim = self.config['model']['feat_dim']).to(self.device)
            #model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            
            #mol_to_feats = {}
            #for data in full_data_loader:
            #    data = data.to(self.device)
            #    __, emb= model(data)
            #    for i,d in enumerate(data.to_data_list()):
            #        mol_to_feats[d.mol_index.item()] = emb[i, :]

            #motif_feats = []
            #with torch.no_grad():
            #    for i in range(len(clique_list)):
            #        mfeats = [mol_to_feats[mol] for mol in clique_to_mol[i]]
            #        motif_feats.append(torch.mean(torch.stack(mfeats), dim=0))
                    
            #    motif_feats = torch.stack(motif_feats)

            with torch.no_grad():

                motif_feats = []
                for c in clique_loader:
                    c = c.to(self.device)
                    __, emb = model(c)
                    motif_feats.append(emb)
            
                motif_feats = torch.cat(motif_feats)

                clique_list.append("EMP")

                label_feats = []
                labels = []
                for d in train_loader:
                    d = d.to(self.device)
                    feat_emb, out_emb = model(d)
                    label_feats.append(out_emb)
                    labels.append(d.y)

                label_feats = torch.cat(label_feats)
                labels = torch.cat(labels)

                linit0 = torch.mean(label_feats[torch.nonzero(labels == 0)[:, 0]], dim=0)
                linit1 = torch.mean(label_feats[torch.nonzero(labels == 1)[:, 0]], dim=0)

                label_feats = torch.vstack((linit0, linit1)).to(self.device)

                dummy_motif = torch.zeros((1, motif_feats.shape[1])).to(self.device)
                if self.config['init'] == 'uniform':
                    nn.init.xavier_uniform_(dummy_motif)
                motif_feats = torch.cat((motif_feats, dummy_motif), dim=0)

            from models.ginet_finetune_mp_link import GINet
            model = GINet(num_motifs, self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            model.init_clique_emb(motif_feats)
            model.init_label_emb(label_feats)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_molclr import GCN
            model = GCN(feat_dim = self.config['model']['feat_dim']).to(self.device)
            model = self._load_pre_trained_weights(model)

            with torch.no_grad():
                motif_feats = []
                for c in clique_loader:
                    c = c.to(self.device)
                    __, emb = model(c)
                    motif_feats.append(emb)

                motif_feats = torch.cat(motif_feats)

                clique_list.append('EMP')

                label_feats = []
                labels = []
                for d in train_loader:
                    d = d.to(self.device)
                    feat_emb, out_emb = model(d)
                    label_feats.append(out_emb)
                    labels.append(d.y)

                label_feats = torch.cat(label_feats)
                labels = torch.cat(labels)

                linit0 = torch.mean(label_feats[torch.nonzero(labels == 0)[:, 0]], dim=0)
                linit1 = torch.mean(label_feats[torch.nonzero(labels == 1)[:, 0]], dim=0)

                label_feats = torch.vstack((linit0, linit1)).to(self.device)

                dummy_motif = torch.zeros((1, motif_feats.shape[1])).to(self.device)
                if self.config['init'] == 'uniform':
                    nn.init.xavier_uniform_(dummy_motif)
                motif_feats = torch.cat((motif_feats, dummy_motif), dim=0)

            from models.gcn_finetune_mp_link import GCN
            model = GCN(num_motifs, self.config['dataset']['task'], **self.config['model']).to(self.device)
            model = self._load_pre_trained_weights(model)
            model.init_clique_emb(motif_feats)
            model.init_label_emb(label_feats)

        layer_list = []
        for name, param in model.named_parameters():
            #if 'motif' in name or 'conc' in name or 'pred' in name or 'prompt' in name:
            if 'clique' in name or 'motif' in name or 'conc' in name or 'pred' in name or 'prompt' in name:    
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        #def motif_initializer(emb):
        #    emb[:] = motif_feats
        #    return emb
        #motif_embed = dgl.nn.NodeEmbedding(motif_feats.shape[0], motif_feats.shape[1], name="motif_embed",
        #                                   init_func=motif_initializer)
        #print("motif embedding is in ", motif_embed.emb_tensor.device)

        optimizer = torch.optim.Adam(
                [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
                self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )
        #motif_optimizer = dgl.optim.SparseAdam(params=[motif_embed], lr=self.config['init_lr'])

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        ret_val = []

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                data = data.to(self.device)
            
                mol_idx, clique_idx = self._extract_train_cliques(data, mol_to_clique, clique_list)

                optimizer.zero_grad()
                #motif_optimizer.zero_grad()

                loss = self._step(model, data, n_iter, mol_idx, clique_idx, epoch=epoch_counter)

                #if n_iter % self.config['log_every_n_steps'] == 0:
                #    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                #    print(epoch_counter, bn, loss.detach().item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                #nn.utils.clip_grad_norm_(model.parameters(), 32)
                optimizer.step()
                #motif_optimizer.step()
                
                n_iter += 1

            #_plot_grad_flow(model.named_parameters(), epoch_counter)
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader, 
                                                           clique_list, mol_to_clique)
                    ret_val.append(valid_cls)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader,
                                                           clique_list, mol_to_clique)
                    ret_val.append(valid_rgr)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                #self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        self._test(model, test_loader, clique_list, mol_to_clique)
        print("Test ROC-AUC: ", self.roc_auc)
        self.roc_auc = sum(ret_val) / len(ret_val)
        print("Average validation ROC-AUC: ", self.roc_auc)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            #print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, clique_list, mol_to_clique):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                mol_idx, clique_idx = self._extract_test_cliques(data, mol_to_clique, clique_list)

                __, pred = model(data, mol_idx, clique_idx)
                loss = self._step(model, data, bn, mol_idx, clique_idx)

                valid_loss += loss.detach().item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                #print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                #print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            #print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, valid_loader, clique_list, mol_to_clique):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        #print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                mol_idx, clique_idx = self._extract_test_cliques(data, mol_to_clique, clique_list)

                __, pred = model(data, mol_idx, clique_idx)
                loss = self._step(model, data, bn, mol_idx, clique_idx)

                test_loss += loss.detach().item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                #print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                #print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:,1])

def main(config, run):
    #torch.manual_seed(42)
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

    fine_tune = FineTune(dataset, config)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return fine_tune.mae
        else:
            return fine_tune.rmse


if __name__ == "__main__":
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    for run in range(10):
        results_list = []
        for target in target_list:
            config['dataset']['target'] = target
            result = main(config, run)
            results_list.append([target, result])

        print(results_list)
    #os.makedirs('experiments', exist_ok=True)
    #df = pd.DataFrame(results_list)
    #df.to_csv(
    #    'experiments/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']), 
    #    mode='a', index=False, header=False
    #)
