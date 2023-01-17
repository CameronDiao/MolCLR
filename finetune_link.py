import os
import shutil
import sys
import yaml
import random
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper
from dataset.dataset_clique import MolCliqueDatasetWrapper
#from dataset.dataset_clique import MolCliqueDataset
from utils import clique
from utils import vocab

from utils.nt_xent import NTXentLoss

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
        dir_name = current_time + '_' + config['task_name']
        self.log_dir = os.path.join('finetune', dir_name)

        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        #print("Running on:", device)

        return device

    def _step(self, model, data, n_iter, cluster_idx, epoch=float('inf')):   
        # get the prediction
        preds, embs = model(data, cluster_idx)

        if self.config['dataset']['task'] == 'classification':
            # normalize projection feature vectors
            preds = F.normalize(preds, dim=1)
            embs = F.normalize(embs, dim=1)

            loss = self.criterion(preds, embs)
            loss += float(self.config['ortho_weight']) * _ortho_constraint(self.device, model.get_label_emb())

        return loss

    def _extract_clusters(self, batch, mol_to_cluster):
        cluster_idx = []
        for d in batch.to_data_list():
            if d.y.item() == 0:
                idx = mol_to_cluster[d.mol_index.item()]
            elif d.y.item() == 1:
                idx = self.config['num_clusters'] + mol_to_cluster[d.mol_index.item()]
            cluster_idx.append(idx)

        cluster_idx = torch.tensor(cluster_idx).to(self.device)
        
        return cluster_idx

    def _gen_clusters(self):
        full_data_loader = self.dataset.get_full_data_loader()
        mol_to_cluster = {}
        cluster_idx_0 = 0
        cluster_idx_1 = 0
        for batch in full_data_loader:
            for d in batch.to_data_list():
                if d.y.item() == 0:
                    mol_to_cluster[d.mol_index.item()] = cluster_idx_0
                    cluster_idx_0 = (cluster_idx_0 + 1) % self.config['num_clusters']
                elif d.y.item() == 1:
                    mol_to_cluster[d.mol_index.item()] = cluster_idx_1
                    cluster_idx_1 = (cluster_idx_1 + 1) % self.config['num_clusters']
        return mol_to_cluster

    def train(self):
        smiles_data, dropped_train_loader, train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        if self.config['task_name'] == 'MUV':
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            if len(torch.unique(labels)) < 2:
                self.roc_auc = 1.
                return
            labels = []
            for d in valid_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            if len(torch.unique(labels)) < 2:
                self.roc_auc = 1.
                return
            labels = []
            for d in test_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            if len(torch.unique(labels)) < 2:
                self.roc_auc = 1.
                return

        mol_to_cluster = self._gen_clusters()

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
            
            with torch.no_grad():
                label_feats = []
                labels = []
                
                for batch in train_loader:
                    batch = batch.to(self.device)
                    feat_emb, out_emb = model(batch)
                    label_feats.append(out_emb)
                    labels.append(batch.y)

                label_feats = torch.cat(label_feats)
                labels = torch.cat(labels)

                linit0 = torch.mean(label_feats[torch.nonzero(labels == 0)[:, 0]], dim=0)
                linit1 = torch.mean(label_feats[torch.nonzero(labels == 1)[:, 0]], dim=0)

                label_feats = torch.vstack((linit0, linit1)).to(self.device)

                cluster_indices = [0 for i in range(self.config['num_clusters'])]
                cluster_indices.extend([1 for i in range(self.config['num_clusters'])])

                label_feats = torch.index_select(label_feats, 0, torch.tensor(cluster_indices).to(self.device))

            from models.ginet_finetune_link import GINet
            model = GINet(self.config['dataset']['task'], self.config['cluster_mode'], self.config['num_clusters'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            model.init_label_emb(label_feats)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_molclr import GCN
            model = GCN(feat_dim = self.config['model']['feat_dim']).to(self.device)
            model = self._load_pre_trained_weights(model)

            with torch.no_grad():
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

            from models.gcn_finetune_link import GCN
            model = GCN(self.config['dataset']['task'], **self.config['model']).to(self.device)
            model = self._load_pre_trained_weights(model)
            model.init_label_emb(label_feats)

        layer_list = []
        for name, param in model.named_parameters():
            if 'out_lin' in name or 'prompt' in name:
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
                [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
                self.config['init_lr'], weight_decay=self.config['weight_decay']
        )

        model_checkpoints_folder = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(model_checkpoints_folder, exist_ok=True)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        ret_val = []
        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(dropped_train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)

                cluster_idx = self._extract_clusters(data, mol_to_cluster)

                loss = self._step(model, data, n_iter, cluster_idx, epoch=epoch_counter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_cls = self._validate(model, valid_loader, mol_to_cluster) 
                    ret_val.append(valid_cls)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                valid_n_iter += 1

        self._test(model, test_loader, mol_to_cluster)
        print("Test ROC-AUC: {}".format(self.roc_auc))
        self.roc_auc = sum(ret_val) / len(ret_val)
        # print("Average validation ROC-AUC: ", self.roc_auc)
        #self.roc_auc = best_valid_cls
        #print("Best validation ROC-AUC: ", self.roc_auc)

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

    def _validate(self, model, valid_loader, mol_to_cluster):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                cluster_idx = self._extract_clusters(data, mol_to_cluster)

                preds, __ = model(data, cluster_idx)

                if self.config['dataset']['task'] == 'classification':
                    if self.config['cluster_mode'] == 'fixed_assign':
                        preds = F.normalize(preds, dim=1)
                        embs = model.get_label_emb()
                        embs = F.normalize(embs, dim=1)

                        new_cluster_idx = cluster_idx.clone()
                        new_cluster_idx[new_cluster_idx > self.config['num_clusters'] - 1] -= self.config['num_clusters']

                        pred = torch.empty((data.num_graphs, 2)).to(self.device)
                        for i in range(self.config['num_clusters']):
                            num_i = torch.count_nonzero(new_cluster_idx == i).item()
                            if num_i > 0:
                                new_preds = preds[new_cluster_idx == i, :]
                                new_embs = torch.cat((embs[i].unsqueeze(0), embs[self.config['num_clusters']].unsqueeze(0)), dim=0)
                                similarities = torch.mm(new_preds, new_embs.T)
                                pred[new_cluster_idx == i, :] = F.softmax(similarities, dim=-1)
                    else:
                        preds = F.normalize(preds, dim=1)
                        embs = model.get_label_emb()
                        embs = F.normalize(embs, dim=1)
                        emb_0 = torch.index_select(embs, 0, torch.tensor([i for i in range(self.config['num_clusters'])]).to(self.device))
                        emb_0 = emb_0.mean(dim=0, keepdim=True)
                        emb_0 = F.normalize(emb_0, dim=1)
                        emb_1 = torch.index_select(embs, 0, torch.tensor([self.config['num_clusters'] + i for i in range(self.config['num_clusters'])]).to(self.device))
                        emb_1 = emb_1.mean(dim=0, keepdim=True)
                        emb_1 = F.normalize(emb_1, dim=1)
                        embs = torch.cat((emb_0, emb_1), dim=0)

                        similarities = torch.mm(preds, embs.T)
                        pred= F.softmax(similarities, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

        
        model.train()

        if self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            return roc_auc

    def _test(self, model, valid_loader, mol_to_cluster):
        model_path = os.path.join(self.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        #print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                cluster_idx = self._extract_clusters(data, mol_to_cluster)

                preds, __ = model(data, cluster_idx)

                if self.config['dataset']['task'] == 'classification':
                    if self.config['cluster_mode'] == 'fixed_assign':
                        preds = F.normalize(preds, dim=1)
                        embs = model.get_label_emb()
                        embs = F.normalize(embs, dim=1)

                        new_cluster_idx = cluster_idx.clone()
                        new_cluster_idx[new_cluster_idx > self.config['num_clusters'] - 1] -= self.config['num_clusters']

                        pred = torch.empty((data.num_graphs, 2)).to(self.device)
                        for i in range(self.config['num_clusters']):
                            num_i = torch.count_nonzero(new_cluster_idx == i).item()
                            if num_i > 0:
                                new_preds = preds[new_cluster_idx == i, :]
                                new_embs = torch.cat((embs[i].unsqueeze(0), embs[self.config['num_clusters']].unsqueeze(0)), dim=0)
                                similarities = torch.mm(new_preds, new_embs.T)
                                pred[new_cluster_idx == i, :] = F.softmax(similarities, dim=-1)
                    else:
                        preds = F.normalize(preds, dim=1)
                        embs = model.get_label_emb()
                        embs = F.normalize(embs, dim=1)
                        emb_0 = torch.index_select(embs, 0, torch.tensor([i for i in range(self.config['num_clusters'])]).to(self.device))
                        emb_0 = emb_0.mean(dim=0, keepdim=True)
                        emb_0 = F.normalize(emb_0, dim=1)
                        emb_1 = torch.index_select(embs, 0, torch.tensor([self.config['num_clusters'] + i for i in range(self.config['num_clusters'])]).to(self.device))
                        emb_1 = emb_1.mean(dim=0, keepdim=True)
                        emb_1 = F.normalize(emb_1, dim=1)
                        embs = torch.cat((emb_0, emb_1), dim=0)

                        similarities = torch.mm(preds, embs.T)
                        pred= F.softmax(similarities, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())
        
        model.train()

        if self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])

def main(config):
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

    for _ in range(5):
        for target in target_list:
            config['dataset']['target'] = target
            result = main(config)