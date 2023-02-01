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

    def _step(self, model, data, n_iter, mol_idx, clique_idx, cluster_idx, epoch=float('inf')):   
        # get the prediction
        preds, embs = model(data, mol_idx, clique_idx, cluster_idx, self.device)
        if self.config['dataset']['task'] == 'classification':
            # normalize projection feature vectors
            # preds = F.normalize(preds, dim=1)
            embs = F.normalize(embs, dim=1)

            loss = self.criterion(preds, embs)
            loss += float(self.config['ortho_weight']) * _ortho_constraint(self.device, model.get_label_emb())

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
        elif self.config['vocab'] == 'brics':
            for i, m in enumerate(smiles_data):
                mol_to_clique[i] = {}
                mol = vocab.get_mol(m)
                cliques, edges = clique.simple_brics_decomp(mol)
                for c in cliques:
                    cmol = clique.get_clique_mol(mol, c)
                    cs = clique.get_smiles(cmol)
                    clique_set.add(cs)
                    if cs not in mol_to_clique[i]:
                        mol_to_clique[i][cs] = 1
                    else:
                        mol_to_clique[i][cs] += 1

        
        return list(clique_set), mol_to_clique

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

    def train(self):
        smiles_data, dropped_train_loader, train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        #full_data_loader = self.dataset.get_full_data_loader()

        labels = []
        for d in train_loader:
            labels.append(d.y)
        labels = torch.cat(labels)
        if len(torch.unique(labels)) < 2:
            self.roc_auc = 0.
            return
        labels = []
        for d in valid_loader:
            labels.append(d.y)
        labels = torch.cat(labels)
        if len(torch.unique(labels)) < 2:
            self.roc_auc = 0.
            return
        labels = []
        for d in test_loader:
            labels.append(d.y)
        labels = torch.cat(labels)
        if len(torch.unique(labels)) < 2:
            self.roc_auc = 0.
            return

        mol_to_cluster = self._gen_clusters()

        clique_list, mol_to_clique = self._gen_cliques(smiles_data)
        clique_to_mol = _gen_clique_to_mol(clique_list, mol_to_clique)
        emp_mol, clique_list, mol_to_clique = self._filter_cliques(self.config['threshold'], train_loader, clique_list, mol_to_clique, clique_to_mol)
        num_motifs = len(clique_list) + 1

        clique_dataset = MolCliqueDatasetWrapper(clique_list, num_motifs, self.config['dataset']['num_workers'])
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

            with torch.no_grad():
                motif_feats = []
                for c in clique_loader:
                    c = c.to(self.device)
                    __, emb = model(c)
                    motif_feats.append(emb)
            
                motif_feats = torch.cat(motif_feats)

                clique_list.append("EMP")

                dummy_motif = torch.zeros((1, motif_feats.shape[1])).to(self.device)
                if self.config['init'] == 'uniform':
                    nn.init.xavier_uniform_(dummy_motif)
                
                motif_feats = torch.cat((motif_feats, dummy_motif), dim=0)

                label_feats = []
                labels = []
                clique_idx_0 = []
                clique_idx_1 = []
                for batch in train_loader:
                    batch = batch.to(self.device)

                    feat_emb, out_emb = model(batch)
                    label_feats.append(out_emb)
                    labels.append(batch.y)

                    for i, d in enumerate(batch.to_data_list()):
                        if d.y.item() == 0:
                            for clique in mol_to_clique[d.mol_index.item()].keys():
                                clique_idx_0.append(clique_list.index(clique))
                        elif d.y.item() == 1:
                            for clique in mol_to_clique[d.mol_index.item()].keys():
                                clique_idx_1.append(clique_list.index(clique))

                label_feats = torch.cat(label_feats)
                labels = torch.cat(labels)

                linit0 = torch.mean(label_feats[torch.nonzero(labels == 0)[:, 0]], dim=0)
                linit1 = torch.mean(label_feats[torch.nonzero(labels == 1)[:, 0]], dim=0)

                label_feats = torch.vstack((linit0, linit1)).to(self.device)
                        
                clique_idx_0 = torch.tensor(clique_idx_0)
                clique_idx_1 = torch.tensor(clique_idx_1)

                clique_feats_0 = torch.index_select(motif_feats, 0, clique_idx_0.to(self.device))
                clique_feats_0 = clique_feats_0.mean(dim=0, keepdim=True)
                clique_feats_1 = torch.index_select(motif_feats, 0, clique_idx_1.to(self.device))
                clique_feats_1 = clique_feats_1.mean(dim=0, keepdim=True)
                clique_feats = torch.vstack((clique_feats_0, clique_feats_1)).to(self.device)

                label_feats = label_feats + clique_feats
                cluster_indices = [0 for i in range(self.config['num_clusters'])]
                cluster_indices.extend([1 for i in range(self.config['num_clusters'])])

                label_feats = torch.index_select(label_feats, 0, torch.tensor(cluster_indices).to(self.device))
                # label_feats = F.normalize(label_feats, dim=1)

            from models.ginet_finetune_mp_link import GINet
            model = GINet(num_motifs, self.config['dataset']['task'], self.config['cluster_mode'], self.config['num_clusters'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            model.init_clique_emb(motif_feats)
            model.init_label_emb(label_feats)
        else:
            raise ValueError('Unsupported model type.')

        layer_list = []
        for name, param in model.named_parameters():
            if 'clique' in name or 'motif' in name or 'label' in name:
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        for p in base_params:
            p.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.config['init_lr'], weight_decay = self.config['weight_decay']
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
            
                mol_idx, clique_idx = self._extract_train_cliques(data, mol_to_clique, clique_list)
                cluster_idx = self._extract_clusters(data, mol_to_cluster)  

                loss = self._step(model, data, n_iter, mol_idx, clique_idx, cluster_idx, epoch=epoch_counter)

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
                    valid_cls = self._validate(model, valid_loader, clique_list, mol_to_clique, mol_to_cluster)
                    ret_val.append(valid_cls)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                
                #self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        self._test(model, test_loader, clique_list, mol_to_clique, mol_to_cluster)
        print(self.roc_auc)
        self.roc_auc = sum(ret_val) / len(ret_val)

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

    def _validate(self, model, valid_loader, clique_list, mol_to_clique, mol_to_cluster):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                mol_idx, clique_idx = self._extract_test_cliques(data, mol_to_clique, clique_list)
                cluster_idx = self._extract_clusters(data, mol_to_cluster)

                preds, __ = model(data, mol_idx, clique_idx, cluster_idx, self.device)

                if self.config['dataset']['task'] == 'classification':
                    # preds = F.normalize(preds, dim=1)
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
            roc_auc = roc_auc_score(labels, predictions[:,1])
            return roc_auc

    def _test(self, model, valid_loader, clique_list, mol_to_clique, mol_to_cluster):
        model_path = os.path.join(self.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        #print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                mol_idx, clique_idx = self._extract_test_cliques(data, mol_to_clique, clique_list)
                cluster_idx = self._extract_clusters(data, mol_to_cluster)

                preds, __ = model(data, mol_idx, clique_idx, cluster_idx, self.device)

                if self.config['dataset']['task'] == 'classification':
                    # preds = F.normalize(preds, dim=1)
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
        for target in target_list:
            config['dataset']['target'] = target
            result = main(config, run)
