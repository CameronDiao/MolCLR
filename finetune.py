import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

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
from utils.clique import get_mol, get_smiles, sanitize, get_clique_mol, brics_decomp

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
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter, mol_idx, motif_samples, label_samples):   
        # get the prediction
        __, pred = model(data, mol_idx, motif_samples, label_samples)
        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def _gen_cliques(self, smiles_data):
        mol_to_clique = {}
        clique_set = set()
        for i, m in enumerate(smiles_data):
            mol_to_clique[i] = {}
            mol = get_mol(m)
            cliques, __  = brics_decomp(mol)
            for c in cliques:
                cmol = get_clique_mol(mol, c)
                cs = get_smiles(cmol)
                clique_set.add(cs)
                if cs not in mol_to_clique[i]:
                    mol_to_clique[i][cs] = 1
                else:
                    mol_to_clique[i][cs] += 1
        return list(clique_set), mol_to_clique

    def train(self):
        smiles_data, train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        #full_data_loader = self.dataset.get_full_data_loader()

        clique_list, mol_to_clique = self._gen_cliques(smiles_data)
        print("Finished generating motif vocabulary")
        #clique_to_mol = _gen_clique_to_mol(clique_list, mol_to_clique)
        num_motifs = len(clique_list)

        clique_dataset = MolCliqueDatasetWrapper(clique_list, self.config['batch_size'], self.config['dataset']['num_workers'])
        clique_loader = clique_dataset.get_data_loaders()

        self.normalizer = None
      
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            
            motif_feats = []
            for c in clique_loader:
                c = c.to(self.device)
                emb, __ = model(c)
                motif_feats.append(emb)
            
            with torch.no_grad():               
                motif_feats = torch.cat(motif_feats)

            clique_list.append("EMP0")
            clique_list.append("EMP1")

            label_feats = []
            labels = []
            for d in train_loader:
                d = d.to(self.device)
                emb, __ = model(d)
                label_feats.append(emb)
                labels.append(d.y)

            with torch.no_grad():
                label_feats = torch.cat(label_feats)
                labels = torch.cat(labels)

                linit1 = torch.mean(label_feats[torch.nonzero(labels == 0)[:, 0]], dim=0)
                linit2 = torch.mean(label_feats[torch.nonzero(labels == 1)[:, 0]], dim=0)

                label_feats = torch.vstack((linit1, linit2)).to(self.device)

            from models.ginet_finetune_mp_link import GINet
            model = GINet(num_motifs, self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
            #model.init_motif_emb(motif_feats)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_lin' in name:
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        def motif_initializer(emb):
            emb[:] = motif_feats
            return emb
        motif_embed = dgl.nn.NodeEmbedding(motif_feats.shape[0], motif_feats.shape[1], name="motif_embed",
                                           init_func=motif_initializer)
        print("motif embedding is in ", motif_embed.emb_tensor.device)

        def label_initializer(emb):
            emb[:] = label_feats
            return emb
        label_embed = dgl.nn.NodeEmbedding(label_feats.shape[0], label_feats.shape[1], name="label_embed",
                                           init_func=label_initializer)
        print("label embedding is in ", label_embed.emb_tensor.device)

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )
        motif_optimizer = dgl.optim.SparseAdam(params=[motif_embed], lr=self.config['init_base_lr'] * 10)
        label_optimizer = dgl.optim.SparseAdam(params=[label_embed], lr=self.config['init_base_lr'] * 10)

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

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                data = data.to(self.device)
             
                mol_idx = []
                clique_idx = []
                for i, d in enumerate(data.to_data_list()):
                    for clique in mol_to_clique[d.mol_index.item()].keys():
                        mol_idx.append(i)
                        clique_idx.append(clique_list.index(clique))
                mol_idx.extend([i for i in range(max(mol_idx) + 1)])
                mol_idx = torch.tensor(mol_idx).to(self.device)
                clique_idx = torch.tensor(clique_idx)
                
                motif_samples = motif_embed(clique_idx).to(self.device)

                lab0 = label_embed(torch.zeros(data.num_graphs, dtype=torch.long)).unsqueeze(0)
                lab1 = label_embed(torch.ones(data.num_graphs, dtype=torch.long)).unsqueeze(0)
                label_samples = torch.cat((lab0, lab1), dim=0).to(self.device)
                #label_samples = label_embed(torch.zeros(data.num_graphs, dtype=torch.long)).unsqueeze(0)
                #label_samples = torch.cat((label_samples, label_embed(torch.ones(data.num_graphs,\
                #        dtype=torch.long)).unsqueeze(0)), dim=0).to(self.device)

                optimizer.zero_grad()
                motif_optimizer.zero_grad()
                label_optimizer.zero_grad()

                loss = self._step(model, data, n_iter,  mol_idx, motif_samples, label_samples)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                motif_optimizer.step()
                label_optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader, motif_embed.emb_tensor,
                                                           label_embed.emb_tensor, clique_list, mol_to_clique)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader, motif_embed.emb_tensor,
                                                           label_embed.emb_tensor, clique_list, mol_to_clique)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        self._test(model, test_loader, motif_embed.emb_tensor, label_embed.emb_tensor, clique_list, mol_to_clique)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, motif_emb_tensor, label_emb_tensor, clique_list, mol_to_clique):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                mol_idx = []
                clique_idx = []
                for i, d in enumerate(data.to_data_list()):
                    for clique in mol_to_clique[d.mol_index.item()].keys():
                        mol_idx.append(i)
                        clique_idx.append(clique_list.index(clique))
                mol_idx.extend([i for i in range(max(mol_idx) + 1)])
                mol_idx = torch.tensor(mol_idx).to(self.device)
                clique_idx = torch.tensor(clique_idx)

                motif_samples = motif_emb_tensor.index_select(0, clique_idx).to(self.device)

                lab0 = label_emb_tensor.index_select(0, torch.zeros(data.num_graphs, dtype=torch.long))\
                        .unsqueeze(0)
                lab1 = label_emb_tensor.index_select(0, torch.ones(data.num_graphs, dtype=torch.long))\
                        .unsqueeze(0)
                label_samples = torch.cat((lab0, lab1), dim=0).to(self.device)
                #label_samples = label_emb_tensor.index_select(0, torch.zeros(data.num_graphs, dtype=torch.long))\
                #        .unsqueeze(0)
                #label_samples = torch.cat((label_samples, label_emb_tensor.index_select(0, torch.ones(data.num_graphs,\
                #        dtype=torch.long)).unsqueeze(0)), dim=0).to(self.device)

                __, pred = model(data, mol_idx, motif_samples, label_samples)
                loss = self._step(model, data, bn, mol_idx, motif_samples, label_samples)

                valid_loss += loss.item() * data.y.size(0)
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
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader, motif_emb_tensor, label_emb_tensor, clique_list, mol_to_clique):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                mol_idx = []
                clique_idx = []
                for i, d in enumerate(data.to_data_list()):
                    for clique in mol_to_clique[d.mol_index.item()].keys():
                        mol_idx.append(i)
                        clique_idx.append(clique_list.index(clique))
                mol_idx.extend([i for i in range(max(mol_idx) + 1)])
                mol_idx = torch.tensor(mol_idx).to(self.device)
                clique_idx = torch.tensor(clique_idx)

                motif_samples = motif_emb_tensor.index_select(0, clique_idx).to(self.device)

                lab0 = label_emb_tensor.index_select(0, torch.zeros(data.num_graphs, dtype=torch.long))\
                        .unsqueeze(0)
                lab1 = label_emb_tensor.index_select(0, torch.ones(data.num_graphs, dtype=torch.long))\
                        .unsqueeze(0)
                label_samples = torch.cat((lab0, lab1), dim=0).to(self.device)
                #label_samples = label_emb_tensor.index_select(0, torch.zeros(data.num_graphs, dtype=torch.long))\
                #        .unsqueeze(0)
                #label_samples = torch.cat((label_samples, label_emb_tensor.index_select(0, torch.ones(data.num_graphs,\
                #        dtype=torch.long)).unsqueeze(0)), dim=0).to(self.device)

                __, pred = model(data, mol_idx, motif_samples, label_samples)
                loss = self._step(model, data, bn, mol_idx, motif_samples, label_samples)

                test_loss += loss.item() * data.y.size(0)
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
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)


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

    for _ in range(10):
        results_list = []
        for target in target_list:
            config['dataset']['target'] = target
            result = main(config)
            results_list.append([target, result])

        print(results_list)
    #os.makedirs('experiments', exist_ok=True)
    #df = pd.DataFrame(results_list)
    #df.to_csv(
    #    'experiments/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']), 
    #    mode='a', index=False, header=False
    #)
