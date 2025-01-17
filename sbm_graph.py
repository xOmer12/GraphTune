import torch
import numpy as np
import torch_geometric.utils
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json


class StochasticBlockModel:

    def __init__(self, p, q, config):

        self.p = p
        self.q = q
        self.task = config['name']
        train_left, train_right, train_labels, train_logits = self.read_data(dset_path=config['trainset'], get_logits=True)
        val_left, val_right, val_labels = self.read_data(dset_path=config['validset'])
        test_left, test_right, test_labels = self.read_data(dset_path=config['testset'])
        train_size, val_size, test_size = len(train_labels), len(val_labels), len(test_labels)
        dset_size = train_size + val_size + test_size

        self.left_samples = np.concatenate([train_left, val_left, test_left], axis=0)
        self.right_samples = np.concatenate([train_right, val_right, test_right], axis=0)
        self.labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)
        self.train_logits = train_logits
        self.train_entropies = self.entropy_vals = np.array([-prob*np.log(prob)-(1-prob)*np.log(1-prob) for prob in self.train_logits])

        self.train_mask = torch.zeros(dset_size, dtype=torch.bool)
        self.val_mask = torch.zeros(dset_size, dtype=torch.bool)
        self.test_mask = torch.zeros(dset_size, dtype=torch.bool)

        self.train_mask[:train_size] = True
        self.val_mask[train_size:train_size + val_size] = True
        self.test_mask[train_size + val_size:] = True


    def read_data(self, dset_path, get_logits=False):

        with open(dset_path) as f:
            lines = f.read().splitlines()

        split_lines = [line.split('\t') for line in lines] 
        split_lines = np.array(split_lines)
        left_samples = split_lines[:, 0].tolist()
        right_samples = split_lines[:, 1].tolist()
        labels = split_lines[:, 2].tolist()
        labels = [int(l) for l in labels]

        if get_logits:
            logits = split_lines[:, 3].tolist()
            logits = [float(l) for l in logits]
            return left_samples, right_samples, labels, logits
        
        else:
            return left_samples, right_samples, labels
        
    def create_conf_node_mask(self, entropy_threshold):
        return np.where(self.train_entropies < entropy_threshold)[0]
    
    def sample_edge_index(self, entropy_threshold):

        conf_nodes = self.create_conf_node_mask(entropy_threshold)
        val_nodes = np.where(self.val_mask == True)[0] 
        test_nodes = np.where(self.test_mask == True)[0]
        nodes_to_sample = np.concatenate([conf_nodes, val_nodes, test_nodes])
        probs = torch.tensor([[self.p, self.q], [self.q, self.p]], dtype=torch.float)
        row, col = torch.combinations(torch.tensor(nodes_to_sample), r=2, with_replacement=True).t()
        mask = torch.bernoulli(probs[self.labels[row], self.labels[col]]).to(torch.bool)
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        if len(edge_index.shape) != 2: 
            edge_index = edge_index.view(2, -1)
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        self.edge_index = edge_index
        

    def generate_graph(self, entropy_threshold, output_path=False):

        self.sample_edge_index(entropy_threshold=entropy_threshold)
        data = Data(x=None, edge_index=self.edge_index, y=torch.tensor(self.labels))

        data.train_mask = torch.tensor(self.train_mask)
        data.val_mask = torch.tensor(self.val_mask)
        data.test_mask = torch.tensor(self.test_mask)
        data.logits = self.train_logits
        data.left = self.left_samples
        data.right = self.right_samples
        data.samples = [l+'[sep]'+r for l, r in zip(data.left, data.right)]

        if output_path:
            torch.save(data, output_path)
        
        else:
            return data

class NoisySBM:

    def __init__(self, p, q, noisy_config, real_config):

        self.p = p
        self.q = q
        self.noisy_config = noisy_config
        self.real_config = real_config
        self.read_data(noisy_config['trainset'], real_config['trainset'])



    def read_data(self, noisy_path, real_path):

        with open(noisy_path) as f:
            noisy_lines = f.read().splitlines()

        with open(real_path) as f:
            real_lines = f.read().splitlines()

        split_noisy_lines = [line.split('\t') for line in noisy_lines]
        split_real_lines = [line.split('\t') for line in real_lines] 
        split_noisy_lines = np.array(split_noisy_lines)
        split_real_lines = np.array(split_real_lines)

        self.left_samples = split_noisy_lines[:, 0].tolist()
        self.right_samples = split_noisy_lines[:, 1].tolist()

        noisy_labels = split_noisy_lines[:, 2].tolist()
        self.noisy_labels = [int(l) for l in noisy_labels]

        real_labels = split_real_lines[:, 2].tolist()
        self.real_labels = [int(l) for l in real_labels]

        logits = split_noisy_lines[:, 3].tolist()
        self.logits = [float(l) for l in logits]
        self.train_entropies = np.array([-prob*np.log(prob)-(1-prob)*np.log(1-prob) for prob in self.logits])

                
    def create_mismatch_mask(self):
        mask = []
        for noisy, real in zip(self.noisy_labels, self.real_labels):
            if noisy != real:
                mask.append(1)
            else:
                mask.append(0)
        self.mismatch_mask = np.array(mask)

    def sample_edge_index(self):
        # Convert noisy_labels to tensor if it isn't already
        if not isinstance(self.noisy_labels, torch.Tensor):
            self.noisy_labels = torch.tensor(self.noisy_labels, dtype=torch.long)
        
        nodes_to_sample = np.arange(len(self.noisy_labels))
        probs = torch.tensor([[self.p, self.q], [self.q, self.p]], dtype=torch.float)
        
        # Convert to tensor and ensure long type
        row, col = torch.combinations(torch.tensor(nodes_to_sample, dtype=torch.long), r=2, with_replacement=True).t()
        
        # Get labels and ensure they're long type for indexing
        row_labels = self.noisy_labels[row].long()
        col_labels = self.noisy_labels[col].long()
        
        # Use the labels for indexing
        mask = torch.bernoulli(probs[row_labels, col_labels]).to(torch.bool)
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        
        if len(edge_index.shape) != 2:
            edge_index = edge_index.view(2, -1)
        
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        self.edge_index = edge_index
    
    def generate_graph(self):
        
        self.create_mismatch_mask()
        self.sample_edge_index()
        data = Data(x=None, edge_index=self.edge_index, y=torch.tensor(self.noisy_labels))
        data.real_labels = torch.tensor(self.real_labels)
        data.logits = self.logits
        data.left = self.left_samples
        data.right = self.right_samples
        data.samples = [l+'[sep]'+r for l, r in zip(data.left, data.right)]
        data.mismatch_mask = self.mismatch_mask

        return data
        


if __name__ == '__main__':

    task = "Dirty/iTunes-AmazonBert"
    og_task = task.replace("Bert", "")
    with open('task_configs.json', 'r') as file:
        configs = json.load(file)
    
    configs = {config['name']: config for config in configs}
    noisy_config = configs[task]
    real_config = configs[og_task]

    sbm = NoisySBM(p=0.75, q=0.25, noisy_config=noisy_config, real_config=real_config)
    graph = sbm.generate_graph()
    print(graph.mismatch_mask)
    