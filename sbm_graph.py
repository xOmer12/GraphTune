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


if __name__ == '__main__':

    task = "Structured/Amazon-GoogleBert"
    with open('task_configs.json', 'r') as file:
        configs = json.load(file)
    
    configs = {config['name']: config for config in configs}
    task_config = configs[task]

    sbm = StochasticBlockModel(p=0.75, q=0.25, config=task_config)
    graph = sbm.generate_graph(entropy_threshold=0.5)

    print(graph.left)