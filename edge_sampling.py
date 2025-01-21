import numpy as np
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
import torch



def read_dataest(dset_path: str): 

    '''
    Read the dataset file and return the left samples, right samples, labels and logits (if get_logits is True)
    '''
    with open(dset_path) as f:
        lines = f.read().splitlines()

    split_lines = [line.split('\t') for line in lines] 
    split_lines = np.array(split_lines)
    left_samples = split_lines[:, 0].tolist()
    right_samples = split_lines[:, 1].tolist()
    labels = split_lines[:, 2].tolist()
    labels = [int(l) for l in labels]
    return left_samples, right_samples, labels

def read_data(config: dict):
    '''
    Read the train, validation and test data from the config paths. Return the left samples, right samples, labels and logits, as well as masks for each set.
    '''

    # Read the data
    train_left, train_right, train_labels = read_dataest(dset_path=config['trainset'])
    val_left, val_right, val_labels = read_dataest(dset_path=config['validset'])
    test_left, test_right, test_labels = read_dataest(dset_path=config['testset'])
    left = np.concatenate([train_left, val_left, test_left], axis=0)
    right = np.concatenate([train_right, val_right, test_right], axis=0)
    labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

    # Get the sizes of each set and create masks
    train_size, val_size, test_size = len(train_labels), len(val_labels), len(test_labels)
    dset_size = train_size + val_size + test_size
    train_mask = torch.zeros(dset_size, dtype=torch.bool)
    val_mask = torch.zeros(dset_size, dtype=torch.bool)
    test_mask = torch.zeros(dset_size, dtype=torch.bool)
    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True

    return left, right, labels, train_mask, val_mask, test_mask

class SBMGraph:
    def __init__(self, p: float, q: float, config_true: dict, config_noisy: dict):
        self.p = p
        self.q = q
        self.left, self.right, self.labels, self.train_mask, self.val_mask, self.test_mask = read_data(config_true)
        _, _, self.labels_noisy, _, _, _ = read_data(config_noisy)
        

    def sample_edge_index(self):
        '''
        Sample the edge index from the dataset according to SBM model
        '''

        train_nodes = np.where(self.train_mask == True)[0]
        val_nodes = np.where(self.val_mask == True)[0]
        test_nodes = np.where(self.test_mask == True)[0]
        nodes_to_sample = np.concatenate([train_nodes, val_nodes, test_nodes])
        probs = torch.tensor([[self.p, self.q], [self.q, self.p]], dtype=torch.float)
        row, col = torch.combinations(torch.tensor(nodes_to_sample), r=2, with_replacement=True).t()
        mask = torch.bernoulli(probs[self.labels[row], self.labels[col]]).to(torch.bool)
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        if len(edge_index.shape) != 2: 
            edge_index = edge_index.view(2, -1)
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        self.edge_index = edge_index
        return edge_index

    def generate_graph(self) -> Data:
        '''
        Generate a graph from the left and right samples, using labels as assginments to communities in SBM model
        '''
        self.sample_edge_index()
        data = Data(x=None, edge_index=self.edge_index, y=torch.tensor(self.labels_noisy))

        data.train_mask = torch.tensor(self.train_mask)
        data.val_mask = torch.tensor(self.val_mask)
        data.test_mask = torch.tensor(self.test_mask)
        data.left = self.left
        data.right = self.right
        data.samples = [l+' [sep] '+r for l, r in zip(data.left, data.right)]
        data.labels_clean = torch.tensor(self.labels)

        return data
        
        