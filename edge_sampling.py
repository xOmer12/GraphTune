import numpy as np
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt



def read_dataest(dset_path: str, noisy=False): 

    '''
    Read the dataset file and return the left samples, right samples, labels and logits (if get_logits is True)
    '''
    if noisy:
        print(dset_path)
    with open(dset_path) as f:
        lines = f.read().splitlines()

    split_lines = [line.split('\t') for line in lines] 
    split_lines = np.array(split_lines)
    left_samples = split_lines[:, 0].tolist()
    right_samples = split_lines[:, 1].tolist()
    labels = split_lines[:, 2].tolist()
    labels = [int(l) for l in labels]
    if noisy:
        logits = split_lines[:, 3].tolist()
        logits = [float(l) for l in logits]
        return left_samples, right_samples, labels, logits
    return left_samples, right_samples, labels

def read_data(config: dict, noisy=False):
    '''
    Read the train, validation and test data from the config paths. Return the left samples, right samples, labels and logits, as well as masks for each set.
    '''

    # Read the data
    if noisy:
        train_left, train_right, train_labels, train_logits = read_dataest(dset_path=config['trainset'], noisy=True)
    else:
        train_left, train_right, train_labels = read_dataest(dset_path=config['trainset'], noisy=False)
    val_left, val_right, val_labels = read_dataest(dset_path=config['validset'])
    test_left, test_right, test_labels = read_dataest(dset_path=config['testset'])
    left = np.concatenate([train_left, val_left, test_left], axis=0)
    right = np.concatenate([train_right, val_right, test_right], axis=0)
    labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

    # Get the sizes of each set and create masks
    train_size, val_size, test_size = len(train_labels), len(val_labels), len(test_labels)
    print(f'train size: {train_size}')
    print(f'val size: {val_size}')
    print(f'test size: {test_size}')
    dset_size = train_size + val_size + test_size
    train_mask = torch.zeros(dset_size, dtype=torch.bool)
    val_mask = torch.zeros(dset_size, dtype=torch.bool)
    test_mask = torch.zeros(dset_size, dtype=torch.bool)
    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True
    if noisy:
        return left, right, labels, train_logits, train_mask, val_mask, test_mask
    else:
        return left, right, labels, train_mask, val_mask, test_mask

class SBMGraph:
    def __init__(self, p: float, q: float, config_true: dict, config_noisy: dict):
        self.p = p
        self.q = q
        self.left, self.right, self.labels, self.train_mask, self.val_mask, self.test_mask = read_data(config_true)
        _, _, self.labels_noisy, self.train_logits, _, _, _ = read_data(config_noisy, noisy=True)
        
    def create_pft_mask(self, k):
        probs = torch.tensor(self.train_logits)
        labels = torch.tensor(self.labels[self.train_mask])
        entropy = torch.tensor([-p*torch.log(p)-(1-p)*torch.log(1-p) for p in probs])  # Avoid log(0) with small epsilon

        # Create mask for each label (assuming labels are 0 and 1)
        mask_0 = (labels == 0)
        mask_1 = (labels == 1)

        # Get indices for each label
        indices_0 = torch.nonzero(mask_0, as_tuple=True)[0]
        indices_1 = torch.nonzero(mask_1, as_tuple=True)[0]

        # Sort entropies within each label group
        sorted_indices_0 = indices_0[torch.argsort(entropy[indices_0])[:k]]
        sorted_indices_1 = indices_1[torch.argsort(entropy[indices_1])[:k]]

        # Create final mask
        final_mask = torch.zeros(self.labels.shape[0], dtype=torch.bool)
        final_mask[sorted_indices_0] = True
        final_mask[sorted_indices_1] = True

        self.pft_mask = final_mask

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

    def analyze_graph(self):
        '''
        Analyze the graph generated from the SBM model
        '''
        G = nx.Graph()
        G.add_edges_from(self.edge_index.t().tolist())
        print(f'Number of nodes: {G.number_of_nodes()}')
        print(f'Number of edges: {G.number_of_edges()}')
        print(f'Average degree: {np.mean(list(dict(G.degree()).values()))}')
        print(f'Density: {nx.density(G)}')
        print(f'Number of connected components: {nx.number_connected_components(G)}')
        print(f'Average clustering coefficient: {nx.average_clustering(G)}')

    def generate_graph(self) -> Data:
        '''
        Generate a graph from the left and right samples, using labels as assginments to communities in SBM model
        '''
        self.sample_edge_index()
        self.create_pft_mask(k=1000)
        data = Data(x=None, edge_index=self.edge_index, y=torch.tensor(self.labels_noisy))

        data.train_mask = torch.tensor(self.train_mask)
        data.val_mask = torch.tensor(self.val_mask)
        data.test_mask = torch.tensor(self.test_mask)
        data.left = self.left
        data.right = self.right
        data.samples = [l+' [sep] '+r for l, r in zip(data.left, data.right)]
        data.labels_clean = torch.tensor(self.labels)
        data.pft_mask = torch.tensor(self.pft_mask)

        return data

class AdaptiveBMGraph(SBMGraph):
    def __init__(self, p: float, q: float, config_true: dict, config_noisy: dict, c0: int, c1, beta: float):
        super().__init__(p, q, config_true, config_noisy)
        self.c0 = c0
        self.c1 = c1
        self.beta = beta
    

    def calc_community_probs(self):
        '''
        Calculate the community probabilities for the Adaptive SBM model
        '''
        n_zeros = np.where(self.labels == 0)[0].shape[0]
        n_ones = np.where(self.labels == 1)[0].shape[0]
        p_intra_0 = self.c0 / n_zeros
        p_intra_1 = self.c1 / n_ones
        p_inter = self.beta / (n_zeros + n_ones)
        self.probs = torch.tensor([[p_intra_0, p_inter], [p_inter, p_intra_1]], dtype=torch.float)

    def sample_edge_index(self):
        '''
        Sample the edge index from the dataset according to Adaptive SBM model
        '''
        train_nodes = np.where(self.train_mask == True)[0]
        val_nodes = np.where(self.val_mask == True)[0]
        test_nodes = np.where(self.test_mask == True)[0]
        nodes_to_sample = np.concatenate([train_nodes, val_nodes, test_nodes])
        probs = self.probs
        row, col = torch.combinations(torch.tensor(nodes_to_sample), r=2, with_replacement=True).t()
        mask = torch.bernoulli(probs[self.labels[row], self.labels[col]]).to(torch.bool)
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        if len(edge_index.shape) != 2: 
            edge_index = edge_index.view(2, -1)
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        self.edge_index = edge_index
        return edge_index
    
    def plot_graph(self):
        '''
        Plot the graph generated from the Adaptive SBM model, with labels as node colors
        '''
        G = nx.Graph()
        G.add_edges_from(self.edge_index.t().tolist())
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=self.labels, with_labels=True, cmap=plt.cm.plasma)
        plt.savefig('graph.png')



        