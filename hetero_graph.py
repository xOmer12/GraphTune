
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from itertools import combinations
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, HeteroConv, GCNConv, SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from umap import UMAP
from pft_encoder import read_data, PreFTDataset, Encoder, pre_fine_tune


def get_data(dset_path, has_labels=True):

    with open(dset_path) as f:
        lines = f.read().splitlines()
        # remove punctuation from the text
    removePunc = list()
    # remove extra spaces and the punctuation and the symbol "-"
    [removePunc.append(line.replace(" -", "").translate(str.maketrans('', '', string.punctuation)).replace("  ", " "))
    for line in lines]

    # split the lines according to "\t" since the different instances in a line of one example are separated by "\t"
    splitList = [line.split('\t') for line in removePunc]
    # turning to numpy array for in order to use numpy functionality
    npArray = np.array(splitList)
    # getting the left side examples
    leftCol = npArray[:, 0].tolist()
    # getting the right side examples
    rightCol = npArray[:, 1].tolist()
    # if the dataset is with labels (train, valid) then extract also the labels from the file
    if has_labels:
        # getting the labels of the pairs
        label = npArray[:, 2].tolist()
        return leftCol, rightCol, label
    else:
        return leftCol, rightCol

def parse_attr(line):
    parsed_line = {}
    splitline = line.split('COL ')[1:]
    for tokens in splitline:
        val_split = tokens.split('VAL ')
        key = val_split[0].strip()
        val = val_split[1].strip()
        parsed_line[key] = val
    return parsed_line

def embed_dict(parsed_line, model):
    embedded_dict = {}
    with torch.no_grad():
        for key, val in parsed_line.items():
            embedded_dict[key] = model.encode(val)
    return embedded_dict

def extract_mutual_agreement(left_dict, right_dict):
    mutual_agreement = {}
    for key in left_dict.keys():
        if key in right_dict.keys():
            left_emb = left_dict[key]
            right_emb = right_dict[key]
            sim = np.dot(left_emb, right_emb)/(np.linalg.norm(left_emb)*np.linalg.norm(right_emb))
            mutual_agreement[key] = sim
    return mutual_agreement

def extract_agreement_pairs(left, right, model):
    attr_agreement = {}
    for i, (l, r) in tqdm(enumerate(zip(left, right))):
        l_dict = embed_dict(parse_attr(l), model)
        r_dict = embed_dict(parse_attr(r), model)
        mutual_agreement = extract_mutual_agreement(l_dict, r_dict)
        attr_agreement[i] = mutual_agreement
    return attr_agreement

def build_mutual_agreement_graph(attr_agreement, threshold=0.75, sample_edge=False, sampling_prob=0.5, null_sampling_prob=0.05):
    data = HeteroData()
    nodes = set(attr_agreement.keys())
    # Add nodes to the graph
    data['node'].num_nodes = len(nodes)
    edges = {attr: [] for attr in next(iter(attr_agreement.values())).keys()}
    edges['null'] = []  # For no attribute agreement
    for node1, node2 in combinations(attr_agreement.keys(), 2):
        no_agreement_left = True
        no_agreement_right = True
        for attr in attr_agreement[node1].keys():
            # Check if both nodes have high agreement for this attribute
            if not sample_edge:
                if attr_agreement[node1][attr] >= threshold:
                    no_agreement_left = False
                    if attr_agreement[node2][attr] >= threshold:
                        no_agreement_right = False
                        edges[attr].append((node1, node2))
            else: # Sample edges based on agreement
                if attr_agreement[node1][attr] >= threshold:
                    no_agreement_left = False
                    if attr_agreement[node2][attr] >= threshold:
                        no_agreement_right = False
                        if np.random.rand() < sampling_prob:
                            edges[attr].append((node1, node2))
        if no_agreement_left and no_agreement_right:
            if np.random.rand() < null_sampling_prob:
                edges['null'].append((node1, node2))
    # Convert to PyTorch tensors and store in HeteroData
    for attr, edge_list in edges.items():
        if edge_list:  # Ensure there are edges of this type
            edge_index = torch.tensor(edge_list, dtype=torch.long).T  # Convert to (2, num_edges)
            data["node", attr, "node"].edge_index = edge_index
    
    return data

def visualize_hetero_graph(g):



    """
    Visualizes a heterogeneous graph with different colors for each edge type.
    
    Parameters:
    g (HeteroData): A PyG HeteroData graph.
    """
    # Create a networkx graph
    G = nx.Graph()
    
    # Assign unique colors to each edge type
    edge_types = list(g.edge_types)
    colors = plt.cm.rainbow(torch.linspace(0, 1, len(edge_types)))
    edge_color_map = {etype: colors[i] for i, etype in enumerate(edge_types)}
    
    # Add nodes
    for node_type in g.node_types:
        G.add_nodes_from(range(g[node_type].num_nodes), node_type=node_type)
    
    # Add edges with colors
    edge_colors = []
    for etype in g.edge_types:
        edge_index = g[etype].edge_index.numpy().T  # Convert to (src, dst) format
        for src, dst in edge_index:
            G.add_edge(src, dst, edge_type=etype)
            edge_colors.append(edge_color_map[etype])
    
    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.kamada_kawai_layout(G)  # Layout for positioning
    
    # Draw nodes
    nx.draw(G, pos, node_size=15, with_labels=False, edge_color=edge_colors, alpha=0.5, width=1)
    
    # Add legend
    handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label=etype) for i, etype in enumerate(edge_types)]
    plt.legend(handles=handles, title="Edge Types")
    plt.title("Mutual Agreement Graph Visualization")
    plt.show()

def init_embeddings(left, right, model, sentence_transformer=False):
    X = []
    if sentence_transformer:
        for l, r in tqdm(zip(left, right)):
            l_emb = model.encode(l, convert_to_tensor=True)
            r_emb = model.encode(r, convert_to_tensor=True)
            init_emb = torch.cat((l_emb, r_emb), axis=0)
            X.append(init_emb)
        X = torch.stack(X)
    else:
        for l, r in tqdm(zip(left, right)):
            with torch.no_grad():
                _, emb = model(l, r, inference=True)
            X.append(emb.squeeze())
        X = torch.stack(X)
    return X

class RGNN(torch.nn.Module):
    ''''
        relational graph neural network for entity matching node classification task on heterogeneous graphs.
    '''
    def __init__(self, init_dim, channels, num_classes, edge_types, conv_type='GCN', agg_type='mean'):
        super(RGNN, self).__init__()
        self.init_dim = init_dim
        self.edge_types = edge_types
        self.num_relations = len(self.edge_types)
        self.num_classes = num_classes
        self.channels = [init_dim] + channels

        # Initialize the convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        for in_channels, out_channels in zip(self.channels[:-1], self.channels[1:]):
            if conv_type == 'GCN':
                conv_dict = {edge_type:GCNConv(in_channels, out_channels) for edge_type in self.edge_types}
            elif conv_type == 'SAGE':
                conv_dict = {edge_type:SAGEConv(in_channels, out_channels) for edge_type in self.edge_types}
            else:
                raise ValueError(f"Unsupported convolution type {conv_type}. Use 'GCN' or 'SAGE'.")
            if agg_type == 'mean':
                self.conv_layers.append(HeteroConv(conv_dict, aggr='mean'))
            elif agg_type == 'sum':
                self.conv_layers.append(HeteroConv(conv_dict, aggr='sum'))
            elif agg_type == 'max':
                self.conv_layers.append(HeteroConv(conv_dict, aggr='max'))
            else:
                raise ValueError(f"Unsupported aggregation type {agg_type}. Use 'mean', 'sum', or 'max'.")
        self.linear = torch.nn.Linear(self.channels[-1], self.num_classes)
        self.activation = torch.nn.ReLU()


    def forward(self, x_dict, edge_index_dict, inference=False):
        for conv in self.conv_layers:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.activation(v) for k, v in x_dict.items()}
        out_dict = {k: self.linear(v) for k, v in x_dict.items()}
        if inference:
            return out_dict, x_dict
        else:
            return out_dict

def train_model(model, data, noisy_labels, true_labels, train_mask, val_mask, epochs=100, lr=0.01, analyze_learning=False):
    embedding_epoch_dict = {}
    labels_epoch_dict = {}
    dist_epoch_dict = {}
    out_epoch_dict = {}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    edge_type_dict = {edge_type: i for i, edge_type in enumerate(data.edge_types)}
    data.edge_index_dict = edge_index_dict
    data.edge_type_dict = edge_type_dict
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)
        out = out['node']
        # Compute loss
        loss = criterion(out[train_mask], noisy_labels[train_mask])
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch) % 5 == 0:
            if analyze_learning:
                sampled_out, sampled_x, sampled_labels, pairwise_distances = analyze_vspace(model, data, true_labels)
                embedding_epoch_dict[epoch] = sampled_x
                labels_epoch_dict[epoch] = sampled_labels
                dist_epoch_dict[epoch] = pairwise_distances
                out_epoch_dict[epoch] = sampled_out
            # Evaluate on validation set
            _, acc, f1 = evaluate_model(model, data, val_mask, true_labels)
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {acc:.4f}, Validation F1: {f1:.4f}')
    
    return embedding_epoch_dict, labels_epoch_dict, dist_epoch_dict, out_epoch_dict

def evaluate_model(model, data, eval_mask, labels):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        out = out['node']
        _, predicted = torch.max(out, 1)
        labels = labels.cpu()
        predicted = predicted.cpu()
        eval_mask = eval_mask.cpu()
        accuracy = accuracy_score(labels[eval_mask], predicted[eval_mask])
        f1 = f1_score(labels[eval_mask], predicted[eval_mask])
    return predicted, accuracy, f1

def analyze_vspace(model, data, labels, sample_frac=0.1):
    model.eval()
    with torch.no_grad():
        out_dict, x_dict = model(data.x_dict, data.edge_index_dict, inference=True)
        out = out_dict['node']
        x = x_dict['node']
        # Sample a fraction of the nodes
        num_nodes = out.shape[0]
        sample_size = int(num_nodes * sample_frac)
        sampled_indices = np.random.choice(num_nodes, sample_size, replace=False)
        sampled_out = out[sampled_indices]
        sampled_x = x[sampled_indices]
        sampled_labels = labels[sampled_indices]
        
        # Compute pairwise distances
        pairwise_distances = []
        for i in range(sample_size):
            for j in range(sample_size):
                dist = torch.norm(sampled_x[i] - sampled_x[j]).item()
                pairwise_distances.append(dist)
        
        # Compute mean and std of distances
        
        
    return sampled_out, sampled_x, sampled_labels, pairwise_distances

def plot_dist_histogram(dists):
    plt.figure(figsize=(10, 6))
    plt.hist(dists, bins=50, alpha=0.7, color='green')
    plt.title('Pairwise Distance Histogram')
    plt.xlabel('Distances')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def plot_vspace_reduced(x, labels):
    umap_red =  UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    x_reduced = umap_red.fit_transform(x)
    plt.figure(figsize=(10, 6))
    scatter_plot = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels, cmap='cool', alpha=0.5)
    plt.legend(*scatter_plot.legend_elements(), title="Classes")
    plt.title('Node Embeddings in UMAP Space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid()
    plt.show()

def agreement_score(v, vspace, labels, k):
    """
    Compute the agreement score between a node and its k-nearest neighbors in the vspace.
    """
    # Compute pairwise distances
    pairwise_distances = torch.cdist(v.unsqueeze(0), vspace)
    # Get the indices of the k nearest neighbors
    knn_indices = torch.topk(pairwise_distances, k=k, largest=False).indices.squeeze(0)
    # Compute the agreement score
    agreement_score = sum(labels[knn_indices] == labels[knn_indices[0]]) / k
    return agreement_score

def plot_vspace_at_epoch(embedding_epoch_dict, labels_epoch_dict, n_cols = 3):
    n = len(embedding_epoch_dict)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    print(len(axes))
    print(n)
    for i, (epoch, embeddings) in enumerate(embedding_epoch_dict.items()):
        labels = labels_epoch_dict[epoch]
        tsne_red =  TSNE(n_components=2, random_state=42)
        x_reduced = tsne_red.fit_transform(embeddings.cpu().numpy())
        scatter_plot = axes[i].scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels.cpu().numpy(), cmap='cool', alpha=0.5)
        axes[i].legend(*scatter_plot.legend_elements(), title="Classes")
        axes[i].set_title(f'Node Embeddings at Epoch {epoch}')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
        axes[i].grid()

    # Hide any unused subplots
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_dist_at_epoch(dist_epoch_dict, n_cols = 3):
    n = len(dist_epoch_dict)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    for i, (epoch, dists) in enumerate(dist_epoch_dict.items()):
        # Plotting the histogram
        axes[i].hist(dists, bins=50, alpha=0.7, color='green')
        axes[i].set_title(f'Pairwise Distance Histogram at Epoch {epoch}')
        axes[i].set_xlabel('Distances')
        axes[i].set_ylabel('Frequency')
        axes[i].grid()

    # Hide any unused subplots
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_agreement_at_epoch(embedding_epoch_dict, labels_epoch_dict, k, n_cols = 3):
    avg_agreement_scores = []
    positive_avg_agreement_scores = []
    negative_avg_agreement_scores = []
    for epoch, embeddings in embedding_epoch_dict.items():
        labels = labels_epoch_dict[epoch].cpu().numpy()
        embeddings = embeddings.cpu()
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        # Compute agreement scores
        agreement_scores = [agreement_score(v, embeddings, labels, k=k) for v in embeddings]
        avg_agreement_scores.append(np.mean(agreement_scores))
        positive_avg_agreement_scores.append(np.mean([agreement_scores[i] for i in pos_indices]))
        negative_avg_agreement_scores.append(np.mean([agreement_scores[i] for i in neg_indices]))
    
    # stacked bar chart of scores
    x = np.arange(len(avg_agreement_scores))
    width = 0.3
    fig, ax = plt.subplots()
    ax.bar(x-0.2, avg_agreement_scores, width, label='Avg Agreement Score', color='blue')
    ax.bar(x, positive_avg_agreement_scores, width, label='Positive Avg Agreement Score', color='green')
    ax.bar(x+0.2, negative_avg_agreement_scores, width, label='Negative Avg Agreement Score', color='red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Agreement Score')
    ax.set_title(f'Mean Agreement@{k} at Each Epoch')
    ax.set_xticks(x)
    ax.set_xticklabels(list(embedding_epoch_dict.keys()))
    ax.legend()
    plt.show()
        
    
