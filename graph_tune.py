import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils import data
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
import torch_geometric.loader as neighbor_loader
import sklearn


class GraphTune(nn.Module):
    """Transformer encoder fine-tuned via GNN classification head"""

    def __init__(self, lm='roberta-base', conv_type='GCN', encoder_channel=768, hidden_channels=[32, 16], seed=42, encoding_batch_size=16, device='cuda'):

        super().__init__()
        self.lm = lm
        self.device = device
        self.transformer = AutoModel.from_pretrained(lm)
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.conv_type = conv_type
        self.hc_config = [encoder_channel]+hidden_channels
        self.seed = seed
        self.conv_layers = nn.ModuleList()
        self.batch_size = encoding_batch_size

        for in_channels, out_channels in zip(self.hc_config[:-1], self.hc_config[1:]):
            if conv_type == 'GCN':
                self.conv_layers.append(GCNConv(in_channels, out_channels))
            elif conv_type == 'GAT':
                self.conv_layers.append(GATConv(in_channels, out_channels))
            elif conv_type == 'GraphSAGE':
                self.conv_layers.append(SAGEConv(in_channels, out_channels))
            else:
                raise ValueError(f"Unsupported graph convolution type: {conv_type}")

        self.classifier = nn.Linear(hidden_channels[-1], 2)

    def signature(self):
        return f'{self.lm}_{self.conv_type}'

    def _forward_transformer(self, input_ids, attention_mask):
        return self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    
    def _encode_text(self, samples):
        """
        Encode all samples in batches as part of the forward propagation.
        Optimized to minimize memory spikes and manage device transfers.
        """
        all_embeddings = []
        
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i:i+self.batch_size]
            inputs = self.tokenizer(batch_samples, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = checkpoint(
                self._forward_transformer,
                inputs['input_ids'],
                inputs['attention_mask'],
                use_reentrant=False
            )
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings) 
            del inputs, outputs, embeddings
            torch.cuda.empty_cache()
        
        # Concatenate on CPU and move to GPU once
        return torch.cat(all_embeddings, dim=0).to(self.device)

    
    def forward(self, samples, adjs):
        # Step 1: Encode text into node features
        x = self._encode_text(samples).to(device=self.device)
        
        # Step 2: Pass through GNN layers

        for conv_layer, edge_index in zip(self.conv_layers, adjs):
            x = conv_layer(x, edge_index)
            x = F.relu(x)
        x = self.classifier(x)
        return x
        
    

def evaluate(model, data, mask, hp, device='cuda'):
    model.eval()
    all_predictions = []
    all_true_labels = []

    val_nodes = torch.where(mask)[0]  # Indices of validation nodes
    val_node_set = set(val_nodes.tolist())

    relevant_edges = [
        (src, dst) for src, dst in data.edge_index.t().tolist()
        if src in val_node_set or dst in val_node_set
    ]
    if not relevant_edges:
        raise ValueError("No edges found for the validation set.")

    subgraph_edge_index = torch.tensor(relevant_edges, dtype=torch.long).t().contiguous().to(device)
    subgraph_nodes = torch.unique(subgraph_edge_index).to(device)

    val_loader = neighbor_loader.NeighborSampler(
        edge_index=subgraph_edge_index,
        node_idx=subgraph_nodes,
        sizes=hp.sizes,  # Sampling sizes
        batch_size=hp.sampling_size,
        shuffle=False
    )

    with torch.no_grad():
        for _, n_id, adjs in tqdm(val_loader):
            val_mask_in_batch = torch.isin(n_id, val_nodes.to(n_id.device))
            batch_samples = [data.samples[i] for i in n_id.tolist()]
            batch_edge_index = [adj.edge_index.to(device) for adj in adjs]
            out = model(batch_samples, batch_edge_index)
            predictions = out.argmax(dim=1)
            true_labels = data.y[n_id].to(device)
            all_predictions.append(predictions[val_mask_in_batch].cpu())
            all_true_labels.append(true_labels[val_mask_in_batch].cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_labels = torch.cat(all_true_labels, dim=0)

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')

    return accuracy, f1

def ensure_val_test_edges(batch_edge_index, n_id, data, val_mask, test_mask):
    """
    Add missing edges for validation and test nodes to ensure proper message passing.

    Args:
        batch_edge_index (torch.Tensor): Current edge index for the batch.
        n_id (torch.Tensor): Node indices in the batch.
        data (Data): Full graph data object.
        val_mask (torch.Tensor): Validation mask.
        test_mask (torch.Tensor): Test mask.

    Returns:
        torch.Tensor: Updated edge index with added edges for val/test nodes.
    """
    device = batch_edge_index.device
    batch_global_ids = n_id.tolist()
    global_to_local = {nid: i for i, nid in enumerate(batch_global_ids)}
    updated_edge_index = batch_edge_index.clone().to(device)

    val_test_global_ids = torch.cat([torch.where(val_mask)[0], torch.where(test_mask)[0]]).tolist()
    val_test_in_batch = [nid for nid in val_test_global_ids if nid in global_to_local]

    for global_id in val_test_in_batch:
        local_id = global_to_local[global_id]
        neighbors = data.edge_index[1, data.edge_index[0] == global_id].tolist()
        for neighbor_global in neighbors:
            if neighbor_global in global_to_local:
                neighbor_local = global_to_local[neighbor_global]
                if not ((updated_edge_index[0] == local_id) & (updated_edge_index[1] == neighbor_local)).any():
                    updated_edge_index = torch.cat(
                        [updated_edge_index, torch.tensor([[local_id], [neighbor_local]]).to(device)], dim=1
                    )
                if not ((updated_edge_index[0] == neighbor_local) & (updated_edge_index[1] == local_id)).any():
                    updated_edge_index = torch.cat(
                        [updated_edge_index, torch.tensor([[neighbor_local], [local_id]]).to(device)], dim=1
                    )

    return updated_edge_index

def train(model, data, optimizer, hp, device='cuda'):
    
    criterion = nn.CrossEntropyLoss()
    print(model.signature())

    model.to(device)  

    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    # Identify validation and test nodes
    val_nodes = torch.where(data.val_mask)[0].to(device)
    test_nodes = torch.where(data.test_mask)[0].to(device)
    val_and_test_nodes = torch.cat([val_nodes, test_nodes]).unique()

    # Ensure all edges connected to val/test nodes are included
    edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool).to(device)
    for node in val_and_test_nodes:
        edge_mask |= (data.edge_index[0] == node) | (data.edge_index[1] == node)

    # Always include edges connected to validation/test nodes in NeighborSampler
    fixed_edge_index = data.edge_index[:, edge_mask]

    loader = neighbor_loader.NeighborSampler(
        data.edge_index, 
        node_idx=torch.arange(len(data.samples)),
        sizes=hp.sizes,
        batch_size=hp.sampling_size,
        shuffle=True
    )

    for epoch in range(1, hp.n_epochs + 1):
        model.train()
        total_loss = 0
        num_train_batches = 0
        
        print('performing train step:')
        for _, n_id, adjs in tqdm(loader):
            batch_edge_index = torch.cat([adj.edge_index for adj in adjs], dim=1).to(device)
            batch_edge_index = ensure_val_test_edges(batch_edge_index, n_id, data, data.val_mask, data.test_mask)
            batch_edge_index = torch.unique(batch_edge_index, dim=1)
            batch_train_mask = data.train_mask[n_id]
            batch_samples = [data.samples[i] for i in n_id]
            print(batch_edge_index.shape)
            
            out = model(batch_samples, batch_edge_index)  # Feed the corresponding sample embeddings
            loss = criterion(out[batch_train_mask], data.y[n_id][batch_train_mask])  # Only use training nodes
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_train_batches += 1
        
        avg_loss = total_loss / num_train_batches

        print('evaluating model:')
        acc, f1 = evaluate(model, data, data.val_mask, hp, device) 
        print(f"Epoch {epoch}/{hp.n_epochs}, Loss: {avg_loss:.4f}, f1: {f1:.4f}, accuracy: {acc:.4f}")
            

