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
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
import torch_geometric.loader as neighbor_loader
import sklearn

# TODO: properly implement evaluation function
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
        
        
        return torch.cat(all_embeddings, dim=0).to(self.device)

    
    def forward(self, samples, edge_index):
        
        x = self._encode_text(samples).to(device=self.device)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = F.relu(x)
        x = self.classifier(x)
        return x
        
    

def evaluate(model, data, samples, mask, hp, device='cuda'):
    model.eval()
    all_predictions = []
    all_true_labels = []

    val_nodes = torch.where(mask)[0] 

    val_loader = neighbor_loader.NeighborLoader(
        data,
        input_nodes=val_nodes,  
        num_neighbors=[-1, -1], 
        batch_size=hp.sampling_size,
        shuffle=False
       )

    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch_samples = [samples[i] for i in batch.n_id.tolist()]
            out = model(batch_samples, batch.edge_index)
            predictions = out.argmax(dim=1)
            true_labels = data.y[batch.n_id].to(device)
            all_predictions.append(predictions.cpu())
            all_true_labels.append(true_labels.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_labels = torch.cat(all_true_labels, dim=0)

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions)

    return accuracy, f1


def transform(data):
    samples = data.samples
    left = data.left
    right = data.right
    clean_data = Data(
    edge_index=data.edge_index,
    y=data.y,
    train_mask=data.train_mask,
    val_mask=data.val_mask,
    test_mask=data.test_mask,
    logits=data.logits,
    num_nodes=len(samples))
    return left, right, samples, clean_data

def train(model, data_obj, optimizer, hp, device='cuda'):

    _, _, samples, data = transform(data_obj)
    criterion = nn.CrossEntropyLoss()
    print(model.signature())

    model.to(device)  

    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    
    loader = neighbor_loader.NeighborLoader(
        data,
        input_nodes=torch.arange(data.num_nodes),  
        num_neighbors=hp.sizes, 
        batch_size=hp.sampling_size,
        shuffle=True
       )

    for epoch in range(1, hp.n_epochs + 1):
        model.train()
        total_loss = 0
        num_train_batches = 0
        
        print('performing train step:')
        for batch in tqdm(loader):
            optimizer.zero_grad()
            batch_samples = [samples[i] for i in batch.n_id]
            batch_train_mask = data.train_mask[batch.n_id]

            out = model(batch_samples, batch.edge_index)  # Feed the corresponding sample embeddings
            loss = criterion(out[batch_train_mask], batch.y[batch_train_mask])  # Only use training nodes
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_train_batches += 1
        
        avg_loss = total_loss / num_train_batches

        print('evaluating model:')
        acc, f1 = evaluate(model, data, samples, data.val_mask, hp, device) 
        train_acc, train_f1 = evaluate(model, data, samples, data.train_mask, hp, device)

        print(f"Epoch {epoch}/{hp.n_epochs}, Loss: {avg_loss:.4f}, f1: {f1:.4f}, accuracy: {acc:.4f}, train_f1: {train_f1:.4f}, train_acc:{train_acc:.4f}")
            

