import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as metrics
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils import data
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
import torch_geometric.loader as neighbor_loader
import sklearn


class GraphNNTransformer(nn.Module):

    def __init__(self, encoder_model, conv_type='GraphSAGE', hidden_channels=[16, 32], batch_size=16, device='cuda:1'):
        super(GraphNNTransformer, self).__init__()
        self.transformer = encoder_model.transformer
        self.transformer.gradient_checkpointing_enable()
        for param in self.transformer.parameters():
            param.requires_grad = True
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model.lm_type)
        self.batch_size = batch_size
        self.device=device

        # Initializng message passing layers
        self.conv_layers = nn.ModuleList()
        self.hc_config = [self.transformer.config.hidden_size] + hidden_channels
        for in_channels, out_channels in zip(self.hc_config[:-1], self.hc_config[1:]):
            if conv_type == 'GCN':
                self.conv_layers.append(GCNConv(in_channels, out_channels))
            elif conv_type == 'GAT':
                self.conv_layers.append(GATConv(in_channels, out_channels))
            elif conv_type == 'GraphSAGE':
                self.conv_layers.append(SAGEConv(in_channels, out_channels))
            elif conv_type == 'Transformer':
                self.conv_layers.append(TransformerConv(in_channels, out_channels))
            else:
                raise ValueError(f"Unsupported graph convolution type: {conv_type}")
            
        self.classifier = nn.Linear(hidden_channels[-1], 2)
    
    def freeze_transformer(self, freeze=True):
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
        else:
            for param in self.transformer.parameters():
                param.requires_grad = True

    def _encode_text(self, left_samples, right_samples):
        torch.cuda.empty_cache()
        x = [] 
        for i in range(0, len(left_samples), self.batch_size):
            left_batch = left_samples[i:i+self.batch_size]
            right_batch = right_samples[i:i+self.batch_size]
            tokenized = self.tokenizer(
                text=left_batch,
                text_pair=right_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=256
            )
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            encodings = self.transformer(input_ids, attention_mask=attention_mask)
            enc = encodings.last_hidden_state[:, 0]  # CLS token representation
            x.append(enc)

        return torch.cat(x, dim=0)
    
    def forward(self, left_samples, right_samples, edge_index, inference=False):
        x = self._encode_text(left_samples, right_samples)
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
        output = self.classifier(x)

        if inference:
            return output, x
        else:
            return output

def negative_sampling(data_mask, negative_label_mask, sample_ratio=0.5):
    '''
    Sample negative nodes from the data mask and return mask for the negative samples
    '''
    n_samples = int(sample_ratio * data_mask.sum().item())
    negative_samples = torch.where(negative_label_mask & data_mask)[0]
    negative_samples = negative_samples[torch.randperm(negative_samples.size(0))[:n_samples]]
    negative_mask = torch.zeros_like(data_mask)
    negative_mask[negative_samples] = True
    return negative_mask

def evaluate(model, edge_index ,left_samples, right_samples, mask, labels):
    with torch.no_grad():
        model.eval()
        with torch.no_grad():
            out = model(left_samples, right_samples, edge_index)
            logits = out[mask==1]
            labels = labels[mask==1]
            preds = torch.argmax(logits, dim=1)
            accuracy = accuracy_score(labels.cpu(), preds.cpu())
            f1 = f1_score(labels.cpu(), preds.cpu())
        return accuracy, f1
    
def train(model, optimizer, criterion, data_object, hp, device):
    model.to(device)
    model.train()
    
    # Unpack data object
    samples = data_object.samples
    left_samples = data_object.left
    right_samples = data_object.right

    if isinstance(left_samples, np.ndarray):
        left_samples = left_samples.tolist()
    if isinstance(right_samples, np.ndarray):
        right_samples = right_samples.tolist()

    edge_index = data_object.edge_index.to(device)
    train_mask = data_object.train_mask.to(device)
    val_mask = data_object.val_mask.to(device)
    test_mask = data_object.test_mask.to(device)
    y = data_object.y.to(device)
    labels_clean = data_object.labels_clean.to(device)

    # Create positive & negative label masks
    negative_label_mask = torch.zeros_like(y)
    positive_sample_mask = torch.zeros_like(y)
    for i, label in enumerate(y):
        if train_mask[i]:
            if label:
                positive_sample_mask[i] = 1
            else:
                negative_label_mask[i] = 1
    
    # Training loop
    for epoch in range(1, hp.n_epochs+1):
        model.train()
        if epoch <= hp.freeze_epoch_ratio * hp.n_epochs:
            model.freeze_transformer(freeze=True) # GNN warmup
        else:
            model.freeze_transformer(freeze=False) # Complete training
        optimizer.zero_grad()
        out = model(left_samples, right_samples, edge_index)

        # Sample nodes with label 0 to be used as negative samples out of the train mask
        negative_sample_mask = negative_sampling(train_mask, negative_label_mask, sample_ratio=hp.sample_ratio)
        train_iter_mask = torch.zeros_like(y)
        for i, label in enumerate(y):
            if negative_sample_mask[i] or positive_sample_mask[i]:
                train_iter_mask[i] = 1
        
        print(f'Epoch {epoch} Training on [{train_iter_mask.sum().item()}/{train_mask.sum().item()}] samples')
        loss = criterion(out[train_iter_mask==1], y[train_iter_mask==1])
        loss.backward()
        optimizer.step()
        noisy_acc, noisy_f1 = evaluate(model, edge_index, left_samples, right_samples, train_mask, y)
        train_acc, train_f1 = evaluate(model, edge_index, left_samples, right_samples, train_mask, labels_clean)
        acc, f1 = evaluate(model, edge_index, left_samples, right_samples, val_mask, labels_clean)
        print(f'Epoch {epoch} Loss: {loss.item()}')
        print(f'Noisy Train Accuracy: {noisy_acc}, Noisy Train F1: {noisy_f1}')
        print(f'Train Accuracy: {train_acc}, Train F1: {train_f1}')
        print(f'Validation Accuracy: {acc}, Validation F1: {f1}')
        print('----------------------------------------')




        