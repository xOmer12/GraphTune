import torch
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
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
import torch_geometric.loader as neighbor_loader
import sklearn

# TODO: Add sampling based on prior on weak label
# TODO: Find a way to prevent model to converge to a trivial solution
# encoder initial embeddings seems too tight for GNN to learn anything, everything is embedded around the same point hence GNN cannot warm up correctly.
# Maybe warmup transformer first?

# Loss seems to converge yet model learns to classify everything as 0 already after the first epoch.

def filter_entity(e, attrs_to_keep):
    tokens = e.split(' ')
    filtered = []
    attr_indices = [i+1 for i, e  in enumerate(tokens) if e=='COL']
    for i, t in enumerate(tokens):
        if t in attrs_to_keep:
            next_attr_index = attr_indices[attr_indices.index(i)+1]
            val = ' '.join(tokens[i+1: next_attr_index-1])
            filtered.append(f'COL {t} {val}')
    return ' '.join(filtered)


class PreFTDataset(Dataset):

    """Basic dataset class for pre-fine-tuning transformer phase"""

    def __init__(self, lefts, rights, labels):
        self.lefts = lefts
        self.rights = rights
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.lefts[index], self.rights[index], self.labels[index]


class GraphTune(nn.Module):
    """Transformer encoder fine-tuned via GNN classification head"""

    def __init__(self, lm='roberta-base', conv_type='GCN', encoder_channel=768, hidden_channels=[32, 16], seed=42, encoding_batch_size=16, device='cuda'):

        super().__init__()
        self.lm = lm
        self.device = device
        self.transformer = AutoModel.from_pretrained(lm)
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.conv_type = conv_type
        self.hc_config = [encoder_channel] + hidden_channels
        self.seed = seed
        self.conv_layers = nn.ModuleList()
        self.batch_size = encoding_batch_size
        self.dropout = nn.Dropout(p=0.5)
        self.training_mode = True

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
        self.pre_fine_tune_classifier = nn.Linear(encoder_channel, 2)

    def signature(self):
        return f'{self.lm}_{self.conv_type}'
    
    def get_optimizer(self, transformer_lr=1e-5, gnn_lr=1e-3):
        # Separate parameters for transformer and GNN
        transformer_params = self.transformer.parameters()
        transformer_params += list(self.pre_fine_tune_classifier.parameters())
        gnn_params = []
        
        # Collect GNN and classifier parameters
        for layer in self.conv_layers:
            gnn_params += list(layer.parameters())
        gnn_params += list(self.classifier.parameters())
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': transformer_params, 'lr': transformer_lr},
            {'params': gnn_params, 'lr': gnn_lr}
        ]
        
        return AdamW(param_groups)

    def _forward_transformer(self, input_ids, attention_mask):
        return self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    
    def _freeze_transformer(self, freeze=True):
        '''
        Freeze the transformer weights (for warming up GNN wieghts)
        '''
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
        else:
            for param in self.transformer.parameters():
                param.requires_grad = True

    def _encode_text(self, left_samples, right_samples):
        all_embeddings = []
        
        for i in range(0, len(left_samples), self.batch_size):
            batch_samples_left = left_samples[i:i+self.batch_size].tolist()
            batch_samples_right = right_samples[i:i+self.batch_size].tolist()
            process_left, process_right = [], []
            for left, right in zip(batch_samples_left, batch_samples_right):
                process_left.append(filter_entity(left, {'title'}))
                process_right.append(filter_entity(right, {'title'}))

            inputs = self.tokenizer(text=process_left, text_pair=process_right, padding=True, truncation=True, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Use inference mode for transformer if not training
            if not self.training:
                with torch.inference_mode():
                    outputs = self._forward_transformer(inputs['input_ids'], inputs['attention_mask'])
            else:
                outputs = checkpoint(
                    self._forward_transformer,
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    use_reentrant=False
                )
            
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings) 
            
            # Clear memory more aggressively
            del inputs, outputs
            torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0)

    
    def forward(self, lefts, rights, edge_index, pre_fine_tune=False):

        if pre_fine_tune:
            process_lefts = []
            process_rights = []
            for l, r in zip(lefts, rights):
                process_lefts.append(filter_entity(l, {'title'}))
                process_rights.append(filter_entity(r, {'title'}))
            inputs = self.tokenizer(text=process_lefts, text_pair=process_rights, padding=True, truncation=True, return_tensors='pt')
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = checkpoint(
                    self._forward_transformer,
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    use_reentrant=False
                )
            embeddings = outputs.last_hidden_state[:, 0, :]
            x = self.pre_fine_tune_classifier(embeddings)

        else:
            x = self._encode_text(lefts, rights).to(device=self.device)
            
            for conv_layer in self.conv_layers:
                x = conv_layer(x, edge_index)
                x = F.relu(x)
            x = self.classifier(x)
        return x
        
    
def evaluate(model, edge_index, lefts, rights, mask, labels_clean, hp, device='cuda:1'):
    '''
    Evaluate the model on the given data mask
    No neighbor sampling
    '''
    model.eval()
    model.training_mode = False # Manual flag for transformer inference mode
    with torch.no_grad():
        out = model(lefts, rights, edge_index)
        logits = out[mask]
        labels = labels_clean[mask]
        preds = torch.argmax(logits, dim=1)
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu())
    return accuracy, f1


def train(model, data_obj, hp, device='cuda:1'):
    '''
    Basic training loop for the model
    No neighbor sampling, evaluate on validation set after each epoch
    '''

    model.to(device)
    model.train()
    model.training_mode = True # Manual flag for transformer inference mode
    optimizer = model.get_optimizer(hp.transformer_lr, hp.gnn_lr)
    criterion = nn.CrossEntropyLoss()
    samples = data_obj.samples # Strings hence no need to move to device
    lefts = data_obj.left
    rights = data_obj.right

    # Move the rest of the data to the device
    edge_index = data_obj.edge_index.to(device)
    train_mask = data_obj.train_mask.to(device)
    val_mask = data_obj.val_mask.to(device)
    test_mask = data_obj.test_mask.to(device)
    y = data_obj.y.to(device)
    positive_label_mask = torch.zeros_like(y)
    positive_label_mask[y == 1] = True # Positive label mask
    negative_label_mask = torch.zeros_like(y)
    negative_label_mask[y == 0] = True # Negative label mask
    labels_clean=data_obj.labels_clean.to(device)

    # --------------pre fine tuning-------------
    ns_mask_pft = negative_sampling(train_mask, negative_label_mask, sample_ratio=hp.sample_ratio)
    ps_mask_pft = (positive_label_mask & train_mask)
    pft_mask = (negative_sample_mask | positive_sample_mask)
    pft_left = torch.tensor(lefts)[pft_mask]
    pft_right = torch.tensor(rights)[pft_mask]
    pft_labels = y[pft_mask]

    pft_dataset = PreFTDataset(lefts=pft_left, rights=pft_right, labels=pft_labels)
    pft_dataloader = DataLoader(pft_dataset, batch_size=16, shuffle=True)

    print(f'amount of positive samples in trainset: [{(positive_label_mask & train_mask).sum().item()}/{train_mask.sum().item()}]')
    print(f'amount of negative samples in trainset: [{(negative_label_mask & train_mask).sum().item()}/{train_mask.sum().item()}]')
    for epcoh in range(1, hp.pft_epochs+1):
        for l, r, y in pft_dataloader:
            out = model(l, r, edge_index, pre_fine_tune=True)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    for epoch in range(1, hp.n_epochs+1):
        if epoch <= hp.freeze_epoch_ratio * hp.n_epochs:
            model._freeze_transformer(freeze=True)
            print('transformer freezed')
        else:
            model._freeze_transformer(freeze=False)
            print('transformer active')
        # model._freeze_transformer(freeze=True) # Test GNN overfitting, temporary
        model.train()
        model.training_mode = True # Manual flag for transformer inference mode
        optimizer.zero_grad()
        out = model(lefts, rights, edge_index)

        # Sample nodes with label 0 to be used as negative samples out of the train mask
        negative_sample_mask = negative_sampling(train_mask, negative_label_mask, sample_ratio=hp.sample_ratio)
        positive_sample_mask = (positive_label_mask & train_mask)
        train_iter_mask = (negative_sample_mask | positive_sample_mask)
        print(f'Epoch {epoch} Training on [{train_iter_mask.sum().item()}/{train_mask.sum().item()}] samples')
        loss = criterion(out[train_iter_mask], y[train_iter_mask])
        loss.backward()
        optimizer.step()
        noisy_acc, noisy_f1 = evaluate(model, edge_index, lefts, rights, train_mask, y, hp, device)
        train_acc, train_f1 = evaluate(model, edge_index, lefts, rights, train_mask, labels_clean, hp, device)
        acc, f1 = evaluate(model, edge_index, lefts, rights, val_mask, labels_clean, hp, device)
        print(f'Epoch {epoch} Loss: {loss.item():.4f} Accuracy: {acc:.4f} F1: {f1:.4f} Train Accuracy: {train_acc:.4f} Train F1: {train_f1:.4f} Noisy Accuracy: {noisy_acc:.4f} Noisy F1: {noisy_f1:.4f}')

def transform(data):
    '''
    Transform the data object into a format that can be used by the neighbor loader
    '''
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