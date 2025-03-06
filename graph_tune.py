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

    def __init__(self, lefts, rights, labels, lm='roberta-base'):
        self.lefts = lefts
        self.rights = rights
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        tokenized_pair = self.tokenizer.encode(self.lefts[index], self.rights[index], truncation=True, padding=True, max_length=512)
        return tokenized_pair, self.labels[index]



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
        transformer_params = list(self.transformer.parameters())
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

    # def _encode_text(self, left_samples, right_samples):
    #     all_embeddings = []
        
    #     for i in range(0, len(left_samples), self.batch_size):
    #         batch_samples_left = left_samples[i:i+self.batch_size].tolist()
    #         batch_samples_right = right_samples[i:i+self.batch_size].tolist()
    #         inputs = self.tokenizer(text=batch_samples_left, text_pair=batch_samples_right, padding=True, truncation=True, return_tensors='pt')
    #         inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
    #         # Use inference mode for transformer if not training
    #         if not self.training:
    #             with torch.inference_mode():
    #                 outputs = self._forward_transformer(inputs['input_ids'], inputs['attention_mask'])
    #         else:
    #             outputs = checkpoint(
    #                 self._forward_transformer,
    #                 inputs['input_ids'],
    #                 inputs['attention_mask'],
    #                 use_reentrant=False
    #             )
            
    #         embeddings = outputs.last_hidden_state[:, 0, :]
    #         all_embeddings.append(embeddings) 
            
    #         # Clear memory more aggressively
    #         del inputs, outputs
    #         torch.cuda.empty_cache()
    #     return torch.cat(all_embeddings, dim=0)

    def _tokenize_text(self, left_samples, right_samples):
        tokenized_samples = []
        for left, right in zip(left_samples, right_samples):
            x = self.tokenizer.encode(text=left, text_pair=right, max_length=512, truncation=True)
            tokenized_samples.append(x)
        return tokenized_samples
    
    def _encode_text(self, left_samples, right_samples):
        encodings = []
        for left, right in zip(left_samples, right_samples):
            x = self.tokenizer.encode(text=left, text_pair=right, max_length=512, truncation=True)
            enc = self.transformer(x)[0][:, 0, :]
            encodings.append(enc)
        encoded_text = torch.stack(encodings)
        return encoded_text

    
    def forward(self, x, lefts, rights, edge_index, pre_fine_tune=False, debug_mode=False):

        if pre_fine_tune:
            emb = self.transformer(x)[0][:, 0, :]
            out = self.pre_fine_tune_classifier(emb)
            if debug_mode:
                return emb
            return out

        else:
            x = self._encode_text(lefts, rights).to(device=self.device)
            
            for conv_layer in self.conv_layers:
                x = conv_layer(x, edge_index)
                x = F.relu(x)
            if debug_mode:
                return x
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
        logits = out[mask==1]
        labels = labels_clean[mask==1]
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
    negative_label_mask = torch.zeros_like(y)
    positive_sample_mask = torch.zeros_like(y)
    for i, label in enumerate(y):
        if train_mask[i]:
            if label:
                positive_sample_mask[i] = 1
            else:
                negative_label_mask[i] = 1
    labels_clean=data_obj.labels_clean.to(device)

    # --------------pre fine tuning-------------
    # pft_mask = data_obj.pft_mask.to(device)
    pft_mask = data_obj.train_mask.to(device)
    pft_left, pft_right, pft_labels = [], [], []
    for i, (l, r, y_pft) in enumerate(zip(lefts, rights, y)):
        if pft_mask[i]:
            pft_left.append(l)
            pft_right.append(r)
            pft_labels.append(y_pft)
    print(len(pft_left))
    pft_labels = torch.tensor(pft_labels).to(device)  
    pft_dataset = PreFTDataset(lefts=pft_left, rights=pft_right, labels=pft_labels)
    pft_full_dataset = PreFTDataset(lefts=lefts, rights=rights, labels=labels_clean)
    pft_dataloader = DataLoader(pft_dataset, batch_size=16, shuffle=True)
    pft_full_dataloader = DataLoader(pft_full_dataset, batch_size=16, shuffle=False)

    for epcoh in range(1, hp.pft_epochs+1):
        model.train()
        model.training_mode = True
        for l, r, y_pft in tqdm(pft_dataloader):
            optimizer.zero_grad()
            out = model(list(l), list(r), edge_index, pre_fine_tune=True)
            loss = criterion(out, y_pft)
            loss.backward()
            optimizer.step()
        match_s = []
        unmatch_s = []
        for l, r, y_pft in pft_full_dataloader:
            emb = model(l, r, edge_index, pre_fine_tune=True, debug_mode=True)
            # out = model(l, r, edge_index, pre_fine_tune=True, debug_mode=False)
            # pred = torch.argmax(out, dim=1)
            # print(f'acc: {accuracy_score(pred.cpu(), y_pft.cpu())}')
            # print(f'f1: {f1_score(pred.cpu(), y_pft.cpu())}')
            for i, label in enumerate(y_pft):
                if label:   
                    match_s.append(emb[i])
                else:
                    unmatch_s.append(emb[i])
        match_centroid = torch.mean(torch.stack(match_s), dim=0)
        unmatch_centroid = torch.mean(torch.stack(unmatch_s), dim=0)
        print(f'gap: {torch.norm(match_centroid-unmatch_centroid)}')

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
        emb = model(lefts, rights, edge_index, debug_mode=True)

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
        match_s, unmatch_s = [], []
        for i, label in enumerate(labels_clean[train_iter_mask==1]):
            if label:
                match_s.append(emb[i])
            else:
                unmatch_s.append(emb[i])
        match_centroid = torch.mean(torch.stack(match_s), dim=0)
        unmatch_centroid = torch.mean(torch.stack(unmatch_s), dim=0)
        print(f'gap: {torch.norm(match_centroid-unmatch_centroid)}')
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