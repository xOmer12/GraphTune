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
from peft import get_peft_model, LoraConfig, TaskType
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

    def signature(self):
        return f'{self.lm}_{self.conv_type}'
    
    def get_optimizer(self, transformer_lr=1e-5, gnn_lr=1e-3):
        # Separate parameters for transformer and GNN
        transformer_params = self.transformer.parameters()
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
    
    def _encode_text(self, samples):
        all_embeddings = []
        
        with torch.no_grad():  # Add this since embeddings don't need gradients during inference
            for i in range(0, len(samples), self.batch_size):
                batch_samples = samples[i:i+self.batch_size]
                inputs = self.tokenizer(batch_samples, return_tensors="pt", padding=True, truncation=True)
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
                all_embeddings.append(embeddings.detach())  # Detach to break computation graph
                
                # Clear memory more aggressively
                del inputs, outputs
                torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0)

    
    def forward(self, samples, edge_index):
        
        x = self._encode_text(samples).to(device=self.device)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = F.layer_norm(x, x.shape[1:])
        x = self.classifier(x)
        return x
        
    """Transformer encoder fine-tuned via LoRA and GNN classification head"""

    def __init__(self, lm='roberta-base', conv_type='GCN', encoder_channel=768, 
                 hidden_channels=[32, 16], seed=42, encoding_batch_size=16, 
                 device='cuda', lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.lm = lm
        self.device = device
        
        # Initialize base transformer
        self.transformer = AutoModel.from_pretrained(lm)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.conv_type = conv_type
        self.hc_config = [encoder_channel]+hidden_channels
        self.seed = seed
        self.conv_layers = nn.ModuleList()
        self.batch_size = encoding_batch_size
        self.dropout = nn.Dropout(p=0.5)

        # Initialize graph layers
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

    def get_optimizer(self, transformer_lr=1e-5, gnn_lr=1e-3):

        # Separate trainable parameters
        lora_params = []
        gnn_params = []
        
        # Get LoRA parameters
        for name, param in self.transformer.named_parameters():
            if 'lora' in name:
                lora_params.append(param)
                
        # Get GNN and classifier parameters
        for layer in self.conv_layers:
            gnn_params.extend(list(layer.parameters()))
        gnn_params.extend(list(self.classifier.parameters()))
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': lora_params, 'lr': transformer_lr},
            {'params': gnn_params, 'lr': gnn_lr}
        ]
        
        return AdamW(param_groups)

    def _forward_transformer(self, input_ids, attention_mask):
        return self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    
    def _encode_text(self, samples):
        """
        Encode text samples using LoRA-adapted transformer
        """
        all_embeddings = []
        
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i:i+self.batch_size]
            inputs = self.tokenizer(batch_samples, return_tensors="pt", 
                                  padding=True, truncation=True)
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
            x = self.dropout(x)
            x = F.layer_norm(x, x.shape[1:])
            
        x = self.classifier(x)
        return x

    def save_lora_weights(self, path):
        self.transformer.save_pretrained(path)
        
    def load_lora_weights(self, path):
        self.transformer.load_adapter(path)    

def evaluate(model, edge_index, samples, mask, labels_clean, hp, device='cuda:1'):
    '''
    Evaluate the model on the given data mask
    No neighbor sampling
    '''
    model.eval()
    model.training_mode = False # Manual flag for transformer inference mode
    with torch.no_grad():
        out = model(samples, edge_index)
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

    # Move the rest of the data to the device
    edge_index = data_obj.edge_index.to(device)
    train_mask = data_obj.train_mask.to(device)
    val_mask = data_obj.val_mask.to(device)
    test_mask = data_obj.test_mask.to(device)
    y = data_obj.y.to(device)
    labels_clean=data_obj.labels_clean.to(device)

    for epoch in range(1, hp.n_epochs+1):
        model.train()
        model.training_mode = True # Manual flag for transformer inference mode
        optimizer.zero_grad()
        out = model(data_obj.samples, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        accuracy, f1 = evaluate(model, edge_index, samples, val_mask, labels_clean, hp, device)
        print(f'Epoch {epoch} Loss: {loss.item()} Accuracy: {accuracy} F1: {f1}')

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