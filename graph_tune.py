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
            x = self.dropout(x)
            x = F.layer_norm(x, x.shape[1:])
        x = self.classifier(x)
        return x
        
class GraphTuneLoRA(nn.Module):
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

def evaluate(model, data, samples, mask, hp, device='cuda:1'):
    model.eval()
    all_predictions = []
    all_true_labels = []

    val_loader = neighbor_loader.NeighborLoader(
        data,  
        num_neighbors=hp.sizes, 
        batch_size=hp.sampling_size,
        shuffle=False
       )

    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch_samples = [samples[i] for i in batch.n_id]
            if mask != None:
                batch_eval_mask = mask[batch.n_id]
            else:
                batch_eval_mask = None
            out = model(batch_samples, batch.edge_index)
            predictions = out.argmax(dim=1)
            true_labels = batch.real_labels.to(device)
            if mask != None:
                all_predictions.append(predictions[batch_eval_mask].cpu())
                all_true_labels.append(true_labels[batch_eval_mask].cpu())
            else:
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

def transform_flip(data):
    samples = data.samples
    left = data.right
    right = data.left
    clean_data = Data(
        edge_index=torch.tensor(data.edge_index) if not isinstance(data.edge_index, torch.Tensor) else data.edge_index,
        y=torch.tensor(data.y) if not isinstance(data.y, torch.Tensor) else data.y,
        logits=torch.tensor(data.logits) if not isinstance(data.logits, torch.Tensor) else data.logits,
        real_labels=torch.tensor(data.real_labels) if not isinstance(data.real_labels, torch.Tensor) else data.real_labels,
        mismatch_mask=torch.tensor(data.mismatch_mask) if not isinstance(data.mismatch_mask, torch.Tensor) else data.mismatch_mask,
        num_nodes=len(samples)
    )
    return left, right, samples, clean_data

def train(model, data_obj, hp, device='cuda:1'):

    optimizer = model.get_optimizer(
        transformer_lr=hp.transformer_lr,
        gnn_lr=hp.gnn_lr 
    )

    _, _, samples, data = transform(data_obj)
    train_labels = data.y[data.train_mask]
    class_counts = torch.bincount(train_labels)
    total_samples = data.y.size(0)
    num_classes = 2
    class_weights = total_samples / (class_counts * num_classes)
    class_weights = class_weights.to(device)
    print(f'Loss weights distribution: {class_weights}')
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model.to(device)  

    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    for name, param in model.transformer.named_parameters():
        if 'lora' not in name:  # Identify non-LoRA parameters
            param.requires_grad = False
        else:
            param.requires_grad = True 
        
    loader = neighbor_loader.NeighborLoader(
        data, 
        num_neighbors=hp.sizes, 
        batch_size=hp.sampling_size,
        shuffle=True
       )

    for epoch in range(1, hp.n_epochs + 1):
        
        model.train()
        total_loss = 0
        num_train_batches = 0

        # Freeze all transformer in early epcohs
        if epoch < hp.n_epochs / 2:
            for param in model.transformer.parameters():
                param.require_grad = False
        else:
            for param in model.transformer.parameters():
                param.require_grad = True

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
    return train_acc, train_f1, acc, f1

def train_flip(model, data_obj, hp, device='cuda:1'): 


    # for name, param in model.transformer.named_parameters():
    #     if 'lora' not in name:
    #         param.requires_grad = False

    if isinstance(model, GraphTuneLoRA):
        optimizer = model.get_optimizer(
            transformer_lr=hp.transformer_lr,
            gnn_lr=hp.gnn_lr 
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

    _, _, samples, data = transform_flip(data_obj)
    train_labels = data.y
    class_counts = torch.bincount(train_labels)
    total_samples = data.y.size(0)
    num_classes = 2
    class_weights = total_samples / (class_counts * num_classes)
    class_weights = class_weights.to(device)
    print(f'Loss weights distribution: {class_weights}')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    

    model.to(device)  

    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.mismatch_mask = data.mismatch_mask.to(device)
    data.real_labels = data.real_labels.to(device)

    print("Real Class Distribution:", torch.bincount(data.real_labels))
    print("Noisy Class Distribution:", torch.bincount(data.y))
        
    loader = neighbor_loader.NeighborLoader(
        data, 
        num_neighbors=hp.sizes, 
        batch_size=hp.sampling_size,
        shuffle=True
    )
    
    val_loader = neighbor_loader.NeighborLoader(
        data,  
        num_neighbors=hp.sizes, 
        batch_size=hp.sampling_size,
        shuffle=False
    )

    for epoch in range(1, hp.n_epochs + 1):
        
        model.train()
        total_loss = 0
        num_train_batches = 0

        print('performing train step:')
        for batch in tqdm(loader):
            optimizer.zero_grad()
            # seed_nodes = batch.n_id[:hp.sampling_size]
            batch_samples = [samples[i] for i in batch.n_id]
            
            # seed_mask = torch.zeros(len(batch_samples), dtype=torch.bool)
            # seed_mask[seed_nodes] = True
            out = model(batch_samples, batch.edge_index)  # Feed the corresponding sample embeddings
            loss = criterion(out, batch.y)  # Only use training nodes
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_train_batches += 1
        
        avg_loss = total_loss / num_train_batches

        print('evaluating model:')
        model.eval()
        all_pred = []
        all_true_labels = []
        all_noisy_labels = []
        mismatch_pred = []
        mismatch_true_labels = []
        mismatch_noisy_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader):

                batch_samples = [samples[i] for i in batch.n_id]
                # seed_nodes = batch.n_id[:hp.sampling_size]
                out = model(batch_samples, batch.edge_index)

                # seed_mask = torch.zeros(len(batch_samples), dtype=torch.bool)
                # seed_mask[seed_nodes] = True
                predictions = out.argmax(dim=1).to(device)
                true_labels = batch.real_labels.to(device)
                noisy_labels = batch.y.to(device)

                all_pred.append(predictions.cpu())
                all_true_labels.append(true_labels.cpu())
                all_noisy_labels.append(noisy_labels.cpu())

                mismatch_seed_mask = batch.mismatch_mask[seed_nodes]

                mismatch_pred.append(predictions[batch.mismatch_mask].cpu())
                mismatch_true_labels.append(true_labels[batch.mismatch_mask].cpu())
                mismatch_noisy_labels.append(noisy_labels[batch.mismatch_mask].cpu())
        
            all_pred = torch.cat(all_pred, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)
            all_noisy_labels = torch.cat(all_noisy_labels, dim=0)
            mismatch_pred = torch.cat(mismatch_pred, dim=0)
            mismatch_true_labels = torch.cat(mismatch_true_labels, dim=0)

            accuracy = accuracy_score(all_true_labels, all_pred)
            f1 = f1_score(all_true_labels, all_pred)
            mm_accuracy = accuracy_score(mismatch_true_labels, mismatch_pred)
            mm_f1 = f1_score(mismatch_true_labels, mismatch_pred)
            noisy_accuracy = accuracy_score(all_noisy_labels, all_pred)
            noisy_f1 = f1_score(all_noisy_labels, all_pred)

            print(f"Epoch {epoch}/{hp.n_epochs}, Loss: {avg_loss:.4f}, mm_f1: {mm_f1:.4f}, mm_accuracy: {mm_accuracy:.4f}, f1: {f1:.4f}, acc:{accuracy:.4f}, n_f1:{noisy_f1:.4f}, n_acc:{noisy_accuracy:.4f}")
    return accuracy, f1, mm_accuracy, mm_f1

def train_flip_no_load(model, data_obj, hp, device='cuda:1'): 

    for name, param in model.transformer.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    optimizer = model.get_optimizer(
        transformer_lr=hp.transformer_lr,
        gnn_lr=hp.gnn_lr 
    )

    _, _, samples, data = transform_flip(data_obj)
    criterion = nn.CrossEntropyLoss()

    model.to(device)  

    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.mismatch_mask = data.mismatch_mask.to(device)
    data.real_labels = data.real_labels.to(device)

    print("Real Class Distribution:", torch.bincount(data.real_labels))
    print("Noisy Class Distribution:", torch.bincount(data.y))
    for epoch in range(1, hp.n_epochs + 1):
        
        model.train()
        optimizer.zero_grad()        
        out = model(samples, data.edge_index)  # Feed the corresponding sample embeddings
        loss = criterion(out, data.y)  # Only use training nodes
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
           
            out = model(samples, data.edge_index)
            predictions = out.argmax(dim=1)
            mismatch_pred = predictions[data.mismatch_mask].cpu()
            mismatch_true_labels = data.real_labels[data.mismatch_mask].cpu()


            accuracy = accuracy_score(y_true=data.real_labels.cpu(), y_pred=predictions.cpu())
            f1 = f1_score(y_true=data.real_labels.cpu(), y_pred=predictions.cpu())
            mm_accuracy = accuracy_score(y_true=mismatch_true_labels, y_pred=mismatch_pred)
            mm_f1 = f1_score(y_true=mismatch_true_labels, y_pred=mismatch_pred)
            noisy_accuracy = accuracy_score(y_true=data.y.cpu(), y_pred=predictions.cpu())
            noisy_f1 = f1_score(y_true=data.y.cpu(), y_pred=predictions.cpu())

            print(f"Epoch {epoch}/{hp.n_epochs}, Loss: {loss.item():.4f}, mm_f1: {mm_f1:.4f}, mm_accuracy: {mm_accuracy:.4f}, f1: {f1:.4f}, acc:{accuracy:.4f}, n_f1:{noisy_f1:.4f}, n_acc:{noisy_accuracy:.4f}")
    return accuracy, f1, mm_accuracy, mm_f1