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



def read_data(path):
    with open(path) as f:
        lines = f.read().splitlines()
    split_lines = [line.split('\t') for line in lines]
    split_lines = np.array(split_lines)
    left_samples = split_lines[:, 0].tolist()
    right_samples = split_lines[:, 1].tolist()
    labels = split_lines[:, 2].tolist()
    labels = [int(l) for l in labels]
    return left_samples, right_samples, labels


class PreFTDataset(Dataset):

    def __init__(self, path, lm='roberta-base', max_len=512):
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.max_len = max_len
        left_samples, right_samples, labels = read_data(path)
        self.left_samples = left_samples
        self.right_samples = right_samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        left_sample = self.left_samples[idx]
        right_sample = self.right_samples[idx]
        label = self.labels[idx]
        return left_sample, right_sample, label
    
class Encoder(nn.Module):

    def __init__(self, lm='roberta-base', device='cuda:1'):
        super(Encoder, self).__init__()
        self.lm_type = lm
        self.transformer = AutoModel.from_pretrained(lm)
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        hidden_size = self.transformer.config.hidden_size
        self.fc = nn.Linear(hidden_size, 2)
        self.device = device

    def forward(self, left_samples, right_samples, inference=False):
        tokenized = self.tokenizer(text=left_samples, text_pair=right_samples, padding=True, truncation=True, return_tensors='pt', max_length=256)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        encodings = self.transformer(input_ids, attention_mask=attention_mask)
        enc = encodings.last_hidden_state[:, 0]
        if inference:
            return self.fc(enc), enc
        else:
            return self.fc(enc)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

def pre_fine_tune(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    
    for epoch in range(1, epochs+1):
        total_loss = 0
        for left_samples, right_samples, labels in tqdm(train_loader):
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(left_samples, right_samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss/len(train_loader)}")


def inference(model, loader):
    x = []
    with torch.no_grad():
        model.eval()
        for left_samples, right_samples, labels in loader:
            _, enc = model(left_samples, right_samples, inference=True)
            x.append(enc)
    return torch.cat(x, dim=0)

        
def evaluate_space(model, train_loader, device):
    with torch.no_grad():
        model.eval()
        positive = []
        negative = []
        for left_samples, right_samples, labels in tqdm(train_loader):
            labels = labels.to(device)
            _, enc = model(left_samples, right_samples, inference=True)
            for v, label in zip(enc, labels):
                if label == 1:
                    positive.append(v)
                else:
                    negative.append(v)
        pos_centroid = torch.mean(torch.stack(positive), dim=0)
        neg_centroid = torch.mean(torch.stack(negative), dim=0)
        gap = torch.norm(pos_centroid - neg_centroid)
    return positive, negative, pos_centroid, neg_centroid, gap