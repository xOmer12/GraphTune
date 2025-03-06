from edge_sampling import *
from gnn_transformer import GraphNNTransformer, train
from pft_encoder import read_data, PreFTDataset, Encoder, pre_fine_tune
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import json
import torch
import numpy as np
import ast
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Amazon-GoogleBert")
    parser.add_argument("--matcher_type", type=str, default="Bert")
    parser.add_argument("--lm", type=str, default="roberta-base")

    # Generative graph model params
    parser.add_argument("--proximity", type=float, default=0.005)
    parser.add_argument("--diversity", type=float, default=0.0005)
    parser.add_argument("--entropy_threshold", type=float, default=0.4)

    # GNN params
    parser.add_argument("--conv_type", type=str, default="GraphSAGE")
    parser.add_argument("--input_layer", type=int, default=768)
    parser.add_argument("--hidden_layers", type=list, default=[32, 16])
    parser.add_argument("--seeds", type=str, default="[42, 24, 7, 30, 15]")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--pft_epochs", type=int, default=1)
    parser.add_argument("--freeze_epoch_ratio", type=float, default=0.3)
    parser.add_argument("--sizes", type=list, default=[50, 10])
    parser.add_argument("--sampling_size", type=int, default=512)
    parser.add_argument("-encoding_size", type=int, default=16)
    parser.add_argument("--sample_ratio", type=float, default=0.1)

    if torch.cuda.is_available():
        print('using GPU')
        device='cuda:1'
    else:
        print('using CPU')
        device='cpu'

    hp = parser.parse_args()

    # Extract task configs
    task = hp.task
    og_task = hp.task.replace(hp.matcher_type, "")
    with open('task_configs.json', 'r') as file:
        configs = json.load(file)
    
    configs = {config['name']: config for config in configs}
    noisy_config = configs[task]
    real_config = configs[og_task]

    sbm = AdaptiveBMGraph(p=hp.proximity, q=hp.diversity, 
    config_true=real_config, config_noisy=noisy_config, c0=50, c1=50,beta=25)
    sbm.calc_community_probs()
    print('Community probabilities:')
    print(sbm.probs)
    print('Generating graph...')
    graph = sbm.generate_graph()
    print('Analyzing graph...')
    sbm.analyze_graph()


    # PFT for encoder:
    pft_dataset = PreFTDataset(noisy_config["trainset"], max_len=128)
    pft_dataloader = DataLoader(pft_dataset, batch_size=32, shuffle=True)
    encoding_model = Encoder(device=device)
    pft_criterion = nn.CrossEntropyLoss()
    pft_optimizer = optim.Adam(encoding_model.parameters(), lr=1e-5)
    encoding_model.to(device)
    pre_fine_tune(encoding_model, pft_dataloader, pft_optimizer, pft_criterion, device, epochs=1)

    # GNNT model:
    model = GraphNNTransformer(encoder_model=encoding_model, conv_type=hp.conv_type, hidden_channels=hp.hidden_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    train(model=model, optimizer=optimizer, criterion=criterion, data_object=graph, hp=hp, device=device)