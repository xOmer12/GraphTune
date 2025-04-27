
from sentence_transformer import SentenceTransformer
from hetero_graph import *
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


    # Read the data:
    similarity_model = SentenceTransformer(hp.sim_lm)
    dset_path = real_config["trainset"]
    noisy_dset_path = noisy_config["trainset"]
    val_dset_path = real_config["valset"]
    test_dset_path = real_config["testset"]
    train_left, train_right, labels = get_data(dset_path)
    val_left, val_right, val_labels = get_data(val_dset_path)
    test_left, test_right, test_labels = get_data(test_dset_path)
    left = train_left + val_left + test_left
    right = train_right + val_right + test_right
    _, _, noisy_labels = get_data(noisy_dset_path)
    labels = [int(l) for l in labels] + [int(l) for l in val_labels] + [int(l) for l in test_labels]
    noisy_labels = [int(l) for l in noisy_labels] + [int(l) for l in val_labels] + [int(l) for l in test_labels]
    
    # Create masks:
    train_mask = [1]*len(train_left) + [0]*len(val_left) + [0]*len(test_left)
    val_mask = [0]*len(train_left) + [1]*len(val_left) + [0]*len(test_left)
    test_mask = [0]*len(train_left) + [0]*len(val_left) + [1]*len(test_left)

    # Create the graph:
    data_attr_agreement = extract_agreement_pairs(left, right, similarity_model)
    mutual_agreement_graph = build_mutual_agreement_graph(data_attr_agreement, threshold=hp.ma_tresh, 
                                                      sample_edge=True, sampling_prob=hp.ma_sample_prob)
    
    # PFT for encoder:
    pft_dataset = PreFTDataset(noisy_config["trainset"], max_len=128)
    pft_dataloader = DataLoader(pft_dataset, batch_size=32, shuffle=True)
    encoding_model = Encoder(device=device)
    pft_criterion = nn.CrossEntropyLoss()
    pft_optimizer = optim.Adam(encoding_model.parameters(), lr=1e-5)
    encoding_model.to(device)
    pre_fine_tune(encoding_model, pft_dataloader, pft_optimizer, pft_criterion, device, epochs=hp.pft_epochs)


    X = X = init_embeddings(left, right, encoding_model, sentence_transformer=False)
    # GNNT model:
    rgnn_model = RGNN(init_dim=768, channels=[512, 256], edge_types=mutual_agreement_graph.edge_types, 
                    conv_type='SAGE', agg_type='sum', num_classes=2)
    rgnn_model = rgnn_model.to(device)

    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool).to(device)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool).to(device)
    test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool).to(device)
    noisy_labels_tensor = torch.tensor(noisy_labels, dtype=torch.long).to(device)
    true_labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    mutual_agreement_graph['node'].x = X
    mutual_agreement_graph['node'].y = noisy_labels_tensor
    mutual_agreement_graph = mutual_agreement_graph.to(device)

    embedding_epoch_dict, labels_epoch_dict, dist_epoch_dict, out_epoch_dict = train_model(rgnn_model, mutual_agreement_graph, noisy_labels=noisy_labels_tensor, 
            true_labels=true_labels_tensor, train_mask=train_mask_tensor, val_mask=val_mask_tensor, epochs=50, lr=1e-3, analyze_learning=True)
    
    # Write model results to files:
    evaluated_labels, accuracy, f1 = evaluate_model(rgnn_model, mutual_agreement_graph, test_mask_tensor, true_labels_tensor)
    