
from sentence_transformers import SentenceTransformer
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

def get_data(dset_path, has_labels=True):
    with open(dset_path) as f:
        lines = f.read().splitlines()
        # remove punctuation from the text
    removePunc = list()
    # remove extra spaces and the punctuation and the symbol "-"
    [removePunc.append(line.replace(" -", "").translate(str.maketrans('', '', string.punctuation)).replace("  ", " "))
    for line in lines]

    # split the lines according to "\t" since the different instances in a line of one example are separated by "\t"
    splitList = [line.split('\t') for line in removePunc]
    # turning to numpy array for in order to use numpy functionality
    npArray = np.array(splitList)
    # getting the left side examples
    leftCol = npArray[:, 0].tolist()
    # getting the right side examples
    rightCol = npArray[:, 1].tolist()
    # if the dataset is with labels (train, valid) then extract also the labels from the file
    if has_labels:
        # getting the labels of the pairs
        label = npArray[:, 2].tolist()
        return leftCol, rightCol, label
    else:
        return leftCol, rightCol

def write_results_to_file(file_path, res_dict):

    with open(file_path, 'r') as file:
        data = json.load(file)
    data.append(res_dict)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="hetero_graph")
    parser.add_argument("--task", type=str, default="Structured/iTunes-AmazonBert")
    parser.add_argument("--matcher_type", type=str, default="Bert")

    # Generative graph model params
    parser.add_argument("--sim_lm", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--q", type=float, default=0.001)
    parser.add_argument("--agreement_threshold", type=float, default=0.85)

    # Encoding model parameters
    parser.add_argument("--encoding_lm", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--encoding_lr", type=int, default=1e-5)
    parser.add_argument("--encoding_epochs", type=int, default=10)

    # GNN params
    parser.add_argument("--conv_type", type=str, default="SAGE")
    parser.add_argument("--agg_type", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--input_layer", type=int, default=768)
    parser.add_argument("--hidden_layers", type=list, default=[256, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gnn_lr", type=float, default=3e-4)
    parser.add_argument("--n_epochs", type=int, default=10)


   

    if torch.cuda.is_available():
        print('using GPU')
        device='cuda'
    else:
        print('using CPU')
        device='cpu'

    hp = parser.parse_args()

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)

    task = hp.task
    og_task = hp.task.replace(hp.matcher_type, "")
    with open('task_configs.json', 'r') as file:
        configs = json.load(file)
    
    configs = {config['name']: config for config in configs}
    noisy_config = configs[task]
    real_config = configs[og_task]

    # Read the data
    print('Reading data...')
    train_left, train_right, labels = get_data(real_config['trainset'])
    val_left, val_right, val_labels = get_data(real_config['validset'])
    test_left, test_right, test_labels = get_data(real_config['testset'])
    left = train_left + val_left + test_left
    right = train_right + val_right + test_right
    _, _, noisy_labels = get_data(noisy_config['trainset'])
    labels = [int(l) for l in labels] + [int(l) for l in val_labels] + [int(l) for l in test_labels]
    noisy_labels = [int(l) for l in noisy_labels] + [int(l) for l in val_labels] + [int(l) for l in test_labels]

    # Create masks:
    train_mask = [1]*len(train_left) + [0]*len(val_left) + [0]*len(test_left)
    val_mask = [0]*len(train_left) + [1]*len(val_left) + [0]*len(test_left)
    test_mask = [0]*len(train_left) + [0]*len(val_left) + [1]*len(test_left)

    # Create the graph
    print('Creating graph...')
    simillarity_model = SentenceTransformer(hp.sim_lm)
    data_attr_agreement = extract_agreement_pairs(left, right, simillarity_model)
    mutual_agreement_graph = build_mutual_agreement_graph(data_attr_agreement, threshold=hp.agreement_threshold, 
                                                      sample_edge=True, sampling_prob=hp.p, null_sampling_prob=hp.q)
    
    # Create initial node embeddings with encoding model
    print('Creating initial node embeddings...')
    pft_dataset = PreFTDataset(noisy_config['trainset'], max_len=hp.max_len)
    pft_dataloader = torch.utils.data.DataLoader(pft_dataset, batch_size=hp.batch_size, shuffle=True)
    encoding_model = Encoder(lm=hp.encoding_lm, device=device)
    pft_criterion = nn.CrossEntropyLoss()
    pft_optimizer = optim.Adam(encoding_model.parameters(), lr=hp.encoding_lr)
    encoding_model.to(device)
    pre_fine_tune(encoding_model, pft_dataloader, pft_optimizer, pft_criterion, device, epochs=hp.encoding_epochs)
    
    X = init_embeddings(left, right, encoding_model, sentence_transformer=False)

    # Run the HeteroGraph model
    print('Running the model...')
    rgnn_model = RGNN(init_dim=768, channels=hp.hidden_layers, edge_types=mutual_agreement_graph.edge_types, 
                  conv_type=hp.conv_type, agg_type=hp.agg_type, num_classes=2)
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
            true_labels=true_labels_tensor, train_mask=train_mask_tensor, val_mask=test_mask_tensor, epochs=hp.n_epochs, lr=hp.gnn_lr, analyze_learning=False)
    
    evaluated_labels, accuracy, f1 = evaluate_model(rgnn_model, mutual_agreement_graph, test_mask_tensor, true_labels_tensor)

    res_dict = {
        'experiment_name': hp.experiment_name,
        'task': og_task,
        'seed': hp.seed,
        'edge_sampling_prob': hp.p,
        'null_sampling_prob': hp.q,
        'agreement_threshold': hp.agreement_threshold,
        'encoding_model': hp.encoding_lm,
        'encoding_lr': hp.encoding_lr,
        'encoding_epochs': hp.encoding_epochs,
        'gnn_model': hp.conv_type,
        'gnn_agg_type': hp.agg_type,
        'gnn_epochs': hp.n_epochs,
        'accuracy': accuracy,
        'f1': f1
    }

    write_results_to_file('results.json', res_dict)
            
