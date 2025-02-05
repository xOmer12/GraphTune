from edge_sampling import *
from graph_tune import GraphTune, train, evaluate
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
    parser.add_argument("--entropy_threshold", type=float, default=1)

    # GNN params
    parser.add_argument("--conv_type", type=str, default="GAT")
    parser.add_argument("--input_layer", type=int, default=768)
    parser.add_argument("--hidden_layers", type=list, default=[32, 16])
    parser.add_argument("--seeds", type=str, default="[42, 24, 7, 30, 15]")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--transformer_lr", type=float, default=1e-3)
    parser.add_argument("--gnn_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--pft_epochs", type=int, default=2)
    parser.add_argument("--freeze_epoch_ratio", type=float, default=0.5)
    parser.add_argument("--sizes", type=list, default=[50, 10])
    parser.add_argument("--sampling_size", type=int, default=512)
    parser.add_argument("-encoding_size", type=int, default=16)
    parser.add_argument("--sample_ratio", type=float, default=0.1)

    # LoRA params
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    if torch.cuda.is_available():
        print('using GPU')
        device='cuda:1'
    else:
        print('using CPU')
        device='cpu'

    hp = parser.parse_args()
    
    task = hp.task
    og_task = hp.task.replace(hp.matcher_type, "")
    with open('task_configs.json', 'r') as file:
        configs = json.load(file)
    
    configs = {config['name']: config for config in configs}
    noisy_config = configs[task]
    real_config = configs[og_task]

    sbm = AdaptiveBMGraph(p=hp.proximity, q=hp.diversity, 
    config_true=noisy_config, config_noisy=real_config, c0=50, c1=50,beta=25)
    sbm.calc_community_probs()
    print('Community probabilities:')
    print(sbm.probs)
    print('Generating graph...')
    graph = sbm.generate_graph()
    print('Analyzing graph...')
    sbm.analyze_graph()
    print('-'*50)
    model = GraphTune(lm = hp.lm,
                      conv_type=hp.conv_type,
                      encoder_channel=hp.input_layer,
                      hidden_channels=hp.hidden_layers,
                      encoding_batch_size=hp.encoding_size,
                      device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    train(model=model, data_obj=graph, hp=hp, device=device)
