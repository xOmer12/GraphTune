from sbm_graph import StochasticBlockModel
from graph_tune import GraphTune, GraphTuneLoRA, train, evaluate
import argparse
import json
import torch
import numpy as np
import ast


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="Structured/iTunes-AmazonBert")
    parser.add_argument("--lm", type=str, default="roberta-base")

    # Generative graph model params
    parser.add_argument("--proximity", type=float, default=0.7)
    parser.add_argument("--diversity", type=float, default=0.3)
    parser.add_argument("--entropy_threshold", type=float, default=0.4)

    # GNN params
    parser.add_argument("--conv_type", type=str, default="GraphSAGE")
    parser.add_argument("--input_layer", type=int, default=768)
    parser.add_argument("--hidden_layers", type=list, default=[64, 32])
    parser.add_argument("--seeds", type=str, default="[42, 24, 7, 30, 15]")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--transformer_lr", type=float, default=1e-5)
    parser.add_argument("--gnn_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--sizes", type=list, default=[50, 10])
    parser.add_argument("--sampling_size", type=int, default=64)
    parser.add_argument("-encoding_size", type=int, default=32)

    # LoRA params
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)


    hp = parser.parse_args()
    
    task = hp.task
    with open('task_configs.json', 'r') as file:
        configs = json.load(file)
    
    configs = {config['name']: config for config in configs}
    task_config = configs[task]

    sbm = StochasticBlockModel(p=hp.proximity, q=hp.diversity, config=task_config)
    graph = sbm.generate_graph(entropy_threshold=hp.entropy_threshold)

    model = GraphTuneLoRA(lm = hp.lm, 
                      conv_type=hp.conv_type, 
                      encoder_channel=hp.input_layer, 
                      hidden_channels=hp.hidden_layers,
                      lora_r=hp.lora_rank,
                      lora_alpha=hp.lora_alpha,
                      lora_dropout=hp.lora_dropout)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    if torch.cuda.is_available():
        print('using GPU')
        device='cuda'
    else:
        print('using CPU')
        device='cpu'
    res = train(model=model, data_obj=graph, hp=hp, device=device)
