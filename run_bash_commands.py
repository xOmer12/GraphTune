import os
import time

seeds = [42, 21, 69, 33, 1]
tasks = ['Structured/iTunes-AmazonBert', 'Structured/Amazon-GoogleBert', 'Structured/DBLP-ACMBert', 'Structured/BeerBert', 'Textual/Abt-BuyBert']
short_tasks = ['Structured/iTunes-AmazonBert', 'Structured/Amazon-GoogleBert', 'Structured/BeerBert']
p_q_combs = [(0.005, 0.001), (0.01, 0.001), (0.005, 0.0001), (0.01, 0.0001)]
p_q_extended = [(0.005, 0.001), (0.01, 0.002), (0.05, 0.01), (0.1, 0.02), (0.5, 0.1)]
agreement_thresholds = [0.75, 0.8, 0.85, 0.9, 0.95]


# # p_q experiments:
for p, q in p_q_extended:
    for task in short_tasks:
        for seed in seeds:
            command = f"python hetero_graph_pipeline.py --task {task} --seed {seed} --p {p} --q {q} --experiment_name edge_sampling_probs_extended"
            print(command)
            os.system(command)


# # agreement threshold experiments:
# for threshold in agreement_thresholds:
#     for task in tasks:
#         for seed in seeds:
#             command = f"python hetero_graph_pipeline.py --task {task} --seed {seed} --agreement_threshold {threshold} --experiment_name agreement_threshold_expr"
#             print(command)
#             os.system(command)


