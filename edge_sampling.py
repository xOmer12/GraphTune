import numpy as np
import networkx as nx



def read_dataest(dset_path: str, get_logits=False: bool): 

    '''
    Read the dataset file and return the left samples, right samples, labels and logits (if get_logits is True)
    '''
    with open(dset_path) as f:
        lines = f.read().splitlines()

    split_lines = [line.split('\t') for line in lines] 
    split_lines = np.array(split_lines)
    left_samples = split_lines[:, 0].tolist()
    right_samples = split_lines[:, 1].tolist()
    labels = split_lines[:, 2].tolist()
    labels = [int(l) for l in labels]

    if get_logits:
        logits = split_lines[:, 3].tolist()
        logits = [float(l) for l in logits]
        return left_samples, right_samples, labels, logits
    
    else:
        return left_samples, right_samples, labels

def read_data(config: dict, get_logits=False: bool):
    '''
    Read the train, validation and test data from the config paths. Return the left samples, right samples, labels and logits, as well as masks for each set.
    '''

    # Read the data
    if get_logits:
        train_left, train_right, train_labels, train_logits = read_dataest(dset_path=config['trainset'], get_logits=True)
    else:
        train_left, train_right, train_labels = read_dataest(dset_path=config['trainset'])
    val_left, val_right, val_labels = read_dataest(dset_path=config['validset'])
    test_left, test_right, test_labels = read_dataest(dset_path=config['testset'])
    left = np.concatenate([train_left, val_left, test_left], axis=0)
    right = np.concatenate([train_right, val_right, test_right], axis=0)
    labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

    # Get the sizes of each set and create masks
    train_size, val_size, test_size = len(train_labels), len(val_labels), len(test_labels)
    dset_size = train_size + val_size + test_size
    train_mask = torch.zeros(dset_size, dtype=torch.bool)
    val_mask = torch.zeros(dset_size, dtype=torch.bool)
    test_mask = torch.zeros(dset_size, dtype=torch.bool)
    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True

    return left, right, labels, train_logits, train_mask, val_mask, test_mask


class SBMGraph:
    def __init__(self, p: float, q: float, config_true: dict, config_noisy: dict):
        self.p = p
        self.q = q
        

    def generate_graph(self):
        '''
        Generate a graph from the left and right samples, using labels as assginments to communities in SBM model
        '''
        
        