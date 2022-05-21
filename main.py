# NAS WOT Algorithm
from typing import Dict, Union

import numpy as np
import torch
import random
import argparse
from searchspace.searchspace import NasBench201
from datasets import datasets as data
from score.score import ScoreAgent as Score

# PARSING ARGUMENT

config = {}
config['data_loc'] = 'C:/Users/Michele/Desktop/FilePersonali/NAS_MachineLearningDeepLearning/files'
config['api_loc'] = 'C:/Users/Michele/Downloads/NAS-Bench-201-v1_1-096897.pth'
config['score'] = 'hook_logdet'
config['nasspace'] = 'nasbench201'
config['augtype'] = 'none'
config['dataset'] = 'cifar10'
config['maxofn'] = 3
config['batch_size'] = 128
config['seed'] = 1
config['dataset'] = 'cifar10'

"""
parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int,
                    help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
"""
#######################

# LOADING SEARCH SPACE ########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

searchspace = NasBench201(config['api_loc'])

################################

# LOADING DATASET ##############

""" Modules for data loading can be found in datasets package"""

train_dt = data.load_dataset(config['dataset'], config['augtype'], config['batch_size'], config['data_loc'])

################################

# SCORE INITIALIZING ##############

score = Score(searchspace, config['score'])
accs = np.zeros(len(searchspace))

################################

# SCORING ARCHITECTURE #########

for i, (uid, network) in enumerate(searchspace):
    try:

        # Add Hook to modules in the current network
        if 'hook_' in config['score']:

            def counting_forward_hook(module_hook, inp, out):
                try:
                    if not module_hook.visited_backwards:
                        return
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    inp = inp.view(inp.size(0), -1)
                    x = (inp > 0).float()
                    K = x @ x.t()
                    K2 = (1. - x) @ (1. - x.t())
                    network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
                except Exception as exception:
                    print(exception)
                    pass

            def counting_backward_hook(module_hook, inp, out):
                module_hook.visited_backwards = True

            for name, module in network.named_modules():
                if 'ReLU' in str(type(module)):
                    module.register_forward_hook(counting_forward_hook)
                    module.register_backward_hook(counting_backward_hook)

        # Starting score algorithm
        network = network.to(device)
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        s = []
        for j in range(config['maxofn']):
            data_iterator = iter(train_dt)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            print(x, target)
            jacobs, labels, y, out = score.get_jacobian(network, x, target, device, config)
            if 'hook_' in config['score']:
                network(x2.to(device))
                s.append(score.get_score(network.K, target))
            else:
                pass
                s.append(score.get_score(config['score'])(jacobs, labels))

        score.register_score(s, i)
        scores = score.search_score(i)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)  # type: ignore
        accs_ = accs[~np.isnan(scores)]
        scores_ = scores[~np.isnan(scores)]
        numnan = np.isnan(scores).sum()
        # tau, p = stats.kendalltau(accs_[:max(i - numnan, 1)], scores_[:max(i - numnan, 1)])
        # print(f'{tau}')
        filename = ''
        accfilename = ''
        if i % 1000 == 0:
            np.save(filename, scores)
            np.save(accfilename, accs)
        """ Check other score in original file and add """

    except Exception as e:
        print(e)
        score.register_score([np.NaN], i)

################################

# SAVING RESULT ################

""" TO IMPLEMENT """

################################

