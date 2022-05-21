import numpy as np
import torch


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()


def random_score(jacob, label=None):
    return np.random.normal()


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld


class ScoreAgent:
    def __init__(self, searchspace, score_type):
        self.scores = np.zeroes(len(searchspace))
        self._scores = {
                            'hook_logdet': hooklogdet,
                            'random': random_score
                        }
        self.get_score = self._scores[score_type]

    def score_network(self, network, metrics, dataset):
        """
        Function that given a network, a metric and a dataset
        calculates the score of the network
        (this scoring is intended to be as training free score)
        """
        pass

    def register_score(self, score: list, index):
        self.scores[index] = np.mean(score)

    def search_score(self, index):
        return self.scores[index]

    def get_score_func(self, score_name):
        return self._scores[score_name]

    def get_jacobian(self, net, x, target, device, args=None):
        return get_batch_jacobian(net, x, target, device, args=None)

