import numpy as np


def random_score(jacob, label=None):
    return np.random.normal()


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld


class ScoreAgent:
    def __init__(self, searchspace):
        self.scores = np.zeroes(len(searchspace))
        self._scores = {
                            'hook_logdet': hooklogdet,
                            'random': random_score
                        }

    def score_network(self, network, metrics, dataset):
        """
        Function that given a network, a metric and a dataset
        calculates the score of the network
        (this scoring is intended to be as training free score)
        """
        pass

    def register_score(self, score: list, index):
        self.scores[index] = np.mean(score)

    def get_score_func(self, score_name):
        return self._scores[score_name]

