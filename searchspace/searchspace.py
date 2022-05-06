

from nas_201_api import NASBench201API as API
from neural_model.neural_model import get_cell_net


class NasBench201:
    def __init__(self, path=''):
        self.api = API(path, verbose=False)

    def get_network(self, uid):
        config = self.api.get_net_config(uid, 'cifar10-valid')
        config['num_classes'] = 1
        network = get_cell_net(config)
        return network

    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_network(uid)
            yield uid, network

    def __len__(self):
        return len(self.api)

