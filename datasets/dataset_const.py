import numpy as np

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'fake': 10,
                 'imagenet-1k-s': 1000,
                 'imagenette2': 10,
                 'imagenet-1k': 1000,
                 'ImageNet16': 1000,
                 'ImageNet16-150': 150,
                 'ImageNet16-120': 120,
                 'ImageNet16-200': 200}

imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}
