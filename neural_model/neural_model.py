################################################
# Developed for MLDL Project by Michele Presti #
# Developed using automl NASLib code           #
################################################

# methods to get cell based networks giving nas bench 201 configuration
from .cell_searchs import nas201_super_nets
from .cell_searchs import nasnet_super_nets
from .cell_searchs import CellStructure
from .cell_infers import TinyNetwork


def get_cell_net(config):
    """
    Function that return a neural network based on a
    cell structure following NASBench201 model
    """
    super_type = getattr(config, 'super_type', 'basic')
    group_names = ['DARTS-V1', 'DARTS-V2', 'GDAS', 'SETN', 'ENAS', 'RANDOM']
    if super_type == 'basic' and config.name in group_names:
        try:
            return nas201_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space, config.affine, config.track_running_stats)

        except:
            return nas201_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space)
    elif super_type == 'nasnet-super':
        return nasnet_super_nets[config.name](config.C, config.N, config.steps, config.multiplier, config.stem_multiplier, config.num_classes, config.space, config.affine,
                                           config.track_running_stats)
    elif config.name == 'infer.tiny':

        if hasattr(config, 'genotype'):
            genotype = config.genotype
        elif hasattr(config, 'arch_str'):
            genotype = CellStructure.str2structure(config.arch_str)
        else:
            raise ValueError('Can not find genotype from this config : {:}'.format(config))
        return TinyNetwork(config.C, config.N, genotype, config.num_classes)
    elif config.name == 'infer.shape.tiny':
        from .shape_infers import DynamicShapeTinyNet
        if isinstance(config.channels, str):
            channels = tuple([int(x) for x in config.channels.split(':')])
        else:
            channels = config.channels
        genotype = CellStructure.str2structure(config.genotype)
        return DynamicShapeTinyNet(channels, genotype, config.num_classes)
    elif config.name == 'infer.nasnet-cifar':
        raise NotImplementedError
    else:
        raise ValueError('invalid network name : {:}'.format(config.name))
