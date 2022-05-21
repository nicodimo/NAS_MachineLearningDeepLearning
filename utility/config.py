import os
import json

support_types = ('str', 'int', 'bool', 'float', 'none')


def convert_param(original_lists):
    assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list: value = [value]
    outs = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            if x.lower() != 'none':
                raise ValueError('For the none type, the value must be none instead of {:}'.format(x))
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    if not is_list: outs = outs[0]
    return outs


def load_config(path='files/config.txt', verbose=True):
    path = str(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    with open(path, 'r') as f:
        data = json.load(f)
    # content = {k: convert_param(v) for k, v in data.items()}
    # if verbose:
        # print('{:}'.format(content))
    return data

