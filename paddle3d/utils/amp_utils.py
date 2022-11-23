from collections.abc import Mapping, Sequence
from typing import List

import paddle


def dtype2float32(src_tensors):
    if isinstance(src_tensors,
                  paddle.Tensor) and src_tensors.dtype != 'float32':
        return src_tensors.astype('float32')
    elif isinstance(src_tensors, Sequence):
        return type(src_tensors)([dtype2float32(x) for x in src_tensors])
    elif isinstance(src_tensors, Mapping):
        return {key: dtype2float32(x) for key, x in src_tensors.items()}
    return src_tensors
