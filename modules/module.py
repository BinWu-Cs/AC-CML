# import torch.nn as nn
# import re

from torchmeta.modules import MetaModule
# from collections import OrderedDict


class MetaModuleMonteCarlo(MetaModule):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.
    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        # return the parameters in model related to meta-learning
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModuleMonteCarlo) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
