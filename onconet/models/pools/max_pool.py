import torch
import torch.nn as nn
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool

@RegisterPool('GlobalMaxPool')
class GlobalMaxPool(AbstractPool):

    def replaces_fc(self):
        return False

    def forward(self, x):
        # print('IN: global_max_pool', x.size())
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        # torch.save(x, 'x_max_pool_idx.pt')
        x, idx = torch.max(x, dim=-1)
        # torch.save(idx, 'max_pool_idx.pt')
        # print('OUT: global_max_pool', x.size())
        return None, x
