import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def naive_unpool(f_regions, region_indices):
    _, _, C = f_regions.shape
    N, L = region_indices.shape
    index = region_indices.view(N, L, 1).expand(N, L, C)
    result = f_regions.gather(1, index)
    return result


class State:
    def __init__(self, unpooling):
        self.unpooling = unpooling
        self.__updated = False

    @property
    def updated(self):
        return self.__updated

    def get(self, name, default=None):
        return getattr(self, name, default)

    def update_state(self, **states: dict):
        self.__updated = True
        for k, v in states.items():
            setattr(self, k, v)

    def call(self, input: torch.Tensor):
        return self.unpooling(input, self)


class UnpoolingBase(nn.Module):
    def forward(self, x, state: State):
        if not state.updated:
            return x, False
        return self._forward(x, state)

    def derive_unpooler(self):
        return State(self)


class NaiveUnpooling(UnpoolingBase):
    def _forward(self, x, state: State):
        return naive_unpool(x, state.hard_labels), False


class TokenReconstructionBlock(UnpoolingBase):
    def __init__(self, k=3, temperture=0.05):
        super().__init__()
        self.k = k
        self.temperture = temperture

    def _forward(self, x, state: State):
        feat = state.feat_before_pooling
        sfeat = state.feat_after_pooling
        ds = (
            (feat * feat).sum(dim=2).unsqueeze(2)
            + (sfeat * sfeat).sum(dim=2).unsqueeze(1)
            - 2 * torch.einsum("bnc, bmc -> bnm", feat, sfeat)
        )  # distance between features and super-features
        weight = torch.exp(-self.temperture * ds)
        if self.k >= 0:
            topk, indices = torch.topk(weight, k=self.k, dim=2)
            mink = torch.min(topk, dim=-1).values
            mink = mink.unsqueeze(-1).repeat(1, 1, weight.shape[-1])
            mask = torch.ge(weight, mink)
            zero = Variable(torch.zeros_like(weight)).cuda()
            attention = torch.where(mask, weight, zero)
        else:
            attention = weight
        attention = F.normalize(attention, dim=2)
        ret = torch.einsum("bnm, bmc -> bnc", attention, x)

        return ret, False
