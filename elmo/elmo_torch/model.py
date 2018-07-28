# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CharacterEmbedding(nn.Module):
    def __init__(self, weight_matrix, filters=[32, 32, 32, 32], kernel_sizes=[2, 3, 4, 5]):
        super(CharacterEmbedding, self).__init__()
        input_size = weight_matrix.shape[1]
        self.embedding = nn.Embedding(input_size, input_size, _weight=weight_matrix)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=input_size, out_channels=filters[i], kernel_size=kernel_sizes[i]) for i in range(len(filters))])
    
    def conv_and_pool(self, x, conv):
        b, w, c, e = tuple(x.shape)
        x_out = conv(x.view(b, e, c, w)).max(dim=-1)[0]
        # x_out = F.relu(x_out)
        return x_out.view(b, w, -1)
    
    def forward(self, x):
        x = self.embedding(x)
        results = list(map(lambda conv: self.conv_and_pool(x, conv), self.convs))
        results = torch.cat(results, dim=-1)
        return results


class Elmo(nn.Module):
    def __init__(self):
        super(Elmo, self).__init__()
