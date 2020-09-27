# from https://github.com/NLPInBLCU/BiaffineDependencyParsing/blob/master/modules/biaffine.py

import torch
import torch.nn as nn


class Bilinear(nn.Module):
    """
    使用版本
    A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)"""

    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super(Bilinear, self).__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())

        intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size),)

        input2 = input2.transpose(1, 2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)

        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

        if self.bias is not None:
            output = output + self.bias

        return output
