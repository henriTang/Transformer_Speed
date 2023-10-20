# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math

import torch
from torch import nn

logger = logging.getLogger(__name__)


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}

BertLayerNorm = torch.nn.LayerNorm


class MSCFFN_step1(nn.Module):
    def __init__(self, config):
        super(MSCFFN_step1, self).__init__()
        self.ffn_head = 12
        self.ffn_head_size = int(config.hidden_size / self.ffn_head)
        self.dense_ffn_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn_intermediate_size = self.ffn_head_size * 6
        self.ffn_intermediate_weights = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.ffn_head,
                                                                                          self.ffn_head_size, self.ffn_intermediate_size)))
        self.ffn_intermediate_bias = nn.Parameter(nn.init.zeros_(torch.empty(self.ffn_head,
                                                                             self.ffn_intermediate_size)))
        # self.dropout = nn.Dropout(dropout_rate)
        self.relu_act = ACT2FN["relu"]

    def transpose_for_scores(self, x):
        x_size = x.size()
        new_x_shape = [x_size[0]*x_size[1], self.ffn_head, self.ffn_head_size]
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2)

    def forward(self, hidden_states):
        hidden_states = self.dense_ffn_head(hidden_states)
        hidden_states = self.transpose_for_scores(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.ffn_intermediate_weights) + self.ffn_intermediate_bias.unsqueeze(1)
        hidden_relu = self.relu_act(hidden_states[0: int(self.ffn_head/2), :, :])
        hidden_linear = hidden_states[int(self.ffn_head/2):self.ffn_head, :, :]
        hidden_states = hidden_relu * hidden_linear
        # hidden_states = self.dropout(hidden_states)
        return hidden_states


class MSCFFN_step2(nn.Module):
    def __init__(self, config):
        super(MSCFFN_step2, self).__init__()
        self.ffn_head = 12
        self.ffn_head_size = int(config.hidden_size / self.ffn_head)
        self.ffn_intermediate_size = self.ffn_head_size * 6
        self.ffn_weights = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(int(self.ffn_head/2),
                                                                             self.ffn_intermediate_size, self.ffn_head_size)))
        self.ffn_bias = nn.Parameter(nn.init.zeros_(torch.empty(int(self.ffn_head/2),
                                                                self.ffn_head_size)))
        self.dense2 = nn.Linear(int(config.hidden_size/2), config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        input_tensor_size = input_tensor.size()
        hidden_states = torch.matmul(hidden_states, self.ffn_weights) + self.ffn_bias.unsqueeze(1)
        hidden_states = torch.reshape(hidden_states.permute(1, 0, 2), [input_tensor_size[0], input_tensor_size[1], self.ffn_head_size*int(self.ffn_head/2)])
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

