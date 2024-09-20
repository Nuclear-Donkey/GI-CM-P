import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, time_steps, kernel_size=2, dropout=0.2, attention=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            # if attention != None and i == 0: # 选择在哪一层添加
            if attention != None:
                if attention == 'F_CBAM':
                    if num_channels[i] > 1:
                        layers += [F_CBAM(out_channels)]
                elif attention == 'T_CBAM':
                    assert time_steps > 1, 'When using time attention, the time_steps must be greater than 1'
                    layers += [T_CBAM(time_steps)]
                elif attention == 'CBAM':
                    if num_channels[i] > 1:
                        assert time_steps > 1, 'When using CBAM attention, the time_steps must be greater than 1'
                        layers += [CBAM(out_channels, time_steps, hidden_size=out_channels)]
                elif attention == 'MLA':
                    if num_channels[i] > 1:
                        assert time_steps > 1, 'When using MLA attention, the time_steps must be greater than 1'
                        layers += [MLA(out_channels, time_steps, hidden_size=out_channels)]
                elif attention == 'F_ECA':
                    if num_channels[i] > 1:
                        layers += [F_ECA(out_channels)]
                elif attention == 'T_ECA':
                    assert time_steps > 1, 'When using time attention, the time_steps must be greater than 1'
                    layers += [T_ECA(time_steps)]
                else:
                    print("Please check the name of attention!!!\n" * 3)
                    os._exit(0)   # 程序终止

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, time_steps, kernel_size, dropout, attention):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, time_steps=time_steps,
                                   kernel_size=kernel_size, dropout=dropout, attention=attention)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, input):
        # ori input shape:[B, T, F]
        # requirement: input x : [B, F, T], where B = batch size, F = features,  T = time steps
        input = input.permute(0, 2, 1)   # [B, T, F] --> [B, F, T]
        y = self.tcn(input)
        output = self.linear(y[:, :, -1])
        return output


class F_CBAM(nn.Module):
    def __init__(self, channels, reduction=2):
        super(F_CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, F, T)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


# class F_CBAM(nn.Module):
#     def __init__(self, channels, reduction=2):
#         super(F_CBAM, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv1d(channels // reduction, channels // reduction, kernel_size=1, bias=False)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.fc3 = nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         # self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         # self.max_pool = nn.AdaptiveMaxPool1d(1)
#         # self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False)
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x shape: (B, F, T)
#         # avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
#         # max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
#         avg_out = self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(self.avg_pool(x))))))
#         max_out = self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(self.max_pool(x))))))
#         attn = self.sigmoid(avg_out + max_out)
#         return x * attn


class T_CBAM(nn.Module):
    def __init__(self, time_steps):
        super(T_CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(time_steps, time_steps, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attn = self.sigmoid(avg_out + max_out)
        return (x * attn).permute(0, 2, 1)   # shape: (B, F, T)


class CBAM(nn.Module):
    def __init__(self, channels, time_steps, hidden_size):
        super(CBAM, self).__init__()
        self.channel_attention = F_CBAM(channels)
        self.time_attention = T_CBAM(time_steps)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.time_attention(out)
        return out


# class UMA-CBAM(nn.Module):
#     def __init__(self, channels, time_steps, hidden_size):
#         super(CBAM, self).__init__()
#         self.channel_attention = F_CBAM(channels)
#         self.time_attention = T_CBAM(time_steps)
#         self.f_eca = F_ECA(hidden_size)
#         # self.t_eca = T_ECA(hidden_size)
#
#     def forward(self, x):
#         out = self.channel_attention(x)
#         out = self.time_attention(out)
#         out = self.f_eca(out)
#         # out = self.t_eca(out)
#         # out = out1 + out2
#         return out

class MLA(nn.Module):
    def __init__(self, channels, time_steps, hidden_size):
        super(MLA, self).__init__()
        self.channel_attention = F_CBAM(channels)
        self.time_attention = T_CBAM(time_steps)
        self.f_eca = F_ECA(hidden_size)
        self.t_eca = T_ECA(hidden_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.time_attention(out)
        # out = out1 + out2
        out1 = self.f_eca(out)
        out2 = self.t_eca(out)
        out = out1 + out2
        return out


class F_ECA(nn.Module):
    def __init__(self, hidden_size, gamma=2, b=1):
        super(F_ECA, self).__init__()
        kernel_size = int(abs((math.log(hidden_size, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: [B, F, T]
        v = self.avg(x)  #
        v = self.conv1d(v.transpose(-1, -2)).transpose(-1, -2)
        v = self.sigmoid(v)
        return x * v


class T_ECA(nn.Module):
    def __init__(self, hidden_size, gamma=2, b=1):
        super(T_ECA, self).__init__()
        kernel_size = int(abs((math.log(hidden_size, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, F]
        v = self.avg(x)
        v = self.conv1d(v.transpose(-1, -2)).transpose(-1, -2)
        v = self.sigmoid(v)
        return (x * v).transpose(1, 2)





