import torch
import torch.nn as nn
from tkinter import _flatten
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.nn.functional import relu

def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

def complex_dropout(input, p=0.05, training=True):#p   dropout率
    # need to have the same dropout mask for real and imaginary part
    mask = torch.ones(*input.shape, dtype = torch.float32).to(input.device)
    mask = dropout(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    return mask*input

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size) # 第二个全连接层
        self.dropout = nn.Dropout(0.05)#dropout率
    def forward(self, x):
        x =  torch.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x

class LTFAD(nn.Module):
    def __init__(self, win_size,d_model=256, local_size=[3, 5, 7],global_size=[3,5,7], channel=55,dropout=0.05, output_attention=True):
        super(LTFAD, self).__init__()
        self.output_attention = output_attention
        self.local_size = local_size
        self.channel = channel
        self.win_size = win_size
        self.mlp_size = nn.ModuleList(
            MLP(localsize, d_model, global_size[index]) for index, localsize in enumerate(self.local_size))

        self.mlp_num = nn.ModuleList(
            MLP(global_size[index], d_model, localsize) for index, localsize in enumerate(self.local_size))
        self.scale = 0.02
        self.sparsity_threshold = 0.01

        # the MLP1's first floor: local_size -> d_model
        self.r1 = nn.ParameterList([nn.Parameter(self.scale * torch.randn(localsize, d_model)) for localsize in (self.local_size)])
        self.i1 = nn.ParameterList( [nn.Parameter(self.scale * torch.randn(localsize, d_model)) for localsize in (self.local_size)])
        self.rb1 = nn.ParameterList( [nn.Parameter(self.scale * torch.randn(d_model)) for localsize in (self.local_size)])
        self.ib1 = nn.ParameterList( [nn.Parameter(self.scale * torch.randn(d_model)) for localsize in (self.local_size)])
        # the MLP1's seconde floor: d_model -> global_size
        self.r2 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(d_model, global_size[index])) for index,localsize in enumerate(self.local_size)])
        self.i2 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(d_model, global_size[index])) for index,localsize in enumerate(self.local_size)])
        self.rb2 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(global_size[index])) for index,localsize in enumerate(self.local_size)])
        self.ib2 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(global_size[index])) for index,localsize in enumerate(self.local_size)])

        # the MLP2's first floor: global_size -> d_model
        self.r3 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(global_size[index], d_model)) for index,localsize in enumerate(self.local_size)])
        self.i3 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(global_size[index], d_model)) for index,localsize in enumerate(self.local_size)])
        self.rb3 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(d_model)) for localsize in (self.local_size)])
        self.ib3 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(d_model)) for localsize in (self.local_size)])
        # the MLP2's second floor: d_model -> local_size
        self.r4 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(d_model, localsize)) for
             localsize in (self.local_size)])
        self.i4 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(d_model,  localsize)) for
             localsize in (self.local_size)])
        self.rb4 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(localsize)) for localsize in (self.local_size)])
        self.ib4 = nn.ParameterList(
            [nn.Parameter(self.scale * torch.randn(localsize)) for localsize in (self.local_size)])
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)



    def FreMLP(self, x, r, i, rb, ib):
        """
            x: input
            r: weight of x.real
            i: weight of x.imag
            rb: bias of x.real
            ib: bias of x.imag
        """
        # relu[ XrWr - XiWi + rb ]


        o1_real = F.relu(
            torch.einsum('bijd,dw->bijw', x.real, r) - \
            torch.einsum('bijd,dw->bijw', x.imag, i) + \
            rb
        )
        # relu[ XiWr + XrWi +ib ]
        o1_imag = F.relu(
            torch.einsum('bijd,dw->bijw', x.imag, r) + \
            torch.einsum('bijd,dw->bijw', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        # 软收缩， lambd：收缩阈值
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def MLP_temporal_size(self, x, L, idx):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(x, self.r1[idx], self.i1[idx], self.rb1[idx], self.ib1[idx])
        y=complex_dropout(y)
        y = self.FreMLP(y, self.r2[idx], self.i2[idx], self.rb2[idx], self.ib2[idx])
        x = torch.fft.irfft(y, n=L, dim=2, norm="ortho")
        return x

    def MLP_temporal_num(self, x, L, idx):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(x, self.r3[idx], self.i3[idx], self.rb3[idx], self.ib3[idx])
        y = complex_dropout(y)
        y = self.FreMLP(y, self.r4[idx], self.i4[idx], self.rb4[idx], self.ib4[idx])
        x = torch.fft.irfft(y, n=L, dim=2, norm="ortho")
        return x


    def forward(self, in_size,in_num):
        local_mean = []
        global_mean = []
        local_mean_1 = []
        global_mean_1 = []
        B, L, M, _ = in_size[0].shape

        #time
        for index, localsize in enumerate(self.local_size):
            x_local, x_global = in_size[index], in_num[index]  # B L M N
            x_local = self.mlp_size[index](x_local)
            x_global = self.mlp_num[index](x_global)

            local_mean.append(x_local), global_mean.append(x_global)

        local_mean = list(_flatten(local_mean))  # 3
        global_mean = list(_flatten(global_mean))  # 3

        #freq
        for index, localsize in enumerate(self.local_size):
            # B L M N -> B M L N
            x_local, x_global = in_size[index].permute(0, 2, 1, 3), in_num[index].permute(0, 2, 1, 3)
            x_local = self.MLP_temporal_size(x_local, L, index)
            x_global = self.MLP_temporal_num(x_global, L, index)

            local_mean_1.append(x_local.permute(0,2,1,3)), global_mean_1.append(x_global.permute(0,2,1,3))

        global_mean_1 = list(_flatten(global_mean_1))  # 3
        local_mean_1 = list(_flatten(local_mean_1))  # 3
        if self.output_attention:
            return local_mean, global_mean,local_mean_1, global_mean_1
        else:
            return None


