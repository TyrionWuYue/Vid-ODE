import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class Conv2dTime(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvODEFunc(nn.Module):
    def __init__(self, nf, time_dependent=False):
        super(ConvODEFunc, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0

        if time_dependent:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = Conv2dTime(nf, nf, 3, 1, 1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = Conv2dTime(nf, nf, 3, 1, 1)
        else:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        
    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            out = self.norm1(x)
            out = F.relu(self.conv1(t, out))
            out = self.norm2(out)
            out = F.relu(self.conv2(t, out))
        else:
            out = self.norm1(x)
            out = F.relu(self.conv1(out))
            out = self.norm2(out)
            out = F.relu(self.conv2(out))
        return out


class ODEBlock(nn.Module):
    def __init__(self, in_channel, hidden_dim, horizon, atol=1e-5, rtol=1e-4, solver='dopri5'):
        super(ODEBlock, self).__init__()
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.atol = atol
        self.rtol = rtol
        self.horizon = horizon
        self.solver = solver

        self.t = torch.linspace(0, 1, self.horizon).float()
        self.func = ConvODEFunc(self.in_channel)
    
    def forward(self, z0):
        t = self.t.to(z0.device)
        zt = odeint(self.func, z0, t, atol=self.atol, rtol=self.rtol, method=self.solver).transpose(0, 1)
        return zt
