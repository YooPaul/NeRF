import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, L1, L2, input_dim=3, layers=8, feature_dim=256, device=None, use_ray_directions=False):
        super(NeRF, self).__init__()
        
        self.use_ray_directions = use_ray_directions
        self.L1 =  L1
        self.L2 = L2
        self.layers = layers
        
        modules = []
        modules += [nn.Linear(3 + 2 * L1 * input_dim, feature_dim), nn.ReLU()]
        for i in range(layers):
            if i%(layers // 2)==0 and i>0:
                modules += [nn.Linear(3 + 2 * L1 * input_dim + feature_dim, feature_dim), nn.ReLU()]
            else:
                modules += [nn.Linear(feature_dim, feature_dim), nn.ReLU()]
            
        self.MLP = nn.ModuleList(modules)
        if use_ray_directions:
            self.linear = nn.Linear(2 * L2 * input_dim + feature_dim, 128)
       
        self.output = nn.Linear(feature_dim, 4)

    def _positional_encoding(self, x, L, device=None):
        x_enc = torch.zeros(*x.shape[:-1], 2 * L * x.shape[-1], device=device)
        
        x_features = x.shape[-1]
        for i in range(L):
            start = x_features  *  2 * i
            x_enc[..., start:start + x_features] =  torch.sin(2**i * x)
            x_enc[...,start + x_features: start + 2 * x_features] =  torch.cos(2**i * x)

        '''
        arange = torch.arange(self.L, device=device)
        x_enc[...,torch.arange(self.L)] = torch.sin(2**arange * self.freq * x[...,0, None]).float()
        x_enc[...,2 * self.L + torch.arange(self.L)] = torch.sin(2**arange * self.freq * x[...,1, None]).float()
        x_enc[...,4 * self.L + torch.arange(self.L)] = torch.sin(2**arange * self.freq * x[...,2, None]).float()

        x_enc[...,torch.arange(self.L) + 1] = torch.cos(2**arange * self.freq * x[...,0, None]).float()
        x_enc[...,2 * self.L + torch.arange(self.L) + 1] = torch.cos(2**arange * self.freq * x[...,1, None]).float()
        x_enc[...,4 * self.L + torch.arange(self.L) + 1] = torch.cos(2**arange * self.freq * x[...,2, None]).float()
        '''

        return x_enc


    def forward(self, x, dirs=None):
        if self.use_ray_directions:
            d = self._positional_encoding(dirs, self.L2, self.device)
        input = torch.concat((x, self._positional_encoding(x, self.L1, self.device)), dim=-1)
        x = input

        for i, module in enumerate(self.MLP):
            if i%(self.layres // 2 + 1)==0 and i>0:
                x = torch.concat([x, input], -1)
            x = module(x)
        
        latent_code = x
        if self.use_ray_directions:
            latent_code = F.relu(self.linear( torch.concat( (d.unsqueeze(1).expand(d.shape[0], latent_code.shape[1], d.shape[-1]), latent_code), dim=-1) ) )
        
        output = self.output(latent_code)

        density, color = output[...,:1], output[...,1:]

        return F.relu(density), torch.sigmoid(color)