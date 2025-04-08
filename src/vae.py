import torch
import torch.nn as nn
from typing import *

class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DownsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VAE(nn.Module):
    expansion = 2
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 block: Union[UpsampleBlock, DownsampleBlock],
                 num_blocks: int=4, 
                 latent_dim: int=128) -> None:
        
        super().__init__()
        self.hidden_channels = hidden_channels
        
        self.encoder = self._make_encoder(in_channels, block[0], num_blocks)
        
        self.mu = nn.Linear(self.hidden_channels, latent_dim)
        self.sigma = nn.Linear(self.hidden_channels, latent_dim)
        
        self.fc_decoder = nn.Linear(latent_dim, self.hidden_channels)
        self.decoder = self._make_decoder(block[1], num_blocks)
    
    # Code can be written in a more elegant way
    # but for the sake of simplicity, I am writing it in a more verbose way
    
    def _make_encoder(self, in_channels: int, block, num_blocks: int):
        # hidden_channels = 3 -> 32 -> 64 -> 128 -> 256
        # image size = 128 -> 64 -> 32 -> 16 -> 8
        
        layers = [block(in_channels, self.hidden_channels)]
        
        for _ in range(num_blocks):
            layers.append(block(self.hidden_channels, self.hidden_channels * self.expansion))
            self.hidden_channels *= self.expansion
            
        layers.append(nn.Flatten())
        self.hidden_channels = self.hidden_channels * 4 * 4 #Atm have to change this manually, defenitely not the best way to do it
        
        return nn.Sequential(*layers)

    def _make_decoder(self, block, num_blocks):
        
        self.hidden_channels = self.hidden_channels // (4 * 4) #Atm have to change this manually, defenitely not the best way to do it
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.hidden_channels, self.hidden_channels // self.expansion))
            self.hidden_channels =  self.hidden_channels // self.expansion
        
        layers.append(nn.ConvTranspose2d(self.hidden_channels, 3, kernel_size=4, stride=2, padding=1))
        return nn.Sequential(*layers)
    
    def encode(self, x):
        out = self.encoder(x)
        mu, sigma = self.mu(out), self.sigma(out)
        
        return mu, sigma
    
    def decode(self, z):
       
        z = self.fc_decoder(z)
        z = z.view(z.size(0), 512, 4, 4) # need to change this manually, defenitely not the best way to do it
        
        # [batch_size, 3, 128, 128] Outputs image between [0, 1]
        out = torch.sigmoid(self.decoder(z))
        return out
        
    def forward(self, x):
        # x = [batch_size, 3, 128, 128]
    
        sigma, mu = self.encode(x)
        
        epsilon = torch.randn_like(sigma) # [batch_size, latent_dim]
        z = mu + sigma * epsilon # [batch_size, latent_dim]
        
        out = self.decode(z) # [batch_size, 3, 128, 128]
       
        return out, mu, sigma     
    
