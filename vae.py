import torch
import torch.nn as nn

import torchvision.transforms as transforms 
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import *

import os
from datetime import datetime

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

def save_params(model: nn.Module, 
               optimizer: torch.optim, 
               epoch: int, 
               lr_scheduler: torch.optim.lr_scheduler, 
               losses: List[float],
               root_path: str) -> None:
    
    """
    Saves the model to the specified path.
    """
    
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "lr_scheduler": lr_scheduler.state_dict(),
        "losses": losses
    }
    
    torch.save(save_dict, f"{root_path}/vae.pth")

def train(model: nn.Module, 
          dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim, 
          criterion: torch.nn, 
          lr_scheduler: torch.optim.lr_scheduler,
          epochs: int=10,
          beta: float=0.5,
          resume: bool=False,
          checkpoint_freq: int=100) -> None:
    
    curr_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", curr_date)
    writer = SummaryWriter(log_dir=log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    start_epoch = 0
    
    if resume:
        
        _, _, files = next(os.walk("checkpoints/"))
        if len(files) == 0:
            raise ValueError("No checkpoints found")
        else:
            last = int(files[-1].split("_")[-1].split(".")[0])
            state = torch.load(f"checkpoints/vae_{last}.pth", map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"]
    
    for epoch in range(start_epoch, epochs):
        
        total_loss = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch}/{epochs}]", miniters=10)
        
        model.train()
        for _, (img, _) in loop:
            img = img.to(device)
            out, mu, sigma = model(img)
            
            # Reconstruction Loss
            reconstructured_loss = criterion(out, img) 
            
            #KL Divergence
            # KL(p||q) = -E[log(q(z|x))] + E[log(p(z))]
            kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
            loss = reconstructured_loss + beta * kl_div
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
        lr_scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_reconstructured_loss = reconstructured_loss.item() / len(dataloader)
        avg_kl_div = kl_div.item() / len(dataloader)
        
        if epoch % checkpoint_freq == 0:
            checkpoint_dir = os.path.join("checkpoints", curr_date)
            save_params(model=model, 
                        optimizer=optimizer, 
                        epoch=epoch, 
                        lr_scheduler=lr_scheduler, 
                        losses=[avg_loss, avg_reconstructured_loss, avg_kl_div], 
                        root_path=checkpoint_dir)
        
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Reconstruction Loss/train", avg_reconstructured_loss, epoch)
        writer.add_scalar("KL Divergence Loss/train", avg_kl_div, epoch)
    
    writer.close()
    
    # Saves the model after training
    model_dir = os.path.join("models", curr_date)
    save_params(model=model, 
                optimizer=optimizer, 
                epoch=epoch, 
                lr_scheduler=lr_scheduler, 
                losses=[avg_loss, avg_reconstructured_loss, avg_kl_div], 
                root_path=model_dir) # For now we are not saving the optimizer and lr_scheduler state_dict

def inference(model, dataset, label, num_examples=1):
    """
    Generates (num_examples) of a particular flower.
    Specifically we extract an example of each flower,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 102:
            break
    
    encodings_imgs = []
    
    model.eval()
    with torch.no_grad():
        
        for d in range(20):
            with torch.no_grad():
                plt.imshow(images[d].permute(1, 2, 0).numpy())
                plt.show()
                
                mu, sigma = model.encode(images[d].view(1, 3, 128, 128))
            encodings_imgs.append((mu, sigma))


        mu, sigma = encodings_imgs[label]
        for example in range(num_examples):
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            out = model.decode(z)
            save_image(out, f"generated_{label}_ex{example}.png")


def main():
    flower_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    flower_dataset = datasets.Flowers102(root="D:/Datasets/cv-datasets", split="train", transform=flower_transform, download=True)
    # plt.imshow(flower_dataset[0][0].permute(1, 2, 0))
    # plt.show()
    
    # train_dataloader = torch.utils.data.DataLoader(flower_dataset, batch_size=128, shuffle=True)
    
    # model = VAE(in_channels=3, 
    #             hidden_channels=32,
    #             block=[UpsampleBlock, DownsampleBlock], 
    #             num_blocks=4, 
    #             latent_dim=32)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # criterion = nn.MSELoss(reduction="sum")
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[75, 150], gamma=0.5)
    
    # train(model, 
    #       train_dataloader, 
    #       optimizer, 
    #       criterion, 
    #       lr_scheduler, 
    #       epochs=500, 
    #       beta=0.4)
    
    state = torch.load("models/vae_6.pth", map_location="cpu", weights_only=True)
    
    model = VAE(in_channels=3, 
                hidden_channels=32,
                block=[UpsampleBlock, DownsampleBlock], 
                num_blocks=4, 
                latent_dim=64)
    
    model.load_state_dict(state["model"])
    
    inference(model, flower_dataset, label=0, num_examples=5)
    
if __name__ == "__main__":
    main()