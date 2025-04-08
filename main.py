import torch
import torch.nn as nn

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import os
from typing import List
from datetime import datetime
from tqdm import tqdm

from src.utils import *
from src.vae import *

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
    best_avg_loss = float("inf")
    
    if resume:
        
        _, _, files = next(os.walk("checkpoints/"))
        if len(files) == 0:
            raise ValueError("No checkpoints found")
        else:
            last = int(files[-1].split("_")[-1].split(".")[0])
            state = torch.load(f"checkpoints/{curr_date}/vae_{last}.pth", map_location=device)
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
        
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
        
        if epoch % checkpoint_freq == 0:
            checkpoint_dir = os.path.join("checkpoints", curr_date)
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            
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
    os.mkdir(model_dir)
    
    save_params(model=model, 
                optimizer=optimizer, 
                epoch=epoch, 
                lr_scheduler=lr_scheduler, 
                losses=[avg_loss, avg_reconstructured_loss, avg_kl_div], 
                root_path=model_dir) # For now we are not saving the optimizer and lr_scheduler state_dict
    
    return best_avg_loss

def inference():
    """
    Generates fake flowers using the trained VAE model.
    """
    flower_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    flower_dataset = datasets.Flowers102(root="D:/Datasets/cv-datasets", split="train", transform=flower_transform, download=True)

    state = torch.load("checkpoints/2025-04-07_02-07-23/vae_0.pth", map_location="cpu", weights_only=True)
    
    model = VAE(in_channels=3,
                hidden_channels=32,
                block=[UpsampleBlock, DownsampleBlock],
                num_blocks=4,
                latent_dim=32)
    
    model.load_state_dict(state["model"])
    
    train_dataloader = torch.utils.data.DataLoader(flower_dataset, batch_size=1, shuffle=True)
    fig, axes = plt.subplots(2, 10)
    fig.subplots_adjust(wspace=0.1, hspace=0.01)
    
    for i in range(10):
        img, _ = next(iter(train_dataloader))
        img = img.to("cpu")
        
        with torch.no_grad():
            out, _, _ = model(img)
        
        axes[0, i].imshow(out[0].permute(1, 2, 0))
        axes[0, i].axis("off")
        
        axes[1, i].imshow(img[0].permute(1, 2, 0))
        axes[1, i].axis("off")
    
    plt.savefig('generated_imgs/generated_flowers.png', dpi=300, bbox_inches='tight')

def objective(trial):
    """
    Create the objective function for the hyperparameter optimization
    using Optuna.
    """
    
    lr = trial.suggest_float("lr", 3e-6, 3e-4, log=True)
    beta = trial.suggest_float("beta", 0.1, 1.0)
    latent_dim = trial.suggest_int("latent_dim", 32, 128)
    gamma = trial.suggest_float("gamma", 0.1, 0.5)
    
    flower_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    flower_dataset = datasets.Flowers102(root="D:/Datasets/cv-datasets", split="train", transform=flower_transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(flower_dataset, batch_size=128, shuffle=True)
    
    model = VAE(in_channels=3, 
                hidden_channels=32,
                block=[UpsampleBlock, DownsampleBlock], 
                num_blocks=4, 
                latent_dim=latent_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="sum")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[250], gamma=gamma)
    
    best_avg_loss = train(model, 
                  train_dataloader, 
                  optimizer, 
                  criterion, 
                  lr_scheduler, 
                  epochs=1000, 
                  beta=beta)
    
    return best_avg_loss

def main():
    flower_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    flower_dataset = datasets.Flowers102(root="D:/Datasets/cv-datasets", split="train", transform=flower_transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(flower_dataset, batch_size=128, shuffle=True)
    
    model = VAE(in_channels=3, 
                hidden_channels=32,
                block=[UpsampleBlock, DownsampleBlock], 
                num_blocks=4, 
                latent_dim=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss(reduction="sum")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[250], gamma=0.5)
    
    train(model, 
          train_dataloader, 
          optimizer, 
          criterion, 
          lr_scheduler, 
          epochs=500, 
          beta=1)
    
if __name__ == "__main__":
    # main()
    # main_vae()
    inference()