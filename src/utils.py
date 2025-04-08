import torch
import torch.nn as nn
import os
from typing import List

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
    
    _, dir_names, files = next(os.walk(root_path))
    if len(dir_names) == 0:
        torch.save(save_dict, f"{root_path}/vae_0.pth")
    else:
        last = int(files[-1].split("_")[-1].split(".")[0])
        torch.save(save_dict, f"{root_path}/vae_{last + 1}.pth")