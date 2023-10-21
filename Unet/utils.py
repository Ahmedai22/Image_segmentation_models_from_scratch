import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_img_dir,
        train_mask_dir,
        batch_size,
        train_transform, # we would also include validation and test data as we did for train but we want to keep it simple for now
        num_workers=4,
        pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir = train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size = batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader # we shall create a val_loader for validation and shall return that as well if model's being validated

