from args import get_args
import pandas as pd
import os
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model
import torch
from augmentations import build_train_transforms, build_val_transforms

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()

    # 1. Read the dataframes
    train_df = pd.read_csv(os.path.join(args.csv_dir, 'train_df.csv'))
    val_df =  pd.read_csv(os.path.join(args.csv_dir, 'val_df.csv'))

    # 2. Prepare datasets
    train_dataset = ObjDetectionDataset(train_df, transforms=build_train_transforms(args.image_size))
    val_dataset = ObjDetectionDataset(val_df, transforms=build_val_transforms(args.image_size))

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    # images, targets = next(iter(train_loader))

    #4 Initializing the model
    model = build_model(args.backbone, num_classes=args.num_classes + 1)

    #5. Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()