import argparse
from utils.datasets import VideoDataset
from utils.models import Ext_Arch_2
from utils.trainer import finetune
from torch.utils.data import DataLoader
import numpy as np

from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from mmaction.apis import init_recognizer

import torch
import torch.nn as nn
import pandas as pd

FRAMES = 0

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video feature extraction")

    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-m",
        "--model",
        help="Type of video model",
        choices=["mvitv2","uniformerv2","videomaev2"],
        required=True,
    )
    required.add_argument(
        "-b",
        "--batch_size",
        type=lambda x: int(x) if x.isdigit() else x,
        default=1,
        required=True,
    )
    required.add_argument(
        "-d",
        "--dataset",
        help="Type of dataset",
        choices=["basketball","aerobic_gymnastics","diving","football","volleyball"],
        required=True,
    )
    required.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs for training",
        type=int,
        default=30,
        required=True,
    )
    return parser.parse_args()


def get_dataloader(video_file, metadata_file,  batch_size=1, shuffle=True, transform=None, num_worker=0):
    dataset = VideoDataset(video_file, metadata_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
    return dataloader



def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_dir = "datasets_numpy"

    print(10*'#')
    print(f"Selected model to be finte-tuned: {args.model}")
    print(f"Dataset: {args.dataset} [Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(10*'#')

    if args.model == "mvitv2":
        FRAMES = 32
        register_all_modules()

        cfg_mvit = dict(type='MViT', arch='base', out_scales=[0, 1, 2, 3], pretrained_type='mvit-base-p244_32x3x1_kinetics400-rgb')

        backbone = MODELS.build(cfg_mvit)
        backbone.init_weights()

        norms = ['norm0.weight','norm0.bias','norm1.weight','norm1.bias','norm2.weight','norm2.bias','norm3.weight','norm3.bias']

        for k,param in backbone.named_parameters():
            if 'blocks.23' in k or k in norms:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model = Ext_Arch_2(backbone=backbone,dataset=args.dataset,model_type=args.model)
        model = model.to(device)

    elif args.model == "uniformerv2":
        FRAMES = 8
        config_file = "mmaction2/mmaction/configs/recognition/uniformerv2/uniformerv2_base_p16_res224_clip_kinetics710_pre_u8_kinetics400_rgb.py"
        checkpoint_file = 'mmaction2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_20230313-75be0806.pth'

        model = init_recognizer(config_file, checkpoint_file, device=device)

        backbone = model.backbone

        for k,param in backbone.named_parameters():
            if 'transformer.dec.3' in k or 'transformer.norm' in k:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model = Ext_Arch_2(backbone=model.backbone, dataset=args.dataset, model_type=args.model)
    
    else:
        FRAMES = 16
        config_file = "mmaction2/configs/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py"
        checkpoint_file = 'mmaction2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth'

        model = init_recognizer(config_file, checkpoint_file, device=device)

        backbone = model.backbone

        layers = ["fc_norm.weight","fc_norm.bias","blocks.11.mlp.layers.1.weight","blocks.11.mlp.layers.1.bias"]

        for k,param in backbone.named_parameters():
            if k in layers or "blocks.11.mlp.layers" in k:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model = Ext_Arch_2(backbone=backbone, dataset=args.dataset, model_type=args.model)
    

    train_video_file = f"{path_dir}/frame_{FRAMES}/frame{FRAMES}_{args.dataset}_train.npy"
    train_metadata_file = f"{path_dir}/frame_{FRAMES}/frame{FRAMES}_{args.dataset}_train_metadata.json"

    val_video_file = f"{path_dir}/frame_{FRAMES}/frame{FRAMES}_{args.dataset}_val.npy"
    val_metadata_file = f"{path_dir}/frame_{FRAMES}/frame{FRAMES}_{args.dataset}_val_metadata.json"

    train_dataloader = get_dataloader(train_video_file, train_metadata_file,shuffle=False, batch_size=args.batch_size)
    val_dataloader = get_dataloader(val_video_file, val_metadata_file, shuffle=False, batch_size=args.batch_size)


    print(f"Size of dataloader (train/val): {len(train_dataloader)} / {len(val_dataloader)}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    histry = finetune(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            EPOCHS=args.epochs,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            device=device,
                            print_metric=True)
    
    df_histry = pd.DataFrame(histry)
    df_histry.to_csv(f"results/finetuned_models/histry_finetuned_model_{args.model}_{args.dataset}.csv",index=False)
    

if __name__ == "__main__":
    main()