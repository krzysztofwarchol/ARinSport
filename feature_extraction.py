import argparse
from utils.datasets import VideoDataset
from utils.models import Ext_Arch, Ext_Arch_2
from utils.utils import str_to_bool
from torch.utils.data import DataLoader
import numpy as np

from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from mmaction.apis import init_recognizer

import torch
from typing import Tuple,List
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

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
        "-fd",
        "--finetuned",
        help="Whether the model is finetuned",
        type=str_to_bool,
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
        "-f",
        "--frame",
        help="Number of frames for a single video",
        type=int,
        choices=[8,16,32],
        default=8,
        required=True,
    )
    return parser.parse_args()


def get_dataloader(video_file, metadata_file,  batch_size=1, shuffle=True, transform=None, num_worker=0):
    dataset = VideoDataset(video_file, metadata_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
    return dataloader



def main():
    args = parse_arguments()

    path_dir = "datasets_numpy"

    train_video_file = f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_train.npy"
    train_metadata_file = f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_train_metadata.json"

    val_video_file = f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_val.npy"
    val_metadata_file = f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_val_metadata.json"

    test_video_file = f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_test.npy"
    test_metadata_file = f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_test_metadata.json"

    train_dataloader = get_dataloader(train_video_file, train_metadata_file,shuffle=False, batch_size=args.batch_size)
    val_dataloader = get_dataloader(val_video_file, val_metadata_file, shuffle=False, batch_size=args.batch_size)
    test_dataloader = get_dataloader(test_video_file, test_metadata_file, shuffle=False, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(10*'#')
    print(f"Selected model: {args.model}")
    print(f"Dataset: {args.dataset} [Frames: {args.frame}], Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(10*'#')


    if args.model == "mvitv2":
        register_all_modules()

        cfg_mvit = dict(type='MViT', arch='base', out_scales=[0, 1, 2, 3], pretrained_type='mvit-base-p244_32x3x1_kinetics400-rgb')

        backbone = MODELS.build(cfg_mvit)
        backbone.init_weights()

        if args.finetuned:
            print(f"Fine-tuned: True")
            model = Ext_Arch_2(backbone=backbone, dataset=args.dataset, model_type=args.model)
            model.load_state_dict(torch.load(f"results/finetuned_models/finetuned_model_{args.model}_{args.dataset}.pt"))
            model = model.to(device)
            backbone = model.backbone

        model = Ext_Arch(backbone=backbone,model_type=args.model)
        model = model.to(device)

    elif args.model == "uniformerv2":
        config_file = "mmaction2/mmaction/configs/recognition/uniformerv2/uniformerv2_base_p16_res224_clip_kinetics710_pre_u8_kinetics400_rgb.py"
        checkpoint_file = 'mmaction2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_20230313-75be0806.pth'

        model = init_recognizer(config_file, checkpoint_file, device=device)

        backbone = model.backbone

        if args.finetuned:
            print(f"Fine-tuned: True")
            model = Ext_Arch_2(backbone=backbone, dataset=args.dataset, model_type=args.model)
            model.load_state_dict(torch.load(f"results/finetuned_models/finetuned_model_{args.model}_{args.dataset}.pt"))
            model = model.to(device)
            backbone = model.backbone

        model = Ext_Arch(model.backbone, model_type=args.model)
    
    else:
        config_file = "mmaction2/configs/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py"
        checkpoint_file = 'mmaction2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth'

        model = init_recognizer(config_file, checkpoint_file, device=device)

        backbone = model.backbone

        if args.finetuned:
            print(f"Fine-tuned: True")
            model = Ext_Arch_2(backbone=backbone, dataset=args.dataset, model_type=args.model)
            model.load_state_dict(torch.load(f"results/finetuned_models/finetuned_model_{args.model}_{args.dataset}.pt"))
            model = model.to(device)
            backbone = model.backbone

        model = Ext_Arch(model.backbone,model_type=args.model)


    print(f"Size of dataloader (train/val/test): {len(train_dataloader)} / {len(val_dataloader)} / {len(test_dataloader)}")

    for dl, split in zip([train_dataloader,val_dataloader,test_dataloader],["train","val","test"]):

        print(f"\nType of split: {split}")
        tensors_list = []

        model.eval()
        with torch.no_grad():
            for batch, _ in tqdm(dl):
                batch = batch.to(device)
                outputs = model(batch)
                tensors_list.append(outputs.squeeze().cpu())
            
        tensors_list = torch.stack(tensors_list).detach().numpy()
        if args.finetuned:
            np.save(f"{path_dir}/video_feature_extraction/vfe_finetuned_{args.model}_frame{args.frame}_{args.dataset}_{split}.npy", tensors_list)
        else:
            np.save(f"{path_dir}/video_feature_extraction/vfe_{args.model}_frame{args.frame}_{args.dataset}_{split}.npy", tensors_list)



if __name__ == "__main__":
    main()