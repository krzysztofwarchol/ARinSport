import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(self, video_file, metadata_file, transform=None):
        self.videos = torch.from_numpy(np.load(video_file))
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.transform = transform

        self.label_to_index = {label: idx for idx, label in enumerate(set(self.metadata['label']))}
    
    def __len__(self):
        return len(self.metadata['video_name'])
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.metadata['label'][idx]
        label_idx = self.label_to_index[label]

        if self.transform:
            video = self.transform(video)
        
        return video / 255., torch.tensor(label_idx, dtype=torch.long)
    

    


