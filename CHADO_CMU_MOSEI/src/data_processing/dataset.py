import torch
from torch.utils.data import Dataset
import pickle

class CMUMOSEIDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return {
            'audio': torch.FloatTensor(sample['acoustic']),
            'visual': torch.FloatTensor(sample['visual']),
            'text': torch.FloatTensor(sample['text']),
            'label': torch.LongTensor([sample['label']])[0]
        }