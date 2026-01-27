#!/usr/bin/env python3
"""
This script creates all necessary Python files for CHADO project
Run this once to set up the entire codebase
"""

import os
from pathlib import Path

# Define all files and their content
files_content = {}

# ==============================================================================
# 1. src/data_processing/load_csd_data.py
# ==============================================================================
files_content['src/data_processing/load_csd_data.py'] = '''
import pickle
import numpy as np
from pathlib import Path

class CMUMOSEILoader:
    def __init__(self, data_root='/CMU-MOSEI/dataset'):
        self.data_root = Path(data_root)
        
    def load_csd_file(self, filepath):
        """Load .csd (pickled) files"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    
    def load_all_modalities(self):
        """Load all modality features"""
        
        # Acoustic features (COVAREP - 74 dims)
        acoustics = self.load_csd_file(
            self.data_root / 'acoustics/CMU_MOSEI_COVAREP.csd'
        )
        
        # Visual features
        visuals_facet = self.load_csd_file(
            self.data_root / 'visuals/CMU_MOSEI_VisualFacet42.csd'
        )
        visuals_openface = self.load_csd_file(
            self.data_root / 'visuals/CMU_MOSEI_VisualOpenFace2.csd'
        )
        
        # Text features
        text_words = self.load_csd_file(
            self.data_root / 'languages/CMU_MOSEI_TimestampedWords.csd'
        )
        text_vectors = self.load_csd_file(
            self.data_root / 'languages/CMU_MOSEI_TimestampedWordVectors.csd'
        )
        text_phones = self.load_csd_file(
            self.data_root / 'languages/CMU_MOSEI_TimestampedPhones.csd'
        )
        
        # Labels
        labels = self.load_csd_file(
            self.data_root / 'labels/CMU_MOSEI_Labels.csd'
        )
        
        return {
            'acoustics': acoustics,
            'visuals_facet': visuals_facet,
            'visuals_openface': visuals_openface,
            'text_words': text_words,
            'text_vectors': text_vectors,
            'text_phones': text_phones,
            'labels': labels
        }
    
    def extract_emotion_labels(self, labels_data):
        """
        CMU-MOSEI has 6 emotions: happy, sad, anger, surprise, disgust, fear
        """
        emotion_dict = {
            'happiness': 0,
            'sadness': 1, 
            'anger': 2,
            'surprise': 3,
            'disgust': 4,
            'fear': 5
        }
        
        processed_labels = {}
        for video_id, segments in labels_data.items():
            processed_labels[video_id] = {}
            for segment_id, annotations in segments.items():
                if 'All Labels' not in annotations:
                    continue
                    
                all_labels = annotations['All Labels']
                
                # Find dominant emotion
                emotion_scores = {}
                for label_type, values in all_labels.items():
                    if label_type in emotion_dict:
                        emotion_scores[label_type] = values
                
                if not emotion_scores:
                    continue
                
                # Get max emotion
                max_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                
                processed_labels[video_id][segment_id] = {
                    'emotion_class': emotion_dict.get(max_emotion, -1),
                    'emotion_scores': emotion_scores,
                    'sentiment': annotations.get('sentiment', [0])[0]
                }
        
        return processed_labels
'''

# ==============================================================================
# 2. src/data_processing/prepare_data.py (SIMPLIFIED VERSION)
# ==============================================================================
files_content['src/data_processing/prepare_data.py'] = '''
import os
import sys
import pickle
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from data_processing.load_csd_data import CMUMOSEILoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:5')
    args = parser.parse_args()
    
    print("="*60)
    print("CMU-MOSEI DATA PREPROCESSING (SIMPLIFIED)")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\\n[1/3] Loading .csd files...")
    loader = CMUMOSEILoader(args.data_root)
    data = loader.load_all_modalities()
    
    acoustics = data['acoustics']
    visuals = data['visuals_facet']
    text_vecs = data['text_vectors']
    labels_data = data['labels']
    
    print(f"✓ Loaded {len(labels_data)} videos")
    
    # Extract labels
    print("\\n[2/3] Processing samples...")
    processed_labels = loader.extract_emotion_labels(labels_data)
    
    processed_data = []
    skipped = 0
    
    for video_id in tqdm(list(processed_labels.keys())[:100], desc="Processing"):  # First 100 for testing
        for segment_id in processed_labels[video_id].keys():
            try:
                label_info = processed_labels[video_id][segment_id]
                emotion_class = label_info['emotion_class']
                
                if emotion_class == -1:
                    skipped += 1
                    continue
                
                # Get acoustic features
                if video_id not in acoustics or segment_id not in acoustics[video_id]:
                    skipped += 1
                    continue
                    
                acoustic_feats = acoustics[video_id][segment_id]['features']
                
                # Get visual features
                if video_id not in visuals or segment_id not in visuals[video_id]:
                    skipped += 1
                    continue
                    
                visual_feats = visuals[video_id][segment_id]['features']
                
                # Get text features
                if video_id not in text_vecs or segment_id not in text_vecs[video_id]:
                    skipped += 1
                    continue
                    
                text_feats = text_vecs[video_id][segment_id]['features']
                
                # Simple padding/truncation to 50 timesteps
                def pad_or_truncate(arr, target_len=50):
                    if len(arr) > target_len:
                        return arr[:target_len]
                    elif len(arr) < target_len:
                        pad = np.zeros((target_len - len(arr), arr.shape[1]))
                        return np.vstack([arr, pad])
                    return arr
                
                acoustic_feats = pad_or_truncate(acoustic_feats)
                visual_feats = pad_or_truncate(visual_feats)
                text_feats = pad_or_truncate(text_feats)
                
                sample = {
                    'acoustic': acoustic_feats,
                    'visual': visual_feats,
                    'text': text_feats,
                    'label': emotion_class,
                    'video_id': video_id,
                    'segment_id': segment_id
                }
                
                processed_data.append(sample)
                
            except Exception as e:
                skipped += 1
                continue
    
    print(f"\\n✓ Processed {len(processed_data)} samples")
    print(f"✗ Skipped {skipped} samples")
    
    # Create splits
    print("\\n[3/3] Creating splits and saving...")
    from sklearn.model_selection import train_test_split
    
    indices = list(range(len(processed_data)))
    labels = [s['label'] for s in processed_data]
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)
    
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    
    # Save
    with open(output_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    for split_name, indices in splits.items():
        split_data = [processed_data[i] for i in indices]
        with open(output_dir / f'{split_name}_data.pkl', 'wb') as f:
            pickle.dump(split_data, f)
        print(f"✓ Saved {split_name}: {len(split_data)} samples")
    
    print("\\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
'''

# ==============================================================================
# 3. src/data_processing/dataset.py
# ==============================================================================
files_content['src/data_processing/dataset.py'] = '''
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
'''

# Create all files
print("Creating all Python files...")
for filepath, content in files_content.items():
    full_path = Path(filepath)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w') as f:
        f.write(content.strip())
    
    print(f"✓ Created: {filepath}")

print("\n" + "="*60)
print("✓ ALL FILES CREATED SUCCESSFULLY")
print("="*60)
print("\nNext steps:")
print("1. Run: python src/data_processing/prepare_data.py ...")
print("="*60)

