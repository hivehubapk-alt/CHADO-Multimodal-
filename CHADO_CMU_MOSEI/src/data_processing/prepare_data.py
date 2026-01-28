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

def align_features(acoustic_data, visual_data, target_len=50):
    """
    Align acoustic and visual features to same length
    Use linear interpolation for resampling
    """
    from scipy.interpolate import interp1d
    
    acoustic_feats = acoustic_data['features']
    visual_feats = visual_data['features']
    
    # Resample to target length
    def resample(feats, target):
        if len(feats) == 0:
            return np.zeros((target, feats.shape[1] if len(feats.shape) > 1 else 1))
        if len(feats) == target:
            return feats
        
        old_indices = np.linspace(0, len(feats)-1, len(feats))
        new_indices = np.linspace(0, len(feats)-1, target)
        
        if len(feats.shape) == 1:
            feats = feats.reshape(-1, 1)
        
        interpolator = interp1d(old_indices, feats, axis=0, kind='linear', fill_value='extrapolate')
        return interpolator(new_indices)
    
    acoustic_resampled = resample(acoustic_feats, target_len)
    visual_resampled = resample(visual_feats, target_len)
    
    return acoustic_resampled, visual_resampled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:5')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max samples to process (for testing)')
    args = parser.parse_args()
    
    print("="*60)
    print("CMU-MOSEI DATA PREPROCESSING")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/3] Loading .csd files...")
    loader = CMUMOSEILoader(args.data_root)
    data = loader.load_all_modalities()
    
    labels_data = data['labels']
    acoustics = data['acoustics']
    visuals = data['visuals_facet']
    
    # Extract labels
    print("\n[2/3] Processing samples...")
    processed_labels = loader.extract_emotion_labels(labels_data)
    
    if len(processed_labels) == 0:
        print("ERROR: No labels found!")
        return
    
    processed_data = []
    skipped = 0
    
    # Process samples
    sample_count = 0
    for video_id in tqdm(processed_labels.keys(), desc="Processing videos"):
        if sample_count >= args.max_samples:
            break
        
        for segment_id in processed_labels[video_id].keys():
            if sample_count >= args.max_samples:
                break
            
            try:
                label_info = processed_labels[video_id][segment_id]
                emotion_class = label_info['emotion_class']
                
                # Get acoustic features
                if video_id not in acoustics:
                    skipped += 1
                    continue
                
                acoustic_data = acoustics[video_id]
                
                # Get visual features
                if video_id not in visuals:
                    skipped += 1
                    continue
                
                visual_data = visuals[video_id]
                
                # Align features
                acoustic_feats, visual_feats = align_features(
                    acoustic_data, visual_data, target_len=50
                )
                
                # Text features (placeholder - use zeros for now)
                # You can add ALBERT encoding here later
                text_feats = np.zeros((50, 300))  # 300-dim GloVe placeholder
                
                sample = {
                    'acoustic': acoustic_feats,
                    'visual': visual_feats,
                    'text': text_feats,
                    'label': emotion_class,
                    'video_id': video_id,
                    'segment_id': segment_id
                }
                
                processed_data.append(sample)
                sample_count += 1
                
            except Exception as e:
                skipped += 1
                continue
    
    print(f"\n✓ Processed {len(processed_data)} samples")
    print(f"✗ Skipped {skipped} samples")
    
    if len(processed_data) == 0:
        print("ERROR: No samples processed!")
        return
    
    # Create splits
    print("\n[3/3] Creating splits...")
    from sklearn.model_selection import train_test_split
    
    indices = list(range(len(processed_data)))
    labels_list = [s['label'] for s in processed_data]
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels_list)
    temp_labels = [labels_list[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=temp_labels)
    
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
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total samples: {len(processed_data)}")
    print(f"Feature shapes:")
    print(f"  Acoustic: {processed_data[0]['acoustic'].shape}")
    print(f"  Visual: {processed_data[0]['visual'].shape}")
    print(f"  Text: {processed_data[0]['text'].shape}")
    print("="*60)

if __name__ == '__main__':
    main()
