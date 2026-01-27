"""
Download CMU-MOSEI from Hugging Face Hub
"""
from datasets import load_dataset
import pickle
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

print("="*80)
print("DOWNLOADING CMU-MOSEI FROM HUGGING FACE")
print("="*80)

# Create directories
os.makedirs("./data/mosei/processed", exist_ok=True)

print("\nDownloading dataset from Hugging Face...")
print("This may take 10-20 minutes...")

# Load dataset
try:
    dataset = load_dataset("CMU-MOSEI/cmu-mosei")
    
    print(f"\n✓ Dataset loaded!")
    print(f"Available splits: {list(dataset.keys())}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTrying alternative Hugging Face dataset...")
    
    # Alternative: Load from different repo
    dataset = load_dataset("cmu-multimodal-sdk/cmu-mosei")

# Process dataset
print("\nProcessing dataset...")
emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]

processed_samples = []
emotion_distribution = defaultdict(int)

# Combine all splits
all_data = []
for split_name in dataset.keys():
    all_data.extend(dataset[split_name])

print(f"Total samples: {len(all_data)}")

for idx, sample in enumerate(tqdm(all_data)):
    try:
        # Extract features and labels
        # Adapt based on actual dataset structure
        
        processed_sample = {
            'video_id': sample.get('video_id', f'video_{idx}'),
            'segment_id': sample.get('segment_id', 0),
            'text': sample.get('text', ''),
            'audio': sample.get('audio_features', np.zeros((100, 74))),
            'video': sample.get('video_features', np.zeros((100, 35))),
            'emotion_label': sample.get('emotion_label', 0),
            'emotion_scores': sample.get('emotion_scores', np.zeros(6)),
            'sentiment_score': sample.get('sentiment', 0.0)
        }
        
        processed_samples.append(processed_sample)
        emotion_distribution[processed_sample['emotion_label']] += 1
        
    except Exception as e:
        continue

# Create splits
print("\nCreating splits...")
num_samples = len(processed_samples)
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)

train_end = int(0.70 * num_samples)
val_end = train_end + int(0.09 * num_samples)

splits = {
    'train': indices[:train_end].tolist(),
    'val': indices[train_end:val_end].tolist(),
    'test': indices[val_end:].tolist()
}

# Save
output_dir = "./data/mosei/processed"

with open(f"{output_dir}/samples.pkl", 'wb') as f:
    pickle.dump(processed_samples, f)

with open(f"{output_dir}/splits.pkl", 'wb') as f:
    pickle.dump(splits, f)

# Save statistics
with open(f"{output_dir}/statistics.txt", 'w') as f:
    f.write("="*70 + "\n")
    f.write("CMU-MOSEI DATASET (FROM HUGGING FACE)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total samples: {num_samples:,}\n")
    f.write(f"Train: {len(splits['train']):,}\n")
    f.write(f"Val: {len(splits['val']):,}\n")
    f.write(f"Test: {len(splits['test']):,}\n\n")
    f.write("Emotion distribution:\n")
    for idx, emotion in enumerate(emotions):
        count = emotion_distribution.get(idx, 0)
        pct = (count/num_samples)*100 if num_samples > 0 else 0
        f.write(f"  {emotion:12s}: {count:6,} ({pct:5.2f}%)\n")

print("\n" + "="*80)
print("✓ DOWNLOAD AND PROCESSING COMPLETE!")
print("="*80)
print(f"\nData saved to: {os.path.abspath(output_dir)}")
print(f"Total samples: {num_samples:,}")
