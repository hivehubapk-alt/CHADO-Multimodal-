#!/usr/bin/env python3
"""
ROBUST CMU-MOSEI Dataset Extractor with Multiple Download Methods
"""
import os
import sys
import subprocess
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)

def download_with_mmsdk():
    """Try downloading with mmsdk"""
    print("\n📥 METHOD 1: Trying mmsdk download...")
    
    try:
        from mmsdk import mmdatasdk
        
        data_path = "./data/mosei"
        os.makedirs(f"{data_path}/raw", exist_ok=True)
        
        # Try downloading to a specific directory
        dataset_path = f"{data_path}/raw"
        
        features = {
            'CMU_MOSEI_TimestampedWords': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/language/CMU_MOSEI_TimestampedWords.csd',
            'CMU_MOSEI_COVAREP': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/acoustic/CMU_MOSEI_COVAREP.csd',
            'CMU_MOSEI_VisualOpenFace2': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd',
            'CMU_MOSEI_Labels': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/labels/CMU_MOSEI_Labels.csd'
        }
        
        datasets = {}
        for feature_name, url in features.items():
            print(f"\nDownloading {feature_name}...")
            try:
                # Download to specific location
                dataset = mmdatasdk.mmdataset({feature_name: url})
                datasets[feature_name] = dataset
                print(f"✓ Downloaded {feature_name}")
            except Exception as e:
                print(f"✗ Failed: {e}")
                return None
        
        return datasets
        
    except Exception as e:
        print(f"✗ mmsdk method failed: {e}")
        return None

def download_alternative_method():
    """Alternative download using direct SDK calls"""
    print("\n📥 METHOD 2: Trying alternative download method...")
    
    try:
        from mmsdk import mmdatasdk
        import hashlib
        
        data_path = "./data/mosei"
        
        # Create SDK instance with custom download path
        print("Setting up custom download directory...")
        
        # Define computational sequences
        visual = "http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd"
        acoustic = "http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/acoustic/CMU_MOSEI_COVAREP.csd"  
        text = "http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/language/CMU_MOSEI_TimestampedWords.csd"
        labels = "http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/labels/CMU_MOSEI_Labels.csd"
        
        recipe = {
            'CMU_MOSEI_VisualOpenFace2': visual,
            'CMU_MOSEI_COVAREP': acoustic,
            'CMU_MOSEI_TimestampedWords': text,
            'CMU_MOSEI_Labels': labels
        }
        
        print("Downloading dataset components...")
        dataset = mmdatasdk.mmdataset(recipe, f"{data_path}/raw/")
        
        print("✓ Download successful!")
        return dataset
        
    except Exception as e:
        print(f"✗ Alternative method failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def download_manual_method():
    """Manual download with urllib"""
    print("\n📥 METHOD 3: Trying manual download...")
    print("⚠ This method downloads raw files and may take longer")
    
    response = input("Continue with manual download? (yes/no): ")
    if response.lower() != 'yes':
        return None
    
    try:
        import urllib.request
        import gzip
        import shutil
        
        data_path = "./data/mosei/raw"
        os.makedirs(data_path, exist_ok=True)
        
        files_to_download = {
            'visual': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd',
            'acoustic': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/acoustic/CMU_MOSEI_COVAREP.csd',
            'text': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/language/CMU_MOSEI_TimestampedWords.csd',
            'labels': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/labels/CMU_MOSEI_Labels.csd'
        }
        
        for name, url in files_to_download.items():
            print(f"\nDownloading {name}...")
            output_file = f"{data_path}/{name}.csd"
            
            try:
                urllib.request.urlretrieve(url, output_file)
                print(f"✓ Downloaded {name}")
            except Exception as e:
                print(f"✗ Failed to download {name}: {e}")
                return None
        
        # Load with mmsdk
        from mmsdk import mmdatasdk
        dataset = mmdatasdk.mmdataset(data_path)
        return dataset
        
    except Exception as e:
        print(f"✗ Manual download failed: {e}")
        return None

def use_cached_or_pretrained():
    """Check if dataset already exists or use pre-processed version"""
    print("\n📥 METHOD 4: Checking for existing dataset...")
    
    data_path = "./data/mosei/processed"
    
    if os.path.exists(f"{data_path}/samples.pkl"):
        print("✓ Found existing processed dataset!")
        return "existing"
    
    print("✗ No existing dataset found")
    return None

def suggest_alternative_sources():
    """Provide alternative download instructions"""
    print("\n" + "="*80)
    print("ALTERNATIVE DOWNLOAD OPTIONS")
    print("="*80)
    
    print("""
The CMU servers may be down or inaccessible. Here are alternatives:

OPTION 1: Download from Google Drive (Recommended)
--------------------------------------------------
1. Download pre-processed CMU-MOSEI from:
   https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk
   
2. Extract to: ./data/mosei/raw/

3. Run this script again


OPTION 2: Use Kaggle Dataset
-----------------------------
1. Visit: https://www.kaggle.com/datasets/tracyyang/cmu-mosei
2. Download dataset
3. Extract to: ./data/mosei/raw/
4. Run this script again


OPTION 3: Download from Hugging Face
-------------------------------------
pip install datasets
python -c "from datasets import load_dataset; load_dataset('cmu-mosei')"


OPTION 4: Clone from GitHub Mirror
-----------------------------------
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK
python mmsdk/download.py


Would you like to:
[1] Retry download
[2] Skip download and proceed with dummy data (for testing only)
[3] Exit and download manually
""")

def create_dummy_data_for_testing():
    """Create minimal dummy data for testing the pipeline"""
    print("\n⚠ Creating DUMMY data for testing purposes only!")
    print("This is NOT real CMU-MOSEI data!")
    
    response = input("Create dummy data? (yes/no): ")
    if response.lower() != 'yes':
        return False
    
    data_path = "./data/mosei/processed"
    os.makedirs(data_path, exist_ok=True)
    
    # Create dummy samples
    num_samples = 1000
    emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
    
    samples = []
    for i in range(num_samples):
        sample = {
            'video_id': f'dummy_video_{i}',
            'segment_id': 0,
            'text': np.random.randn(10, 300),  # Dummy text features
            'audio': np.random.randn(100, 74),  # Dummy audio features (COVAREP)
            'video': np.random.randn(100, 35),  # Dummy video features (OpenFace)
            'emotion_label': np.random.randint(0, 6),
            'emotion_scores': np.random.rand(6),
            'sentiment_score': np.random.rand()
        }
        samples.append(sample)
    
    # Create splits
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(0.7 * num_samples)
    val_end = train_end + int(0.09 * num_samples)
    
    splits = {
        'train': indices[:train_end].tolist(),
        'val': indices[train_end:val_end].tolist(),
        'test': indices[val_end:].tolist()
    }
    
    # Save
    with open(f"{data_path}/samples.pkl", 'wb') as f:
        pickle.dump(samples, f)
    
    with open(f"{data_path}/splits.pkl", 'wb') as f:
        pickle.dump(splits, f)
    
    # Save stats
    from collections import Counter
    emotion_dist = Counter([s['emotion_label'] for s in samples])
    
    with open(f"{data_path}/statistics.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("DUMMY CMU-MOSEI DATASET (FOR TESTING ONLY)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples: {num_samples}\n")
        f.write(f"Train: {len(splits['train'])}\n")
        f.write(f"Val: {len(splits['val'])}\n")
        f.write(f"Test: {len(splits['test'])}\n\n")
        f.write("Emotion distribution:\n")
        for idx, emotion in enumerate(emotions):
            count = emotion_dist.get(idx, 0)
            f.write(f"  {emotion}: {count}\n")
    
    print("✓ Dummy data created!")
    return True

def main():
    print_header("CMU-MOSEI DATASET DOWNLOADER (ROBUST VERSION)")
    
    # Try Method 4 first: Check existing
    result = use_cached_or_pretrained()
    if result == "existing":
        print("\n✓ Using existing dataset!")
        print("Location: ./data/mosei/processed/")
        return
    
    # Try Method 1: mmsdk
    dataset = download_with_mmsdk()
    if dataset is not None:
        print("\n✓ Download successful with mmsdk!")
        # Process dataset here
        return
    
    # Try Method 2: Alternative
    dataset = download_alternative_method()
    if dataset is not None:
        print("\n✓ Download successful with alternative method!")
        # Process dataset here
        return
    
    # All methods failed
    print("\n" + "="*80)
    print("❌ ALL DOWNLOAD METHODS FAILED")
    print("="*80)
    
    suggest_alternative_sources()
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nRetrying download...")
        main()
    elif choice == '2':
        if create_dummy_data_for_testing():
            print("\n✓ Dummy data created successfully!")
            print("You can now test the model training pipeline")
            print("⚠ Remember: This is NOT real data!")
    else:
        print("\nExiting. Please download dataset manually and place in ./data/mosei/raw/")
        print("\nRecommended: Download from Google Drive link above")

if __name__ == "__main__":
    main()
