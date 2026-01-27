#!/usr/bin/env python3
"""
MASTER SCRIPT: Download, Extract, and Verify CMU-MOSEI Dataset
"""
import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)

def install_dependencies():
    """Install all required packages"""
    print_header("STEP 1: INSTALLING DEPENDENCIES")
    
    print("\nInstalling required packages...")
    
    # Install in order with correct package names
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('torchaudio', 'torchaudio'),
        ('transformers', 'transformers'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'scikit-learn'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('h5py', 'h5py'),
        ('tensorboard', 'tensorboard'),
        ('CMU-MultimodalDataSDK', 'CMU-MultimodalDataSDK'),  # FIXED!
        ('tabulate', 'tabulate')
    ]
    
    failed = []
    for display_name, pip_name in packages:
        try:
            print(f"Installing {display_name}...", end=' ')
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', pip_name], 
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓")
        except subprocess.CalledProcessError as e:
            print(f"✗")
            failed.append((display_name, str(e.stderr)))
    
    if failed:
        print("\n⚠ Some packages failed to install:")
        for name, error in failed:
            print(f"  - {name}: {error[:100]}")
        
        # Try to verify if mmsdk can be imported anyway
        try:
            from mmsdk import mmdatasdk
            print("\n✓ But mmsdk is available, continuing...")
        except ImportError:
            print("\n✗ mmsdk not available. Please install manually:")
            print("  pip install CMU-MultimodalDataSDK")
            return False
    
    print("\n✓ Dependency installation complete!")
    return True

def download_dataset():
    """Download and process CMU-MOSEI dataset"""
    print_header("STEP 2: DOWNLOADING CMU-MOSEI DATASET")
    
    try:
        from mmsdk import mmdatasdk
    except ImportError:
        print("✗ Error: CMU-MultimodalDataSDK not installed!")
        print("Please run: pip install CMU-MultimodalDataSDK")
        return False
    
    import pickle
    import numpy as np
    from tqdm import tqdm
    from collections import defaultdict
    
    # Set data path
    data_path = "./data/mosei"
    os.makedirs(f"{data_path}/raw", exist_ok=True)
    os.makedirs(f"{data_path}/processed", exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(f"{data_path}/processed/samples.pkl"):
        print("\n⚠ Dataset already exists!")
        response = input("Do you want to re-download? (yes/no): ")
        if response.lower() != 'yes':
            print("Skipping download...")
            return True
    
    print("\n📥 Downloading CMU-MOSEI dataset...")
    print("⏰ Estimated time: 30-60 minutes depending on internet speed")
    print("💾 Download size: ~5-10 GB")
    
    # Dataset URLs
    features = {
        'text': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/language/CMU_MOSEI_TimestampedWords.csd',
        'audio': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/acoustic/CMU_MOSEI_COVAREP.csd',
        'video': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd',
        'labels': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/labels/CMU_MOSEI_Labels.csd'
    }
    
    # Download each feature
    datasets = {}
    for feature_name, url in features.items():
        print(f"\n{'='*60}")
        print(f"Downloading {feature_name}...")
        print(f"{'='*60}")
        try:
            datasets[feature_name] = mmdatasdk.mmdataset(url)
            print(f"✓ Successfully downloaded {feature_name}")
        except Exception as e:
            print(f"✗ Error downloading {feature_name}: {e}")
            return False
    
    print("\n✓ All features downloaded!")
    
    # Process and align data
    print("\n" + "="*60)
    print("Processing and aligning data...")
    print("="*60)
    
    emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
    processed_samples = []
    emotion_distribution = defaultdict(int)
    
    # Get label dataset
    label_dataset = datasets['labels']
    label_key = 'CMU_MOSEI_Labels'
    video_ids = list(label_dataset.computational_sequences[label_key].keys())
    
    print(f"\nTotal videos to process: {len(video_ids)}")
    
    missing_count = {'text': 0, 'audio': 0, 'video': 0}
    
    for vid_id in tqdm(video_ids, desc="Processing videos"):
        try:
            # Get labels
            labels = label_dataset.computational_sequences[label_key][vid_id]['features']
            
            # Get features
            text_key = 'CMU_MOSEI_TimestampedWords'
            audio_key = 'CMU_MOSEI_COVAREP'
            video_key = 'CMU_MOSEI_VisualOpenFace2'
            
            if vid_id not in datasets['text'].computational_sequences[text_key]:
                missing_count['text'] += 1
                continue
            if vid_id not in datasets['audio'].computational_sequences[audio_key]:
                missing_count['audio'] += 1
                continue
            if vid_id not in datasets['video'].computational_sequences[video_key]:
                missing_count['video'] += 1
                continue
            
            text_features = datasets['text'].computational_sequences[text_key][vid_id]['features']
            audio_features = datasets['audio'].computational_sequences[audio_key][vid_id]['features']
            video_features = datasets['video'].computational_sequences[video_key][vid_id]['features']
            
            # Process each segment
            num_segments = labels.shape[0]
            
            for seg_idx in range(num_segments):
                # Get emotion label (first 6 columns)
                emotion_scores = labels[seg_idx, :6]
                emotion_label = int(np.argmax(emotion_scores))
                
                # Get sentiment score
                sentiment_score = labels[seg_idx, 6] if labels.shape[1] > 6 else 0.0
                
                sample = {
                    'video_id': vid_id,
                    'segment_id': seg_idx,
                    'text': text_features,
                    'audio': audio_features,
                    'video': video_features,
                    'emotion_label': emotion_label,
                    'emotion_scores': emotion_scores,
                    'sentiment_score': sentiment_score
                }
                
                processed_samples.append(sample)
                emotion_distribution[emotion_label] += 1
                
        except Exception as e:
            continue
    
    print(f"\n✓ Successfully processed {len(processed_samples)} segments")
    print(f"\nMissing modalities (videos skipped):")
    for modality, count in missing_count.items():
        print(f"  - {modality}: {count} videos")
    
    # Create splits
    print("\n" + "="*60)
    print("Creating train/val/test splits...")
    print("="*60)
    
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
    
    print(f"  Train: {len(splits['train']):,} samples (70.0%)")
    print(f"  Val:   {len(splits['val']):,} samples (9.0%)")
    print(f"  Test:  {len(splits['test']):,} samples (21.0%)")
    
    # Save processed data
    print("\n" + "="*60)
    print("Saving processed data...")
    print("="*60)
    
    with open(f"{data_path}/processed/samples.pkl", 'wb') as f:
        pickle.dump(processed_samples, f)
    print(f"✓ Saved samples ({len(processed_samples):,} samples)")
    
    with open(f"{data_path}/processed/splits.pkl", 'wb') as f:
        pickle.dump(splits, f)
    print(f"✓ Saved splits")
    
    # Save statistics
    stats = {
        'num_samples': num_samples,
        'num_train': len(splits['train']),
        'num_val': len(splits['val']),
        'num_test': len(splits['test']),
        'emotion_distribution': dict(emotion_distribution),
        'emotions': emotions
    }
    
    with open(f"{data_path}/processed/statistics.pkl", 'wb') as f:
        pickle.dump(stats, f)
    print(f"✓ Saved statistics")
    
    # Save readable statistics
    with open(f"{data_path}/processed/statistics.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("CMU-MOSEI DATASET STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total samples: {num_samples:,}\n")
        f.write(f"Train samples: {len(splits['train']):,} (70.0%)\n")
        f.write(f"Val samples:   {len(splits['val']):,} (9.0%)\n")
        f.write(f"Test samples:  {len(splits['test']):,} (21.0%)\n\n")
        f.write("Emotion distribution:\n")
        f.write("-"*70 + "\n")
        for emotion_idx, count in sorted(emotion_distribution.items()):
            emotion_name = emotions[emotion_idx]
            percentage = (count / num_samples) * 100
            f.write(f"  {emotion_name:12s}: {count:6,} ({percentage:5.2f}%)\n")
    
    print(f"✓ Saved readable statistics")
    
    print("\n✓✓✓ Dataset download and processing complete! ✓✓✓")
    return True

def verify_dataset():
    """Verify dataset integrity"""
    print_header("STEP 3: VERIFYING DATASET")
    
    import pickle
    import numpy as np
    from collections import Counter
    
    data_path = "./data/mosei"
    
    try:
        # Load data
        print("\n📂 Loading processed data...")
        with open(f"{data_path}/processed/samples.pkl", 'rb') as f:
            samples = pickle.load(f)
        with open(f"{data_path}/processed/splits.pkl", 'rb') as f:
            splits = pickle.load(f)
        
        print(f"✓ Loaded {len(samples):,} samples")
        print(f"✓ Loaded splits: Train={len(splits['train']):,}, Val={len(splits['val']):,}, Test={len(splits['test']):,}")
        
        # Verify structure
        print("\n🔍 Checking data structure...")
        
        required_fields = ['video_id', 'text', 'audio', 'video', 'emotion_label']
        sample = samples[0]
        
        all_ok = True
        for field in required_fields:
            if field in sample:
                print(f"  ✓ Field '{field}' present")
            else:
                print(f"  ✗ Field '{field}' missing")
                all_ok = False
        
        if not all_ok:
            return False
        
        # Check splits
        print("\n🔍 Checking split integrity...")
        train_set = set(splits['train'])
        val_set = set(splits['val'])
        test_set = set(splits['test'])
        
        if not (train_set & val_set or train_set & test_set or val_set & test_set):
            print("  ✓ No overlap between splits")
        else:
            print("  ✗ Found overlap between splits")
            return False
        
        # Emotion distribution
        print("\n📊 Emotion distribution:")
        emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
        emotion_counts = Counter([s['emotion_label'] for s in samples])
        
        print("-"*60)
        for emotion_idx in range(len(emotions)):
            count = emotion_counts.get(emotion_idx, 0)
            percentage = (count / len(samples)) * 100
            print(f"  {emotions[emotion_idx].capitalize():12s}: {count:6,} ({percentage:5.2f}%)")
        print("-"*60)
        
        print("\n✓ Dataset verification complete!")
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    print_header("CMU-MOSEI DATASET EXTRACTION PIPELINE")
    print("This script will:")
    print("  1. Install required dependencies")
    print("  2. Download CMU-MOSEI dataset (~5-10GB)")
    print("  3. Process and align modalities")
    print("  4. Verify data integrity")
    print("\n⚠  This process may take 30-60 minutes on first run")
    print("⚠  Make sure you have stable internet connection")
    
    response = input("\nDo you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Exiting...")
        return
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    for subdir in ['data/mosei/raw', 'data/mosei/processed', 'models', 'utils', 'configs', 'logs', 'checkpoints', 'results']:
        os.makedirs(subdir, exist_ok=True)
    print("✓ Directories created")
    
    # Run pipeline
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Downloading dataset", download_dataset),
        ("Verifying dataset", verify_dataset),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*80}")
        print(f"Starting: {step_name}")
        print(f"{'='*80}")
        
        try:
            success = step_func()
            if not success:
                print(f"\n✗ Failed at step: {step_name}")
                print("Please fix the error and run again.")
                return
        except KeyboardInterrupt:
            print(f"\n\n⚠ Process interrupted by user!")
            return
        except Exception as e:
            print(f"\n✗ Unexpected error at step '{step_name}': {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Final summary
    print_header("✓✓✓ DATASET EXTRACTION COMPLETE ✓✓✓")
    print(f"\n📁 Dataset location:")
    print(f"   {os.path.abspath('./data/mosei/processed/')}")
    print(f"\n📊 Files created:")
    print(f"   - samples.pkl (all processed samples)")
    print(f"   - splits.pkl (train/val/test indices)")
    print(f"   - statistics.pkl (dataset statistics)")
    print(f"   - statistics.txt (human-readable stats)")
    print(f"\n📖 View statistics:")
    print(f"   cat ./data/mosei/processed/statistics.txt")
    print("\n🎉 Dataset ready for model training!")
    print("="*80)

if __name__ == "__main__":
    main()
