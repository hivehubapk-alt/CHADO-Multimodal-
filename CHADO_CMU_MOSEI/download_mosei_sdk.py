"""
Download CMU-MOSEI using official SDK
"""
import sys
import os

# Add SDK to path
sys.path.insert(0, './CMU-MultimodalSDK')

from mmsdk import mmdatasdk

# Create data directory
data_dir = "./data/mosei/raw"
os.makedirs(data_dir, exist_ok=True)

print("="*80)
print("DOWNLOADING CMU-MOSEI DATASET")
print("="*80)
print("\nThis will download approximately 5-10 GB")
print("Estimated time: 30-60 minutes\n")

# Dataset components
dataset_components = {
    'CMU_MOSEI_TimestampedWords': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/language/CMU_MOSEI_TimestampedWords.csd',
    'CMU_MOSEI_COVAREP': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/acoustic/CMU_MOSEI_COVAREP.csd',
    'CMU_MOSEI_VisualOpenFace2': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd',
    'CMU_MOSEI_Labels': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/labels/CMU_MOSEI_Labels.csd'
}

print("Components to download:")
for name in dataset_components.keys():
    print(f"  - {name}")

print("\nStarting download...")
print("="*80)

try:
    # Download dataset
    dataset = mmdatasdk.mmdataset(dataset_components, data_dir)
    
    print("\n" + "="*80)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nDataset saved to: {os.path.abspath(data_dir)}")
    
    # Show what was downloaded
    print("\nDownloaded components:")
    for comp_name in dataset.computational_sequences.keys():
        comp = dataset.computational_sequences[comp_name]
        print(f"  - {comp_name}: {len(comp.data.keys())} videos")
    
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: Download one by one
    for name, url in dataset_components.items():
        print(f"\nDownloading {name}...")
        try:
            comp_dataset = mmdatasdk.mmdataset({name: url}, data_dir)
            print(f"✓ {name} downloaded")
        except Exception as e2:
            print(f"✗ Failed to download {name}: {e2}")

print("\n" + "="*80)
print("Download script completed")
print("="*80)
