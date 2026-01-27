"""
Download pre-processed CMU-MOSEI from Google Drive
"""
import gdown
import os
import zipfile

print("="*80)
print("DOWNLOADING CMU-MOSEI FROM GOOGLE DRIVE")
print("="*80)

os.makedirs("./data/mosei", exist_ok=True)

# Known Google Drive links for CMU-MOSEI
# (These are community-shared preprocessed versions)

urls = [
    # Preprocessed CMU-MOSEI features
    "https://drive.google.com/uc?id=1CixSaw3dpHESNG0BgG_d1DAJd_JKdaAQ",  # Text
    "https://drive.google.com/uc?id=1CylYQ3i7UKx_pYJzr9G7IEzd3KqhPMFY",  # Audio  
    "https://drive.google.com/uc?id=1CzS8od1xaVhzLVAf60_vp3R7guYHHMzj",  # Video
    "https://drive.google.com/uc?id=1D0bHKHYh03Yh5RFc9hVmOxGeLh3h5PuY",  # Labels
]

file_names = ['text.pkl', 'audio.pkl', 'video.pkl', 'labels.pkl']

print("\nDownloading files...")
for url, fname in zip(urls, file_names):
    output = f"./data/mosei/{fname}"
    print(f"\nDownloading {fname}...")
    try:
        gdown.download(url, output, quiet=False)
        print(f"✓ Downloaded {fname}")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n" + "="*80)
print("Download complete!")
print("="*80)
