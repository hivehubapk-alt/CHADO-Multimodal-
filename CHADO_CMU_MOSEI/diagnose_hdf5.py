import h5py
import numpy as np

def explore_hdf5(filepath, max_depth=3):
    """Recursively explore HDF5 file structure"""
    
    print(f"\n{'='*60}")
    print(f"Exploring: {filepath}")
    print(f"{'='*60}")
    
    def print_structure(name, obj, depth=0):
        indent = "  " * depth
        
        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 GROUP: {name}")
            if depth < max_depth:
                for key in list(obj.keys())[:5]:  # Show first 5 keys
                    print_structure(key, obj[key], depth + 1)
                if len(obj.keys()) > 5:
                    print(f"{indent}  ... ({len(obj.keys())} total keys)")
        
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}📄 DATASET: {name}")
            print(f"{indent}   Shape: {obj.shape}, Dtype: {obj.dtype}")
            if obj.size < 20:  # Show small datasets
                print(f"{indent}   Data: {obj[:]}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"\nRoot keys: {list(f.keys())[:10]}")
        print(f"Total root keys: {len(f.keys())}\n")
        
        # Explore first few items
        for i, key in enumerate(list(f.keys())[:3]):
            print(f"\n--- Exploring key {i+1}: '{key}' ---")
            print_structure(key, f[key])

# Diagnose each file
files = [
    '/home/tahirahmad/CMU-MOSEI/dataset/labels/CMU_MOSEI_Labels.csd',
    '/home/tahirahmad/CMU-MOSEI/dataset/acoustics/CMU_MOSEI_COVAREP.csd',
    '/home/tahirahmad/CMU-MOSEI/dataset/visuals/CMU_MOSEI_VisualFacet42.csd',
]

for filepath in files:
    try:
        explore_hdf5(filepath)
    except Exception as e:
        print(f"Error exploring {filepath}: {e}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
