import h5py
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CMUMOSEILoader:
    def __init__(self, data_root='/home/tahirahmad/CMU-MOSEI/dataset'):
        self.data_root = Path(data_root)
        
    def load_csd_file(self, filepath, modality_key):
        """
        Load .csd files (HDF5 format)
        modality_key: 'All Labels', 'COVAREP', 'FACET 4.2', etc.
        """
        print(f"  Loading: {filepath.name}...")
        
        data_dict = {}
        
        with h5py.File(filepath, 'r') as f:
            # Navigate to data group
            if modality_key not in f:
                print(f"    Warning: Key '{modality_key}' not found. Available: {list(f.keys())}")
                return data_dict
            
            modality_group = f[modality_key]
            
            if 'data' not in modality_group:
                print(f"    Warning: 'data' not found in {modality_key}")
                return data_dict
            
            data_group = modality_group['data']
            
            # Iterate through video IDs
            for video_id in data_group.keys():
                video_group = data_group[video_id]
                
                segment_data = {}
                
                if 'features' in video_group:
                    segment_data['features'] = video_group['features'][:]
                
                if 'intervals' in video_group:
                    segment_data['intervals'] = video_group['intervals'][:]
                
                data_dict[video_id] = segment_data
        
        print(f"    ✓ Loaded {len(data_dict)} videos")
        return data_dict
    
    def load_all_modalities(self):
        """Load all modality features"""
        
        print("\nLoading CMU-MOSEI modalities:")
        print("-" * 60)
        
        labels = self.load_csd_file(
            self.data_root / 'labels' / 'CMU_MOSEI_Labels.csd',
            'All Labels'
        )
        
        acoustics = self.load_csd_file(
            self.data_root / 'acoustics' / 'CMU_MOSEI_COVAREP.csd',
            'COVAREP'
        )
        
        visuals_facet = self.load_csd_file(
            self.data_root / 'visuals' / 'CMU_MOSEI_VisualFacet42.csd',
            'FACET 4.2'
        )
        
        # OpenFace might have different key
        try:
            visuals_openface = self.load_csd_file(
                self.data_root / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd',
                'OpenFace2'
            )
        except:
            print("  Note: OpenFace2 not loaded (optional)")
            visuals_openface = {}
        
        # Text features (optional for now)
        try:
            text_words = self.load_csd_file(
                self.data_root / 'languages' / 'CMU_MOSEI_TimestampedWords.csd',
                'words'
            )
        except:
            print("  Note: Text words not loaded (will use GloVe)")
            text_words = {}
        
        try:
            text_vectors = self.load_csd_file(
                self.data_root / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd',
                'glove_vectors'
            )
        except:
            print("  Note: Text vectors not loaded")
            text_vectors = {}
        
        print("-" * 60)
        
        return {
            'labels': labels,
            'acoustics': acoustics,
            'visuals_facet': visuals_facet,
            'visuals_openface': visuals_openface,
            'text_words': text_words,
            'text_vectors': text_vectors
        }
    
    def extract_emotion_labels(self, labels_data):
        """
        Extract emotion labels from CMU-MOSEI
        Labels structure: [sentiment, happy, sad, anger, surprise, disgust, fear]
        7 values total
        """
        
        processed_labels = {}
        total_segments = 0
        valid_segments = 0
        
        for video_id, data in labels_data.items():
            if 'features' not in data:
                continue
            
            features = data['features']  # Shape: (num_segments, 7)
            intervals = data.get('intervals', None)
            
            # Each row is a segment
            for seg_idx in range(len(features)):
                total_segments += 1
                
                # Extract 7 values: [sentiment, happy, sad, anger, surprise, disgust, fear]
                segment_features = features[seg_idx]
                
                if len(segment_features) < 7:
                    continue
                
                sentiment = float(segment_features[0])
                emotions = {
                    'happiness': float(segment_features[1]),
                    'sadness': float(segment_features[2]),
                    'anger': float(segment_features[3]),
                    'surprise': float(segment_features[4]),
                    'disgust': float(segment_features[5]),
                    'fear': float(segment_features[6])
                }
                
                # Get dominant emotion (highest score)
                max_emotion = max(emotions.items(), key=lambda x: x[1])
                
                # Map to class index
                emotion_dict = {
                    'happiness': 0,
                    'sadness': 1,
                    'anger': 2,
                    'surprise': 3,
                    'disgust': 4,
                    'fear': 5
                }
                
                # Create segment ID
                segment_id = f"seg_{seg_idx}"
                
                if video_id not in processed_labels:
                    processed_labels[video_id] = {}
                
                processed_labels[video_id][segment_id] = {
                    'emotion_class': emotion_dict[max_emotion[0]],
                    'emotion_scores': emotions,
                    'sentiment': sentiment,
                    'interval': intervals[seg_idx] if intervals is not None else None
                }
                
                valid_segments += 1
        
        print(f"\nLabel extraction: {valid_segments}/{total_segments} valid segments")
        print(f"Videos with labels: {len(processed_labels)}")
        
        return processed_labels
