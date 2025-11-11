import torch
import torch.nn.functional as F
from data.scripts.data_loader import EEGProcessor 
from data.scripts.data_loader_preictal import EEGProcessorPreictal
import numpy as np

class EEGPooler:
    def __init__(self, results, target_time_points=100):
        """
        Initializes the EEGPooler with EEG clips and applies adaptive pooling.

        Args:
            results (dict): Dictionary containing EEG clips and labels from EEGProcessor.
            target_time_points (int): Desired number of time points after pooling.
        """
        self.results = results
        self.target_time_points = target_time_points

    def adaptive_pool_clip(self, clip):
        """
        Applies adaptive pooling to an EEG clip to ensure it has a fixed time dimension.

        Args:
            clip (numpy.ndarray or torch.Tensor): EEG clip of shape (T, num_nodes, feature_dim).

        Returns:
            torch.Tensor: Resized EEG clip with shape (target_time_points, num_nodes, feature_dim).
        """
        if isinstance(clip, np.ndarray):
            clip = torch.tensor(clip, dtype=torch.float32)

        if clip.ndim != 3:
            raise ValueError(f"Expected clip shape (T, num_nodes, feature_dim), got {clip.shape}")

        # Change shape from (T, num_nodes, feature_dim) to (num_nodes, feature_dim, T)
        clip = clip.permute(1, 2, 0)  
        # Apply adaptive pooling to match target_time_points
        pooled_clip = F.adaptive_avg_pool1d(clip, self.target_time_points)  
        # Back to (target_T, num_nodes, feature_dim)
        pooled_clip = pooled_clip.permute(2, 0, 1)  

        return pooled_clip

    def apply_pooling_DC(self):
        """
        Applies adaptive pooling to all EEG clips in results.

        NEW: Works with file paths (lazy mode) or traditional results dict.

        Returns:
            dict or list: Dictionary with pooled EEG clips and corresponding labels,
                         OR list of file_path dicts if in lazy mode.
        """
        # Check if results is a list of file paths (lazy mode)
        if isinstance(self.results, list) and len(self.results) > 0:
            if isinstance(self.results[0], dict) and 'h5_path' in self.results[0]:
                print("Lazy loading mode detected - returning file paths")
                return self.results  # Return paths unchanged for lazy loading

        # Traditional mode - load everything
        pooled_results = {}

        for key, (eeg_clips, labels, _,_,_) in self.results.items():
            pooled_clips = [self.adaptive_pool_clip(clip) for clip in eeg_clips]
            pooled_results[key] = (pooled_clips, labels)

        return pooled_results

    # def apply_pooling_RC(self):
    #     """
    #     Applies adaptive pooling to all EEG clips in results.

    #     Returns:
    #         dict: Dictionary with pooled EEG clips, labels, start times, and stop times.
    #     """
    #     pooled_results = {}

    #     for key, (eeg_clips, labels, start_times, stop_times) in self.results.items():
    #         pooled_clips = [self.adaptive_pool_clip(clip) for clip in eeg_clips]
    #         pooled_results[key] = (pooled_clips, labels, start_times, stop_times)

    #     return pooled_results

    
    def apply_pooling_RC(self):
        """
        Applies adaptive pooling to all EEG clips in results for RC tasks.

        NEW: Works with file paths (lazy mode) or traditional results dict.

        Returns:
            dict or list: Dictionary with pooled EEG clips, labels, start times, and stop times,
                         OR list of file_path dicts if in lazy mode.
        """
        # Check if results is a list of file paths (lazy mode)
        if isinstance(self.results, list) and len(self.results) > 0:
            if isinstance(self.results[0], dict) and 'h5_path' in self.results[0]:
                print("Lazy loading mode detected - returning file paths")
                return self.results  # Return paths unchanged for lazy loading

        # Traditional mode - load everything
        pooled_results = {}

        for key, value in self.results.items():
            try:
                # Validate the input is a 4-element tuple
                if not isinstance(value, tuple) or len(value) != 4:
                    print(f"Skipping {key}: Invalid data format, expected 4-element tuple, got {value}")
                    continue
                eeg_clips, labels, start_times, stop_times = value
                # Validate that all elements are lists and non-empty
                if not (isinstance(eeg_clips, list) and isinstance(labels, list) and
                        isinstance(start_times, list) and isinstance(stop_times, list)):
                    print(f"Skipping {key}: Invalid data types: clips={type(eeg_clips)}, "
                        f"labels={type(labels)}, start_times={type(start_times)}, "
                        f"stop_times={type(stop_times)}")
                    continue
                if not (eeg_clips and labels and start_times and stop_times):
                    print(f"Skipping {key}: Empty data: clips={len(eeg_clips)}, "
                        f"labels={len(labels)}, start_times={len(start_times)}, "
                        f"stop_times={len(stop_times)}")
                    continue
                # Ensure lengths match
                if not (len(eeg_clips) == len(labels) == len(start_times) == len(stop_times)):
                    print(f"Skipping {key}: Mismatched lengths: clips={len(eeg_clips)}, "
                        f"labels={len(labels)}, start_times={len(start_times)}, "
                        f"stop_times={len(stop_times)}")
                    continue
                # Apply pooling
                pooled_clips = [self.adaptive_pool_clip(clip) for clip in eeg_clips]
                pooled_results[key] = (pooled_clips, labels, start_times, stop_times)
            except Exception as e:
                print(f"Error pooling {key}: {e} (skipping)")

        if not pooled_results:
            print("Warning: No valid data after pooling. Check input files and preprocessing.")
        return pooled_results
