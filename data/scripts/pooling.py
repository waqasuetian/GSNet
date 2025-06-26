import torch
import torch.nn.functional as F
from data.scripts.data_loader import EEGProcessor 
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

        Returns:
            dict: Dictionary with pooled EEG clips and corresponding labels.
        """
        pooled_results = {}

        for key, (eeg_clips, labels, _, _, _) in self.results.items():
            pooled_clips = [self.adaptive_pool_clip(clip) for clip in eeg_clips]
            pooled_results[key] = (pooled_clips, labels)

        return pooled_results

    def apply_pooling_RC(self):
        """
        Applies adaptive pooling to all EEG clips in results.

        Returns:
            dict: Dictionary with pooled EEG clips, labels, start times, and stop times.
        """
        pooled_results = {}

        for key, (eeg_clips, labels, start_times, stop_times) in self.results.items():
            pooled_clips = [self.adaptive_pool_clip(clip) for clip in eeg_clips]
            pooled_results[key] = (pooled_clips, labels, start_times, stop_times)

        return pooled_results



## Example usage



# # Step 1: Process EEG files
# directory = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
# processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
# processor.convert_edf_to_h5()
# results = processor.process_directory()

# # Step 2: Apply Adaptive Pooling
# pooler = EEGPooler(results, target_time_points=100)
# pooled_results = pooler.apply_pooling()
