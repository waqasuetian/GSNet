from typing import Union
import logging
import torch
from data.scripts.data_loader_preictal import EEGProcessorPreictal
from data.scripts.pooling import EEGPooler
from data.scripts.seizure_dataset import SeizureDataset

class AdjacencyMatrixProcessor:
    def __init__(self, pooled_results, data_directory, top_k=1, standardize=True):
        """
        Initializes the AdjacencyMatrixProcessor with pooled EEG clips and creates a SeizureDataset instance.

        Args:
            pooled_results (dict or list): Dictionary containing pooled EEG clips and labels, or list of file path dicts.
            data_directory (str): Path to the dataset directory.
            top_k (int, optional): Number of top connections to keep in adjacency computation. Defaults to 1.
            standardize (bool, optional): Whether to standardize the EEG data. Defaults to True.
        """
        self.pooled_results = pooled_results
        self.data_directory = data_directory  # Store for use in lazy loading
        self.dataset = SeizureDataset(data_directory=data_directory, top_k=top_k, standardize=standardize)
        self.num_nodes = 19  # EEG channel count (assumed fixed)
        self.xs, self.ys = torch.tril_indices(row=self.num_nodes, col=self.num_nodes, offset=0)

    def compute_all_edge_weights(self, DC: bool = False, RC: bool = False) -> torch.Tensor:
        """
        Extracts adjacency matrices from EEG clips, computes edge weights, 
        and returns a tensor of all edge weights for all adjacency matrices.

        Args:
            DC (bool): Whether to process for detection/classification tasks.
            RC (bool): Whether to process for regression/classification tasks.

        Returns:
            torch.Tensor: Tensor containing all extracted edge weights. Shape: (num_samples, num_edges).
        """
        if not (DC or RC):
            logging.error("Neither DC nor RC flag is set. Returning empty tensor.")
            return torch.empty(0)

        edge_weights_list = []

        if isinstance(self.pooled_results, dict):
            # Traditional mode: pooled_results is a dict
            if DC:
                for base, (pooled_clips, clip_labels) in self.pooled_results.items():
                    logging.info(f"Processing Base: {base} with {len(pooled_clips)} clips.")
                    for idx, clip in enumerate(pooled_clips):
                        label = clip_labels[idx]
                        adj_mat, clip_label = self.dataset._get_indiv_graphs(clip, label)
                        if adj_mat is None:
                            logging.warning(f"Skipping None adjacency matrix at index {idx} in base {base}")
                            continue
                        try:
                            adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)
                            edge_weights = adj_tensor[self.xs, self.ys]
                            edge_weights_list.append(edge_weights)
                        except Exception as e:
                            logging.warning(f"Error at index {idx} in base {base}: {e}")
                            continue
            elif RC:
                for base, (pooled_clips, clip_labels, start_time, stop_time) in self.pooled_results.items():
                    logging.info(f"Processing Base: {base} with {len(pooled_clips)} clips.")
                    for idx, clip in enumerate(pooled_clips):
                        label = clip_labels[idx]
                        adj_mat, clip_label = self.dataset._get_indiv_graphs(clip, label)
                        if adj_mat is None:
                            logging.warning(f"Skipping None adjacency matrix at index {idx} in base {base}")
                            continue
                        try:
                            adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)
                            edge_weights = adj_tensor[self.xs, self.ys]
                            edge_weights_list.append(edge_weights)
                        except Exception as e:
                            logging.warning(f"Error at index {idx} in base {base}: {e}")
                            continue
        else:
            # Lazy loading mode: pooled_results is a list of dicts with 'h5_path'
            processor = EEGProcessorPreictal(
                root_directory=self.data_directory,  # Use stored data_directory
                resampled_freq=200,
                time_step_size=1,
                apply_fft=True,
                overlap=0.5,
                min_channels_per_event=1,
                topk_events_by_duration=10,
                pad_short_segments=True,
                min_short_frac=0.30
            )
            for idx, file_dict in enumerate(self.pooled_results):
                if 'h5_path' not in file_dict:
                    logging.warning(f"Skipping invalid file dict: {file_dict}")
                    continue
                h5_path = file_dict['h5_path']
                logging.info(f"Processing H5 file: {h5_path}")
                try:
                    # Load data from H5 file
                    eeg_clips, clip_labels, start_times, stop_times = processor.load_h5_file(h5_path)
                    # Pool clips since apply_pooling_RC didn't pool them
                    pooler = EEGPooler({"temp": (eeg_clips, clip_labels, start_times, stop_times)}, target_time_points=100)
                    pooled_data = pooler.apply_pooling_RC()
                    if not pooled_data:
                        logging.warning(f"No valid data after pooling for {h5_path}")
                        continue
                    pooled_clips, clip_labels, start_times, stop_times = pooled_data["temp"]
                    # Process clips
                    for clip_idx, clip in enumerate(pooled_clips):
                        label = clip_labels[clip_idx]
                        adj_mat, clip_label = self.dataset._get_indiv_graphs(clip, label)
                        if adj_mat is None:
                            logging.warning(f"Skipping None adjacency matrix at index {clip_idx} in {h5_path}")
                            continue
                        try:
                            adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)
                            edge_weights = adj_tensor[self.xs, self.ys]
                            edge_weights_list.append(edge_weights)
                        except Exception as e:
                            logging.warning(f"Error at index {clip_idx} in {h5_path}: {e}")
                            continue
                except Exception as e:
                    logging.warning(f"Error processing {h5_path}: {e}")
                    continue

        if not edge_weights_list:
            logging.warning("No valid edge weights found. Returning empty tensor.")
            return torch.empty(0)

        edge_weights_tensor = torch.stack(edge_weights_list)
        return edge_weights_tensor

