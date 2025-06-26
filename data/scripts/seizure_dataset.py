import os
import pickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset ,DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import *

# Define the expected frequency
FREQUENCY = 200

class SeizureDataset(Dataset):
    def __init__(self, data_directory, top_k=8, time_step_size=1, input_len=50, output_len=12, standardize=False):
        self.data_directory = data_directory
        self.time_step_size = time_step_size
        self.input_len = input_len
        self.output_len = output_len
        self.standardize = standardize
        self.top_k = top_k
        self.sensor_ids = [x.split(' ')[-1] for x in INCLUDED_CHANNELS]

        # Fetch all matching .h5 and .edf files
        self.file_pairs = self._find_file_pairs()

    def _find_file_pairs(self):
        """Find only .edf files that have a corresponding .h5 file in the directory."""
        file_pairs = []
        
        # Find all .h5 files first
        h5_files = {f.replace(".h5", ""): os.path.join(root, f) 
                    for root, _, files in os.walk(self.data_directory) 

                    for f in files if f.endswith(".h5")}
        # print("------------"*50)
        # print(len(h5_files))
        # Find only those .edf files that have a corresponding .h5 file
        for root, _, files in os.walk(self.data_directory):
            for f in files:
                if f.endswith(".edf"):
                    base_name = f.replace(".edf", "")
                    if base_name in h5_files:  # Ensure corresponding .h5 file exists
                        edf_path = os.path.join(root, f)
                        file_pairs.append((h5_files[base_name], edf_path))
        # print("------------"*50)
        # print(len(file_pairs))


        return file_pairs


    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """Load and process EEG data from a given file pair."""
        h5_file, edf_file = self.file_pairs[idx]

        # Load EEG data
        with h5py.File(h5_file, 'r') as h5f:
            eeg_data = np.array(h5f['resampled_signal'])

        # Ensure correct shape (channels, time)
        if len(eeg_data.shape) == 1:
            eeg_data = eeg_data[np.newaxis, :]

        num_channels, num_timesteps = eeg_data.shape

        # Skip files with less than 19 channels
        if num_channels < 19:
            raise IndexError(f"Skipping {h5_file}: Only {num_channels} channels available")

        # Truncate to 19 channels if there are more
        eeg_data = eeg_data[:19, :]

        # Standardize if required
        if self.standardize:
            eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)

        # Define the number of available samples
        num_samples = (num_timesteps - (self.input_len + self.output_len)) // self.time_step_size
        if num_samples <= 0:
            raise IndexError(f"Skipping {h5_file}: Not enough samples for input/output lengths")

        # Randomly select a valid index
        start_idx = np.random.randint(0, num_samples) * self.time_step_size
        x = eeg_data[:, start_idx:start_idx + self.input_len]  # Input sequence
        y = eeg_data[:, start_idx + self.input_len:start_idx + self.input_len + self.output_len]  # Output sequence

        # Apply random scaling
        #x = self._random_scale(x)
        #y = self._random_scale(y)

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

    def _get_indiv_graphs(self, eeg_clip, clip_label, swap_nodes=None):
        """
        Compute adjacency matrix for correlation graph using only 19 channels 
        and return its corresponding label.

        Returns:
            adj_mat (np.ndarray): the adjacency matrix of shape (19, 19)
            clip_label: The label corresponding to this EEG clip
        """
        num_sensors = 19
        adj_mat = np.eye(num_sensors, dtype=np.float32)

        # Truncate to first 19 channels if more
        eeg_clip = eeg_clip[:, :num_sensors, :]

        # Rearrange to (num_nodes, seq_len, input_dim)
        eeg_clip = np.transpose(eeg_clip, (1, 0, 2))

        # ❗ Skip clip if not exactly 19 channels
        if eeg_clip.shape[0] != num_sensors:
            print(f"[SKIP] Clip skipped: expected {num_sensors} channels, got {eeg_clip.shape[0]}")
            return None, None

        # Flatten: (num_nodes, seq_len * input_dim)
        eeg_clip = eeg_clip.reshape((num_sensors, -1))

        # Sensor ID remapping
        sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(self.sensor_ids[:num_sensors])}

        if swap_nodes is not None:
            for node_pair in swap_nodes:
                if node_pair[0] in sensor_id_to_ind and node_pair[1] in sensor_id_to_ind:
                    i0 = sensor_id_to_ind[node_pair[0]]
                    i1 = sensor_id_to_ind[node_pair[1]]
                    sensor_id_to_ind[node_pair[0]], sensor_id_to_ind[node_pair[1]] = i1, i0

        # Correlation graph
        for i in range(num_sensors):
            for j in range(i + 1, num_sensors):
                xcorr = comp_xcorr(eeg_clip[i, :], eeg_clip[j, :], mode='valid', normalize=True)
                adj_mat[i, j] = xcorr
                adj_mat[j, i] = xcorr

        adj_mat = np.abs(adj_mat)

        if self.top_k is not None:
            adj_mat = keep_topk(adj_mat, top_k=self.top_k, directed=True)
        else:
            raise ValueError('Invalid top_k value!')

        return adj_mat, clip_label

    
    
    def _get_combined_graph(self, swap_nodes=None):
        """
        Get adjacency matrix for pre-computed distance graph
        Returns:
            adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
        """
        with open(self.adj_mat_dir, 'rb') as pf:
            adj_mat = pickle.load(pf)[-1]

        adj_mat_new = adj_mat.copy()
        if swap_nodes is not None:
            for node_pair in swap_nodes:
                for i in range(adj_mat.shape[0]):
                    adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                    adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                    adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                    adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                    adj_mat_new[i, i] = 1
                adj_mat_new[node_pair[0], node_pair[1]] = adj_mat[node_pair[1], node_pair[0]]
                adj_mat_new[node_pair[1], node_pair[0]] = adj_mat[node_pair[0], node_pair[1]]

        return adj_mat_new   
