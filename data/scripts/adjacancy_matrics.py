
import torch
from data.scripts.seizure_dataset import SeizureDataset
from data.scripts.data_loader import EEGProcessor 
from data.scripts.pooling import EEGPooler


class AdjacencyMatrixProcessor:
    def __init__(self, pooled_results, data_directory, top_k=1, standardize=True):
        """
        Initializes the AdjacencyMatrixProcessor with pooled EEG clips and creates a SeizureDataset instance.

        Args:
            pooled_results (dict): Dictionary containing pooled EEG clips and labels.
            data_directory (str): Path to the dataset directory.
            top_k (int, optional): Number of top connections to keep in adjacency computation. Defaults to 1.
            standardize (bool, optional): Whether to standardize the EEG data. Defaults to True.
        """
        self.pooled_results = pooled_results
        self.dataset = SeizureDataset(data_directory=data_directory, top_k=top_k, standardize=standardize)
        self.num_nodes = 19  # EEG channel count (assumed fixed)
        self.xs, self.ys = torch.tril_indices(row=self.num_nodes, col=self.num_nodes, offset=0)

    def compute_all_edge_weights(self, DC, RC):
        """
        Extracts adjacency matrices from EEG clips, computes edge weights, 
        and returns a tensor of all edge weights for all adjacency matrices.

        Returns:
            torch.Tensor: Tensor containing all extracted edge weights.
                          Shape: (num_samples, num_edges).
        """
        edge_weights_list = []

        if DC:
            for base, (pooled_clips, clip_labels) in self.pooled_results.items():
                print(f"Processing Base: {base} with {len(pooled_clips)} clips.")

                for idx, clip in enumerate(pooled_clips):
                    label = clip_labels[idx]
                    adj_mat, clip_label = self.dataset._get_indiv_graphs(clip, label)

                    if adj_mat is None:
                        print(f"⚠️ Skipping None adjacency matrix at index {idx} in base {base}")
                        continue

                    try:
                        adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)
                        edge_weights = adj_tensor[self.xs, self.ys]
                        edge_weights_list.append(edge_weights)
                    except Exception as e:
                        print(f"⚠️ Error at index {idx} in base {base}: {e}")
                        continue

        elif RC:
            for base, (pooled_clips, clip_labels, start_time, stop_time) in self.pooled_results.items():
                print(f"Processing Base: {base} with {len(pooled_clips)} clips.")

                for idx, clip in enumerate(pooled_clips):
                    label = clip_labels[idx]
                    adj_mat, clip_label = self.dataset._get_indiv_graphs(clip, label)

                    if adj_mat is None:
                        print(f"⚠️ Skipping None adjacency matrix at index {idx} in base {base}")
                        continue

                    try:
                        adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)
                        edge_weights = adj_tensor[self.xs, self.ys]
                        edge_weights_list.append(edge_weights)
                    except Exception as e:
                        print(f"⚠️ Error at index {idx} in base {base}: {e}")
                        continue

        else:
            print("❌ Neither DC nor RC flag is set. Returning empty tensor.")
            return torch.empty(0)

        if not edge_weights_list:
            print("⚠️ No valid edge weights found. Returning empty tensor.")
            return torch.empty(0)

        edge_weights_tensor = torch.stack(edge_weights_list)
        return edge_weights_tensor
