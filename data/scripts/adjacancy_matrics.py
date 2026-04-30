from typing import List, Tuple, Union, Dict, Any  # Add Dict and Any
from typing import Union, List, Optional, Dict
import logging
import torch
import numpy as np
from scipy.signal import coherence as scipy_coherence
from scipy.signal import correlate
from sklearn.feature_selection import mutual_info_regression
from data.scripts.data_loader_preictal import EEGProcessorPreictal
from data.scripts.pooling import EEGPooler
from data.scripts.seizure_dataset import SeizureDataset
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add to adjacancy_matrics.py

# Add to adjacancy_matrics.py

class HybridGraphBuilder:
    """
    Builds a hybrid graph combining:
    1. Distance-based geometric connectivity (structural prior)
    2. Correlation-based functional connectivity (data-driven)
    
    Formula: W_hybrid = α * W_distance + (1-α) * W_functional
    
    Where α controls the trade-off between structure and function.
    """
    
    def __init__(self, channel_names, alpha=0.5, distance_threshold=0.9, 
                 functional_method='pearson', tau=None):
        """
        Args:
            channel_names: List of EEG channel names
            alpha: Weight for distance graph (0-1). Higher = more structure, lower = more function
            distance_threshold: Threshold for distance graph (κ)
            functional_method: 'pearson', 'cross_corr', 'plv', 'coherence', 'mi'
            tau: For sparse correlation (top-τ neighbors), None = full graph
        """
        self.channel_names = channel_names
        self.alpha = alpha
        self.distance_threshold = distance_threshold
        self.functional_method = functional_method
        self.tau = tau
        
        # Initialize component builders
        self.distance_builder = GeometricGraphBuilder(
            channel_names, threshold=distance_threshold
        )
        self.functional_builder = None  # Will be created per signal
        
    def compute_distance_graph(self):
        """Compute static distance-based graph"""
        return self.distance_builder.compute_distance_graph()
    
    def compute_functional_graph(self, signal):
        """Compute functional graph from EEG signal"""
        factory = GraphConstructionFactory(signal, self.channel_names, fs=200)
        
        if self.functional_method == 'pearson':
            corr = factory.compute_pearson_correlation()
        elif self.functional_method == 'cross_corr':
            corr = factory.compute_cross_correlation()
        elif self.functional_method == 'plv':
            corr = factory.compute_phase_locking_value()
        elif self.functional_method == 'coherence':
            corr = factory.compute_coherence()
        elif self.functional_method == 'mi':
            corr = factory.compute_mutual_information()
        else:
            raise ValueError(f"Unknown functional method: {self.functional_method}")
        
        # Apply sparsity if tau is specified
        if self.tau is not None and self.tau > 0:
            n = corr.shape[0]
            sparse_corr = np.zeros_like(corr)
            for i in range(n):
                top_indices = np.argsort(np.abs(corr[i]))[-self.tau:]
                for j in top_indices:
                    sparse_corr[i, j] = corr[i, j]
            corr = sparse_corr
        
        return corr
    
    def compute_hybrid_graph(self, signal):
        """
        Compute hybrid graph: α * W_distance + (1-α) * W_functional
        """
        #print(f"Signal shape in hybrid builder: {signal.shape}")
        # Compute individual graphs
        W_distance = self.compute_distance_graph()
        W_functional = self.compute_functional_graph(signal)
        #print(f"W_distance shape: {W_distance.shape}")
        #print(f"W_functional shape: {W_functional.shape}")
        # Normalize both to [0, 1] range
        if W_distance.max() > 0:
            W_distance = W_distance / W_distance.max()
        if W_functional.max() > 0:
            W_functional = W_functional / W_functional.max()
        
        # Combine
        W_hybrid = self.alpha * W_distance + (1 - self.alpha) * W_functional
        
        # # Ensure symmetry
        # W_hybrid = (W_hybrid + W_hybrid.T) / 2
        
        # # No self-connections
        # np.fill_diagonal(W_hybrid, 0)
        
        return W_hybrid
    
    def compute_adaptive_hybrid(self, signal):
        """
        Adaptive hybrid where α is determined by signal characteristics
        Higher SNR → more weight to functional connectivity
        """
        # Estimate signal quality (e.g., SNR, variance)
        signal_quality = np.std(signal) / (np.mean(np.abs(signal)) + 1e-8)
        signal_quality = np.clip(signal_quality, 0, 1)
        
        # Adaptive alpha: better signal → less weight on distance
        alpha_adaptive = 1 - signal_quality
        
        W_distance = self.compute_distance_graph()
        W_functional = self.compute_functional_graph(signal)
        
        if W_distance.max() > 0:
            W_distance = W_distance / W_distance.max()
        if W_functional.max() > 0:
            W_functional = W_functional / W_functional.max()
        
        W_hybrid = alpha_adaptive * W_distance + (1 - alpha_adaptive) * W_functional
        W_hybrid = (W_hybrid + W_hybrid.T) / 2
        np.fill_diagonal(W_hybrid, 0)
        
        return W_hybrid, alpha_adaptive


class GeometricGraphBuilder:
    """
    Builds graph based on physical electrode positions (10-20 system)
    """
    
    # Standard 10-20 electrode positions (normalized coordinates)
    # Based on 19-channel montage
    ELECTRODE_POSITIONS = {
        'FP1': (-0.5, 0.8), 'FP2': (0.5, 0.8),
        'F3': (-0.4, 0.5), 'F4': (0.4, 0.5), 'FZ': (0, 0.5),
        'F7': (-0.7, 0.4), 'F8': (0.7, 0.4),
        'C3': (-0.4, 0), 'C4': (0.4, 0), 'CZ': (0, 0),
        'T3': (-0.7, -0.1), 'T4': (0.7, -0.1),
        'P3': (-0.4, -0.4), 'P4': (0.4, -0.4), 'PZ': (0, -0.4),
        'T5': (-0.6, -0.6), 'T6': (0.6, -0.6),
        'O1': (-0.3, -0.8), 'O2': (0.3, -0.8)
    }
    
    def __init__(self, channel_names, threshold=0.9, sigma=None):
        self.channel_names = channel_names
        self.threshold = threshold
        self.sigma = sigma
        
    def compute_euclidean_distances(self):
        """Compute Euclidean distances between electrode positions"""
        n = len(self.channel_names)
        distances = np.zeros((n, n))
        
        for i, ch_i in enumerate(self.channel_names):
            pos_i = self.ELECTRODE_POSITIONS.get(ch_i, (0, 0))
            for j, ch_j in enumerate(self.channel_names):
                pos_j = self.ELECTRODE_POSITIONS.get(ch_j, (0, 0))
                distances[i, j] = np.sqrt((pos_i[0]-pos_j[0])**2 + (pos_i[1]-pos_j[1])**2)
        
        return distances
    
    def compute_distance_graph(self):
        """Build graph using thresholded Gaussian kernel"""
        distances = self.compute_euclidean_distances()
        
        if self.sigma is None:
            nonzero_dist = distances[distances > 0]
            self.sigma = np.std(nonzero_dist) if len(nonzero_dist) > 0 else 1.0
        
        # Apply Gaussian kernel
        weights = np.exp(-(distances ** 2) / (self.sigma ** 2))
        
        # Apply threshold
        weights[distances > self.threshold] = 0
        
        # Set diagonal to 0
        np.fill_diagonal(weights, 0)
        
        # Symmetrize
        weights = (weights + weights.T) / 2
        
        return weights


class GraphConstructionFactory:
    """
    Factory class for different graph construction methods.
    Supports 5 different edge weight schemes.
    """
    
    # Available graph methods
    METHODS = {
        'pearson': 'Pearson correlation coefficient',
        'cross_corr': 'Cross-correlation with time lag',
        'plv': 'Phase Locking Value (PLV)',
        'coherence': 'Spectral coherence',
        'mi': 'Mutual Information'
    }
    
    def __init__(self, signal_array: np.ndarray, ch_names: List[str], fs: int = 200):
        """
        Args:
            signal_array: (n_channels, n_samples) raw EEG signal
            ch_names: list of channel names
            fs: sampling frequency
        """
        self.signal = signal_array.astype(np.float32)
        self.ch_names = ch_names
        self.fs = fs
        self.n_channels = signal_array.shape[0]
        #print(f"GraphConstructionFactory: signal shape = {signal_array.shape}")
        #print(f"Number of channels = {self.n_channels}")
    def compute_pearson_correlation(self) -> np.ndarray:
        """Pearson correlation coefficient (original method)"""
        corr_matrix = np.corrcoef(self.signal)
        np.fill_diagonal(corr_matrix, 0)
        # Clip to [-1, 1] and handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix = np.clip(corr_matrix, -1, 1)
        # Convert to [0, 1] range for edge weights
        corr_matrix = (corr_matrix + 1) / 2
        return corr_matrix
    
    def compute_cross_correlation(self, max_lag: int = 50) -> np.ndarray:
        """Cross-correlation with time lag"""
        n_channels = self.n_channels
        n_samples = self.signal.shape[1]
        cross_corr_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                try:
                    corr = correlate(self.signal[i], self.signal[j], mode='same')
                    denom = n_samples * np.std(self.signal[i]) * np.std(self.signal[j])
                    if denom > 0:
                        corr = corr / denom
                        max_corr = np.max(np.abs(corr))
                    else:
                        max_corr = 0
                    cross_corr_matrix[i, j] = max_corr
                    cross_corr_matrix[j, i] = max_corr
                except Exception:
                    cross_corr_matrix[i, j] = 0
                    cross_corr_matrix[j, i] = 0
        
        # Clip to [0, 1]
        cross_corr_matrix = np.clip(cross_corr_matrix, 0, 1)
        return cross_corr_matrix
    
    def compute_phase_locking_value(self) -> np.ndarray:
        """Phase Locking Value (PLV) - measures phase synchrony"""
        try:
            from scipy.signal import hilbert
            
            n_channels = self.n_channels
            plv_matrix = np.zeros((n_channels, n_channels))
            
            # Compute Hilbert transform to get phase
            analytic_signals = np.array([hilbert(self.signal[i]) for i in range(n_channels)])
            phases = np.angle(analytic_signals)
            
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    phase_diff = phases[i] - phases[j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_matrix[i, j] = plv
                    plv_matrix[j, i] = plv
            
            return np.clip(plv_matrix, 0, 1)
        except Exception as e:
            print(f"PLV computation failed: {e}")
            return np.zeros((self.n_channels, self.n_channels))
    
    def compute_coherence(self, nperseg: int = 256) -> np.ndarray:
        """Spectral coherence averaged across frequencies"""
        n_channels = self.n_channels
        coherence_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                try:
                    f, Cxy = scipy_coherence(self.signal[i], self.signal[j], fs=self.fs, nperseg=nperseg)
                    avg_coherence = np.mean(Cxy[1:])  # Exclude DC
                    coherence_matrix[i, j] = avg_coherence
                    coherence_matrix[j, i] = avg_coherence
                except Exception:
                    coherence_matrix[i, j] = 0
                    coherence_matrix[j, i] = 0
        
        return np.clip(coherence_matrix, 0, 1)
    
    def compute_mutual_information(self) -> np.ndarray:
        """Mutual Information normalized to [0,1]"""
        n_channels = self.n_channels
        mi_matrix = np.zeros((n_channels, n_channels))
        
        # Normalize signals
        signals_norm = (self.signal - self.signal.mean(axis=1, keepdims=True)) / (self.signal.std(axis=1, keepdims=True) + 1e-8)
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                try:
                    X = signals_norm[i].reshape(-1, 1)
                    y = signals_norm[j]
                    mi = mutual_info_regression(X, y, random_state=42)[0]
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
                except Exception:
                    mi_matrix[i, j] = 0
                    mi_matrix[j, i] = 0
        
        # Normalize to [0, 1]
        if mi_matrix.max() > 0:
            mi_matrix = mi_matrix / mi_matrix.max()
        
        return np.clip(mi_matrix, 0, 1)
    
    def compute_method(self, method_name: str) -> np.ndarray:
        """Compute a specific graph construction method"""
        method_map = {
            'pearson': self.compute_pearson_correlation,
            'cross_corr': self.compute_cross_correlation,
            'plv': self.compute_phase_locking_value,
            'coherence': self.compute_coherence,
            'mi': self.compute_mutual_information
        }
        
        if method_name not in method_map:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(method_map.keys())}")
        
        return method_map[method_name]()
    
    def compute_all_methods(self) -> Dict[str, np.ndarray]:
        """Compute all graph construction methods"""
        results = {}
        for method_name in self.METHODS.keys():
            try:
                results[method_name] = self.compute_method(method_name)
            except Exception as e:
                print(f"Warning: {method_name} computation failed: {e}")
                results[method_name] = np.zeros((self.n_channels, self.n_channels))
        
        return results


class AdjacencyMatrixProcessor:
    """
    Adjacency processor that supports multiple graph construction methods.
    """
    def __init__(self, pooled_results, data_directory, channel_names: Optional[List[str]] = None, 
                 graph_method: str = 'pearson', graph_params: Optional[Dict] = None,
                 top_k: int = 1, standardize: bool = True):
        """
        Args:
            pooled_results: Dictionary or list of pooled EEG clips
            data_directory: Path to dataset directory
            channel_names: List of channel names
            graph_method: One of 'pearson', 'cross_corr', 'plv', 'coherence', 'mi'
            top_k: Number of top connections to keep (0 = keep all)
            standardize: Whether to standardize EEG data
        """
        self.pooled_results = pooled_results
        self.data_directory = data_directory
        self.graph_method = graph_method
        self.graph_params = graph_params or {}
        self.top_k = top_k
        self.standardize = standardize
        
        # Handle channel names
        if channel_names is None:
            channel_names = [
                'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
            ]
        else:
            while isinstance(channel_names, list) and len(channel_names) > 0 and isinstance(channel_names[0], list):
                channel_names = [item for sublist in channel_names for item in sublist]
        
        self.channel_names = channel_names
        self.num_nodes = len(self.channel_names)
        self.dataset = SeizureDataset(data_directory=data_directory, top_k=top_k, standardize=standardize)
        self.xs, self.ys = torch.tril_indices(row=self.num_nodes, col=self.num_nodes, offset=-1)
    
    def compute_edge_weights_from_signal(self, signal: np.ndarray) -> np.ndarray:
        """Compute edge weights using selected graph method"""

       # print(f"DEBUG: signal shape = {signal.shape}")
       # print(f"DEBUG: num_nodes = {self.num_nodes}")
        if self.graph_method == 'hybrid':
            builder = HybridGraphBuilder(
                self.channel_names,
                alpha=self.graph_params.get('alpha', 0.5),
                distance_threshold=self.graph_params.get('distance_threshold', 0.5),
                functional_method=self.graph_params.get('functional_method', 'cross_corr'),
                tau=self.graph_params.get('tau', None)
            )
            adj_matrix = builder.compute_hybrid_graph(signal)

            
        elif self.graph_method == 'adaptive_hybrid':
            builder = HybridGraphBuilder(
                self.channel_names,
                distance_threshold=self.graph_params.get('distance_threshold', 0.5),
                functional_method=self.graph_params.get('functional_method', 'cross_corr')
            )
            adj_matrix, alpha = builder.compute_adaptive_hybrid(signal)
            print(f"Adaptive α = {alpha:.3f}")
            
        elif self.graph_method == 'distance':
            builder = GeometricGraphBuilder(
                self.channel_names,
                threshold=self.graph_params.get('threshold', 0.9)
            )
            adj_matrix = builder.compute_distance_graph()
            
        # ... other methods ...
        edge_weights = adj_matrix[self.xs.numpy(), self.ys.numpy()]
        return edge_weights
    
    # def compute_edge_weights_from_signal(self, signal: np.ndarray) -> np.ndarray:
    #     """
    #     Compute edge weights from raw signal using selected graph method.
        
    #     Args:
    #         signal: (n_channels, n_samples) raw EEG signal
        
    #     Returns:
    #         edge_weights: (num_edges,) 1D array of edge weights
    #     """
    #     # Ensure signal has correct shape
    #     if signal.ndim == 3:
    #         # (T, N, F) -> average over time
    #         signal = signal.mean(axis=0).T
        
    #     if signal.ndim == 2 and signal.shape[0] != self.num_nodes:
    #         signal = signal.T
        
    #     if signal.shape[0] != self.num_nodes:
    #         print(f"Signal shape {signal.shape} doesn't match nodes {self.num_nodes}")
    #         return np.ones(len(self.xs))
        
    #     factory = GraphConstructionFactory(signal, self.channel_names, fs=200)
    #     corr_matrix = factory.compute_method(self.graph_method)
        
    #     # Extract lower triangular edge weights
    #     edge_weights = corr_matrix[self.xs.numpy(), self.ys.numpy()]
        
    #     # Apply top-k sparsification if requested
    #     if self.top_k > 0 and self.top_k < len(edge_weights):
    #         threshold = np.sort(edge_weights)[-self.top_k]
    #         edge_weights = np.where(edge_weights >= threshold, edge_weights, 0)
        
    #     return edge_weights.astype(np.float32)
    
    def compute_all_edge_weights(self, DC: bool = False, RC: bool = False) -> Union[torch.Tensor, None]:
        """
        Extracts adjacency matrices from EEG clips using the selected graph method.
        
        Args:
            DC: Detection/classification flag
            RC: Regression/classification flag
        
        Returns:
            torch.Tensor or None: Edge weights tensor or None for uniform weights
        """
        if not (DC or RC):
            logging.error("Neither DC nor RC flag is set. Returning None.")
            return None
        
        # Lazy loading mode
        if isinstance(self.pooled_results, list):
            # Create temporary processor to load raw signals
            temp_processor = EEGProcessorPreictal(
                root_directory=self.data_directory,
                resampled_freq=200,
                window_sec=1.0,
                feature_mode='raw',
                enable_channel_filter=True,
                included_channels=self.channel_names
            )
            
            edge_weights_list = []
            count = 0
            
            for file_dict in self.pooled_results:
                if 'h5_path' not in file_dict:
                    continue
                h5_path = file_dict['h5_path']
                csv_path = file_dict.get('csv_path', '')
                
                try:
                    # Load clips for this file
                    eeg_clips, clip_labels, start_times, stop_times = temp_processor.process_h5_and_csv(h5_path, csv_path)
                    
                    if not eeg_clips:
                        continue
                    
                    # Use the first clip for graph construction
                    for clip in eeg_clips[:3]:  # Use up to 3 clips per file
                        if clip.ndim == 3:
                            # Average over time dimension
                            signal = clip.mean(axis=0)
                           # print(f"Signal shape after extraction: {signal.shape}")
                            edge_weights = self.compute_edge_weights_from_signal(signal)
                            
                            if not np.isnan(edge_weights).any() and edge_weights.mean() > 0:
                                edge_weights_list.append(edge_weights)
                                count += 1
                                break
                        
                except Exception as e:
                    logging.warning(f"Error processing {h5_path}: {e}")
                    continue
            
            if count == 0:
                logging.warning(f"No valid edge weights found for method {self.graph_method}. Using uniform weights.")
                return None
            
            # Average across files
            edge_weights_tensor = torch.tensor(np.mean(edge_weights_list, axis=0), dtype=torch.float32)
            print(f"Computed edge weights using {self.graph_method}: mean={edge_weights_tensor.mean():.4f}")
            print(edge_weights_tensor)
            return edge_weights_tensor
        
        else:
            # Traditional dictionary mode
            edge_weights_list = []
            
            if DC:
                for base, (pooled_clips, clip_labels) in self.pooled_results.items():
                    for idx, clip in enumerate(pooled_clips):
                        if clip.ndim == 3:
                            signal = clip.mean(axis=0)
                            edge_weights = self.compute_edge_weights_from_signal(signal)
                            edge_weights_list.append(edge_weights)
            elif RC:
                for base, (pooled_clips, clip_labels, start_time, stop_time) in self.pooled_results.items():
                    for idx, clip in enumerate(pooled_clips):
                        if clip.ndim == 3:
                            signal = clip.mean(axis=0)
                            edge_weights = self.compute_edge_weights_from_signal(signal)
                            edge_weights_list.append(edge_weights)
            
            if not edge_weights_list:
                logging.warning(f"No valid edge weights found for method {self.graph_method}. Using uniform weights.")
                return None
            
            edge_weights_tensor = torch.tensor(np.mean(edge_weights_list, axis=0), dtype=torch.float32)
            
            return edge_weights_tensor


def flatten_list(lst):
    """Recursively flatten a list of lists into a flat list."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result
