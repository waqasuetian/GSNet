import torch

class EEGDataProcessor:
    def __init__(self, pooled_results, target_time_points=50, num_nodes=19, detection=True, classification=False,
                 lazy_loading=False, processor=None, pooler=None):
        """
        Initializes the EEGDataProcessor to handle EEG clips with fixed-length processing.

        Args:
            pooled_results (dict or list): Dictionary of pooled EEG data from EEGPooler,
                                           OR list of file paths for lazy loading.
            target_time_points (int): Fixed length for EEG clips.
            num_nodes (int): Number of EEG channels to retain (default=19).
            detection (bool): Whether the processor is used for detection.
            classification (bool): Whether the processor is used for classification.
            lazy_loading (bool): NEW - If True, load data on-the-fly instead of all at once.
            processor (EEGProcessor): Required if lazy_loading=True.
            pooler (EEGPooler): Required if lazy_loading=True.
        """
        self.pooled_results = pooled_results
        self.target_time_points = target_time_points
        self.num_nodes = num_nodes
        self.detection = detection
        self.classification = classification
        self.lazy_loading = lazy_loading
        self.processor = processor
        self.pooler = pooler

        if self.detection or not self.classification:
            self.label_mapping = {
                'bckg': 0, 'gnsz': 1, 'fnsz': 1, 'tcsz': 1,
                'absz': 1, 'mysz': 1, 'cpsz': 1, 'tnsz': 1
            }
        else:
            self.label_mapping = {
                'gnsz': 0, 'fnsz': 1, 'tcsz': 2,
                'absz': 3, 'mysz': 4, 'cpsz': 5, 'tnsz': 6
            }


    def process_fixed_length_clips(self, classification=False, batch_size=32):
            """
            Processes EEG clips ensuring fixed length, 19 channels, and mapped labels.

            NEW: Supports lazy loading mode to prevent RAM overflow.

            Args:
                classification (bool): If True, skips clips labeled as 'bckg'.
                batch_size (int): NEW - Number of files to process at once in lazy mode.

            Returns:
                tuple:
                    X (torch.Tensor): EEG clips (num_samples, target_T, num_nodes, feature_dim).
                    Y (torch.Tensor): Mapped integer labels (num_samples,).
            """
            # NEW: Check if we're in lazy loading mode
            if self.lazy_loading and isinstance(self.pooled_results, list):
                return self._process_lazy(classification, batch_size)

            # Traditional mode - load all data
            X_list, Y_list = [], []
            label_counts = {key: 0 for key in self.label_mapping.keys()}  # Count clips per label

            for base, (eeg_clips, eeg_labels) in self.pooled_results.items():
                print(f"Processing Base: {base} with {len(eeg_clips)} clips")

                for clip, label in zip(eeg_clips, eeg_labels):
                    # Skip 'bckg' labeled clips if classification is True
                    if classification and label == "bckg":
                        continue

                    # Ensure only 19 channels are kept
                    fixed_clip = clip[:, :self.num_nodes, :]  # Shape: (target_T, 19, feature_dim)
                    
                    # Map label from string to integer
                    if isinstance(label, str):  
                        if label in self.label_mapping:
                            mapped_label = self.label_mapping[label]
                            label_counts[label] += 1  # Track label count
                        else:
                            print(f"Warning: Label '{label}' not found in label mapping. Skipping clip.")
                            continue  # Skip the clip if label is not found
                    else:
                        mapped_label = label
                    
                    # Append only valid clips and labels
                    X_list.append(fixed_clip)
                    Y_list.append(mapped_label)

                print(f"Processed {len(eeg_clips)} clips from {base}")

            # Print label distribution
            print("\n📊 Number of Clips per Label:")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")

            # Filter clips with exactly 19 nodes
            X_list, Y_list = zip(*[(x, y) for x, y in zip(X_list, Y_list) if x.shape[1] == 19])

            # Ensure X and Y have compatible dimensions
            if len(X_list) == 0 or len(Y_list) == 0:
                raise ValueError("No valid EEG clips with 19 nodes found after filtering.")
            if len(X_list) != len(Y_list):
                raise ValueError("Mismatch between number of EEG clips and labels after filtering.")

            # Convert lists to tensors
            X = torch.stack(X_list)  # Shape: (num_samples, target_T, 19, feature_dim)
            Y = torch.tensor(Y_list, dtype=torch.long)  # Shape: (num_samples,)

            
            print('--'*50)
            print(X.shape)
            print('--'*50)
            print(Y.shape)

            return X, Y

    def _process_lazy(self, classification=False, batch_size=32):
        """
        NEW: Process files in batches to avoid RAM overflow.
        Loads and processes files incrementally.
        """
        import numpy as np

        X_list, Y_list = [], []
        label_counts = {key: 0 for key in self.label_mapping.keys()}

        print(f"[LAZY MODE] Processing {len(self.pooled_results)} files in batches of {batch_size}")

        # Process files in batches
        for batch_idx in range(0, len(self.pooled_results), batch_size):
            batch_files = self.pooled_results[batch_idx:batch_idx + batch_size]
            print(f"[LAZY MODE] Processing batch {batch_idx//batch_size + 1}/{(len(self.pooled_results) + batch_size - 1)//batch_size}")

            for file_info in batch_files:
                base_name = file_info['base_name']
                h5_path = file_info['h5_path']
                csv_path = file_info['csv_path']

                try:
                    # Load data for this file only
                    eeg_clips, labels, _, _, _ = self.processor.process_h5_and_csv(h5_path, csv_path)

                    # Pool clips
                    for clip, label in zip(eeg_clips, labels):
                        if classification and label == "bckg":
                            continue

                        # Pool the clip
                        pooled_clip = self.pooler.adaptive_pool_clip(clip)

                        # Ensure only 19 channels are kept
                        fixed_clip = pooled_clip[:, :self.num_nodes, :]

                        # Map label from string to integer
                        if isinstance(label, str):
                            if label in self.label_mapping:
                                mapped_label = self.label_mapping[label]
                                label_counts[label] += 1
                            else:
                                print(f"Warning: Label '{label}' not found in label mapping. Skipping clip.")
                                continue
                        else:
                            mapped_label = label

                        # Append only valid clips and labels
                        X_list.append(fixed_clip)
                        Y_list.append(mapped_label)

                    print(f"[LAZY MODE] Processed {base_name}: {len(eeg_clips)} clips")

                except Exception as e:
                    print(f"[LAZY MODE] Error processing {base_name}: {e}")
                    continue

        # Print label distribution
        print("\n📊 Number of Clips per Label:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        # Filter clips with exactly 19 nodes
        X_list, Y_list = zip(*[(x, y) for x, y in zip(X_list, Y_list) if x.shape[1] == 19])

        # Ensure X and Y have compatible dimensions
        if len(X_list) == 0 or len(Y_list) == 0:
            raise ValueError("No valid EEG clips with 19 nodes found after filtering.")
        if len(X_list) != len(Y_list):
            raise ValueError("Mismatch between number of EEG clips and labels after filtering.")

        # Convert lists to tensors
        X = torch.stack(X_list)  # Shape: (num_samples, target_T, 19, feature_dim)
        Y = torch.tensor(Y_list, dtype=torch.long)  # Shape: (num_samples,)

        print('--'*50)
        print(f"[LAZY MODE] Final X shape: {X.shape}")
        print('--'*50)
        print(f"[LAZY MODE] Final Y shape: {Y.shape}")

        return X, Y
