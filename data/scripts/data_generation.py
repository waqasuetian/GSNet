import torch

class EEGDataProcessor: 
    def __init__(self, pooled_results, target_time_points=50, num_nodes=19, detection=True, classification=False):
        """
        Initializes the EEGDataProcessor to handle EEG clips with fixed-length processing.

        Args:
            pooled_results (dict): Dictionary of pooled EEG data from EEGPooler.
            target_time_points (int): Fixed length for EEG clips.
            num_nodes (int): Number of EEG channels to retain (default=19).
            detection (bool): Whether the processor is used for detection.
            classification (bool): Whether the processor is used for classification.
        """
        self.pooled_results = pooled_results
        self.target_time_points = target_time_points
        self.num_nodes = num_nodes
        self.detection = detection
        self.classification = classification
        
        if self.detection or not self.classification:
            self.label_mapping = {
                'bckg': 0, 'gnsz': 1, 'fnsz': 1, 'tcsz': 1,  
                'absz': 1, 'seiz': 1, 'cpsz': 1, 'tnsz': 1
            }
        else:
            self.label_mapping = {
                'gnsz': 1, 'fnsz': 2, 'tcsz': 3,  
                'absz': 4, 'seiz': 5, 'cpsz': 6, 'tnsz': 7
            }


    def process_fixed_length_clips(self, classification=False):
            """
            Processes EEG clips ensuring fixed length, 19 channels, and mapped labels.
            
            Args:
                classification (bool): If True, skips clips labeled as 'bckg'.

            Returns:
                tuple:
                    X (torch.Tensor): EEG clips (num_samples, target_T, num_nodes, feature_dim).
                    Y (torch.Tensor): Mapped integer labels (num_samples,).
            """
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


# """if classification is TRUE then skip those clips  whose label is bckg and only stack rest of the clips  in X and labelY"""
# # Example Usage
# if __name__ == "__main__":
#     from data_loader import EEGProcessor

#     # Step 1: Process EEG files
#     directory = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
#     processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
#     processor.convert_edf_to_h5()
#     results = processor.process_directory()

#     # Step 2: Apply Adaptive Pooling
#     pooler = EEGPooler(results, target_time_points=100)
#     pooled_results = pooler.apply_pooling()

#     # Step 3: Process Fixed-Length Clips
#     processor = EEGDataProcessor(pooled_results, target_time_points=50)
#     X, Y = processor.process_fixed_length_clips()

#     print("Final X shape:", X.shape)  # Expected: (num_samples, 50, 19, feature_dim)
#     print("Final Y shape:", Y.shape)  # Expected: (num_samples,)



