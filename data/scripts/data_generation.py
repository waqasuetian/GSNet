import torch

class EEGDataProcessor:
    def __init__(self, pooled_results, target_time_points=50, num_nodes=19, 
                 detection=True, classification=False,
                 lazy_loading=False, processor=None, pooler=None,
                 patient_wise_split=True,      # NEW
                 test_patient_ratio=0.2,        # NEW
                 random_seed=42):               # NEW
        """
        Args:
            patient_wise_split: Whether to split by patient (True) or by file (False)
            test_patient_ratio: Proportion of patients for validation
            random_seed: Random seed for reproducible splits
        """
        self.pooled_results = pooled_results
        self.target_time_points = target_time_points
        self.num_nodes = num_nodes
        self.detection = detection
        self.classification = classification
        self.lazy_loading = lazy_loading
        self.processor = processor
        self.pooler = pooler
        self.patient_wise_split = patient_wise_split
        self.test_patient_ratio = test_patient_ratio
        self.random_seed = random_seed
        
        # Store validation data separately
        self.val_X = None
        self.val_Y = None
        

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
            # if self.lazy_loading and isinstance(self.pooled_results, list):
            #     return self._process_lazy(classification, batch_size)
            if self.lazy_loading and isinstance(self.pooled_results, list):
                return self._process_lazy_with_patient_split(classification, batch_size)

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

    def _process_lazy_with_patient_split(self, classification=False, batch_size=32):
            """
            Process files with patient-wise splitting to prevent data leakage.
            """
            from data.scripts.data_loader import (
                patient_wise_split_file_paths, 
                validate_patient_split_no_leakage,
                extract_patient_id_from_path
            )
            
            print(f"\n{'='*60}")
            print("PATIENT-WISE SPLIT FOR DETECTION/CLASSIFICATION")
            print(f"{'='*60}")
            
            # Split files by patient
            train_files, val_files, train_patients, val_patients = patient_wise_split_file_paths(
                self.pooled_results,
                test_size=self.test_patient_ratio,
                random_state=self.random_seed
            )
            
            print(f"Training patients: {len(train_patients)}")
            print(f"Validation patients: {len(val_patients)}")
            print(f"Training files: {len(train_files)}")
            print(f"Validation files: {len(val_files)}")
            
            # Validate no leakage
            validate_patient_split_no_leakage(train_files, val_files)
            
            # Process training files
            print(f"\n{'='*60}")
            print("PROCESSING TRAINING DATA")
            print(f"{'='*60}")
            X_train, Y_train = self._process_file_list(train_files, classification, batch_size, is_training=True)
            
            # Process validation files
            print(f"\n{'='*60}")
            print("PROCESSING VALIDATION DATA")
            print(f"{'='*60}")
            X_val, Y_val = self._process_file_list(val_files, classification, batch_size, is_training=False)
            
            # Store validation data
            self.val_X = X_val
            self.val_Y = Y_val
            
            print(f"\n{'='*60}")
            print("PATIENT-WISE SPLIT COMPLETE")
            print(f"{'='*60}")
            print(f"Training: X={X_train.shape}, Y={Y_train.shape}")
            print(f"Validation: X={X_val.shape}, Y={Y_val.shape}")
            print(f"{'='*60}\n")
            
            return X_train, Y_train
        
    def _process_file_list(self, file_list, classification, batch_size, is_training=True):
        """
        Process a list of files and return stacked tensors.
        """
        X_list = []
        Y_list = []
        label_counts = {key: 0 for key in self.label_mapping.keys()}
        
        for file_info in file_list:
            base_name = file_info['base_name']
            h5_path = file_info['h5_path']
            csv_path = file_info['csv_path']
            
            try:
                # Load data for this file
                eeg_clips, labels, _, _, _ = self.processor.process_h5_and_csv(h5_path, csv_path)
                
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
                            print(f"Warning: Label '{label}' not found. Skipping.")
                            continue
                    else:
                        mapped_label = label
                    
                    X_list.append(fixed_clip)
                    Y_list.append(mapped_label)
                    
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                continue
        
        # Filter clips with exactly 19 nodes
        filtered = [(x, y) for x, y in zip(X_list, Y_list) if x.shape[1] == 19]
        if not filtered:
            raise ValueError("No valid EEG clips with 19 nodes found.")
        
        X_list, Y_list = zip(*filtered)
        
        # Convert to tensors
        X = torch.stack(X_list)
        Y = torch.tensor(Y_list, dtype=torch.long)
        
        # Print label distribution
        split_name = "Training" if is_training else "Validation"
        print(f"\n{split_name} Label Distribution:")
        for label, count in label_counts.items():
            if count > 0:
                print(f"  {label}: {count}")
        
        return X, Y

    def get_validation_data(self):
        """Return validation data if patient-wise split was used."""
        return self.val_X, self.val_Y
