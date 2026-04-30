
import os
import mne
import h5py
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import os
import numpy as np
from collections import defaultdict

def extract_patient_id_from_path(file_path: str) -> str:
    """
    Extract patient ID from file path.
    Pattern: {patient_id}_s{session}_t{trial}.h5
    Example: aaaaaaac_s001_t000.h5 -> aaaaaaac
    """
    base_name = os.path.basename(file_path)
    # Remove extension
    base_name = base_name.replace('.h5', '').replace('.edf', '')
    # Extract patient ID (everything before first underscore)
    patient_id = base_name.split('_')[0]
    return patient_id


def patient_wise_split_file_paths(file_paths, test_size=0.2, random_state=42):
    """
    Split file paths by patient, not by file.
    
    Args:
        file_paths: List of file path dictionaries (with 'h5_path' key)
        test_size: Proportion of patients to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        train_files, val_files: Lists of file dicts for training and validation
        train_patients, val_patients: Lists of patient IDs for each split
    """
    # Extract patient IDs from file paths
    patient_ids = []
    for file_dict in file_paths:
        h5_path = file_dict.get('h5_path', '')
        patient_id = extract_patient_id_from_path(h5_path)
        patient_ids.append(patient_id)
    
    # Get unique patients
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    n_val_patients = max(1, int(n_patients * test_size))
    
    # Split patients
    np.random.seed(random_state)
    val_patients = set(np.random.choice(unique_patients, n_val_patients, replace=False))
    train_patients = [p for p in unique_patients if p not in val_patients]
    
    # Assign files based on patient
    train_files = []
    val_files = []
    
    for file_dict, p in zip(file_paths, patient_ids):
        if p in train_patients:
            train_files.append(file_dict)
        else:
            val_files.append(file_dict)
    
    return train_files, val_files, train_patients, list(val_patients)


def validate_patient_split_no_leakage(train_files, val_files):
    """Ensure no patient appears in both train and validation sets."""
    train_patients = set()
    val_patients = set()
    
    for f in train_files:
        patient_id = extract_patient_id_from_path(f.get('h5_path', ''))
        train_patients.add(patient_id)
    
    for f in val_files:
        patient_id = extract_patient_id_from_path(f.get('h5_path', ''))
        val_patients.add(patient_id)
    
    overlap = train_patients.intersection(val_patients)
    
    if overlap:
        raise ValueError(f"Data leakage detected! Patients {overlap} appear in both sets.")
    
    print(f"✓ No data leakage: {len(train_patients)} train patients, {len(val_patients)} val patients, overlap=0")
    return True

class EEGProcessor:
    def __init__(self, root_directory, resampled_freq=200, time_step_size=1, apply_fft=True, max_files=8000):
        self.root_directory = root_directory
        print(f"EEGProcessor initialized with root_directory: {self.root_directory}")  # Debug print
        self.resampled_freq = resampled_freq
        self.time_step_size = time_step_size
        self.apply_fft = apply_fft
        self.max_files = max_files  # Default maximum number of files to process
        self.included_channels = [
#     "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
#     "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
#     "FZ-CZ", "CZ-PZ", "P7-T5" #"P8-T6"
# ]
        
        
        

         'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
        ]
    
    def convert_edf_to_h5(self):
        failed_files = []
        successfully_converted = []
        total_edf_files = 0
        
        for root, _, files in os.walk(self.root_directory):
            edf_files = [f for f in files if f.endswith(".edf")]
            total_edf_files += len(edf_files)
            for file in edf_files:
                edf_path = os.path.join(root, file)
                h5_path = os.path.join(root, file.replace(".edf", ".h5"))
                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=True)
                    eeg_data = raw.get_data().astype(np.float32)  # Optimize memory
                    original_freq = raw.info["sfreq"]
                    
                    if original_freq != self.resampled_freq:
                        raw.resample(self.resampled_freq)
                        eeg_data = raw.get_data().astype(np.float32)
                    
                    with h5py.File(h5_path, "w") as f:
                        f.create_dataset("resampled_signal", data=eeg_data, compression="gzip", compression_opts=4, chunks=True)
                        f.attrs["resample_freq"] = float(self.resampled_freq)  # Store as attribute
                        f.create_dataset("channel_names", data=np.array(raw.ch_names, dtype="S"))
                    
                    successfully_converted.append(edf_path)
                    print(f"Processed and saved: {h5_path}")
                except Exception as e:
                    print(f"Failed to process {edf_path}: {e}")
                    failed_files.append((edf_path, str(e)))
        
        print(f"Total EDF files found: {total_edf_files}")
        print(f"Successfully converted: {len(successfully_converted)} files")
        if failed_files:
            print(f"Failed to process {len(failed_files)} files. Details: {failed_files}")
            with open(os.path.join(self.root_directory, "failed_conversions.log"), "w") as log_file:
                for edf_path, error in failed_files:
                    log_file.write(f"{edf_path}: {error}\n")
        else:
            print("All EDF files processed successfully.")

        return successfully_converted, failed_files
    
    def _process_edf_file(self, edf_path, h5_path):
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            eeg_data = raw.get_data().astype(np.float32)  # Optimize memory
            original_freq = raw.info["sfreq"]
            
            if original_freq != self.resampled_freq:
                raw.resample(self.resampled_freq)
                eeg_data = raw.get_data().astype(np.float32)
            
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("resampled_signal", data=eeg_data, compression="gzip", compression_opts=4, chunks=True)
                f.attrs["resample_freq"] = float(self.resampled_freq)  # Store as attribute
                f.create_dataset("channel_names", data=np.array(raw.ch_names, dtype="S"))
            
            print(f"Processed and saved: {h5_path}")
        except Exception as e:
            print(f"Failed to process {edf_path}: {e}")
            raise  # Re-raise to allow caller to handle
    
    def compute_fft(self, signals, n):
        fourier_signal = fft(signals, n=n, axis=-1)
        idx_pos = int(np.floor(n / 2))
        fourier_signal = fourier_signal[:, :idx_pos]
        amplitude = np.abs(fourier_signal)
        amplitude[amplitude == 0.0] = 1e-8  # Avoid log of 0
        return np.log(amplitude), np.angle(fourier_signal)
    
    def process_h5_and_csv(self, h5_path, csv_path):
        try:
            with open(csv_path, 'r', encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()
                total_duration = next((float(line.split('=')[1].replace("secs", "").strip())
                                    for line in lines if line.startswith("# duration =")), None)
        
            df = pd.read_csv(csv_path, header=5, on_bad_lines='skip')
            grouped_df = df.groupby(["start_time", "stop_time", "label"], as_index=False)
            grouped_df = grouped_df.agg({"channel": lambda x: list(x)})
            grouped_df["channel_count"] = grouped_df["channel"].apply(len)
            
            top5_durations_df = grouped_df[grouped_df["channel_count"] >= 5]
            top5_durations_df = top5_durations_df.nlargest(5, "channel_count").reset_index(drop=True)
            
            with h5py.File(h5_path, 'r') as f:
                signal_array = f["resampled_signal"][()].astype(np.float32)  # Optimize memory
                # Check for resample_freq as dataset or attribute
                if "resample_freq" in f:
                    res_f = float(f["resample_freq"][()])
                else:
                    res_f = float(f.attrs.get("resample_freq", self.resampled_freq))
                if not np.isclose(res_f, self.resampled_freq, atol=1e-3):
                    print(f"Warning: Resampled frequency mismatch ({res_f} vs {self.resampled_freq}) in {h5_path}. Using {res_f}.")
                self.resampled_freq = res_f  # Sync with actual frequency
            
            eeg_clips, clip_labels = [], []
            for _, row in top5_durations_df.iterrows():
                start_time, stop_time, label = float(row["start_time"]), float(row["stop_time"]), row["label"]
                start_sample, end_sample = int(start_time * self.resampled_freq), int(stop_time * self.resampled_freq)
                end_sample = min(end_sample, signal_array.shape[1])
                
                slice_array = signal_array[:, start_sample:end_sample]
                physical_time_step_size = int(self.resampled_freq * self.time_step_size)
                
                time_steps = []
                for i in range(0, slice_array.shape[1] - physical_time_step_size + 1, physical_time_step_size):
                    slice_segment = slice_array[:, i:i + physical_time_step_size]
                    if slice_segment.shape[1] == physical_time_step_size:
                        if self.apply_fft:
                            time_steps.append(self.compute_fft(slice_segment, n=physical_time_step_size)[0])
                        else:
                            time_steps.append(slice_segment)
                    else:
                        print(f"Skipping short slice for {label} at sample {i} (length {slice_segment.shape[1]} < {physical_time_step_size})")
                
                if time_steps:
                    eeg_clips.append(np.stack(time_steps, axis=0).astype(np.float32))
                    clip_labels.append(label)
            
            return eeg_clips, clip_labels, total_duration, grouped_df, top5_durations_df
        except (OSError, ValueError) as e:
            print(f"Error processing {h5_path}: {e}. Skipping this file.")
            return [], [], None, pd.DataFrame(), pd.DataFrame()  # Return empty results to skip gracefully
    
    def process_directory(self, max_files=None, return_patient_ids=False):
        """
        Returns file paths instead of loading all data into memory.
        
        Args:
            max_files: Maximum number of files to process
            return_patient_ids: If True, also return patient IDs for each file
        
        Returns:
            file_paths: List of file dicts
            patient_ids: (Optional) List of patient IDs
        """
        file_paths = []
        patient_ids_list = []  # NEW
        count = 0

        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith(".edf"):
                    base_name = os.path.splitext(file)[0]
                    h5_path = os.path.join(root, base_name + ".h5")
                    csv_path = os.path.join(root, base_name + ".csv")

                    if os.path.exists(h5_path) and os.path.exists(csv_path):
                        file_paths.append({
                            'base_name': base_name,
                            'h5_path': h5_path,
                            'csv_path': csv_path
                        })
                        patient_ids_list.append(extract_patient_id_from_path(h5_path))  # NEW
                        count += 1
                        if count >= max_files:
                            print(f"Reached limit of {max_files} files. Stopping.")
                            if return_patient_ids:
                                return file_paths, patient_ids_list
                            return file_paths
                    else:
                        print(f"Skipping {base_name}: Missing H5 or CSV file.")

        print(f"Found {count} files for processing.")
        if return_patient_ids:
            return file_paths, patient_ids_list
        return file_paths
