import os
import mne
import h5py
import numpy as np
import pandas as pd
from scipy.fftpack import fft

class EEGProcessor:
    def __init__(self, root_directory, resampled_freq=200, time_step_size=1, apply_fft=False):
        self.root_directory = root_directory
        self.resampled_freq = resampled_freq
        self.time_step_size = time_step_size
        self.apply_fft = apply_fft
        self.included_channels = [
            'A1-T3', 'C3-CZ', 'C3-P3', 'C4-P4', 'C4-T4', 'CZ-C4',
            'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'FP1-F7',
            'FP2-F4', 'FP2-F8', 'P3-O1', 'P4-O2', 'T3-C3', 'T3-T5',
            'T4-A2', 'T4-T6', 'T5-O1', 'T6-O2'
        ]
    
    def convert_edf_to_h5(self):
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith("_cleaned.edf"):
                    edf_path = os.path.join(root, file)
                    h5_path = os.path.join(root, file.replace(".edf", ".h5"))
                    self._process_edf_file(edf_path, h5_path)
    
    def _process_edf_file(self, edf_path, h5_path):
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            eeg_data = raw.get_data()
            original_freq = raw.info["sfreq"]
            
            if original_freq != self.resampled_freq:
                raw.resample(self.resampled_freq)
                eeg_data = raw.get_data()
            
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("resampled_signal", data=eeg_data)
                f.create_dataset("resample_freq", data=self.resampled_freq)
            
            print(f"Processed and saved: {h5_path}")
        except Exception as e:
            print(f"Failed to process {edf_path}: {e}")
    
    def compute_fft(self, signals, n):
        fourier_signal = fft(signals, n=n, axis=-1)
        idx_pos = int(np.floor(n / 2))
        fourier_signal = fourier_signal[:, :idx_pos]
        amplitude = np.abs(fourier_signal)
        amplitude[amplitude == 0.0] = 1e-8  # Avoid log of 0
        return np.log(amplitude), np.angle(fourier_signal)
    
    def process_h5_and_csv(self, h5_path, csv_path):
        with open(csv_path, 'r') as file:
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
            signal_array = f["resampled_signal"][()]
            assert f["resample_freq"][()] == self.resampled_freq, "Resampled frequency mismatch."
        
        eeg_clips, clip_labels = [], []
        for _, row in top5_durations_df.iterrows():
            start_time, stop_time, label = float(row["start_time"]), float(row["stop_time"]), row["label"]
            start_sample, end_sample = int(start_time * self.resampled_freq), int(stop_time * self.resampled_freq)
            end_sample = min(end_sample, signal_array.shape[1])
            
            slice_array = signal_array[:, start_sample:end_sample]
            physical_time_step_size = int(self.resampled_freq * self.time_step_size)
            
            time_steps = [
                self.compute_fft(slice_array[:, i:i+physical_time_step_size], n=physical_time_step_size)[0]
                if self.apply_fft else slice_array[:, i:i+physical_time_step_size]
                for i in range(0, slice_array.shape[1] - physical_time_step_size + 1, physical_time_step_size)
            ]
            
            if time_steps:
                eeg_clips.append(np.stack(time_steps, axis=0))
                clip_labels.append(label)
        
        return eeg_clips, clip_labels, total_duration, grouped_df, top5_durations_df
    
    def process_directory(self):
        results = {}
        max_files = 5000
        count = 0
        
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith(".edf"):
                    base_name = os.path.splitext(file)[0]
                    h5_path = os.path.join(root, base_name + ".h5")
                    csv_path = os.path.join(root, base_name + ".csv")
                    
                    if os.path.exists(h5_path) and os.path.exists(csv_path):
                        print(f"Processing {base_name}...")
                        results[base_name] = self.process_h5_and_csv(h5_path, csv_path)
                        count += 1
                        if count >= max_files:
                            print(f"Reached limit of {max_files} files. Stopping.")
                            return results
                    else:
                        print(f"Skipping {base_name}: Missing H5 or CSV file.")
        
        return results
    
# training main data ko aisy preprocesss karain gy

# # Example usage
# directory = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
# processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
# processor.convert_edf_to_h5()
# results = processor.process_directory()





# import os
# import mne
# import h5py
# import numpy as np

# def process_edf_files(root_directory, resampled_freq=200):
#     for root, dirs, files in os.walk(root_directory):
#         for file in files:
#             if file.endswith(".edf"):
#                 edf_path = os.path.join(root, file)
#                 h5_path = os.path.join(root, file.replace(".edf", ".h5"))
                
#                 try:
#                     # Load the EDF file
#                     raw = mne.io.read_raw_edf(edf_path, preload=True)
#                     eeg_data = raw.get_data()  # Shape: (channels, time)
#                     original_freq = raw.info["sfreq"]
                    
#                     # Resample the data if needed
#                     if original_freq != resampled_freq:
#                         raw.resample(resampled_freq)
#                         eeg_data = raw.get_data()
                    
#                     # Save data into HDF5 file
#                     with h5py.File(h5_path, "w") as f:
#                         f.create_dataset("resampled_signal", data=eeg_data)
#                         f.create_dataset("resample_freq", data=resampled_freq)
                    
#                     print(f"Processed and saved: {h5_path}")
#                 except Exception as e:
#                     print(f"Failed to process {edf_path}: {e}")

# # Example usage
# root_dir = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
# process_edf_files(root_dir)



# import os
# import h5py
# import numpy as np
# import pandas as pd

# FREQUENCY = 200


# from scipy.fftpack import fft

# def computeFFT(signals, n):
#     """
#     Args:
#         signals: EEG signals, (number of channels, number of data points)
#         n: length of positive frequency terms of fourier transform
#     Returns:
#         FT: log amplitude of FFT of signals, (number of channels, number of data points)
#         P: phase spectrum of FFT of signals, (number of channels, number of data points)
#     """
#     # fourier transform
#     fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

#     # only take the positive freq part
#     idx_pos = int(np.floor(n / 2))
#     fourier_signal = fourier_signal[:, :idx_pos]
#     amp = np.abs(fourier_signal)
#     amp[amp == 0.0] = 1e-8  # avoid log of 0

#     FT = np.log(amp)
#     P = np.angle(fourier_signal)

#     return FT, P


# INCLUDED_CHANNELS=['A1-T3',
#  'C3-CZ',
#  'C3-P3',
#  'C4-P4',
#  'C4-T4',
#  'CZ-C4',
#  'F3-C3',
#  'F4-C4',
#  'F7-T3',
#  'F8-T4',
#  'FP1-F3',
#  'FP1-F7',
#  'FP2-F4',
#  'FP2-F8',
#  'P3-O1',
#  'P4-O2',
#  'T3-C3',
#  'T3-T5',
#  'T4-A2',
#  'T4-T6',
#  'T5-O1',
#  'T6-O2']


# import pandas as pd
# import h5py
# import numpy as np

# def computeSliceMatrix(
#     h5_fn,
#     edf_fn,
#     csv_fn,
#     time_step_size=1,
#     is_fft=False
# ):
#     """
#     Reads the CSV file to:
#       1) Parse the total recording duration from line 3 (e.g. "# duration= 301.0 secs")
#       2) Group rows by (start_time, stop_time, label). All rows sharing the same 
#          start_time, stop_time, and label are merged into one group. (Hence one clip.)
#       3) For each group, create an EEG clip from start_time to stop_time 
#          for *all channels* in the H5 file, break it into time steps, and optionally apply FFT.
#       4) Create a second DataFrame with only the top 5 durations (groups) 
#          that have the highest channel count (only including groups with channel_count >= 5),
#          sorted in descending order.
    
#     Only EEG clips (and their labels) corresponding to these top 5 groups are returned.
    
#     Args:
#         h5_fn: Full path of the resampled signal H5 file.
#         edf_fn: Full path of the corresponding EDF file (unused here but kept for consistency).
#         csv_fn: Full path of the CSV file containing intervals.
#         time_step_size: Duration (in seconds) of each smaller time step within the clip.
#         is_fft: Whether to perform FFT on the raw EEG data.
        
#     Returns:
#         eeg_clips: A list of EEG clips (one per group in top5_durations_df). 
#                    Each clip has shape (num_time_steps, num_channels, time_step_size * FREQUENCY).
#         clip_labels: A list of labels corresponding to each returned clip.
#         total_duration: The total recording duration (in seconds) parsed from line 3 of the CSV.
#         grouped_df: A DataFrame showing how rows were grouped by 
#                     (start_time, stop_time, label), including a 'channel' column listing the channels.
#         top5_durations_df: A DataFrame containing the top 5 durations (groups) 
#                            with the highest channel count (only including groups with channel_count >= 5),
#                            sorted in descending order.
#     """

#     # ---------------------------
#     # 1) Parse total duration from line 3
#     # ---------------------------
#     total_duration = None
#     with open(csv_fn, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             if line.startswith("# duration ="):
#                 # e.g. "# duration= 301.0 secs"
#                 parts = line.strip().split('=')
#                 if len(parts) > 1:
#                     duration_str = parts[1].replace("secs", "").strip()
#                     try:
#                         total_duration = float(duration_str)
#                     except ValueError:
#                         total_duration = None
#                 break
    
#     print(f"Total Duration (from CSV line 3): {total_duration} seconds")
    
#     # ---------------------------
#     # 2) Read CSV rows (header=5) and group by (start_time, stop_time, label)
#     # ---------------------------
#     df = pd.read_csv(csv_fn, header=5, on_bad_lines='skip')
#     # Expect columns: channel, start_time, stop_time, label, confidence
#     # Group by (start_time, stop_time, label) so that channels with the same
#     # interval & label form one group.
#     grouped_df = (
#         df
#         .groupby(["start_time", "stop_time", "label"], as_index=False)
#         .agg({"channel": lambda x: list(x)})
#     )
#     # Now grouped_df has columns: start_time, stop_time, label, channel (list of channels)
    
#     # Add a 'channel_count' column to see how many channels are in each group
#     grouped_df["channel_count"] = grouped_df["channel"].apply(len)
    
#     # ---------------------------
#     # 4) Create top5_durations_df filtering only groups with channel_count >= 5
#     # ---------------------------
#     grouped_df_filtered = grouped_df[grouped_df["channel_count"] >= 5]
#     top5_durations_df = grouped_df_filtered.nlargest(5, "channel_count").reset_index(drop=True)
    
#     # ---------------------------
#     # 3) Open the H5 file and load the EEG data
#     # ---------------------------
#     with h5py.File(h5_fn, 'r') as f:
#         signal_array = f["resampled_signal"][()]  # shape: (num_channels, total_samples)
#         resampled_freq = f["resample_freq"][()]
    
#     # Assume FREQUENCY is a global constant defined elsewhere.
#     assert resampled_freq == FREQUENCY, "Resampled frequency does not match expected FREQUENCY."
    
#     # We'll now collect EEG clips and labels only for groups present in top5_durations_df.
#     eeg_clips = []
#     clip_labels = []
    
#     # Iterate over the rows in top5_durations_df (only the top 5 groups meeting the criteria)
#     for idx, row in top5_durations_df.iterrows():
#         start_time = float(row["start_time"])
#         stop_time = float(row["stop_time"])
#         label = row["label"]
#         channels_in_group = row["channel"]  # list of channel names for this group
        
#         duration = stop_time - start_time
#         if duration <= 0:
#             print(f"Skipping group {idx}: Non-positive duration.")
#             continue
        
#         # Convert times to sample indices
#         start_sample = int(start_time * FREQUENCY)
#         end_sample = int(stop_time * FREQUENCY)
#         # Clamp if end_sample is beyond the available data
#         if end_sample > signal_array.shape[1]:
#             end_sample = signal_array.shape[1]
        
#         # Slice the data for all channels (one clip for the group)
#         slice_array = signal_array[:, start_sample:end_sample]
        
#         # Break this slice into smaller time steps
#         physical_time_step_size = int(FREQUENCY * time_step_size)
#         time_steps = []
#         st = 0
#         while st <= slice_array.shape[1] - physical_time_step_size:
#             ed = st + physical_time_step_size
#             curr_time_step = slice_array[:, st:ed]
            
#             if is_fft:
#                 curr_time_step, _ = computeFFT(curr_time_step, n=physical_time_step_size)
            
#             time_steps.append(curr_time_step)
#             st = ed
        
#         if len(time_steps) == 0:
#             print(f"Skipping group {idx}: No time steps in slice.")
#             continue
        
#         # Stack into (num_time_steps, num_channels, physical_time_step_size)
#         eeg_clip = np.stack(time_steps, axis=0)
#         eeg_clips.append(eeg_clip)
#         clip_labels.append(label)
        
#         print(f"Group {idx}: time={start_time}-{stop_time}s, label={label}, "
#               f"channels={channels_in_group}, shape={eeg_clip.shape}, "
#               f"channel_count={len(channels_in_group)}")
    
#     # Return only the clips and labels corresponding to the top5 groups,
#     # along with the total duration, the full grouping DataFrame, and top5_durations_df.
#     return eeg_clips, clip_labels, total_duration, grouped_df, top5_durations_df




# def computeDirectorySliceMatrix(directory, time_step_size=1, is_fft=False):
#     """
#     Process up to 100 EDF/H5/CSV triplets in the given directory.
#     For each EDF file found, it constructs the corresponding H5 and CSV file paths
#     (by replacing the ".edf" extension) and processes them.
    
#     Args:
#         directory: The root directory containing the EEG files.
#         time_step_size: Duration of each time step (in seconds).
#         is_fft: Whether to perform FFT on the raw EEG data.
        
#     Returns:
#         results: A dictionary where keys are base filenames (without extension)
#                  and values are tuples of the form:
#                    (eeg_clips, clip_labels, grouped_df, top5_durations_df, total_duration)
#                  as returned by computeSliceMatrix.
#     """
#     results = {}
#     count = 0        # Keep track of how many files have been processed
#     max_files = 800  # Limit to 100 file triplets

#     # Walk through the entire directory tree.
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".edf"):
#                 edf_fn = os.path.join(root, file)
#                 h5_fn = os.path.join(root, file.replace(".edf", ".h5"))
#                 csv_fn = os.path.join(root, file.replace(".edf", ".csv"))
#                 base = os.path.splitext(file)[0]
                
#                 if os.path.exists(h5_fn) and os.path.exists(csv_fn):
#                     print(f"Processing {base}...")
#                     eeg_clips, clip_labels, total_duration, grouped_df, top5_durations_df = computeSliceMatrix(
#                         h5_fn, edf_fn, csv_fn,
#                         time_step_size=time_step_size,
#                         is_fft=is_fft
#                     )
#                     results[base] = (eeg_clips, clip_labels, grouped_df, top5_durations_df, total_duration)
                    
#                     count += 1
#                     if count >= max_files:
#                         print(f"Reached limit of {max_files} files. Stopping.")
#                         return results
#                 else:
#                     print(f"Skipping {base}: Corresponding H5 or CSV file not found.")
    
#     return results



# directory = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
# results = computeDirectorySliceMatrix(directory, time_step_size=1, is_fft=True)



