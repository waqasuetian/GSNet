import os
import mne
import h5py
import numpy as np
import pandas as pd
from scipy.fftpack import fft

class EEGProcessorPreictal:
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
        amplitude[amplitude == 0.0] = 1e-8
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

        eeg_clips, clip_labels, clip_start_times, clip_stop_times = [], [], [], []
        for _, row in top5_durations_df.iterrows():
            start_time = float(row["start_time"])
            stop_time = float(row["stop_time"])
            label = row["label"]
            start_sample = int(start_time * self.resampled_freq)
            end_sample = min(int(stop_time * self.resampled_freq), signal_array.shape[1])
            slice_array = signal_array[:, start_sample:end_sample]

            physical_time_step_size = int(self.resampled_freq * self.time_step_size)
            time_steps = []

            for i in range(0, slice_array.shape[1] - physical_time_step_size + 1, physical_time_step_size):
                chunk = slice_array[:, i:i+physical_time_step_size]
                if self.apply_fft:
                    chunk, _ = self.compute_fft(chunk, n=physical_time_step_size)
                time_steps.append(chunk)

            if time_steps:
                eeg_clips.append(np.stack(time_steps, axis=0))
                clip_labels.append(label)
                clip_start_times.append(start_time)
                clip_stop_times.append(stop_time)

        return eeg_clips, clip_labels, clip_start_times, clip_stop_times

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
                        eeg_clips, clip_labels, start_times, stop_times = self.process_h5_and_csv(h5_path, csv_path)
                        results[base_name] = (eeg_clips, clip_labels, start_times, stop_times)
                        count += 1
                        if count >= max_files:
                            print(f"Reached limit of {max_files} files. Stopping.")
                            return results
                    else:
                        print(f"Skipping {base_name}: Missing H5 or CSV file.")
        return results
