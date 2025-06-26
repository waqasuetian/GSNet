import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, butter

class EEGFilter:
    def __init__(self, train_directory):
        self.train_directory = train_directory

    def load_edf(self, file_path):
        return mne.io.read_raw_edf(file_path, preload=True)

    def plot_eeg(self, raw, title="EEG Signals"):
        raw.plot(scalings='auto', title=title)
        plt.show()

    def apply_notch_filter(self, data, fs, freq=50.0, quality=30.0):
        b, a = iirnotch(freq, quality, fs)
        return filtfilt(b, a, data, axis=1)

    def apply_bandpass_filter(self, data, fs, low=1.0, high=40.0, order=4):
        b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
        return filtfilt(b, a, data, axis=1)

    def save_to_edf(self, cleaned_data, raw, output_file):
        new_raw = mne.io.RawArray(cleaned_data, raw.info)
        new_raw.export(output_file, fmt='edf', overwrite=True)

    def process_eeg(self, input_edf, output_edf):
        raw = self.load_edf(input_edf)
        fs = raw.info['sfreq']
        
        # Apply filters
        notch_filtered = self.apply_notch_filter(raw.get_data(), fs)
        cleaned_data = self.apply_bandpass_filter(notch_filtered, fs)
        
        # Save filtered EEG
        self.save_to_edf(cleaned_data, raw, output_edf)
        print(f"Processed and saved: {output_edf}")

    def process_directory(self):
        for root, _, files in os.walk(self.train_directory):
            for file in files:
                if file.endswith(".edf"):
                    input_edf = os.path.join(root, file)
                    output_edf = os.path.join(root, file.replace(".edf", "_cleaned.edf"))
                    self.process_eeg(input_edf, output_edf)

# Example usage
if __name__ == "__main__":
    train_directory = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
    eeg_filter = EEGFilter(train_directory)
    eeg_filter.process_directory()
