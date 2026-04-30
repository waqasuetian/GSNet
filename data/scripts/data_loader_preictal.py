import os
import mne
import h5py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class EEGProcessorPreictal:
    """
    Preictal EEG dataloader that:
      - Converts all *.edf to *.h5 (gzip, chunks) with resampling to self.resampled_freq
      - Skips conversion if the corresponding .h5 already exists
      - Extracts per-event clips based on CSV (start_time, stop_time, label, channel)
      - Supports raw windows or rFFT(log-amp) features with Hann window
      - Handles overlap, short-segment padding, top-k selection by duration

    Returns for each file:
        eeg_clips: List[np.ndarray]   # each of shape (T, N, F) for long events, or (1, N, F) if short padded
        clip_labels: List[str]
        clip_start_times: List[float]
        clip_stop_times: List[float]
    """

    def __init__(self,
                 root_directory: str,
                 # core params
                 resampled_freq: float = 200.0,
                 window_sec: float | None = None,
                 overlap: float = 0.0,
                 feature_mode: str | None = None,   # "raw" | "rfft"
                 # event selection
                 min_channels_per_event: int = 1,
                 topk_events_by_duration: int = 5,
                 # legacy aliases (kept for main.py compatibility)
                 time_step_size: float | None = None,
                 apply_fft: bool | None = None,
                 # rfft feature shaping
                 drop_dc_bin: bool = True,          # drop DC so 1s@200Hz -> 100 bins
                 # accept short events by padding
                 pad_short_segments: bool = True,
                 min_short_frac: float = 0.30,
                 # optional channel filter
                 enable_channel_filter: bool = False,
                 included_channels: List[str] | None = None):
        self.root_directory = root_directory
        print(f"EEGProcessorPreictal initialized with root_directory: {self.root_directory}")

        # Map legacy -> new
        if window_sec is None and time_step_size is not None:
            window_sec = float(time_step_size)
        if window_sec is None:
            window_sec = 1.0
        if feature_mode is None:
            if apply_fft is None:
                feature_mode = "raw"
            else:
                feature_mode = "rfft" if apply_fft else "raw"

        # Core config
        self.resampled_freq = float(resampled_freq)
        self.window_sec = float(window_sec)
        self.overlap = float(overlap)
        assert 0.0 <= self.overlap < 1.0, "overlap must be in [0,1)."
        self.feature_mode = feature_mode.lower()
        assert self.feature_mode in {"raw", "rfft"}

        print(f"EEGProcessorPreictal feature_mode: {self.feature_mode}")


        # Event selection
        self.min_channels = int(min_channels_per_event)
        self.topk = int(topk_events_by_duration)

        # rFFT shaping
        self.drop_dc_bin = bool(drop_dc_bin)

        # Short-segment handling
        self.pad_short_segments = bool(pad_short_segments)
        self.min_short_frac = float(min_short_frac)

        # Precompute defaults (we’ll recompute per-file using eff_fs)
        self.win_default = int(round(self.resampled_freq * self.window_sec))
        self.step_default = max(1, int(round(self.win_default * (1.0 - self.overlap))))

        # Channel filter
        self.enable_channel_filter = bool(enable_channel_filter)
        self.included_channels = included_channels or [
          'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
        ]
    def _strip_channel_suffix(self, ch_name: str) -> str:
        """Remove suffix like '-LE', '-REF', etc. from channel name."""
        if '-' in ch_name:
            return ch_name.split('-')[0]
        return ch_name
    # ---------------------------------------------------------------------
    # EDF → H5
    # ---------------------------------------------------------------------
    def convert_edf_to_h5(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        failed_files: List[Tuple[str, str]] = []
        successfully_converted: List[str] = []
        total_edf_files = 0
        skipped_existing = 0

        for root, _, files in os.walk(self.root_directory):
            edf_files = [f for f in files if f.lower().endswith(".edf")]   # process ALL .edf
            total_edf_files += len(edf_files)
            for file in edf_files:
                edf_path = os.path.join(root, file)
                h5_path = os.path.join(root, file[:-4] + ".h5")

                # ✅ Skip conversion if H5 already exists
                if os.path.exists(h5_path):
                    skipped_existing += 1
                    continue

                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                    if float(raw.info["sfreq"]) != self.resampled_freq:
                        raw.resample(self.resampled_freq, verbose=False)
                    eeg_data = raw.get_data().astype("float32")  # (N, T)

                    with h5py.File(h5_path, "w") as f:
                        f.create_dataset("resampled_signal", data=eeg_data,
                                         compression="gzip", compression_opts=4, chunks=True)
                        f.create_dataset("channel_names", data=np.array(raw.ch_names, dtype="S"))
                        f.attrs["resample_freq"] = float(self.resampled_freq)

                    successfully_converted.append(edf_path)
                    print(f"Processed and saved: {h5_path}")
                except Exception as e:
                    print(f"Failed to process {edf_path}: {e}")
                    failed_files.append((edf_path, str(e)))

        print(f"Total EDF files found: {total_edf_files}")
        print(f"Successfully converted: {len(successfully_converted)} files")
        if skipped_existing:
            print(f"Skipped existing H5 files: {skipped_existing}")
        if failed_files:
            print(f"Failed to process {len(failed_files)} files. Details written to failed_conversions.log")
            try:
                with open(os.path.join(self.root_directory, "failed_conversions.log"), "w") as log_file:
                    for edf_path, error in failed_files:
                        log_file.write(f"{edf_path}: {error}\n")
            except Exception:
                pass
        else:
            print("All EDF files processed successfully or already present.")

        return successfully_converted, failed_files

    def _process_edf_file(self, edf_path: str, h5_path: str) -> None:
        # Used for on-the-fly conversion if a specific H5 is missing
        if os.path.exists(h5_path):
            return
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if float(raw.info["sfreq"]) != self.resampled_freq:
                raw.resample(self.resampled_freq, verbose=False)
            eeg_data = raw.get_data().astype("float32")  # (N, T)

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("resampled_signal", data=eeg_data,
                                 compression="gzip", compression_opts=4, chunks=True)
                f.create_dataset("channel_names", data=np.array(raw.ch_names, dtype="S"))
                f.attrs["resample_freq"] = float(self.resampled_freq)

            print(f"Processed and saved: {h5_path}")
        except Exception as e:
            print(f"Failed to process {edf_path}: {e}")
            raise

    # ---------------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------------
    def _rfft_logamp(self, signals: np.ndarray) -> np.ndarray:
        """
        signals: (N, win) -> returns (N, F)
        With 1s@200Hz: raw F=101; if drop_dc_bin=True, returns 100 (bins 1..100).
        """
        n = int(signals.shape[-1])
        if n <= 0:
            raise ValueError("Window length must be positive.")
        window = np.hanning(n).astype(signals.dtype)
        X = np.fft.rfft(signals * window[None, :], n=n, axis=-1)   # (N, n//2+1)
        amp = np.log(np.maximum(np.abs(X), 1e-8))
        if self.drop_dc_bin and amp.shape[-1] > 1:
            amp = amp[:, 1:]  # drop DC
        return amp
    def process_file(self, file_path: str) -> Dict[str, Tuple[List[np.ndarray], List[str], List[float], List[float]]]:
        """
        Process a single H5 file and its corresponding CSV file.
        Args:
            file_path (str): Path to the H5 file.
        Returns:
            Dict[str, Tuple[List[np.ndarray], List[str], List[float], List[float]]]:
                Dictionary with base filename as key and tuple of (eeg_clips, clip_labels, clip_start_times, clip_stop_times).
        """
        import os

        if not file_path.lower().endswith(".h5"):
            print(f"Skipping {file_path}: Not an H5 file.")
            return {}

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_path = os.path.join(os.path.dirname(file_path), base_name + ".csv")
        edf_path = os.path.join(os.path.dirname(file_path), base_name + ".edf")

        if not os.path.exists(file_path):
            print(f"H5 missing for {base_name}; attempting on-the-fly conversion...")
            try:
                self._process_edf_file(edf_path, file_path)
            except Exception as e:
                print(f"Conversion failed for {base_name}: {e}")
                return {}

        if not os.path.exists(csv_path):
            print(f"Skipping {base_name}: CSV not found at {csv_path}")
            return {}

        try:
            print(f"Processing {base_name}...")
            result = self.process_h5_and_csv(file_path, csv_path)
            # Validate the output is a 4-element tuple
            if not isinstance(result, tuple) or len(result) != 4:
                print(f"Error: process_h5_and_csv for {base_name} returned invalid format: {result}")
                return {}
            eeg_clips, clip_labels, clip_start_times, clip_stop_times = result
            # Ensure all elements are lists
            if not (isinstance(eeg_clips, list) and isinstance(clip_labels, list) and 
                    isinstance(clip_start_times, list) and isinstance(clip_stop_times, list)):
                print(f"Error: Invalid data types for {base_name}: clips={type(eeg_clips)}, "
                    f"labels={type(clip_labels)}, start_times={type(clip_start_times)}, "
                    f"stop_times={type(clip_stop_times)}")
                return {}
            # Warn if data is empty
            if not (eeg_clips and clip_labels and clip_start_times and clip_stop_times):
                print(f"Warning: Empty data for {base_name}: clips={len(eeg_clips)}, "
                    f"labels={len(clip_labels)}, start_times={len(clip_start_times)}, "
                    f"stop_times={len(clip_stop_times)}")
                return {}
            # Ensure lengths match
            if not (len(eeg_clips) == len(clip_labels) == len(clip_start_times) == len(clip_stop_times)):
                print(f"Error: Mismatched lengths for {base_name}: clips={len(eeg_clips)}, "
                    f"labels={len(clip_labels)}, start_times={len(clip_start_times)}, "
                    f"stop_times={len(clip_stop_times)}")
                return {}
            print(f"Success: Processed {base_name} with {len(eeg_clips)} clips")
            return {base_name: (eeg_clips, clip_labels, clip_start_times, clip_stop_times)}
        except Exception as e:
            print(f"Error while processing {base_name}: {e} (skipping)")
            return {}

    # ---------------------------------------------------------------------
    # H5 + CSV → clips
    # ---------------------------------------------------------------------
    def process_h5_and_csv(self, h5_path: str, csv_path: str
                           ) -> Tuple[List[np.ndarray], List[str], List[float], List[float]]:
        # Optional duration from header lines
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("# duration"):
                        # kept for possible logging; not otherwise used
                        _ = float(line.split("=")[1].replace("secs", "").strip())
                        break
        except Exception:
            pass

        # Robust CSV parse
        df = pd.read_csv(csv_path, comment="#", on_bad_lines="skip")
        required = {"start_time", "stop_time", "label", "channel"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")

        # Normalize / coerce
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["stop_time"] = pd.to_numeric(df["stop_time"], errors="coerce")
        df = df.dropna(subset=["start_time", "stop_time"])
        df["duration"] = df["stop_time"] - df["start_time"]
        df = df[df["duration"] > 0]

        print(f"[{os.path.basename(csv_path)}] label_counts={df['label'].value_counts().to_dict()}")

        grouped_df = (
            df.groupby(["start_time", "stop_time", "label"], as_index=False)
              .agg(channel=("channel", list),
                   channel_count=("channel", "size"),
                   duration=("duration", "first"))
        )

        top_df = (
            grouped_df.query("channel_count >= @self.min_channels")
                      .nlargest(self.topk, "duration")
                      .reset_index(drop=True)
        )

        # Load H5 arrays
        # with h5py.File(h5_path, "r") as f:
        #     signal_array = f["resampled_signal"][()]
        #     signal_array = signal_array.astype(np.float32, copy=False)
        #     file_res_f = float(f.attrs.get("resample_freq", self.resampled_freq))
        #     ch_names = f["channel_names"][()]
        #     ch_names = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in ch_names]

        # Load H5 arrays
        with h5py.File(h5_path, 'r') as f:
            signal_array = f["resampled_signal"][()].astype(np.float32, copy=False)
            file_res_f = float(f.attrs.get("resample_freq", self.resampled_freq))
            ch_names = f["channel_names"][()]
            ch_names = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in ch_names]

        # Strip 'EEG ' prefix and remove suffix
        ch_names = [name.replace('EEG ', '') for name in ch_names]
        ch_names = [name.split('-')[0] if '-' in name else name for name in ch_names]   # remove suffix
                # Effective sampling for THIS file only
        if not np.isclose(file_res_f, self.resampled_freq, atol=1e-3):
            print(f"Warning: Resampled frequency mismatch ({file_res_f} vs {self.resampled_freq}) in {os.path.basename(h5_path)}. Using file value.")
        eff_fs = file_res_f

        # Optional channel filter
        if self.enable_channel_filter and self.included_channels:
            pick_map = {ch: i for i, ch in enumerate(ch_names)}
            idx = [pick_map[ch] for ch in self.included_channels if ch in pick_map]
            if len(idx) > 0:
                signal_array = signal_array[np.array(idx)]
                ch_names = [ch_names[i] for i in idx]
            else:
                print("Channel filter enabled but none of the included channels were found; using all channels.")

        # Compute per-file window & step
        win = int(round(eff_fs * self.window_sec))
        step = max(1, int(round(win * (1.0 - self.overlap))))

        eeg_clips: List[np.ndarray] = []
        clip_labels: List[str] = []
        clip_start_times: List[float] = []
        clip_stop_times: List[float] = []

        for _, row in top_df.iterrows():
            start_time = float(row["start_time"])
            stop_time = float(row["stop_time"])
            label = str(row["label"])

            start_sample = int(round(start_time * eff_fs))
            end_sample = min(int(round(stop_time * eff_fs)), signal_array.shape[1])

            L = end_sample - start_sample
            if L <= 0:
                continue

            # Short event path
            # Short event path
            if L < win:
                if not (self.pad_short_segments and L >= int(self.min_short_frac * win)):
                    continue
                slice_array = signal_array[:, start_sample:end_sample]  # (N, L)
                pad = win - slice_array.shape[1]
                if pad > 0:
                    slice_array = np.pad(slice_array, ((0, 0), (0, pad)), mode="constant")
                
                # CORRECTED: Apply rFFT only if feature_mode == "rfft"
                if self.feature_mode == "rfft":
                    feats = self._rfft_logamp(slice_array)  # (N, F)
                else:
                    feats = slice_array  # (N, win) raw signals
                
                eeg_clips.append(feats[np.newaxis, ...].astype(np.float32))  # (1, N, F) or (1, N, win)
                clip_labels.append(label)
                clip_start_times.append(start_time)
                clip_stop_times.append(stop_time)
                continue

            # Long event: rolling windows with overlap
# Long event: rolling windows with overlap
            slice_array = signal_array[:, start_sample:end_sample]  # (N, L)
            time_steps = []
            for i in range(0, slice_array.shape[1] - win + 1, step):
                chunk = slice_array[:, i:i + win]
                if chunk.shape[1] < win:
                    chunk = np.pad(chunk, ((0, 0), (0, win - chunk.shape[1])), mode="constant")
                
                # CORRECTED: Apply rFFT only if feature_mode == "rfft"
                if self.feature_mode == "rfft":
                    feats = self._rfft_logamp(chunk)  # (N, F)
                else:
                    feats = chunk  # (N, win) raw signals
                
                time_steps.append(feats.astype(np.float32))
            if time_steps:
                eeg_clips.append(np.stack(time_steps, axis=0))  # (T, N, F) or (T, N, win)
                clip_labels.append(label)
                clip_start_times.append(start_time)
                clip_stop_times.append(stop_time)

            # print(eeg_clips)
            # print(clip_labels)
            # print(clip_start_times)
            # print(clip_stop_times)

        return eeg_clips, clip_labels, clip_start_times, clip_stop_times
  

    # --- Traditional dictionary mode (unchanged) ---
    # ... existing code for dictionary mode ...

    def compute_correlation_matrix(self, h5_path: str, desired_channels: List[str]) -> np.ndarray:
        """
        Load raw EEG signal from H5 file, compute Pearson correlation matrix,
        and return a matrix of size len(desired_channels) x len(desired_channels)
        by mapping the channels present in the file to the desired channel list.
        Missing channels are set to zero correlation.
        """
        while isinstance(desired_channels, list) and len(desired_channels) > 0 and isinstance(desired_channels[0], list):
            desired_channels = [item for sublist in desired_channels for item in sublist]
    # Convert all items to strings (safety)
        desired_channels = [str(ch) for ch in desired_channels]
        import h5py
        import numpy as np
        
        # with h5py.File(r"F:\tuh_data\train\aaaaaaac\s001_2002\02_tcp_le\aaaaaaac_s001_t000.h5", 'r') as f:
        #     ch = [c.decode() for c in f['channel_names'][()]]
        # print(ch)

        # Load raw signal and channel names from H5
        with h5py.File(h5_path, 'r') as f:
            signal = f['resampled_signal'][()].astype(np.float32)
            ch_names = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in f['channel_names'][()]]

        # Normalize channel names: remove 'EEG ' prefix and suffix
        ch_names = [name.replace('EEG ', '') for name in ch_names]
        ch_names = [name.split('-')[0] if '-' in name else name for name in ch_names]
        # Map channel name → index in the signal array
        name_to_idx = {name: i for i, name in enumerate(ch_names)}
       # print(name_to_idx)

        # Identify which desired channels are present
        present = [ch_names for ch_names in desired_channels if ch_names in name_to_idx]
        
        missing = [ch_names for ch_names in desired_channels if ch_names not in name_to_idx]

        n = len(desired_channels)
        corr_full = np.zeros((n, n), dtype=np.float32)

        if present:
            # Extract the signal for the present channels
            indices = [name_to_idx[ch_names] for ch_names in present]
            signal_present = signal[indices]                     # (n_present, time)

            # Compute correlation matrix for these channels
            corr_present = np.corrcoef(signal_present)           # (n_present, n_present)

            # Fill the full matrix at the positions corresponding to the present channels
            for i, ch_i in enumerate(present):
                for j, ch_j in enumerate(present):
                    corr_full[desired_channels.index(ch_i), desired_channels.index(ch_j)] = corr_present[i, j]

        # Zero out diagonal (no self‑connections)
        np.fill_diagonal(corr_full, 0.0)

        if missing:
            print(f"Missing channels in {h5_path}: {missing} (set to zero correlation)")
        
          # ADD THIS DEBUG PRINT
        # print(f"Returning correlation matrix for {h5_path}:")
        # print(f"  Shape: {corr_full.shape}")
        # print(f"  Non-zero entries: {np.count_nonzero(corr_full)}")
        # print(f"  Min value: {corr_full.min():.4f}, Max value: {corr_full.max():.4f}")
        # print(f"  First 5 rows, first 5 columns:\n{corr_full[:5, :5]}")

        return corr_full.astype(np.float32)
        # ----------------------------------``-----------------------------------
    # Directory driver
    # ---------------------------------------------------------------------
    def process_directory(self, max_files: int | None = None
                          ) -> Dict[str, Tuple[List[np.ndarray], List[str], List[float], List[float]]]:
        """
        NEW: Returns file paths instead of loading all data into memory.
        This prevents RAM crashes with large datasets.
        """
        file_paths = []  # Store paths instead of data
        count = 0

        print("Ensuring all EDF files are converted to H5...")
        converted, failed = self.convert_edf_to_h5()
        failed_set = {os.path.basename(p) for (p, _) in failed}

        for root, _, files in os.walk(self.root_directory):
            edf_files = [f for f in files if f.lower().endswith(".edf")]
            for file in edf_files:
                if max_files is not None and count >= max_files:
                    print(f"Reached max_files limit ({max_files}). Stopping directory traversal.")
                    return file_paths

                base_name = os.path.splitext(file)[0]
                if file in failed_set:
                    print(f"Skipping {base_name}: marked as failed during conversion.")
                    continue

                h5_path = os.path.join(root, base_name + ".h5")
                csv_path = os.path.join(root, base_name + ".csv")

                # Ensure H5 exists (on-the-fly conversion if missing)
                if not os.path.exists(h5_path):
                    print(f"H5 missing for {base_name}; attempting on-the-fly conversion...")
                    try:
                        self._process_edf_file(os.path.join(root, base_name + ".edf"), h5_path)
                    except Exception as e:
                        print(f"Conversion failed for {base_name}: {e}")
                        continue

                # CSV required
                if not os.path.exists(csv_path):
                    print(f"Skipping {base_name}: CSV not found at {csv_path}")
                    continue

                # Store paths only, don't load data yet
                file_paths.append({
                    'base_name': base_name,
                    'h5_path': h5_path,
                    'csv_path': csv_path
                })
                count += 1

        print(f"Found {count} files for processing.")
        return file_paths
