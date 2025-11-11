import re
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

_ID_RE = re.compile(r"^(?P<pid>[a-z0-9]+)_s(?P<sess>\d+)_t(?P<trial>\d+)$")


def _parse_base_key(base: str):
    """
    Parse keys like 'aaaaaaac_s001_t000' -> ('aaaaaaac', '001', 0).
    If it doesn't match, return (base, '000', 0) to isolate it.
    """
    m = _ID_RE.match(base)
    if not m:
        return (base, "000", 0)
    pid = m.group("pid")
    sess = m.group("sess")
    trial = int(m.group("trial"))
    return (pid, sess, trial)


class preictal_dataLoader:
    def __init__(self, pooled_results_dictionary, target_time_points=100, num_nodes=19,
                 early_reg=True, early_label=False,
                 allow_intermediate_labels=None, max_gap_between_bckg_and_ictal_sec=0.0,
                 min_preictal_clip_sec=0.0,
                 preictal_strategy: str = "auto",            # "adjacent" | "window" | "auto"
                 preictal_window_sec: float = 600.0,         # 10 min default
                 # stabilize skewed regression targets
                 reg_target_log: bool = True,
                 # optionally expand window in "auto" if adjacent+window find nothing
                 auto_expand_windows: tuple = (),            # e.g., (1200.0, 1800.0, 3600.0)
                 lazy_loading: bool = False,                 # NEW: Enable lazy loading
                 processor=None,                             # NEW: Required for lazy loading
                 pooler=None):                               # NEW: Required for lazy loading
        """
        early_reg=True  -> regression on time-to-seizure (seconds)
        early_label=True -> classification of upcoming ictal type

        preictal_strategy:
          - "adjacent": only bckg immediately preceding ictal (original behavior)
          - "window": any bckg whose STOP is within preictal_window_sec of NEXT ictal START in same session
          - "auto": try "adjacent"; if a session yields 0 preictal, fall back to "window"
                    (optionally expanding the window using auto_expand_windows)

        Notes:
        - This loader expects pooled_results_dictionary keyed by base like '<pid>_s###_t###'.
        - Labels are normalized to lowercase and stripped.
        """
        if early_reg == early_label:
            raise ValueError("Exactly one of early_reg or early_label must be True.")

        # NEW: Check if lazy loading mode
        self.lazy_loading = lazy_loading
        self.processor = processor
        self.pooler = pooler

        if self.lazy_loading and isinstance(pooled_results_dictionary, list):
            print("[LAZY MODE] Preictal data loader initialized with file paths")
            self.file_paths = pooled_results_dictionary
            self.results = {}  # Will be populated incrementally
        else:
            self.results = pooled_results_dictionary
            self.file_paths = None

        self.target_time_points = int(target_time_points)
        self.num_nodes = int(num_nodes)
        self.early_reg = bool(early_reg)
        self.early_label = bool(early_label)

        self.allow_intermediate = set(allow_intermediate_labels or [])
        self.max_gap_sec = float(max_gap_between_bckg_and_ictal_sec)
        self.min_preictal_sec = float(min_preictal_clip_sec)
        self.preictal_strategy = preictal_strategy.lower()
        assert self.preictal_strategy in {"adjacent", "window", "auto"}
        self.preictal_window_sec = float(preictal_window_sec)

        self.reg_target_log = bool(reg_target_log)
        self.auto_expand_windows = tuple(float(x) for x in auto_expand_windows)

        self.target_labels = {"gnsz", "fnsz", "tnsz", "absz", "mysz", "cpsz", "tnsz"}

        # outputs
        self.y_scaler = None
        self.label_to_numeric = {"gnsz": 0, "fnsz": 1, "tcsz": 2, "absz": 3, "mysz": 4, "cpsz":5, "tnsz": 6}

        # NEW: Process data differently based on mode
        if self.lazy_loading and self.file_paths is not None:
            print("[LAZY MODE] Will process files incrementally in get_data()")
            self.filtered_results = {}
            self.preictal_dict = {}
        else:
            # 1) normalize & sort per base
            self.filtered_results = self._filter_and_sort()

            # 2) build preictal dict with cross-file session awareness
            self.preictal_dict = self._generate_preictal_dict()

            if self.early_reg:
                self.x_reg, self.y_reg = self._prepare(mode="early_reg")
            else:
                self.x_clf, self.y_clf = self._prepare(mode="early_label")

    # ... (rest of the code remains the same)
    def _filter_and_sort(self):
        filtered = {}
        for key, value in self.results.items():
            if not (isinstance(value, tuple) and len(value) == 4):
                raise ValueError(f"Expected tuple of length 4 for key '{key}', got {type(value)}")
            clips, labels, start_times, stop_times = value

            def ensure_list(v):
                if isinstance(v, list):
                    return v
                return [v]

            clips = ensure_list(clips)
            labels = [str(l).strip().lower() for l in ensure_list(labels)]  # normalize here
            start_times = list(map(float, ensure_list(start_times)))
            stop_times = list(map(float, ensure_list(stop_times)))

            if len(clips) >= 1:
                combined = list(zip(clips, labels, start_times, stop_times))
                combined_sorted = sorted(combined, key=lambda x: x[2])  # by start_time
                sorted_clips, sorted_labels, sorted_starts, sorted_stops = zip(*combined_sorted)
                filtered[key] = (
                    list(sorted_clips),
                    list(sorted_labels),
                    list(sorted_starts),
                    list(sorted_stops)
                )
        return filtered

    def _session_groups(self):
        """
        Group bases into (pid, sess) and order by trial -> (global timeline per session).
        Creates a list of events with global_start/global_stop (synthetic), so bckg in t000 can precede ictal in t001.
        """
        # Map session -> list of (base, trial, clips, labels, starts, stops)
        sess_map = {}
        for base, (clips, labels, starts, stops) in self.filtered_results.items():
            pid, sess, trial = _parse_base_key(base)
            sess_key = (pid, sess)
            sess_map.setdefault(sess_key, []).append((trial, base, clips, labels, starts, stops))

        # Order by trial within each session, and build global time offsets by concatenation
        session_events = {}
        for sess_key, entries in sess_map.items():
            entries.sort(key=lambda x: x[0])  # by trial
            global_events = []
            offset = 0.0
            for trial, base, clips, labels, starts, stops in entries:
                file_len = max(stops) if stops else 0.0
                for clip, lab, st, sp in zip(clips, labels, starts, stops):
                    global_events.append({
                        "base": base,
                        "clip": clip,
                        "label": lab,
                        "start": st,
                        "stop": sp,
                        "gstart": offset + st,
                        "gstop": offset + sp
                    })
                offset += file_len
            global_events.sort(key=lambda e: e["gstart"])
            session_events[sess_key] = global_events
        return session_events

    def _adjacent_rule(self, events):
        """
        Contiguous 'bckg' blocks immediately before an ictal become 'bckg preictal'.
        """
        labels = [e["label"] for e in events]
        new_labels = labels.copy()
        n = len(labels)
        i = 0
        while i < n:
            if labels[i] == "bckg":
                j = i + 1
                while j < n and labels[j] == "bckg":
                    j += 1
                # allow small intermediates (e.g., 'artf') between bckg and ictal
                k = j
                gap_ok = True
                while k < n and labels[k] not in self.target_labels:
                    if labels[k] in self.allow_intermediate:
                        if self.max_gap_sec > 0.0 and k > 0:
                            gap = max(0.0, events[k]["start"] - events[k - 1]["stop"])
                            if gap > self.max_gap_sec:
                                gap_ok = False
                                break
                        k += 1
                    else:
                        gap_ok = False
                        break
                if k < n and labels[k] in self.target_labels and gap_ok:
                    for t in range(i, j):
                        new_labels[t] = "bckg preictal"
                    new_labels[k] = f"{labels[k]} ictal"
                    i = k + 1
                    continue
            i += 1
        return new_labels

    def _window_rule(self, events):
        """
        Mark 'bckg' whose STOP is within preictal_window_sec of NEXT ictal START.
        Safer: don't mark backgrounds that straddle a previous ictal.
        """
        labels = [e["label"] for e in events]
        new_labels = labels.copy()

        # Next ictal start/type per index (scan backwards)
        next_ictal_start = [None] * len(events)
        next_ictal_type = [None] * len(events)
        next_start = None
        next_type = None
        for i in range(len(events) - 1, -1, -1):
            if labels[i] in self.target_labels:
                next_start = events[i]["gstart"]
                next_type = labels[i]
            next_ictal_start[i] = next_start
            next_ictal_type[i] = next_type

        # Previous ictal stop per index (scan forwards)
        prev_ictal_stop = [None] * len(events)
        prev_stop = None
        for i in range(len(events)):
            if labels[i] in self.target_labels:
                prev_stop = events[i]["gstop"]
            prev_ictal_stop[i] = prev_stop

        for i, e in enumerate(events):
            if labels[i] == "bckg" and next_ictal_start[i] is not None:
                dist = next_ictal_start[i] - e["gstop"]  # distance from bckg STOP to next ictal START
                # don't mark if this bckg crosses a previous ictal
                crosses_prev_ictal = (prev_ictal_stop[i] is not None and e["gstart"] < prev_ictal_stop[i] < e["gstop"])
                if 0.0 <= dist <= self.preictal_window_sec and not crosses_prev_ictal:
                    new_labels[i] = "bckg preictal"
            elif labels[i] in self.target_labels:
                new_labels[i] = f"{labels[i]} ictal"
        return new_labels

    def _generate_preictal_dict(self):
        preictal_dict = {}
        session_events = self._session_groups()

        for sess_key, events in session_events.items():
            # Choose strategy
            if self.preictal_strategy == "adjacent":
                new_labels = self._adjacent_rule(events)
            elif self.preictal_strategy == "window":
                new_labels = self._window_rule(events)
            else:  # auto
                new_labels = self._adjacent_rule(events)
                if "bckg preictal" not in new_labels:
                    # try current window
                    new_labels = self._window_rule(events)
                    # optionally expand windows if still nothing
                    if "bckg preictal" not in new_labels and self.auto_expand_windows:
                        original = self.preictal_window_sec
                        for w in self.auto_expand_windows:
                            self.preictal_window_sec = float(w)
                            new_labels = self._window_rule(events)
                            if "bckg preictal" in new_labels:
                                break
                        self.preictal_window_sec = original

            # Build outputs per BASE (keep downstream contract: {base: (...)} )
            by_base = {}
            session_label_counts = {}  # Track what labels we're finding

            for e, lab in zip(events, new_labels):
                base = e["base"]
                by_base.setdefault(base, {"clips": [], "dur": [], "labels": [], "gstart": [], "gstop": []})
                if lab == "bckg preictal":
                    # Upcoming ictal type for early_label
                    next_type = None
                    for f in events:
                        if f["gstart"] >= e["gstop"] and f["label"] in self.target_labels:
                            next_type = f["label"]
                            break
                    if next_type is None:
                        print(f"[DEBUG] Preictal at {base} has no following ictal event! Skipping.")
                        continue

                    # Track labels
                    session_label_counts[next_type] = session_label_counts.get(next_type, 0) + 1

                    # time-to-ictal (seconds) from end of this background segment
                    tti = 0.0
                    for f in events:
                        if f["gstart"] >= e["gstop"] and f["label"] in self.target_labels:
                            tti = max(0.0, f["gstart"] - e["gstop"])
                            break

                    by_base[base]["clips"].append(e["clip"])
                    by_base[base]["dur"].append(tti)
                    by_base[base]["labels"].append(next_type)
                    by_base[base]["gstart"].append(e["gstart"])
                    by_base[base]["gstop"].append(e["gstop"])

            # Merge into preictal_dict (may be empty for some bases)
            for base, dat in by_base.items():
                preictal_dict[base] = {
                    "clips": dat["clips"],
                    "durations": dat["dur"],     # time-to-ictal seconds
                    "labels": dat["labels"],
                    "gstarts": dat["gstart"],
                    "gstops": dat["gstop"],
                }

            # Debug summary per session (use 'dur' here; 'durations' is set only after merge)
            dur_lists = [v["dur"] for v in by_base.values() if v["dur"]]
            total_preictal = sum(len(d) for d in dur_lists)
            if total_preictal:
                flat = [d for lst in dur_lists for d in lst]
                print(
                    f"[session {sess_key[0]}_s{sess_key[1]}] preictal_total={total_preictal} | "
                    f"tti(s): min={min(flat):.2f} max={max(flat):.2f} mean={np.mean(flat):.2f}"
                )
                # Show label distribution for this session
                print(f"  → Upcoming seizure types in this session: {dict(session_label_counts)}")
            else:
                print(f"[session {sess_key[0]}_s{sess_key[1]}] no preictal clips (strategy={self.preictal_strategy})")
        return preictal_dict

    def _prepare(self, mode):
        X_list, Y_list = [], []
        num_nodes_target = self.num_nodes
        scaler = MinMaxScaler()

        # Use predefined label mapping (DON'T create a new one!)
        if mode == "early_label":
            label_to_numeric = self.label_to_numeric.copy()
            seen_labels = set()  # Track which labels actually appear
            print(f"[DEBUG] Using predefined label mapping: {label_to_numeric}")
        else:
            label_to_numeric = {}

        for key, entry in self.preictal_dict.items():
            clips = entry["clips"]
            targets = entry["durations"] if mode == "early_reg" else entry["labels"]
            if len(clips) != len(targets):
                raise ValueError(f"length mismatch for {key}: {len(clips)} clips vs {len(targets)} targets")

            print(f"Processing {key}: {len(clips)} clips, targets={targets[:5] if len(targets) > 5 else targets}")  # Debug

            for clip, target in zip(clips, targets):
                clip = torch.tensor(np.asarray(clip), dtype=torch.float32)
                if clip.ndim != 3:
                    print(f"Warning: clip for {key} is not (T,N,F); got {tuple(clip.shape)}. Skipping.")
                    continue

                T, N, F = clip.shape
                # time to target length
                if T > self.target_time_points:
                    clip = clip[:self.target_time_points]
                elif T < self.target_time_points:
                    padding = torch.zeros((self.target_time_points - T, N, F), dtype=torch.float32)
                    clip = torch.cat([clip, padding], dim=0)
                # channels to target
                if N > num_nodes_target:
                    clip = clip[:, :num_nodes_target, :]
                    N = num_nodes_target
                elif N < num_nodes_target:
                    padding = torch.zeros((self.target_time_points, num_nodes_target - N, F), dtype=torch.float32)
                    clip = torch.cat([clip, padding], dim=1)

                X_list.append(clip)
                if mode == "early_reg":
                    # time-to-ictal in seconds (float)
                    Y_list.append(float(target))
                else:
                    # Use PREDEFINED mapping
                    if target not in label_to_numeric:
                        print(f"⚠️ WARNING: Label '{target}' not in predefined mapping {list(label_to_numeric.keys())}. Skipping this sample.")
                        X_list.pop()  # Remove the clip we just added
                        continue
                    Y_list.append(label_to_numeric[target])
                    seen_labels.add(target)

        if not X_list:
            total_bases = len(self.preictal_dict)
            total_preictal = sum(len(v["clips"]) for v in self.preictal_dict.values())
            label_set = set()
            for v in self.preictal_dict.values():
                label_set.update(v["labels"])
            raise ValueError(
                "No preictal samples built.\n"
                f"- bases seen: {total_bases}\n"
                f"- preictal clips found: {total_preictal}\n"
                f"- upcoming ictal labels seen: {sorted(label_set)}\n"
                "Likely causes: (1) labels aren’t 'bckg'/'fnsz'...'gnsz' after normalization, "
                "(2) each event is shorter than window and padding is off, "
                "(3) filters too strict (min_channels), "
                "(4) no bckg→ictal in same session or naming doesn’t match '<pid>_s###_t###'."
            )

        X = torch.stack(X_list)  # (B, T, N, F)
        if mode == "early_reg":
            y = np.array(Y_list, dtype=float).reshape(-1, 1)
            if self.reg_target_log:
                # small epsilon to avoid log(0); log1p keeps small values well-behaved
                eps = 1e-3
                y = np.log1p(np.maximum(y, 0.0) + eps)
            Y = scaler.fit_transform(y).astype(np.float32).flatten()
            Y = torch.tensor(Y, dtype=torch.float32)
            self.y_scaler = scaler
        else:
            Y = torch.tensor(Y_list, dtype=torch.long)
            # DON'T overwrite self.label_to_numeric - keep the predefined one!
            # self.label_to_numeric stays as the original predefined mapping

            # Print diagnostic info
            from collections import Counter
            label_counts = Counter(Y_list)
            print(f"\n{'='*60}")
            print(f"[DEBUG] Label Statistics for {mode}:")
            print(f"  Predefined mapping: {self.label_to_numeric}")
            print(f"  Labels seen in data: {sorted(seen_labels)}")
            print(f"  Labels NOT seen: {set(label_to_numeric.keys()) - seen_labels}")
            print(f"  Label distribution (by index):")
            for idx in sorted(label_counts.keys()):
                # Find the label name for this index
                label_name = [k for k, v in label_to_numeric.items() if v == idx][0]
                print(f"    {idx} ({label_name}): {label_counts[idx]} samples")
            print(f"  Total samples: {len(Y_list)}")
            print(f"{'='*60}\n")

            if len(seen_labels) < 2:
                print(f"⚠️ WARNING: Only {len(seen_labels)} unique seizure type found: {sorted(seen_labels)}")
                print(f"⚠️ This means all preictal clips are followed by the same seizure type!")
                print(f"⚠️ Possible solutions:")
                print(f"   1. Increase max_files to get more diverse seizure types")
                print(f"   2. Check your CSV files - do they contain different seizure types?")
                print(f"   3. Increase preictal_window_sec to capture more varied transitions")
                print(f"   4. Try preictal_strategy='window' instead of 'adjacent'")

        print(f"Mode {mode}: X shape={tuple(X.shape)}, Y shape={tuple(Y.shape)}, unique numeric values={sorted(set(Y_list))}")
        return X, Y

    def get_data(self):
        # NEW: Process lazily if in lazy mode
        if self.lazy_loading and self.file_paths is not None:
            print("[LAZY MODE] Processing files in batches for preictal data...")
            return self._process_lazy_preictal()

        if self.early_reg:
            return self.x_reg, self.y_reg
        else:
            return self.x_clf, self.y_clf

    def _process_lazy_preictal(self, batch_size=10):
        """
        NEW: Process preictal data in batches to avoid RAM overflow.
        """
        import torch
        X_list, Y_list = [], []
        skipped_batches = 0
        processed_batches = 0

        print(f"[LAZY MODE] Processing {len(self.file_paths)} files in batches of {batch_size}")

        for batch_idx in range(0, len(self.file_paths), batch_size):
            batch_files = self.file_paths[batch_idx:batch_idx + batch_size]
            print(f"[LAZY MODE] Batch {batch_idx//batch_size + 1}/{(len(self.file_paths) + batch_size - 1)//batch_size}")

            # Process batch
            batch_results = {}
            for file_info in batch_files:
                try:
                    base_name = file_info['base_name']
                    h5_path = file_info['h5_path']
                    csv_path = file_info['csv_path']

                    # Load and pool data for this file
                    eeg_clips, labels, start_times, stop_times = self.processor.process_h5_and_csv(h5_path, csv_path)

                    if not eeg_clips:  # Skip if no clips
                        print(f"[LAZY MODE] Skipping {base_name}: No clips found")
                        continue

                    pooled_clips = [self.pooler.adaptive_pool_clip(clip) for clip in eeg_clips]
                    batch_results[base_name] = (pooled_clips, labels, start_times, stop_times)

                except Exception as e:
                    print(f"[LAZY MODE] Error processing {file_info['base_name']}: {e}")
                    continue

            # Skip batch if no valid files
            if not batch_results:
                print(f"[LAZY MODE] Skipping batch {batch_idx//batch_size + 1}: No valid files")
                skipped_batches += 1
                continue

            # Process this batch through the preictal pipeline
            try:
                temp_loader = preictal_dataLoader(
                    batch_results,
                    target_time_points=self.target_time_points,
                    num_nodes=self.num_nodes,
                    early_reg=self.early_reg,
                    early_label=self.early_label,
                    allow_intermediate_labels=self.allow_intermediate,
                    max_gap_between_bckg_and_ictal_sec=self.max_gap_sec,
                    min_preictal_clip_sec=self.min_preictal_sec,
                    lazy_loading=False  # Use traditional mode for batch processing
                )

                batch_X, batch_Y = temp_loader.get_data()
                X_list.append(batch_X)
                Y_list.append(batch_Y)
                processed_batches += 1

                print(f"[LAZY MODE] Batch processed: {batch_X.shape[0]} samples")

            except ValueError as e:
                # Handle batches with no preictal samples
                if "No preictal samples built" in str(e):
                    print(f"[LAZY MODE] Skipping batch {batch_idx//batch_size + 1}: No preictal samples (this is normal for some batches)")
                    skipped_batches += 1
                else:
                    raise

            # Clear batch from memory
            if 'batch_results' in locals():
                del batch_results
            if 'temp_loader' in locals():
                del temp_loader
            if 'batch_X' in locals():
                del batch_X, batch_Y
            import gc
            gc.collect()

        # Check if we got any data
        if not X_list:
            raise ValueError(
                f"No preictal samples found in ANY batch!\n"
                f"- Total batches: {(len(self.file_paths) + batch_size - 1)//batch_size}\n"
                f"- Skipped batches: {skipped_batches}\n"
                f"- Processed batches: {processed_batches}\n\n"
                f"Possible solutions:\n"
                f"1. Increase max_files to get more diverse data\n"
                f"2. Check that CSV files contain both 'bckg' and seizure labels\n"
                f"3. Reduce min_channels_per_event in EEGProcessorPreictal\n"
                f"4. Increase preictal_window_sec (currently using default)\n"
                f"5. Check file naming matches pattern '<patient>_s<session>_t<trial>'"
            )

        # Concatenate all batches
        X_final = torch.cat(X_list, dim=0)
        Y_final = torch.cat(Y_list, dim=0)

        print(f"[LAZY MODE] Final preictal data: X={X_final.shape}, Y={Y_final.shape}")
        print(f"[LAZY MODE] Batches: {processed_batches} successful, {skipped_batches} skipped")
        return X_final, Y_final
