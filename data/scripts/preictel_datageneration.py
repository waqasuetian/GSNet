""" i have updated the pooled_results_dict, now it contains 4 values in the tuple corresponding to each key 
update the code below"""
import torch

class preictal_dataLoader:
    def __init__(self, pooled_results_dictionary, target_time_points=50, num_nodes=19, early_reg=True, early_label=True):
        # print("***********************************************")
        # for key in pooled_results_dictionary:
        #     print(f"Top-level key: {key}")
        #     print("  Inner value type:", type(pooled_results_dictionary[key]))
        #     print("  Inner value contents:", pooled_results_dictionary[key])
        self.results = pooled_results_dictionary
        self.target_time_points = target_time_points
        self.num_nodes = num_nodes
        self.early_reg = early_reg
        self.early_label = early_label

        self.target_labels = {"fnsz", "tcsz", "absz", "cpsz", "seiz", "tnsz", "gnsz"}

        self.filtered_results = self._filter_and_sort()
        self.preictal_dict = self._generate_preictal_dict()

        # Load data based on mode
        if early_reg and not early_label:
            self.x_reg, self.y_reg = self._prepare(mode="early_reg")
        elif early_label and not early_reg:
            self.x_clf, self.y_clf = self._prepare(mode="early_label")
        else:
            raise ValueError("Exactly one of early_reg or early_label must be True.")

    def _filter_and_sort(self):
        filtered = {}

        for key, value in self.results.items():
            # print(">>>>>>>>>>>>>>>>>>>>")
            # print(f"Key: {key}")
            # print(f"Type of value: {type(value)}")
            # print(f"Length of value: {len(value)}")

            if not (isinstance(value, tuple) and len(value) == 4):
                raise ValueError(f"Expected tuple of length 4 for key '{key}', but got {type(value)} with len={len(value)}")

            clips, labels, start_times, stop_times = value

            # Type normalization
            def ensure_list(v, name):
                if isinstance(v, list):
                    return v
                elif isinstance(v, (float, int, str)):  # Acceptable singletons
                    return [v]
                else:
                    raise TypeError(f"For key '{key}', expected list or single float/int/str for '{name}', but got {type(v)}")

            clips = ensure_list(clips, "clips")
            labels = ensure_list(labels, "labels")
            start_times = ensure_list(start_times, "start_times")
            stop_times = ensure_list(stop_times, "stop_times")

            # Check for sufficient data
            if len(clips) >= 2:
                combined = list(zip(clips, labels, start_times, stop_times))
                combined_sorted = sorted(combined, key=lambda x: x[2])  # x[2] is start_time
                sorted_clips, sorted_labels, sorted_starts, sorted_stops = zip(*combined_sorted)
                filtered[key] = (
                    list(sorted_clips),
                    list(sorted_labels),
                    list(sorted_starts),
                    list(sorted_stops)
                )

        return filtered


    def _generate_preictal_dict(self):
        preictal_dict = {}

        for key, value in self.filtered_results.items():
            clips, labels, start_times, stop_times = value
            durations = [stop - start for start, stop in zip(start_times, stop_times)]

            new_labels = labels.copy()
            n = len(labels)
            i = 0
            while i < n:
                if labels[i] == "bckg":
                    j = i + 1
                    while j < n and labels[j] == "bckg":
                        j += 1
                    if j < n and labels[j] in self.target_labels:
                        target_label = labels[j]
                        for k in range(i, j):
                            new_labels[k] = "bckg preictal"
                        new_labels[j] = f"{target_label} ictal"
                        i = j + 1
                        continue
                i += 1

            for i in range(len(new_labels) - 1):
                if new_labels[i] == "bckg preictal" and new_labels[i + 1].endswith("ictal"):
                    preictal_stop = stop_times[i]
                    ictal_start = start_times[i + 1]
                    if preictal_stop > ictal_start:
                        stop_times[i] = ictal_start
                        durations[i] = ictal_start - start_times[i]

            preictal_clips, preictal_durs, preictal_labels = [], [], []
            for idx, (clip, lab, dur) in enumerate(zip(clips, new_labels, durations)):
                if lab == "bckg preictal":
                    next_ictal = None
                    for m in range(idx + 1, len(new_labels)):
                        if new_labels[m].endswith("ictal"):
                            next_ictal = new_labels[m].replace(" ictal", "")
                            break
                    preictal_clips.append(clip)
                    preictal_durs.append(dur)
                    print(preictal_durs)
                    preictal_labels.append(next_ictal)

            preictal_dict[key] = {
                "clips": preictal_clips,
                "durations": preictal_durs,
                "labels": preictal_labels
            }

        return preictal_dict

    # def _prepare(self, mode="early_reg"):
    #     X, Y = [], []
    #     for key, entry in self.preictal_dict.items():
    #         clips = entry["clips"]
    #         targets = entry["durations"] if mode == "early_reg" else entry["labels"]

    #         if len(clips) != len(targets):
    #             raise ValueError(f"length mismatch for {key}: {len(clips)} clips vs {len(targets)} targets")

    #         X.extend(clips)
    #         Y.extend(targets)

    #     return X, Y
    def _prepare(self, mode): 
        X_list, Y_list = [], []
        num_nodes_target = 33  # Desired number of nodes

        label_to_numeric = {}  # Dictionary to store label-to-numeric mapping
        label_counter = 1  # Start from 1

        for key, entry in self.preictal_dict.items():
            clips = entry["clips"]
            targets = entry["durations"] if mode == "early_reg" else entry["labels"]
            print(targets)
            if len(clips) != len(targets):
                raise ValueError(f"length mismatch for {key}: {len(clips)} clips vs {len(targets)} targets")

            for clip, target in zip(clips, targets):
                # clip shape: (T, N, F)
                T, N, F = clip.shape

                if N > num_nodes_target:
                    # Truncate
                    clip = clip[:, :num_nodes_target, :]
                elif N < num_nodes_target:
                    # Pad with zeros
                    padding = torch.zeros((T, num_nodes_target - N, F), dtype=clip.dtype)
                    clip = torch.cat([clip, padding], dim=1)

                X_list.append(clip)

                # Check if target is string type, then map to numeric
                if isinstance(target, str):
                    if target not in label_to_numeric:
                        label_to_numeric[target] = label_counter
                        label_counter += 1
                    Y_list.append(label_to_numeric[target])
                else:
                    Y_list.append(target)

        # Convert to tensors
        X = torch.stack(X_list)  # Shape: (num_samples, T, num_nodes_target, F)
        Y = torch.tensor(Y_list, dtype=torch.float if mode == "early_reg" else torch.long)

        print(Y)
        return X, Y




    def get_data(self):
        if self.early_reg:
            return self.x_reg, self.y_reg
        elif self.early_label:
            return self.x_clf, self.y_clf


# if name = __main__:
#     loader = preictal_dataLoader(pooled_results_dictionary=results, early_reg=True, early_label=False)
#     x_reg, y_reg = loader.get_data()

#     loader = preictal_dataLoader(pooled_results_dictionary=results, early_reg=False, early_label=True)
#     x_clf, y_clf = loader.get_data()
