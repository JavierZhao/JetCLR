from torch.utils.data import Dataset
import os
import torch


class JetClassDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        flag=None,
        transform=None,
        args=None,
        logfile=None,
        load_labels=False,
    ):
        """
        This Dataset only loads the jets from the data files, and does not load the labels.
        Args:
            data_dir (string): Directory with all the smaller data files.
            transform (callable, optional): Optional transform to be applied on a sample.
            args (argparse.Namespace): Command line arguments.
            logfile (file): File to write logs to.
            load_labels (bool): Whether to load labels or not.
        """
        assert flag is not None, "flag must be specified"
        assert args is not None, "args must be specified"
        assert logfile is not None, "logfile must be specified"

        self.data_dir = dataset_path
        self.load_labels = load_labels
        if args.raw:
            if args.percent == 100:
                self.data_dir = f"{dataset_path}/raw/raw_{flag}/data/*"
                self.label_dir = f"{dataset_path}/raw/raw_{flag}/label/*"
            else:
                self.data_dir = f"{dataset_path}/raw/{flag}/data/*"
                self.label_dir = f"{dataset_path}/raw/{flag}/label/*"
        else:
            if args.percent == 100:
                self.data_dir = f"{dataset_path}/{flag}_{args.percent}%/data/*"
                self.label_dir = f"{dataset_path}/{flag}_{args.percent}%/label/*"
            else:
                self.data_dir = f"{dataset_path}/{flag}/data/*"
                self.label_dir = f"{dataset_path}/{flag}/label/*"

        # Assuming data and label files have the same naming convention
        self.data_files = sorted(
            [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if os.path.isfile(os.path.join(self.data_dir, f))
            ]
        )
        print(f"Number of data files: {len(self.data_files)}", flush=True, file=logfile)
        print(f"Data files: {self.data_files}", flush=True, file=logfile)
        if self.load_labels:
            # Ensure the corresponding label files exist and are in order
            self.label_files = [
                os.path.join(self.label_dir, os.path.basename(f))
                for f in self.data_files
            ]
            print(
                f"Number of label files: {len(self.label_files)}",
                flush=True,
                file=logfile,
            )
            print(f"Label files: {self.label_files}", flush=True, file=logfile)
        self.transform = transform
        self.raw = args.raw
        self.percent = args.percent

        # Calculate total number of jets across all files
        self.jets_per_file = 100e3  # 100k jets per file
        self.total_jets = self.jets_per_file * len(self.data_files)
        self.cache = {}  # Initialize an empty cache

    def __len__(self):
        return self.total_jets

    def __getitem__(self, idx):
        file_idx, jet_idx = self._find_file_and_jet_index(idx)
        data_path = self.data_files[file_idx]
        # Check cache for data; load and cache if not present
        if data_path not in self.cache:
            self.cache[data_path] = torch.load(data_path)
        data = self.cache[data_path]
        sample = data[jet_idx]

        if self.load_labels:
            label_path = self.label_files[file_idx]
            # Check cache for labels; load and cache if not present
            if label_path not in self.cache:
                self.cache[label_path] = torch.load(label_path)
            labels = self.cache[label_path]
            label = labels[jet_idx]

        if self.transform:
            sample = self.transform(sample)

        if self.load_labels:
            return sample, label
        else:
            return sample

    def _find_file_and_jet_index(self, global_idx):
        """
        Given a global index, find which file and which index within that file
        corresponds to this global index, assuming each file contains exactly
        100,000 jets.
        """
        # Calculate which file the jet is in
        file_idx = global_idx // self.jets_per_file

        # Calculate the index of the jet within that file
        jet_idx = global_idx % self.jets_per_file

        return file_idx, jet_idx
