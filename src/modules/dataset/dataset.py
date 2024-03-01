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
        if args.percent == 100:
            self.data_dir = f"{dataset_path}/raw_{flag}/shuffled/data/"
            self.label_dir = f"{dataset_path}/raw_{flag}/shuffled/label/"
        else:
            self.data_dir = f"{dataset_path}/raw_{flag}_{args.percent}%/shuffled/data/"
            self.label_dir = (
                f"{dataset_path}/raw_{flag}_{args.percent}%/shuffled/label/"
            )
        # for testing (LCT), only use the 1% dataset
        if flag == "test":
            self.data_dir = f"{dataset_path}/raw_{flag}_1%/shuffled/data/"
            self.label_dir = f"{dataset_path}/raw_{flag}_1%/shuffled/label/"

        # Assuming data and label files have the same naming convention
        self.data_files = sorted(
            [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if os.path.isfile(os.path.join(self.data_dir, f))
            ]
        )
        print(f"Number of data files: {len(self.data_files)}", flush=True, file=logfile)
        print(
            f"Data files: {[file.split('/')[-1] for file in self.data_files]}",
            flush=True,
            file=logfile,
        )
        if self.load_labels:
            # Ensure the corresponding label files exist and are in order
            #             self.label_files = [
            #                 os.path.join(self.label_dir, os.path.basename(f))
            #                 for f in self.data_files
            #             ]
            self.label_files = sorted(
                [
                    os.path.join(self.label_dir, f)
                    for f in os.listdir(self.label_dir)
                    if os.path.isfile(os.path.join(self.label_dir, f))
                ]
            )
            print(
                f"Number of label files: {len(self.label_files)}",
                flush=True,
                file=logfile,
            )
            print(
                f"Label files: {[file.split('/')[-1] for file in self.label_files]}",
                flush=True,
                file=logfile,
            )
        self.transform = transform
        self.percent = args.percent

        # Calculate total number of jets across all files
        self.jets_per_file = int(100e3)  # 100k jets per file
        if flag == "val" and args.percent in [1, 5]:
            self.jets_per_file = int(50e3)  # 50k jets per file
        self.total_jets = int(self.jets_per_file * len(self.data_files))
        print(f"jets per file: {self.jets_per_file}", flush=True, file=logfile)
        print(f"total jets: {self.total_jets}", flush=True, file=logfile)
        self.cache = {}  # Initialize an empty cache

    def __len__(self):
        return self.total_jets

    def __getitem__(self, idx):
        file_idx, jet_idx = self._find_file_and_jet_index(idx)
        data_path = self.data_files[file_idx]
        # Check cache for data; load and cache if not present
        if data_path not in self.cache:
            print(data_path)
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
        file_idx = int(global_idx // self.jets_per_file)

        # Calculate the index of the jet within that file
        jet_idx = int(global_idx % self.jets_per_file)

        return file_idx, jet_idx

    def get_sample_shape(self):
        # Load a single sample from the dataset
        if self.load_labels:
            sample, _ = self.__getitem__(0)
        else:
            sample = self.__getitem__(0)
        # Return the shape of the sample
        return sample.shape

    def get_dataset_shape(self):
        # Load a single sample to get its shape
        if self.load_labels:
            sample, _ = self.__getitem__(0)
        else:
            sample = self.__getitem__(0)

        # Create a shape tuple that starts with self.total_jets and extends with the sample shape
        dataset_shape = (self.total_jets,) + sample.shape

        return dataset_shape

    def get_labels_shape(self):
        # Load a single sample to get its shape
        if self.load_labels:
            _, label = self.__getitem__(0)
        else:
            raise ValueError("This dataset does not contain labels")

        # Create a shape tuple that starts with self.total_jets and extends with the sample shape
        labels_shape = (self.total_jets,) + label.shape

        return labels_shape

    def _load_data(self):
        """
        Loads data from files in the specified dataset path and concatenates them along the first axis.
        """
        data_list = []
        label_list = []
        # Assuming data files are directly under self.data_path
        for data_filename, label_file_name in zip(
            sorted(os.listdir(self.data_dir)), sorted(os.listdir(self.label_dir))
        ):
            data_file_path = os.path.join(self.data_dir, data_filename)
            label_file_path = os.path.join(self.label_dir, label_file_name)
            # Load the tensor from file
            data_tensor = torch.load(data_file_path)
            label_tensor = torch.load(label_file_path)
            data_list.append(data_tensor)
            label_list.append(label_tensor)

        # Concatenate all tensors along the first axis
        entire_dataset = torch.cat(data_list, dim=0)
        entire_labels = torch.cat(label_list, dim=0)
        return entire_dataset, entire_labels

    def fetch_entire_dataset(self):
        """
        Wrapper method to call load_data and potentially perform additional processing.
        """
        data, labels = self._load_data()
        # Apply transformations if needed and if 'transform' is provided
        if self.transform is not None:
            data = self.transform(data)
        return data, labels
