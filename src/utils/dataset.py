import torch
import torchaudio
import os
import pathlib


class MixturesDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            root: str | pathlib.Path, 
            subset_name: str | pathlib.Path,
            num_samples: int | None = None,
            fixed_ref_mic: int | None = 0,
            file_ext: str = 'flac',
            hard_mode: bool = False,
    ):
        """ Dataset class to handle the MIXTURES database.

        Args:
            root (str, pathlib.Path): Path to the root directory of the dataset.
            subset_name (str, pathlib.Path): Name of the subset to load.
            num_samples (int, optional): Number of samples to load. Default: `None` (all samples). 
            fixed_ref_mic (int, optional): Index for the reference microphone. If `None`, chosen at random. Default: `0`.
            file_ext (str, optional): Format of the audio files. Default: `'flac'`.
            hard_mode (bool, optional): Whether to load only mixture signals with both distractor and ambient noise present. Default: `False`.
        """
        super().__init__()
        self.root = str(root)
        self.subset_name = str(subset_name)
        self.subset_path = os.path.join(root, subset_name)
        self.num_samples = num_samples
        self.fixed_ref_mic = fixed_ref_mic
        self.file_ext = file_ext
        self.hard_mode = hard_mode  
        # Collect sample names.
        self.sample_names = self._collect_samples()
    
    def _collect_samples(self):
        if self.hard_mode:
            # sample_list = [] # [s for s in os.listdir(self.subset_path) if not s.startswith('.')]
            # for sample_name in os.listdir(self.subset_path):
            #     metadata_path = os.path.join(self.subset_path, sample_name, 'metadata.txt')
            #     with open(metadata_path, 'r') as metadata_file:
            #         sample_data = metadata_file.read()
            #     if not 'INF' in sample_data:
            #         sample_list.append(sample_name)
            with open(os.path.join(self.root, f'{self.subset_name}_samples_hard_mode.txt'), 'r') as f:
                sample_list = f.read().splitlines()
        else:
            sample_list = [s for s in os.listdir(self.subset_path) if not s.startswith('.')]            
        return sorted(sample_list)[:self.num_samples]

    def _get_file_path(self, sample_name, file_name):
        return os.path.join(self.subset_path, sample_name, f'{file_name}.{self.file_ext}')

    def _load_sample(self, n: int):

        # Load mixture audio signals.
        sample_name = self.sample_names[n]
        file_path_mixtr = self._get_file_path(sample_name, 'mixtr')
        waveform_mixtr, sr_mixtr = torchaudio.load(file_path_mixtr)

        # Generate a fixed or random reference microphone vector.
        num_channels = waveform_mixtr.size(0)
        if self.fixed_ref_mic is None:
            ref_mic = int(torch.randint(0, num_channels, (1,)))
        else:
            ref_mic = self.fixed_ref_mic
        ref_mic_vect = torch.eye(num_channels)[ref_mic]

        # Load ground-truth audio signals.
        file_path_truth = self._get_file_path(sample_name, 'clean')
        waveform_truth, sr_truth = torchaudio.load(file_path_truth)
        waveform_truth = waveform_truth[ref_mic]

        # Assert audio file consistency.
        assert (sr_mixtr == sr_truth), f"Sampling rates do not match across files."
        assert (waveform_mixtr.size(-1) == waveform_truth.size(-1)), "Signal length does not match across files."
        
        # Load additional multi-channel audio signals.
        file_path_clean = self._get_file_path(sample_name, 'clean')
        waveform_clean, _ = torchaudio.load(file_path_clean)
        file_path_noise = self._get_file_path(sample_name, 'noise')
        waveform_noise, _ = torchaudio.load(file_path_noise)
        
        return waveform_mixtr, waveform_truth, waveform_clean, waveform_noise, ref_mic_vect, sample_name

    def __getitem__(self, n: int):
        """ Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            torch.Tensor: Noisy mixture signal. Size `[M, L]`.
            torch.Tensor: Ground-truth speech signal. Size `[L]`.
            torch.Tensor, optional: Clean speech signal. Size `[M, L]`.
            torch.Tensor, optional: Noise signal. Size `[M, L]`.
            torch.Tensor: Reference microphone vector. Size `[M]`.
            str, optional: Sample name.
        """
        return self._load_sample(n)  

    def __len__(self):
        return len(self.sample_names)
    

def dataloaders(
        root: str | pathlib.Path, 
        batch_size: int, 
        num_samples_trn: int | None = None, 
        num_samples_val: int | None = None, 
        num_samples_tst: int | None = None, 
        **kwargs_dataset,
    ):
    """ Define PyTorch DataLoaders for the training, validation and test subsets.

    Args:
        root (str, pathlib.Path): Path to the root directory of the database.
        batch_size (int): Batch size for PyTorch DataLoaders.
        num_samples_trn (int, optional): Number of samples to load for the training dataset. Default: `None` (all samples).
        num_samples_val (int, optional): Number of samples to load for the validation dataset. Default: `None` (all samples).
        num_samples_tst (int, optional): Number of samples to load for the test dataset. Default: `None` (all samples).
        **dataset_kwargs: Other keyword arguments for the Datasets.

    Returns:
        loaders (dict): Dictionary containing the DataLoaders.
    """
    trn_dataset = MixturesDataset(root, 'trn', num_samples=num_samples_trn, **kwargs_dataset)
    val_dataset = MixturesDataset(root, 'val', num_samples=num_samples_val, **kwargs_dataset)
    tst_dataset = MixturesDataset(root, 'tst', num_samples=num_samples_tst, **kwargs_dataset)
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)
    return {'trn_loader': trn_loader, 'val_loader': val_loader, 'tst_loader': tst_loader}
