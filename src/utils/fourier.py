import torch
import torch.nn as nn
import torch.nn.functional as ff
import torchaudio


class STFT(nn.Module):
    def __init__(
            self,
            win_size: int = 1024,
            hop_size: int = 512,
            win_func: str = 'hann',
            complex_output: bool = False,
            magnitude_only: bool = False,
            device: str | torch.device = 'cpu',
            **kwargs,
    ):
        """Extended STFT module.

        Args:
            win_size (int): Window size for the STFT. Default: `1024`.
            hop_size (int): Hop size for the STFT. Default: `512`.
            win_func (str): Window function to use. Default: `'hann'`.
            complex_output (bool): Whether to return the complex-valued STFT or stacked real & imaginary parts. Default: `False`.
            magnitude_only (bool): Whether to return the magnitude spectrogram. Default: `False`.
            device (str, torch.device): Device to use. Default: `'cpu'`.
            **kwargs: Additional keyword arguments to pass to the Spectrogram transform.
        """
        super().__init__()
        self.complex_output = complex_output
        self.magnitude_only = magnitude_only
        self.device = device
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft = win_size,
            win_length = win_size,
            hop_length = hop_size,
            window_fn = self._map_window_fn(win_func),
            power = 1 if magnitude_only else None,
            normalized = False,
            center = (win_func != 'hamming'),
            **kwargs,
        )

    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Input waveform tensor.
                Tensor of size `[B, M, L]`.
        Returns:
            torch.Tensor: STFT of the input waveform.
                Tensor of size `[B, F, T]` if `complex_output` or `magnitude_only` is `True`, else `[B, F, T, 2]`.
        """
        if self.complex_output or self.magnitude_only:
            return self.stft(waveform)
        else:
            return torch.view_as_real(self.stft(waveform))

    def _map_window_fn(self, window_name):
        """Map window name to PyTorch window function to correctly handle device."""
        window_func = {
            'hann': lambda *args: torch.hann_window(*args).to(self.device),
            'sqrt_hann': lambda *args: torch.sqrt(torch.hann_window(*args).to(self.device)),
            'hamming': lambda *args: torch.hamming_window(*args).to(self.device),
            'bartlett': lambda *args: torch.bartlett_window(*args).to(self.device),
            'blackman': lambda *args: torch.blackman_window(*args).to(self.device),
            'kaiser': lambda *args: torch.kaiser_window(*args).to(self.device),
        }
        assert window_name in window_func, f"Unknown window function: '{window_name}'."
        return window_func[window_name]
    

class InverseSTFT(nn.Module):
    def __init__(
            self,
            win_size: int = 1024,
            hop_size: int = 512,
            win_func: str = 'hann',
            complex_input: bool = False,
            device: str | torch.device = 'cpu',
            **kwargs,
    ):
        """ Extended inverse STFT module.

        Args:
            win_size (int): Window size for the STFT. Default: `1024`.
            hop_size (int): Hop size for the STFT. Default: `512`.
            win_func (str): Window function to use. Default: `'hann'`.
            complex_input (bool): Whether the input STFT is complex. Default: `False`.
            device (str): Device to use. Default: `'cpu'`.
            **kwargs (): Additional keyword arguments for InverseSpectrogram.
        """
        super().__init__()
        self.device = device
        self.complex_input = complex_input
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft = win_size,
            win_length = win_size,
            hop_length = hop_size,
            window_fn = self._map_window_fn(win_func),
            normalized = False,
            center = (win_func != 'hamming'),
            **kwargs,
        )

    def forward(self, specgram, length=None):
        """
        Args:
            specgram (torch.Tensor): Input spectrogram tensor.
                Tensor of size `[B, F, T]` if `complex_input` is `True`, else `[B, F, T, 2]`.
            length (int, optional): Length of the output waveform.
                Default: `None`.
        Returns:
            torch.Tensor: Inverse STFT of the input spectrogram.
                Tensor of size `[B, M, L]`.
        """        
        if self.complex_input:
            return self.istft(specgram, length=length)
        else:
            return self.istft(torch.view_as_complex(specgram), length=length)

    def _map_window_fn(self, window_name):
        """Map window name to PyTorch window function to correctly handle device."""
        window_func = {
            'hann': lambda *args: torch.hann_window(*args).to(self.device),
            'sqrt_hann': lambda *args: torch.sqrt(torch.hann_window(*args).to(self.device)),
            'hamming': lambda *args: torch.hamming_window(*args).to(self.device),
            'bartlett': lambda *args: torch.bartlett_window(*args).to(self.device),
            'blackman': lambda *args: torch.blackman_window(*args).to(self.device),
            'kaiser': lambda *args: torch.kaiser_window(*args).to(self.device),
        }
        assert window_name in window_func, f"Unknown window function: '{window_name}'."
        return window_func[window_name]
    

class FeaturesSTFT(nn.Module):
    def __init__(
            self,
            features: str = 'real+imag',
            eps: float = 1e-8,
    ):
        """ Module that computes spectral features from STFT.

        Args:
            features (str): Name of spectral features to compute.
                Default: `'real+imag'`.
        """        
        super().__init__()
        base_features = {
            'real': torch.real,
            'imag': torch.imag,
            'abs': torch.abs,
            'ang': torch.angle,
            'log': lambda s: torch.log(torch.abs(s).pow(2) + eps),
            'cos': lambda s: torch.cos(torch.angle(s)),
            'sin': lambda s: torch.sin(torch.angle(s)),
        }
        list_features = []
        for feat in features.split('+'):
            if feat in base_features:
                list_features.append(base_features[feat])
            elif feat.startswith('d'):
                list_features.append(lambda s, d=len(feat.split('d')[:-1]), f=base_features[feat.split('d')[-1]]: ff.pad(torch.diff(input=f(s), n=d), (d, 0)))
            else:
                raise ValueError(f"Unknown feature '{feat}'.")
        self.list_features = list_features    

    def forward(self, specgram: torch.Tensor):
        """
        Args:
            specgram (torch.Tensor): Complex-valued spectrogram tensor.
                Tensor of size `[B, M, F, T]`.
        Returns:
            torch.Tensor: Real-valued tensor of spectral features.
                Tensor of size `[B, M, F, T, X]` where `X` is the number of features.
        """
        return torch.stack([f(specgram) for f in self.list_features], dim=-1)
    