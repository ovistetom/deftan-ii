import torch
import os
import sys
sys.path.append(os.path.abspath(''))
import src.utils.fourier as fourier

Loss = torch.nn.modules.loss._Loss


################################################################################################################################
class MultiLoss(Loss):
    def __init__(
            self, 
            list_criterion: list[Loss],
            list_criterion_scale: list[float],
    ):
        """Class to combine multiple losses computed on different input quantites."""
        super().__init__()
        self.list_criterion = list_criterion
        self.list_criterion_scale = list_criterion_scale

    def forward(
            self,
            waveform_clean_estim=None,
            waveform_clean_truth=None,
            waveform_noise_estim=None,
            waveform_noise_truth=None,
    ):
        loss = 0.0
        for a, criterion in zip(self.list_criterion_scale, self.list_criterion):
            loss += a * criterion(
                waveform_clean_estim=waveform_clean_estim,
                waveform_clean_truth=waveform_clean_truth,
                waveform_noise_estim=waveform_noise_estim,
                waveform_noise_truth=waveform_noise_truth,
            )
        return loss



################################################################################################################################
class LossL1(Loss):
    def __init__(
            self,
            *args,
            **kwargs,
        ):
        """Criterion that measures the L1 loss between estimated and ground-truth waveforms."""
        super().__init__()
        self.loss = torch.nn.L1Loss(*args, **kwargs)

    def forward(
            self, 
            waveform_clean_estim: torch.Tensor, 
            waveform_clean_truth: torch.Tensor,
            *args,
            **kwargs,
        ):
        return self.loss(input=waveform_clean_estim, target=waveform_clean_truth)
    

class LossMSE(Loss):
    def __init__(self, *args, **kwargs):
        """Criterion that measures the MSE loss between estimated and ground-truth waveforms."""
        super().__init__()
        self.loss = torch.nn.MSELoss(*args, **kwargs)
    def forward(self, waveform_clean_estim: torch.Tensor, waveform_clean_truth: torch.Tensor):
        return self.loss(input=waveform_clean_estim, target=waveform_clean_truth)


class LossSDR(Loss):
    def __init__(
            self,
            reduction: str = 'mean',
        ):
        """Criterion that measures the SDR loss between estimated and ground-truth waveforms."""
        super().__init__()
        self.reduction = reduction

    def forward(self, waveform_clean_estim: torch.Tensor, waveform_clean_truth: torch.Tensor):
        error = - 10 * torch.log10( waveform_clean_truth.pow(2.0).sum(-1) / (waveform_clean_truth - waveform_clean_estim).pow(2.0).sum(-1) )
        
        match self.reduction:
            case 'sum':
                return torch.sum(error)
            case 'mean':
                return torch.mean(error)
            case _:
                raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")     


class LossSTFT(Loss):
    def __init__(
            self,
            reduction: str = 'mean',
            win_size: int = 1024,
            hop_size: int = 512,
            win_func: str = 'hann',
            comp: float = 1.0,
            norm: int | float = 2,
            beta: float = 0.25,
            input_specgram: bool = False,
            device: str | torch.device = 'cpu',
            *args,
            **kwargs,
    ):
        """Criterion that measures the error between estimated and ground-truth spectrograms.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output.
                Options: `'mean'`, `'sum'`. Default: `'mean'`.
            win_size (int, optional): Window size for the STFT. Default: `1024`.
            hop_size (int, optional): Hop size for the STFT. Default: `512`.
            win_func (str, optional): Window function for the STFT. Default: `'hann'`.
            comp (float): Compression factor for the STFT error. Default: `1.0`.
            norm (int, float): Norm for the STFT error. Default: `2`.
            beta (float): Weight for the complex part of the STFT error. Default: `0.25`.
            input_specgram (bool, optional): Flag to indicate if the input is a spectrogram. Default: `False`.
            device (str): PyTorch device to use. Default: `'cpu'`.
        """
        super().__init__()
        self.reduction = reduction
        self.comp = comp
        self.norm = norm
        self.beta = beta
        self.stft = fourier.STFT(
            win_size = win_size,
            hop_size = hop_size,
            win_func = win_func,
            complex_output = bool(beta),
            magnitude_only = not bool(beta),
            device = device,
        )
        self.input_specgram = input_specgram

    def forward(
            self,
            waveform_clean_estim: torch.Tensor,
            waveform_clean_truth: torch.Tensor,
            *args,
            **kwargs,
    ):
        """
        Args:
            waveform_estim (torch.Tensor): Target waveform tensor with size `(B, L)`.
            waveform_truth (torch.Tensor): Input waveform tensor with size `(B, L)`.
        Returns:
            torch.Tensor: Output loss tensor with size `()`.
        """
        if self.input_specgram:
            specgram_estim = waveform_clean_estim.abs()
            specgram_truth = self.stft(waveform_clean_truth).abs()
        else:
            specgram_estim = self.stft(waveform_clean_estim)
            specgram_truth = self.stft(waveform_clean_truth)

        if self.beta:
            # Complex spectrograms.
            specgram_truth = torch.pow(specgram_truth.abs() + 1e-8, self.comp) * torch.exp(1j * specgram_truth.angle())
            specgram_estim = torch.pow(specgram_estim.abs() + 1e-8, self.comp) * torch.exp(1j * specgram_estim.angle())
            complex_specgram_error = torch.pow(torch.abs(specgram_estim - specgram_truth), self.norm)
            magnitude_specgram_error = torch.pow(specgram_estim.abs() - specgram_truth.abs(), self.norm)
            error = self.beta * complex_specgram_error + (1-self.beta) * magnitude_specgram_error
        else:
            # Magnitude-only spectrograms.
            specgram_truth = torch.pow(specgram_truth, self.comp)
            specgram_estim = torch.pow(specgram_estim, self.comp)
            error = torch.pow(torch.abs(specgram_estim - specgram_truth), self.norm)

        match self.reduction:
            case 'sum':
                return torch.sum(error)
            case 'mean':
                return torch.mean(error)
            case _:
                raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")
            
class LossPCM(Loss):
    def __init__(
            self, 
            reduction: str = 'mean',
            win_size: int = 1024,
            hop_size: int = 512,
            win_func: str = 'hann',            
            device: str | torch.device = 'cpu',            
    ):
        """Criterion that measures the PCM loss between estimated and ground-truth waveforms."""
        super().__init__()
        self.reduction = reduction
        self.stft = fourier.STFT(
            win_size = win_size,
            hop_size = hop_size,
            win_func = win_func,
            complex_output = False,
            magnitude_only = False,
            device = device,
        )

    def forward(
            self,
            waveform_clean_estim: torch.Tensor, 
            waveform_clean_truth: torch.Tensor, 
            waveform_noise_estim: torch.Tensor,
            waveform_noise_truth: torch.Tensor,
    ):
        specgram_clean_estim = self.stft(waveform_clean_estim)                  # (B, F, T, 2)
        specgram_clean_truth = self.stft(waveform_clean_truth)                  # (B, F, T, 2)
        specgram_noise_estim = self.stft(waveform_noise_estim)                  # (B, F, T, 2)
        specgram_noise_truth = self.stft(waveform_noise_truth)                  # (B, F, T, 2)

        error_clean = specgram_clean_estim.abs() - specgram_clean_truth.abs()   # (B, F, T, 2)
        error_noise = specgram_noise_estim.abs() - specgram_noise_truth.abs()   # (B, F, T, 2)
        error_clean = error_clean.sum(dim=-1).abs()                             # (B, F, T)
        error_noise = error_noise.sum(dim=-1).abs()                             # (B, F, T)
        error = 0.5*error_clean + 0.5*error_noise                               # (B, F, T)

        match self.reduction:
            case 'sum':
                return torch.sum(error)                                         # (B, F, T) -> ()
            case 'mean':
                return torch.mean(error)                                        # (B, F, T) -> ()
            case _:
                raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")