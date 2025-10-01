import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from torchmetrics.audio.sdr import SignalDistortionRatio as SDR
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio.snr import SignalNoiseRatio as SNR
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio as SISNR
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore as DNSMOS


METRIC_DICT = {
    'PESQ': lambda sample_rate, device, ref, est: PESQ(fs=sample_rate, mode='wb').to(device)(preds=est, target=ref),
    'STOI': lambda sample_rate, device, ref, est: STOI(fs=sample_rate, extended=False).to(device)(preds=est, target=ref),
    'ESTOI': lambda sample_rate, device, ref, est: STOI(fs=sample_rate, extended=True).to(device)(preds=est, target=ref),
    'SDR': lambda sample_rate, device, ref, est: SDR().to(device)(preds=est, target=ref),
    'SISDR': lambda sample_rate, device, ref, est: SISDR().to(device)(preds=est, target=ref),
    'SNR': lambda sample_rate, device, ref, est: SNR().to(device)(preds=est, target=ref),
    'SISNR': lambda sample_rate, device, ref, est: SISNR().to(device)(preds=est, target=ref),
    'DNSMOS': lambda sample_rate, device, ref, est: DNSMOS(fs=sample_rate, personalized=True).to(device)(preds=est),
    'DNSMOS-P808': lambda sample_rate, device, ref, est: DNSMOS(fs=sample_rate, personalized=True).to(device)(preds=est)[0],
    'DNSMOS-SIG': lambda sample_rate, device, ref, est: DNSMOS(fs=sample_rate, personalized=True).to(device)(preds=est)[1],
    'DNSMOS-BAK': lambda sample_rate, device, ref, est: DNSMOS(fs=sample_rate, personalized=True).to(device)(preds=est)[2],
    'DNSMOS-OVR': lambda sample_rate, device, ref, est: DNSMOS(fs=sample_rate, personalized=True).to(device)(preds=est)[3],
}


def compute_one_metric(
        metric_name: str, 
        reference: torch.Tensor, 
        estimate: torch.Tensor, 
        sample_rate: int = 16000,
        eps: float = 1e-8,
    ) -> torch.Tensor:
    """Compute a single metric given reference and estimate signals.
    
    Args:
        metric_name (str): Metric name.
            Example: `'SDR'`.
        reference (torch.Tensor): Reference signal.
            Tensor of size `[waveform_length]`.    
        estimate (torch.Tensor): Estimate signals.
            Tensor of size `[waveform_length]`. 
        sample_rate (int, optional): Sample rate of the signals.
            Default: `16000`.
        eps (float, optional): Small value to replace NaN values.
            Default: `1e-8`.
    
    Returns:
        torch.Tensor: Metric score.
    """
    metric_func = METRIC_DICT[metric_name]
    estimate = torch.nan_to_num(estimate, nan=eps)
    return metric_func(sample_rate=sample_rate, device=reference.device, ref=reference, est=estimate)


def compute_all_metrics(
        metrics: list[str], 
        reference: torch.Tensor, 
        estimate: torch.Tensor, 
        sample_rate: int = 16000,
    ) -> list[float]:
    """Compute multiple metrics given reference and estimate signals.

    Args:
        metrics (list[str]): List of metric names.
            Example: `['SDR', 'STOI']`.
        reference (torch.Tensor): Reference signal.
            Tensor of size `[L]`.    
        estimate (torch.Tensor): Estimate signals.
            Tensor of size `[L]`.
        sample_rate (int, optional): Sample rate of the signals.
            Default: `16000`.

    Returns:
        list[float]: List of metric scores.
    """
    metric_score_list = []
    for metric_name in metrics:
        match metric_name:
            case 'DNSMOS':
                metric_score_list.extend(compute_one_metric(metric_name.upper(), reference, estimate, sample_rate=sample_rate).tolist())
            case _:
                metric_score_list.append(compute_one_metric(metric_name.upper(), reference, estimate, sample_rate=sample_rate).item())
    return metric_score_list