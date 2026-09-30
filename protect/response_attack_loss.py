"""Three-branch response energy, frozen spatial weights and layer normalization."""
import torch

BRANCHES = ('full', 'face', 'lip')


def response_energy_maps(audio, silent):
    """Layer -> [T,H,W] channel MSE. Sum vectors BEFORE squaring.

    Batch size is one in this attack. Silent is a stop-gradient target.
    """
    if audio.keys() != silent.keys():
        raise ValueError('Audio/silent layer mismatch')
    energies = {}
    for layer in audio:
        a = sum(audio[layer][b].float() for b in BRANCHES)
        s = sum(silent[layer][b].detach().float() for b in BRANCHES)
        energies[layer] = (a - s).square().mean(dim=1)
    return energies


def weighted_energy(energy, weight):
    """Explicit broadcast includes T in denominator; never sum loss over time."""
    weight = weight.to(device=energy.device, dtype=torch.float32).expand_as(energy)
    denominator = weight.sum()
    if not torch.isfinite(weight).all() or (weight < 0).any() or denominator <= 0:
        raise ValueError('Spatial weights must be finite, nonnegative and nonempty')
    return (energy * weight).sum() / denominator


def make_spatial_weights(mean_energy, mean_rms):
    """Frozen clean RMS weights, averaged over calibration samples and frames."""
    result = {}
    for layer, energy in mean_energy.items():
        weight = mean_rms[layer].mean(dim=0, keepdim=True).to(energy.device).detach()
        if not torch.isfinite(weight).all() or weight.min() < 0 or weight.sum() <= 0:
            raise ValueError(f'Empty or invalid spatial weights for {layer}')
        result[layer] = weight / weight.sum()
    return result


def clean_baselines(mean_energy, weights, floor=1e-8):
    baselines = {l: weighted_energy(e, weights[l]).detach() for l, e in mean_energy.items()}
    invalid = [l for l, d in baselines.items() if not torch.isfinite(d) or d < floor]
    if invalid:
        raise ValueError(f'Clean response below normalization floor {floor}: {invalid}; exclude these layers')
    return baselines


def compute_response_loss(energies, weights, baselines):
    """Frozen clean-energy normalization followed by equal layer averaging."""
    if not energies or energies.keys() != weights.keys() or energies.keys() != baselines.keys():
        raise ValueError('Energy/weight/baseline selections must match')
    per_layer = {l: weighted_energy(e, weights[l]) / baselines[l] for l, e in energies.items()}
    total = torch.stack(list(per_layer.values())).mean()
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite response loss')
    return total, per_layer
