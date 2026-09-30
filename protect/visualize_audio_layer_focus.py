#!/usr/bin/env python3
"""Compare spatial focus of audio layers using RMS / spatial mean RMS.

No attack weights or PCA. Default: all audio layers, selected-frame mean,
Audio-minus-silent difference only; one figure per input, saved as PNG and PDF.
An aligned lip mask is required for the per-layer metrics CSV.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--compare', nargs='*', default=[])
    p.add_argument('--output', default='protect/response_layer_focus')
    p.add_argument('--config', default='protect/configs/response.yaml')
    p.add_argument('--audio', required=True, help='Precomputed audio embedding (.pt)')
    p.add_argument('--audio-start', type=int, default=2)
    p.add_argument('--clip-length', type=int, default=3)
    p.add_argument('--timestep', type=int, default=350)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layers', nargs='+', help='Default: every available audio layer, in network order; labels retain full-network indices')
    p.add_argument('--frames', nargs='+', type=int, help='Frames averaged before spatial normalization; default all clip frames')
    p.add_argument('--lip-mask', help='Aligned binary/soft lip mask; default: <mask-dir>/<image_stem>_lip_mask.png')
    p.add_argument('--mask-dir', default='cache_tmp')
    p.add_argument('--layer-cols', type=int, default=8)
    p.add_argument('--panel-gap', type=float, default=0.06)
    p.add_argument('--overlay-alpha', type=float, default=0.5)
    p.add_argument('--no-overlay', action='store_true',
                   help='Plot only the normalized response heatmap, without the input image')
    p.add_argument('--vmax', type=float, default=3., help='Shared upper color limit in multiples of layer mean; default 3')
    p.add_argument('--cmap', default='coolwarm')
    a = p.parse_args()
    if a.clip_length < 1 or a.audio_start < 0 or a.timestep < 0 or a.layer_cols < 1:
        p.error('Invalid clip length, audio start, timestep or layer columns')
    if not 0 <= a.overlay_alpha <= 1 or not 0 <= a.panel_gap < float('inf') or not 1 < a.vmax < float('inf'):
        p.error('Require alpha in [0,1], finite nonnegative gap, and finite vmax > 1')
    a.frames = list(dict.fromkeys(a.frames if a.frames is not None else range(a.clip_length)))
    if any(f < 0 or f >= a.clip_length for f in a.frames):
        p.error('Frames must be within the clip')
    if a.lip_mask is None:
        a.lip_mask = str(Path(a.mask_dir) / f'{Path(a.image).stem}_lip_mask.png')
    for value in (a.image, a.output, a.config, a.audio, a.mask_dir, a.lip_mask, *a.compare):
        if Path(value).is_absolute() or '..' in Path(value).parts:
            p.error('Use project-relative paths without parent traversal')
    for path in (a.image, a.config, a.audio, a.lip_mask, *a.compare):
        if not Path(path).is_file():
            p.error(f'File not found: {path}')
    return a


def difference_maps(audio, silent, frames):
    """Selected-frame RMS of the summed audio-minus-silent branch response."""
    import torch
    from protect.response_attack_loss import response_energy_maps
    maps = {}
    for layer, energy in response_energy_maps(audio, silent).items():
        rms = energy[frames].sqrt()
        if not torch.isfinite(rms).all():
            raise ValueError(f'Non-finite response in {layer}')
        maps[layer] = rms.detach().cpu().numpy()
    return maps


def relative_response(value):
    """Exact mean normalization for nonzero maps; zero maps remain zero."""
    import numpy as np
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Expected a finite nonnegative 2D RMS map')
    mean = float(value.mean())
    return (value / mean if mean > 0 else np.zeros_like(value)), mean


def resize_mask_area(mask, shape):
    """Area-resize an aligned input mask, retaining fractional cell coverage."""
    import torch
    import torch.nn.functional as F
    value = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    value = F.interpolate(value, size=shape, mode='area')[0, 0]
    return value.clamp(0, 1).cpu().numpy()


def lip_response_metrics(raw, mask):
    """Return area share, response mass share, and area-normalized enrichment."""
    import numpy as np
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    total_response = float(raw.sum())
    area_share = float(mask.mean())
    response_share = float((raw * mask).sum()) / total_response if total_response > 0 else 0.
    enrichment = response_share / area_share if area_share > 0 else float('nan')
    return area_share, response_share, enrichment


def plot_focus(record, selectors, input_mask, args, layer_indices, record_index=0):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    columns = min(args.layer_cols, len(selectors))
    rows = math.ceil(len(selectors) / columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(1.65 * columns, 1.85 * rows))
    norm = Normalize(vmin=0, vmax=args.vmax)
    metric_rows = []
    try:
        fig.subplots_adjust(left=.005, right=.995, bottom=.01, top=.93,
                            wspace=args.panel_gap, hspace=.22 + args.panel_gap)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, layer in zip(axes.flat, selectors):
            layer_index = layer_indices[layer]
            # Mean of frame RMS, NOT RMS of temporally averaged feature vectors.
            raw = record['maps'][layer].mean(axis=0)
            relative, mean = relative_response(raw)
            h, w = raw.shape
            layer_mask = resize_mask_area(input_mask, (h, w))
            area_share, response_share, enrichment = lip_response_metrics(raw, layer_mask)
            metric_rows.append({
                'layer_index': layer_index, 'selector': layer, 'height': h, 'width': w,
                'raw_mean_rms': mean, 'lip_area_percent': 100 * area_share,
                'lip_response_percent': 100 * response_share,
                'lip_enrichment': enrichment,
            })
            extent = (-.5, w-.5, h-.5, -.5)
            if args.no_overlay:
                ax.imshow(relative, cmap=args.cmap, norm=norm, extent=extent,
                          origin='upper', interpolation='nearest')
            else:
                # Precompose once: PDF and Agg must not blend two image artists differently.
                background = np.asarray(record['image'], dtype=np.float64)
                ih, iw = background.shape[:2]
                yi = np.minimum(((np.arange(ih) + .5) * h / ih).astype(int), h - 1)
                xi = np.minimum(((np.arange(iw) + .5) * w / iw).astype(int), w - 1)
                heat_rgb = plt.get_cmap(args.cmap)(norm(relative[yi[:, None], xi[None, :]]))[..., :3]
                composite = ((1 - args.overlay_alpha) * background
                             + args.overlay_alpha * heat_rgb).clip(0, 1)
                ax.imshow(composite, extent=extent, origin='upper', interpolation='nearest')
            ax.set_title(f'layer{layer_index}', fontsize=10, pad=3)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        frame_tag = '-'.join(map(str, args.frames))
        mode = '_no_overlay' if args.no_overlay else ''
        stem = output / f'layer_focus_t{args.timestep}_difference_frames{frame_tag}_input{record_index}{mode}'
        for suffix in ('.png', '.pdf'):
            fig.savefig(str(stem) + suffix, dpi=180, bbox_inches='tight', facecolor='white')
        with Path(str(stem) + '_lip_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f'Saved: {stem}.png / .pdf / _lip_metrics.csv', flush=True)
        print('layer  selector  lip area  lip response  enrichment', flush=True)
        for row in metric_rows:
            print(f"{row['layer_index']:>5}  {row['selector']:<8}  "
                  f"{row['lip_area_percent']:>7.2f}%  "
                  f"{row['lip_response_percent']:>11.2f}%  "
                  f"{row['lip_enrichment']:>10.3f}x", flush=True)
    finally:
        plt.close(fig)


def main():
    # Intended final location: protect/visualize_audio_layer_focus.py.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    args = parse_args()
    args.output = str(Path(args.output) / Path(args.image).stem)
    import numpy as np
    import torch
    from PIL import Image
    from hallo.datasets.image_processor import ImageProcessor
    from protect.hallo_response_runtime import HalloResponseRuntime, process_audio_emb
    from protect.response_attack_controller import ResponseAttackController, audio_response_layers

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; select GPU with CUDA_VISIBLE_DEVICES')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = HalloResponseRuntime(args.config)
    if args.timestep >= runtime.scheduler.config.num_train_timesteps:
        raise ValueError('Timestep outside scheduler range')
    available = audio_response_layers(runtime.net.denoising_unet)
    layer_indices = {selector: index for index, selector in enumerate(available)}
    selectors = list(dict.fromkeys(args.layers or available))
    if not selectors or any(s not in available for s in selectors):
        raise ValueError(f'Unknown layer; available: {list(available)}')
    print('Layers:', {s: available[s][0] for s in selectors}, flush=True)
    audio_all = process_audio_emb(torch.load(args.audio, map_location='cpu'))
    if args.audio_start + args.clip_length > len(audio_all):
        raise ValueError('Audio clip exceeds available embeddings')
    audio = audio_all[args.audio_start:args.audio_start + args.clip_length][None].to(runtime.device, runtime.dtype)
    with Image.open(args.lip_mask) as image:
        input_mask = np.asarray(image.convert('L'), dtype=np.float32) / 255.
    if input_mask.shape != (512, 512):
        raise ValueError(f'Lip mask must align with the 512x512 model input; got {input_mask.shape}')
    if not np.isfinite(input_mask).all() or input_mask.max() <= 0:
        raise ValueError('Lip mask is empty or non-finite')
    controller = ResponseAttackController(runtime.net.denoising_unet, selectors)
    try:
        with tempfile.TemporaryDirectory(prefix='hallo_response_') as cache:
            with ImageProcessor((512, 512), runtime.cfg.face_analysis_model_path) as processor:
                values = processor.preprocess(args.image, cache, runtime.cfg.face_expand_ratio)
                source = dict(zip(('image', 'face_region', 'face_emb', 'full_mask', 'face_mask', 'lip_mask'), values))
                source['face_emb'] = torch.as_tensor(source['face_emb'])
                shape = runtime.latent_shape(source['image'], args.clip_length)
                noise = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed)).to(runtime.device, runtime.dtype)
                timestep = torch.tensor([args.timestep], device=runtime.device, dtype=torch.long)
                for index, path in enumerate((args.image, *args.compare)):
                    print(f'Forward: {path}', flush=True)
                    with Image.open(path) as img:
                        pixels = processor.pixel_transform(img.convert('RGB'))
                    # Share original face conditioning, noise and timestep across inputs.
                    with torch.no_grad():
                        audio_features, silent_features = runtime.pair(
                            pixels, source, audio, noise, timestep, controller)
                        maps = difference_maps(audio_features, silent_features, args.frames)
                    controller.reset()
                    del audio_features, silent_features
                    record = {
                        'image': ((pixels.float().permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy(),
                        'maps': maps,
                    }
                    plot_focus(record, selectors, input_mask, args, layer_indices, index)
    finally:
        controller.close()
        runtime.net.clear()


if __name__ == '__main__':
    main()
