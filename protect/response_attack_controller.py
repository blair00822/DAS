"""Gradient-preserving feature collection; loss math lives in response_attack_loss.

Captures the three post-mask/post-zero-conv branches of selected audio blocks.
No attention forward is replaced, no A/AV is stored, and audio features are not
detached. Use an eval-mode model: reentrant checkpoint recomputation must not
be used to collect these features.
"""
import inspect

BRANCHES = ('full', 'face', 'lip')


def audio_response_layers(unet):
    """Stable selectors identify transformer blocks, not individual branches."""
    layers = {}
    for place, blocks in (
        ('down', list(enumerate(unet.down_blocks))),
        ('mid', [(None, unet.mid_block)]),
        ('up', list(enumerate(unet.up_blocks))),
    ):
        index = 0
        for block_index, block in blocks:
            prefix = 'mid_block' if place == 'mid' else f'{place}_blocks.{block_index}'
            for j, audio in enumerate(getattr(block, 'audio_modules', [])):
                for k, transformer in enumerate(getattr(audio, 'transformer_blocks', [])):
                    if all(getattr(transformer, f'attn2_{i}', None) is not None for i in range(3)):
                        path = f'{prefix}.audio_modules.{j}.transformer_blocks.{k}'
                        layers[f'{place}:{index}'] = (path, transformer)
                        index += 1
    return layers



class ResponseAttackController:
    def __init__(self, unet, selectors):
        available = audio_response_layers(unet)
        selectors = list(dict.fromkeys(selectors))
        if not selectors or any(s not in available for s in selectors):
            raise ValueError(f'Invalid selection {selectors}; available: {list(available)}')
        self.layers = {s: available[s] for s in selectors}
        self.handles = []
        self.features = {}
        self.condition = None
        self.scales = {}
        for name, (_, block) in self.layers.items():
            signature = inspect.signature(block.forward)

            def pre(module, args, kwargs, name=name, signature=signature):
                bound = signature.bind_partial(*args, **kwargs)
                scale = bound.arguments.get('motion_scale')
                self.scales[name] = (1., 1., 1.) if scale is None else scale

            self.handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
            for i, branch in enumerate(BRANCHES):
                def capture(module, args, output, name=name, branch=branch, i=i):
                    if self.condition is None:
                        return
                    target = self.features[self.condition].setdefault(name, {})
                    if branch in target:
                        raise RuntimeError(f'Duplicate capture: {name}/{branch}; disable checkpoint replay')
                    if output.ndim != 4:
                        raise ValueError('Expected [B*T,C,H,W] branch output')
                    value = output * self.scales[name][i]
                    target[branch] = value if self.condition == 'audio' else value.detach()

                self.handles.append(getattr(block, f'zero_conv_{branch}').register_forward_hook(capture))

    def begin(self, condition):
        if condition not in ('audio', 'silent') or condition in self.features:
            raise RuntimeError('reset() before each audio/silent pair')
        self.condition = condition
        self.features[condition] = {}

    def pair(self):
        for condition in ('audio', 'silent'):
            features = self.features.get(condition, {})
            if set(features) != set(self.layers):
                raise RuntimeError(f'Missing selected layers for {condition}')
            for name, branches in features.items():
                if set(branches) != set(BRANCHES):
                    raise RuntimeError(f'Missing branches: {condition}/{name}')
        self.condition = None
        return self.features['audio'], self.features['silent']

    def reset(self):
        self.features = {}
        self.condition = None
        self.scales.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.reset()
