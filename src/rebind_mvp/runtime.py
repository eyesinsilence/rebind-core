"""Explicit device selection and local model identity for portable experiments."""
import json
import os
import pathlib

import torch

from .audit import digest


def resolve_device(config=None, role='retriever', override=None):
    config = config or {}
    value = override or os.environ.get('REBIND_' + role.upper() + '_DEVICE') or config.get('devices', {}).get(role, 'auto')
    if value == 'auto':
        value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda':
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f'Configured {role} device {value} is unavailable; set devices.{role} or REBIND_{role.upper()}_DEVICE')
        device = torch.device('cuda', index)
    return str(device)


def encoder_dimension(encoder):
    width = getattr(encoder, 'embedding_dim', None)
    if width is None:
        width = getattr(getattr(getattr(encoder, 'model', None), 'config', None), 'hidden_size', None)
    if width is None or int(width) < 1:
        raise ValueError('Encoder must expose embedding_dim or model.config.hidden_size')
    return int(width)


def peak_memory(device):
    return torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else 0


_REVISION_CACHE = {}


def model_revision(path):
    """Hash actual selected weights, not merely the sharded index filename."""
    root = pathlib.Path(path)
    files = [root / 'config.json']
    files += [root / n for n in ['tokenizer.json', 'tokenizer_config.json', 'tokenizer.model',
                                'special_tokens_map.json', 'vocab.json', 'merges.txt', 'generation_config.json'] if (root / n).is_file()]
    for index_name, single_name in [('model.safetensors.index.json', 'model.safetensors'),
                                     ('pytorch_model.bin.index.json', 'pytorch_model.bin')]:
        index = root / index_name
        if index.exists():
            files.append(index)
            for name in sorted(set(json.loads(index.read_text())['weight_map'].values())):
                relative = pathlib.PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid checkpoint shard path')
                files.append(root / name)
            break
        if (root / single_name).exists():
            files.append(root / single_name)
            break
    else:
        raise FileNotFoundError('No supported local safetensors or PyTorch checkpoint in ' + str(root))
    key = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    if key not in _REVISION_CACHE:
        _REVISION_CACHE[key] = digest({str(p.relative_to(root)): digest(p) for p in files})
    return _REVISION_CACHE[key]


def generation_finish(provider_reason, output_tokens, limit):
    if provider_reason is not None:
        return dict(finish=provider_reason, truncated=provider_reason == 'length', finish_source='provider')
    return dict(finish='length' if output_tokens >= limit else 'stop',
                truncated=output_tokens >= limit, finish_source='token_count_fallback')
