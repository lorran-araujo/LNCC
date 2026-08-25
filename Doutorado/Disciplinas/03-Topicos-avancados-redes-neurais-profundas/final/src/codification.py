import math

import torch


def _validate_num_qubits(num_qubits):
    if not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer.")


def _ensure_2d(features):
    if not torch.is_tensor(features):
        features = torch.as_tensor(features, dtype=torch.float32)

    if features.ndim == 1:
        return features.unsqueeze(0), True
    if features.ndim == 2:
        return features, False

    raise ValueError("features must be a 1D tensor or a 2D batch tensor.")


def _restore_shape(features, was_1d):
    return features.squeeze(0) if was_1d else features


def _pad_or_truncate(features, target_dim):
    current_dim = features.size(1)
    if current_dim == target_dim:
        return features
    if current_dim > target_dim:
        return features[:, :target_dim]

    padding = features.new_zeros(features.size(0), target_dim - current_dim)
    return torch.cat([features, padding], dim=1)


def angle_encoding(features, num_qubits, value_range=(-math.pi, math.pi), eps=1e-12):
    """Prepare features to be used as rotation angles in a quantum circuit.

    For the standard angle encoding used here, ``num_qubits`` qubits receive
    ``num_qubits`` scalar values. If the input has more features, it is
    truncated. If it has fewer features, it is padded with zeros.

    The output is min-max scaled per sample to ``value_range``. The returned
    tensor can be used later as angles for gates such as RX, RY or RZ.
    """
    _validate_num_qubits(num_qubits)
    if len(value_range) != 2 or value_range[0] >= value_range[1]:
        raise ValueError("value_range must be a tuple like (min_angle, max_angle).")

    features, was_1d = _ensure_2d(features)
    features = _pad_or_truncate(features.float(), num_qubits)

    min_values = features.min(dim=1, keepdim=True).values
    max_values = features.max(dim=1, keepdim=True).values
    scale = (max_values - min_values).clamp_min(eps)

    lower, upper = value_range
    encoded = (features - min_values) / scale
    encoded = encoded * (upper - lower) + lower
    return _restore_shape(encoded, was_1d)


def dense_angle_encoding(features, num_qubits, value_range=(-math.pi, math.pi), eps=1e-12):
    """Prepare three rotation angles per qubit for dense angle encoding."""
    _validate_num_qubits(num_qubits)
    if len(value_range) != 2 or value_range[0] >= value_range[1]:
        raise ValueError("value_range must be a tuple like (min_angle, max_angle).")

    features, was_1d = _ensure_2d(features)
    features = _pad_or_truncate(features.float(), 3 * num_qubits)

    min_values = features.min(dim=1, keepdim=True).values
    max_values = features.max(dim=1, keepdim=True).values
    scale = (max_values - min_values).clamp_min(eps)

    lower, upper = value_range
    encoded = (features - min_values) / scale
    encoded = encoded * (upper - lower) + lower
    return _restore_shape(encoded, was_1d)


def amplitude_encoding(features, num_qubits, eps=1e-12):
    """Prepare normalized amplitudes for a quantum state with ``num_qubits``.

    A system with ``num_qubits`` has ``2 ** num_qubits`` computational basis
    states. Therefore, the feature vector is padded or truncated to this size
    and normalized to unit L2 norm so it can be interpreted as amplitudes.
    """
    _validate_num_qubits(num_qubits)

    features, was_1d = _ensure_2d(features)
    target_dim = 2 ** num_qubits
    amplitudes = _pad_or_truncate(features.float(), target_dim)

    norms = torch.linalg.vector_norm(amplitudes, ord=2, dim=1, keepdim=True)
    zero_norm = norms <= eps

    if zero_norm.any():
        amplitudes = amplitudes.clone()
        amplitudes[zero_norm.squeeze(1)] = 0.0
        amplitudes[zero_norm.squeeze(1), 0] = 1.0
        norms = torch.linalg.vector_norm(amplitudes, ord=2, dim=1, keepdim=True)

    amplitudes = amplitudes / norms.clamp_min(eps)
    return _restore_shape(amplitudes, was_1d)


def encode_features(features, encoding, num_qubits, **kwargs):
    """Dispatch feature preparation by encoding name."""
    encoding = encoding.lower()
    if encoding == "angle":
        return angle_encoding(features, num_qubits, **kwargs)
    if encoding == "reupload_angle":
        return angle_encoding(features, num_qubits, **kwargs)
    if encoding == "dense_angle":
        return dense_angle_encoding(features, num_qubits, **kwargs)
    if encoding == "amplitude":
        return amplitude_encoding(features, num_qubits, **kwargs)

    raise ValueError(f"Unsupported encoding: {encoding}")
