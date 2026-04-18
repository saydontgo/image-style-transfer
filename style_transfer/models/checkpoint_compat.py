from __future__ import annotations

from collections.abc import Mapping

import torch


StateDict = Mapping[str, torch.Tensor]


def extract_compatible_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    state_dict = _unwrap_state_dict(checkpoint)
    state_dict = _strip_module_prefix_if_present(state_dict)
    if _is_local_layout(state_dict):
        return dict(state_dict)
    if _is_pytorch_examples_layout(state_dict):
        return _remap_pytorch_examples_layout(state_dict)
    if _is_gordicaleksa_layout(state_dict):
        return _remap_gordicaleksa_layout(state_dict)
    raise RuntimeError(_build_incompatible_checkpoint_message(state_dict))


def _unwrap_state_dict(checkpoint: object) -> StateDict:
    if isinstance(checkpoint, Mapping):
        nested = checkpoint.get("state_dict")
        if isinstance(nested, Mapping):
            return nested  # type: ignore[return-value]
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            tensor_values = sum(isinstance(value, torch.Tensor) for value in checkpoint.values())
            if tensor_values >= max(1, len(checkpoint) // 2):
                return checkpoint  # type: ignore[return-value]
    raise TypeError("Checkpoint does not contain a PyTorch state_dict.")


def _strip_module_prefix_if_present(state_dict: StateDict) -> StateDict:
    if not state_dict:
        return state_dict
    # Common when saving from DataParallel/DistributedDataParallel.
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _is_local_layout(state_dict: StateDict) -> bool:
    return "downsampling.0.conv2d.weight" in state_dict and "upsampling.6.conv2d.weight" in state_dict


def _is_pytorch_examples_layout(state_dict: StateDict) -> bool:
    required = {"conv1.conv2d.weight", "res1.conv1.conv2d.weight", "deconv3.conv2d.weight"}
    return required.issubset(state_dict)


def _is_gordicaleksa_layout(state_dict: StateDict) -> bool:
    required = {"conv1.conv2d.weight", "res1.conv1.conv2d.weight", "up3.conv2d.weight"}
    return required.issubset(state_dict)


def _remap_pytorch_examples_layout(state_dict: StateDict) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    _map_downsampling(remapped, state_dict)
    _map_residuals(remapped, state_dict)
    _copy_conv(remapped, state_dict, "deconv1", "upsampling.0.layer")
    _copy_norm(remapped, state_dict, "in4", "upsampling.1")
    _copy_conv(remapped, state_dict, "deconv2", "upsampling.3.layer")
    _copy_norm(remapped, state_dict, "in5", "upsampling.4")
    _copy_conv(remapped, state_dict, "deconv3", "upsampling.6")
    return remapped


def _remap_gordicaleksa_layout(state_dict: StateDict) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    _map_downsampling(remapped, state_dict)
    _map_residuals(remapped, state_dict)
    _copy_nested_conv(remapped, state_dict, "up1.conv2d", "upsampling.0.layer")
    _copy_norm(remapped, state_dict, "in4", "upsampling.1")
    _copy_nested_conv(remapped, state_dict, "up2.conv2d", "upsampling.3.layer")
    _copy_norm(remapped, state_dict, "in5", "upsampling.4")
    _copy_conv(remapped, state_dict, "up3", "upsampling.6")
    return remapped


def _map_downsampling(remapped: dict[str, torch.Tensor], state_dict: StateDict) -> None:
    _copy_conv(remapped, state_dict, "conv1", "downsampling.0")
    _copy_norm(remapped, state_dict, "in1", "downsampling.1")
    _copy_conv(remapped, state_dict, "conv2", "downsampling.3")
    _copy_norm(remapped, state_dict, "in2", "downsampling.4")
    _copy_conv(remapped, state_dict, "conv3", "downsampling.6")
    _copy_norm(remapped, state_dict, "in3", "downsampling.7")


def _map_residuals(remapped: dict[str, torch.Tensor], state_dict: StateDict) -> None:
    for index in range(5):
        block = f"res{index + 1}"
        target = f"residuals.{index}.block"
        _copy_conv(remapped, state_dict, f"{block}.conv1", f"{target}.0")
        _copy_norm(remapped, state_dict, f"{block}.in1", f"{target}.1")
        _copy_conv(remapped, state_dict, f"{block}.conv2", f"{target}.3")
        _copy_norm(remapped, state_dict, f"{block}.in2", f"{target}.4")


def _copy_conv(remapped: dict[str, torch.Tensor], state_dict: StateDict, source: str, target: str) -> None:
    remapped[f"{target}.conv2d.weight"] = state_dict[f"{source}.conv2d.weight"]
    bias_key = f"{source}.conv2d.bias"
    if bias_key in state_dict:
        remapped[f"{target}.conv2d.bias"] = state_dict[bias_key]
    else:
        # Some checkpoints store conv weights only. Default Conv2d uses bias=True,
        # so initialize missing bias to zeros for compatibility.
        weight = state_dict[f"{source}.conv2d.weight"]
        remapped[f"{target}.conv2d.bias"] = torch.zeros(weight.size(0), dtype=weight.dtype, device=weight.device)


def _copy_nested_conv(remapped: dict[str, torch.Tensor], state_dict: StateDict, source: str, target: str) -> None:
    remapped[f"{target}.conv2d.weight"] = state_dict[f"{source}.conv2d.weight"]
    bias_key = f"{source}.conv2d.bias"
    if bias_key in state_dict:
        remapped[f"{target}.conv2d.bias"] = state_dict[bias_key]
    else:
        weight = state_dict[f"{source}.conv2d.weight"]
        remapped[f"{target}.conv2d.bias"] = torch.zeros(weight.size(0), dtype=weight.dtype, device=weight.device)


def _copy_norm(remapped: dict[str, torch.Tensor], state_dict: StateDict, source: str, target: str) -> None:
    remapped[f"{target}.weight"] = state_dict[f"{source}.weight"]
    remapped[f"{target}.bias"] = state_dict[f"{source}.bias"]


def _build_incompatible_checkpoint_message(state_dict: StateDict) -> str:
    sample_keys = ", ".join(sorted(state_dict.keys())[:8])
    hint = (
        "Checkpoint architecture is incompatible with this project. "
        "Supported layouts are: this repo's own checkpoints, "
        "`pytorch/examples/fast_neural_style`, and "
        "`gordicaleksa/pytorch-neural-style-transfer-johnson`."
    )
    if "conv2.conv1.weight" in state_dict or "upconv1.conv1.conv1.weight" in state_dict:
        hint += " The loaded file looks like a different style-transfer network implementation, not a Johnson TransformerNet checkpoint."
    return f"{hint} Sample checkpoint keys: {sample_keys}"
