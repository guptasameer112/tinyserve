"""Helpers to compare our modules with the Hugging Face model on real activations."""
import torch

from tinyserve.models.norm import RMSNorm


@torch.no_grad()
def capture_module_io(model, input_ids, module_type):
    """Run one forward pass on input_ids (N,) and record (name, input, output) of every module of module_type."""
    records, handles = [], []
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            def hook(_module, args, output, name=name):
                records.append((name, args[0].detach(), output.detach()))

            handles.append(module.register_forward_hook(hook))
    try:
        model(input_ids[None, :])
    finally:
        for handle in handles:
            handle.remove()
    return records


@torch.no_grad()
def compare_rmsnorm(model, input_ids, module_type, eps):
    """For every RMSNorm in the model: feed the real captured input to OUR RMSNorm (with the same weight)
    and report the max |difference| from the Hugging Face output, plus the input's size."""
    modules = dict(model.named_modules())
    rows = []
    for name, norm_input, reference_output in capture_module_io(model, input_ids, module_type):
        ours = RMSNorm(norm_input.shape[-1], eps, device=norm_input.device, dtype=modules[name].weight.dtype)
        ours.weight.copy_(modules[name].weight)
        our_output = ours(norm_input)
        rows.append({
            "name": name,
            "input_rms": norm_input.float().pow(2).mean().sqrt().item(),
            "input_max_abs": norm_input.float().abs().max().item(),
            "max_abs_diff": (our_output.float() - reference_output.float()).abs().max().item(),
            "exactly_equal": bool(torch.equal(our_output, reference_output)),
        })
    return rows
