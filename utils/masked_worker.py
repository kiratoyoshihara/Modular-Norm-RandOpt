"""Module-subset controls with unchanged full-method scales."""
from __future__ import annotations

from utils.perturbation_norms import parameter_group
from utils.worker_extn_ablation import WorkerExtension as FullWorker


def mask_scales(scales, family):
    if family not in ("attention", "mlp"):
        raise ValueError("Mask must be attention or mlp")
    return {name: value if parameter_group(name) == family else float("inf")
            for name, value in scales.items()}


class WorkerExtension(FullWorker):
    def configure_mask(self, family):
        if family not in ("attention", "mlp"):
            raise ValueError("Mask must be attention or mlp")
        self.parameter_mask = family
        return {"parameter_mask": family, "renormalize": False}

    def _get_modular_scales(self, *args, **kwargs):
        scales = super()._get_modular_scales(*args, **kwargs)
        return mask_scales(scales, self.parameter_mask)
