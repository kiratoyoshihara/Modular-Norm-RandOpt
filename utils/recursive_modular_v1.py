"""Version-1 recursive modular scaling for Qwen-style decoder models.

This module builds a *shadow* module tree from parameter names. It does not
change the model forward pass. The purpose of version 1 is to isolate the
architecture/mass recursion effect while using unit-sensitivity proxies for
Qwen-specific nonlinear/bond modules.

Important scope:
    * Atomic tensors keep the practical natural norms implemented in
      ``utils.perturbation_norms``.
    * Qwen residual blocks are assigned proxy sensitivity 1.0 to avoid the
      exponential sensitivity growth obtained by naively composing unscaled
      residual additions.
    * Attention, RMSNorm, RoPE, SwiGLU, and other parameter-free operations use
      proxy sensitivity 1.0.
    * The existing six-entry mass configuration is interpreted hierarchically:
      embedding/head are whole-model masses; attention/mlp are distributed
      across decoder layers and logical projections; norm/other are distributed
      across matching physical parameter modules.

Consequently, this is best described as "recursive modular scaling adapted to
Qwen (unit-sensitivity version)", rather than a theorem-preserving reproduction
of the Modula GPT architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import statistics
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch

_EPS = 1e-12
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ParameterSpec:
    """Architecture role inferred for one physical parameter tensor."""

    name: str
    shape: Tuple[int, ...]
    group: str
    role: str
    logical_module: str
    layer_index: int | None = None
    multiplicity: float = 1.0


@dataclass
class ModuleNode:
    """Minimal shadow-module node used for recursive scale propagation."""

    name: str
    kind: str  # atom, compose, concat, residual
    mass: float
    sensitivity: float
    children: List["ModuleNode"] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "mass": float(self.mass),
            "sensitivity": float(self.sensitivity),
            "parameter_names": list(self.parameter_names),
            "metadata": dict(self.metadata),
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class RecursiveModularResult:
    scales: Dict[str, float]
    assigned_masses: Dict[str, float]
    specs: Dict[str, ParameterSpec]
    tree: ModuleNode
    diagnostics: Dict[str, Any]


def _module_prefix(name: str) -> str:
    if name.endswith(".weight") or name.endswith(".bias"):
        return name.rsplit(".", 1)[0]
    return name


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def infer_qwen_parameter_spec(name: str, shape: Sequence[int]) -> ParameterSpec:
    """Infer a robust Qwen/HF/vLLM role from a parameter name.

    The matcher supports both unfused Hugging Face projection names and common
    vLLM fused names such as ``qkv_proj`` and ``gate_up_proj``.
    """

    lowered = name.lower()
    layer = _layer_index(lowered)
    module = _module_prefix(name)

    if _contains_any(lowered, ("lm_head", "output_projection", "unembed")):
        return ParameterSpec(name, tuple(shape), "head", "lm_head", module, None, 1.0)

    if _contains_any(
        lowered,
        (
            "embed_tokens",
            "tok_embeddings",
            "word_embeddings",
            ".wte.",
            "position_embeddings",
            ".wpe.",
        ),
    ):
        return ParameterSpec(name, tuple(shape), "embedding", "embedding", module, None, 1.0)

    if layer is not None:
        if _contains_any(lowered, ("input_layernorm", "input_layer_norm")):
            return ParameterSpec(name, tuple(shape), "norm", "input_norm", module, layer, 1.0)
        if _contains_any(
            lowered,
            (
                "post_attention_layernorm",
                "post_attention_layer_norm",
                "post_attn_layernorm",
            ),
        ):
            return ParameterSpec(name, tuple(shape), "norm", "post_attention_norm", module, layer, 1.0)

        if _contains_any(lowered, ("qkv_proj", "query_key_value", "w_pack", "c_attn")):
            return ParameterSpec(name, tuple(shape), "attention", "qkv", module, layer, 3.0)
        if _contains_any(lowered, ("q_proj", ".query.")):
            return ParameterSpec(name, tuple(shape), "attention", "q", module, layer, 1.0)
        if _contains_any(lowered, ("k_proj", ".key.")):
            return ParameterSpec(name, tuple(shape), "attention", "k", module, layer, 1.0)
        if _contains_any(lowered, ("v_proj", ".value.")):
            return ParameterSpec(name, tuple(shape), "attention", "v", module, layer, 1.0)
        if _contains_any(
            lowered,
            ("o_proj", "out_proj", "attention.dense", "self_attn.dense"),
        ):
            return ParameterSpec(name, tuple(shape), "attention", "o", module, layer, 1.0)

        if _contains_any(lowered, ("gate_up_proj", "gate_up", "w13")):
            return ParameterSpec(name, tuple(shape), "mlp", "gate_up", module, layer, 2.0)
        if _contains_any(lowered, ("gate_proj", ".w1.")):
            return ParameterSpec(name, tuple(shape), "mlp", "gate", module, layer, 1.0)
        if _contains_any(lowered, ("up_proj", ".w3.")):
            return ParameterSpec(name, tuple(shape), "mlp", "up", module, layer, 1.0)
        if _contains_any(lowered, ("down_proj", ".w2.", ".fc2.")):
            return ParameterSpec(name, tuple(shape), "mlp", "down", module, layer, 1.0)
        if _contains_any(lowered, (".mlp.", "feed_forward", "feedforward", ".fc1.")):
            return ParameterSpec(name, tuple(shape), "mlp", "mlp_other", module, layer, 1.0)
        if _contains_any(lowered, ("self_attn", "attention", ".attn.")):
            return ParameterSpec(name, tuple(shape), "attention", "attention_other", module, layer, 1.0)
        if _contains_any(lowered, ("norm", "layernorm", "layer_norm", "rmsnorm", "rms_norm")):
            return ParameterSpec(name, tuple(shape), "norm", "layer_norm_other", module, layer, 1.0)

    if _contains_any(lowered, ("model.norm", "final_layernorm", "final_layer_norm", "final_norm")):
        return ParameterSpec(name, tuple(shape), "norm", "final_norm", module, None, 1.0)

    if _contains_any(lowered, ("layernorm", "layer_norm", "rmsnorm", "rms_norm", ".norm.")):
        return ParameterSpec(name, tuple(shape), "norm", "norm_other", module, layer, 1.0)

    return ParameterSpec(name, tuple(shape), "other", "other", module, layer, 1.0)


def infer_parameter_spec(
    name: str,
    shape: Sequence[int],
    architecture: str = "qwen2",
) -> ParameterSpec:
    """Classify one HF/vLLM tensor into an architecture-independent role."""

    spec = infer_qwen_parameter_spec(name, shape)
    if architecture not in {"olmo3", "gemma3"} or spec.layer_index is None:
        return spec

    lowered = name.lower()
    module = _module_prefix(name)
    layer = spec.layer_index
    if _contains_any(lowered, (".self_attn.q_norm.", ".attention.q_norm.")):
        return ParameterSpec(name, tuple(shape), "norm", "q_norm", module, layer, 1.0)
    if _contains_any(lowered, (".self_attn.k_norm.", ".attention.k_norm.")):
        return ParameterSpec(name, tuple(shape), "norm", "k_norm", module, layer, 1.0)
    if architecture == "gemma3" and _contains_any(
        lowered, ("pre_feedforward_layernorm", "pre_feedforward_layer_norm")
    ):
        return ParameterSpec(
            name, tuple(shape), "norm", "pre_feedforward_norm", module, layer, 1.0
        )
    if _contains_any(
        lowered,
        ("post_feedforward_layernorm", "post_feedforward_layer_norm"),
    ):
        return ParameterSpec(
            name, tuple(shape), "norm", "mlp_output_norm", module, layer, 1.0
        )
    if _contains_any(
        lowered,
        (
            "post_attention_layernorm",
            "post_attention_layer_norm",
            "post_attn_layernorm",
        ),
    ):
        return ParameterSpec(
            name, tuple(shape), "norm", "attention_output_norm", module, layer, 1.0
        )
    if architecture == "gemma3" and _contains_any(
        lowered, ("input_layernorm", "input_layer_norm")
    ):
        return ParameterSpec(name, tuple(shape), "norm", "input_norm", module, layer, 1.0)
    return spec


def _atom(spec: ParameterSpec, mass: float) -> ModuleNode:
    return ModuleNode(
        name=spec.logical_module,
        kind="atom",
        mass=float(mass),
        sensitivity=1.0,
        parameter_names=[spec.name],
        metadata={
            "group": spec.group,
            "role": spec.role,
            "layer_index": spec.layer_index,
            "shape": list(spec.shape),
            "multiplicity": float(spec.multiplicity),
        },
    )


def _combine(
    kind: str,
    name: str,
    children: Sequence[ModuleNode],
    *,
    sensitivity_override: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModuleNode:
    active = [child for child in children if child.mass > 0.0]
    if not active:
        return ModuleNode(name, kind, 0.0, 1.0, [], [], dict(metadata or {}))
    if len(active) == 1 and kind != "residual":
        # Keep a named wrapper so diagnostics retain architecture boundaries.
        child = active[0]
        return ModuleNode(
            name=name,
            kind=kind,
            mass=child.mass,
            sensitivity=(
                float(sensitivity_override)
                if sensitivity_override is not None
                else child.sensitivity
            ),
            children=[child],
            metadata=dict(metadata or {}),
        )

    mass = sum(child.mass for child in active)
    if sensitivity_override is not None:
        sensitivity = float(sensitivity_override)
    elif kind == "compose":
        sensitivity = math.prod(child.sensitivity for child in active)
    elif kind == "concat":
        sensitivity = sum(child.sensitivity for child in active)
    elif kind == "residual":
        sensitivity = 1.0
    else:
        raise ValueError(f"Unsupported node kind: {kind}")

    return ModuleNode(
        name=name,
        kind=kind,
        mass=float(mass),
        sensitivity=float(sensitivity),
        children=list(active),
        metadata=dict(metadata or {}),
    )


def _compose(name: str, children: Sequence[ModuleNode], **kwargs: Any) -> ModuleNode:
    return _combine("compose", name, children, **kwargs)


def _concat(name: str, children: Sequence[ModuleNode], **kwargs: Any) -> ModuleNode:
    return _combine("concat", name, children, **kwargs)


def _residual(name: str, trainable_branch: ModuleNode) -> ModuleNode:
    # Qwen uses an unscaled residual addition. Naively applying the paper's
    # concatenation sensitivity rule would double sensitivity in every block,
    # causing exponential depth factors. Version 1 deliberately uses a unit-
    # sensitivity proxy while retaining the trainable branch's modular norm.
    return _combine(
        "residual",
        name,
        [trainable_branch],
        sensitivity_override=1.0,
        metadata={"identity_branch_mass": 0.0, "sensitivity_proxy": "unit"},
    )


def _group_by_logical_module(specs: Sequence[ParameterSpec]) -> Dict[str, List[ParameterSpec]]:
    grouped: Dict[str, List[ParameterSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.logical_module, []).append(spec)
    return grouped


def _assign_group_mass(
    specs: Sequence[ParameterSpec],
    total_mass: float,
    *,
    weight_by_multiplicity: bool,
) -> Dict[str, float]:
    """Assign one group mass across logical modules and physical tensors."""

    if not specs:
        return {}
    if total_mass <= 0.0:
        return {spec.name: 0.0 for spec in specs}

    modules = _group_by_logical_module(specs)
    module_weights: Dict[str, float] = {}
    for module_name, module_specs in modules.items():
        if weight_by_multiplicity:
            module_weights[module_name] = max(spec.multiplicity for spec in module_specs)
        else:
            module_weights[module_name] = 1.0

    weight_sum = sum(module_weights.values())
    assigned: Dict[str, float] = {}
    for module_name, module_specs in modules.items():
        module_mass = total_mass * module_weights[module_name] / weight_sum
        tensor_mass = module_mass / len(module_specs)
        for spec in module_specs:
            assigned[spec.name] = float(tensor_mass)
    return assigned


def _allocate_masses(
    specs: Sequence[ParameterSpec],
    mass_config: Mapping[str, float],
) -> Dict[str, float]:
    assigned = {spec.name: 0.0 for spec in specs}

    by_group: Dict[str, List[ParameterSpec]] = {
        group: [spec for spec in specs if spec.group == group]
        for group in mass_config
    }

    # Whole-model groups.
    for group in ("embedding", "head", "norm", "other"):
        assigned.update(
            _assign_group_mass(
                by_group.get(group, []),
                float(mass_config.get(group, 0.0)),
                weight_by_multiplicity=False,
            )
        )

    # Attention/MLP are first divided equally across represented layers, then
    # distributed inside each layer by logical projection multiplicity.
    for group in ("attention", "mlp"):
        group_specs = by_group.get(group, [])
        layer_ids = sorted({spec.layer_index for spec in group_specs if spec.layer_index is not None})
        non_layer_specs = [spec for spec in group_specs if spec.layer_index is None]
        total_mass = float(mass_config.get(group, 0.0))

        partitions: List[List[ParameterSpec]] = [
            [spec for spec in group_specs if spec.layer_index == layer_id]
            for layer_id in layer_ids
        ]
        if non_layer_specs:
            partitions.append(non_layer_specs)

        if not partitions:
            continue
        partition_mass = total_mass / len(partitions)
        for partition in partitions:
            assigned.update(
                _assign_group_mass(
                    partition,
                    partition_mass,
                    weight_by_multiplicity=True,
                )
            )

    return assigned


def _nodes_for_specs(
    specs: Sequence[ParameterSpec],
    assigned_masses: Mapping[str, float],
) -> List[ModuleNode]:
    return [_atom(spec, assigned_masses[spec.name]) for spec in specs if assigned_masses[spec.name] > 0.0]


def _build_qwen_tree(
    specs: Sequence[ParameterSpec],
    assigned_masses: Mapping[str, float],
) -> ModuleNode:
    by_name = {spec.name: spec for spec in specs}
    layer_ids = sorted({spec.layer_index for spec in specs if spec.layer_index is not None})

    embedding_nodes = _nodes_for_specs(
        [spec for spec in specs if spec.group == "embedding"], assigned_masses
    )
    head_nodes = _nodes_for_specs(
        [spec for spec in specs if spec.group == "head"], assigned_masses
    )
    final_norm_nodes = _nodes_for_specs(
        [spec for spec in specs if spec.role == "final_norm"], assigned_masses
    )

    block_nodes: List[ModuleNode] = []
    consumed: set[str] = set()
    consumed.update(node.parameter_names[0] for node in embedding_nodes)
    consumed.update(node.parameter_names[0] for node in head_nodes)
    consumed.update(node.parameter_names[0] for node in final_norm_nodes)

    for layer_id in layer_ids:
        layer_specs = [spec for spec in specs if spec.layer_index == layer_id]
        input_norm_specs = [spec for spec in layer_specs if spec.role == "input_norm"]
        post_norm_specs = [spec for spec in layer_specs if spec.role == "post_attention_norm"]
        attention_specs = [spec for spec in layer_specs if spec.group == "attention"]
        mlp_specs = [spec for spec in layer_specs if spec.group == "mlp"]

        input_norm = _concat(
            f"layer_{layer_id}.input_norm",
            _nodes_for_specs(input_norm_specs, assigned_masses),
            sensitivity_override=1.0,
        )
        attention_projection_nodes = _nodes_for_specs(attention_specs, assigned_masses)
        qkv_nodes = [
            node
            for node in attention_projection_nodes
            if by_name[node.parameter_names[0]].role in {"q", "k", "v", "qkv", "attention_other"}
        ]
        o_nodes = [
            node
            for node in attention_projection_nodes
            if by_name[node.parameter_names[0]].role == "o"
        ]
        attention_parallel = _concat(
            f"layer_{layer_id}.qkv_parallel",
            qkv_nodes,
            sensitivity_override=1.0,
            metadata={"functional_attention_sensitivity_proxy": 1.0},
        )
        attention_core = _compose(
            f"layer_{layer_id}.attention_core",
            [attention_parallel, *o_nodes],
            sensitivity_override=1.0,
        )
        attention_branch = _compose(
            f"layer_{layer_id}.attention_branch",
            [input_norm, attention_core],
            sensitivity_override=1.0,
            metadata={"rope_sensitivity_proxy": 1.0},
        )
        attention_residual = _residual(
            f"layer_{layer_id}.attention_residual", attention_branch
        )

        post_norm = _concat(
            f"layer_{layer_id}.post_attention_norm",
            _nodes_for_specs(post_norm_specs, assigned_masses),
            sensitivity_override=1.0,
        )
        mlp_projection_nodes = _nodes_for_specs(mlp_specs, assigned_masses)
        gate_up_nodes = [
            node
            for node in mlp_projection_nodes
            if by_name[node.parameter_names[0]].role
            in {"gate", "up", "gate_up", "mlp_other"}
        ]
        down_nodes = [
            node
            for node in mlp_projection_nodes
            if by_name[node.parameter_names[0]].role == "down"
        ]
        gate_up_parallel = _concat(
            f"layer_{layer_id}.gate_up_parallel",
            gate_up_nodes,
            sensitivity_override=1.0,
            metadata={"swiglu_sensitivity_proxy": 1.0},
        )
        mlp_core = _compose(
            f"layer_{layer_id}.mlp_core",
            [gate_up_parallel, *down_nodes],
            sensitivity_override=1.0,
        )
        mlp_branch = _compose(
            f"layer_{layer_id}.mlp_branch",
            [post_norm, mlp_core],
            sensitivity_override=1.0,
        )
        mlp_residual = _residual(f"layer_{layer_id}.mlp_residual", mlp_branch)

        block = _compose(
            f"layer_{layer_id}.block",
            [attention_residual, mlp_residual],
            sensitivity_override=1.0,
            metadata={"block_sensitivity_proxy": 1.0},
        )
        if block.mass > 0.0:
            block_nodes.append(block)

        consumed.update(spec.name for spec in input_norm_specs)
        consumed.update(spec.name for spec in post_norm_specs)
        consumed.update(spec.name for spec in attention_specs)
        consumed.update(spec.name for spec in mlp_specs)

    # Norm tensors not recognized as the two standard per-block norms or final
    # norm, plus any other unmatched tensors, remain explicit fallback atoms.
    fallback_specs = [
        spec
        for spec in specs
        if spec.name not in consumed and assigned_masses[spec.name] > 0.0
    ]
    fallback_nodes = _nodes_for_specs(fallback_specs, assigned_masses)

    ordered_nodes: List[ModuleNode] = []
    if embedding_nodes:
        ordered_nodes.append(
            _concat("token_embedding", embedding_nodes, sensitivity_override=1.0)
        )
    ordered_nodes.extend(block_nodes)
    if final_norm_nodes:
        ordered_nodes.append(
            _concat("final_norm", final_norm_nodes, sensitivity_override=1.0)
        )
    if head_nodes:
        ordered_nodes.append(_concat("lm_head", head_nodes, sensitivity_override=1.0))
    if fallback_nodes:
        ordered_nodes.append(
            _concat(
                "structural_fallback_parameters",
                fallback_nodes,
                sensitivity_override=1.0,
                metadata={"warning": "parameters not placed in canonical Qwen subtrees"},
            )
        )

    root = _compose(
        "qwen_recursive_modular_v1",
        ordered_nodes,
        sensitivity_override=1.0,
        metadata={
            "version": 1,
            "sensitivity_policy": "unit proxies for Qwen bonds and residual blocks",
        },
    )
    if root.mass <= 0.0:
        raise ValueError("Recursive modular tree has zero active mass")
    return root


def _propagate_scales(
    node: ModuleNode,
    incoming_scale: float,
    output: MutableMapping[str, float],
) -> None:
    if node.kind == "atom":
        if not node.parameter_names:
            raise ValueError(f"Atom {node.name!r} has no parameter")
        for name in node.parameter_names:
            if name in output:
                raise ValueError(f"Parameter {name!r} appears multiple times in module tree")
            output[name] = float(incoming_scale)
        return

    children = [child for child in node.children if child.mass > 0.0]
    if not children:
        return

    if node.kind == "compose":
        for index, child in enumerate(children):
            downstream_sensitivity = math.prod(
                later.sensitivity for later in children[index + 1 :]
            )
            local_factor = downstream_sensitivity * node.mass / child.mass
            _propagate_scales(child, incoming_scale * local_factor, output)
        return

    if node.kind == "concat":
        for child in children:
            local_factor = node.mass / child.mass
            _propagate_scales(child, incoming_scale * local_factor, output)
        return

    if node.kind == "residual":
        # Identity branch has zero mass; the trainable branch receives the full
        # residual-module mass. The node's proxy sensitivity only affects any
        # *upstream* modules when this residual is used in composition.
        for child in children:
            local_factor = node.mass / child.mass
            _propagate_scales(child, incoming_scale * local_factor, output)
        return

    raise ValueError(f"Unsupported node kind: {node.kind}")


def _finite_summary(values: Sequence[float]) -> Dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "median": statistics.median(finite),
        "max": max(finite),
    }


def build_recursive_modular_result(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    mass_config: Mapping[str, float],
    architecture: str = "qwen2",
) -> RecursiveModularResult:
    """Build the version-1 Qwen shadow tree and per-tensor scales."""

    pairs = list(named_parameters)
    if not pairs:
        raise ValueError("No perturbable parameters were provided")

    specs_list = [
        infer_parameter_spec(name, tuple(param.shape), architecture)
        for name, param in pairs
    ]
    specs = {spec.name: spec for spec in specs_list}
    assigned_masses = _allocate_masses(specs_list, mass_config)
    tree = _build_qwen_tree(specs_list, assigned_masses)

    active_scales: Dict[str, float] = {}
    _propagate_scales(tree, 1.0, active_scales)

    scales: Dict[str, float] = {}
    for spec in specs_list:
        mass = float(assigned_masses.get(spec.name, 0.0))
        if mass <= 0.0:
            scales[spec.name] = float("inf")
        else:
            if spec.name not in active_scales:
                raise RuntimeError(
                    f"Positive-mass parameter {spec.name!r} was not placed in the recursive tree"
                )
            scale = float(active_scales[spec.name])
            if not math.isfinite(scale) or scale <= 0.0:
                raise RuntimeError(f"Invalid recursive scale for {spec.name!r}: {scale}")
            scales[spec.name] = scale

    group_rows: Dict[str, Dict[str, Any]] = {}
    for group in mass_config:
        group_names = [spec.name for spec in specs_list if spec.group == group]
        group_rows[group] = {
            "configured_mass": float(mass_config[group]),
            "present_parameter_count": len(group_names),
            "active_parameter_count": sum(
                1 for name in group_names if assigned_masses.get(name, 0.0) > 0.0
            ),
            "assigned_mass_sum": float(
                sum(assigned_masses.get(name, 0.0) for name in group_names)
            ),
            "scale_summary": _finite_summary([scales[name] for name in group_names]),
        }

    parameter_rows = []
    for spec in specs_list:
        parameter_rows.append(
            {
                "parameter_name": spec.name,
                "shape": list(spec.shape),
                "group": spec.group,
                "role": spec.role,
                "logical_module": spec.logical_module,
                "layer_index": spec.layer_index,
                "multiplicity": float(spec.multiplicity),
                "assigned_mass": float(assigned_masses[spec.name]),
                "scale": (
                    float(scales[spec.name])
                    if math.isfinite(scales[spec.name])
                    else "inf"
                ),
            }
        )

    diagnostics = {
        "method": "recursive_modular_shell",
        "version": 1,
        "description": "Qwen shadow-tree recursion with unit sensitivity proxies",
        "assumptions": [
            "Qwen forward pass is unchanged.",
            "RMSNorm, RoPE, functional attention, SwiGLU, and residual blocks use sensitivity 1.0 proxies.",
            "Attention/MLP masses are divided across layers and projection multiplicities.",
            "Fused qkv and gate_up tensors receive multiplicities 3 and 2 respectively.",
            "Zero-mass groups are frozen for recursive modular perturbations.",
        ],
        "root_mass": float(tree.mass),
        "root_sensitivity": float(tree.sensitivity),
        "num_parameters": len(specs_list),
        "num_active_parameters": sum(
            1 for value in assigned_masses.values() if value > 0.0
        ),
        "num_layers_detected": len(
            {spec.layer_index for spec in specs_list if spec.layer_index is not None}
        ),
        "mass_config": {key: float(value) for key, value in mass_config.items()},
        "group_statistics": group_rows,
        "scale_statistics": _finite_summary(list(scales.values())),
        "parameters": parameter_rows,
        "tree": tree.to_dict(),
    }

    return RecursiveModularResult(
        scales=scales,
        assigned_masses=assigned_masses,
        specs=specs,
        tree=tree,
        diagnostics=diagnostics,
    )


def build_recursive_modular_scales(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    mass_config: Mapping[str, float],
) -> Dict[str, float]:
    return build_recursive_modular_result(named_parameters, mass_config).scales


def recursive_modular_diagnostics(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    mass_config: Mapping[str, float],
) -> Dict[str, Any]:
    return build_recursive_modular_result(named_parameters, mass_config).diagnostics


_RMSNORM_ONLY_TARGET_ROLES: Tuple[str, ...] = ("input_norm", "post_attention_norm")


def build_recursive_modular_result_rmsnorm_only(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    mass_config: Mapping[str, float],
    rmsnorm_multiplier: float = 2.0,
) -> RecursiveModularResult:
    """Ablation: V1 shadow-tree scales, with only per-block RMSNorm scales multiplied.

    This isolates whether V2's improvement over V1 is explained purely by
    shrinking the perturbation applied to per-block RMSNorm weights (V2's
    ``final_v2_to_v1_ratio`` was exactly 2.0 for these parameters and 1.0 for
    the final ``model.norm.weight``). Every other parameter keeps its exact V1
    scale, noise, and denominator, so a candidate generated with a given seed
    differs from its V1 counterpart only in the RMSNorm sub-tensors.
    """

    if rmsnorm_multiplier <= 0.0 or not math.isfinite(rmsnorm_multiplier):
        raise ValueError("rmsnorm_multiplier must be finite and positive")

    v1_result = build_recursive_modular_result(named_parameters, mass_config)

    target_names = {
        spec.name
        for spec in v1_result.specs.values()
        if spec.role in _RMSNORM_ONLY_TARGET_ROLES
    }
    num_layers = v1_result.diagnostics["num_layers_detected"]
    expected = 2 * num_layers
    if len(target_names) != expected:
        raise RuntimeError(
            "RMSNorm-only ablation expected "
            f"{expected} target parameters (2 per detected layer), found "
            f"{len(target_names)}. Refusing to proceed with a mismatched allowlist."
        )
    if any(spec.group != "norm" for spec in v1_result.specs.values() if spec.name in target_names):
        raise RuntimeError("RMSNorm-only target parameters must all belong to the 'norm' group")

    scales: Dict[str, float] = dict(v1_result.scales)
    for name in target_names:
        v1_scale = v1_result.scales[name]
        if math.isfinite(v1_scale):
            scales[name] = v1_scale * rmsnorm_multiplier

    parameter_rows = []
    for row in v1_result.diagnostics["parameters"]:
        name = row["parameter_name"]
        v1_scale = row["scale"]
        is_target = name in target_names
        multiplier = rmsnorm_multiplier if is_target else 1.0
        effective_scale = scales[name] if math.isfinite(scales[name]) else "inf"
        parameter_rows.append(
            {
                **row,
                "v1_scale": v1_scale,
                "rmsnorm_multiplier": multiplier,
                "scale": effective_scale,
            }
        )

    # v1_result.diagnostics["scale_statistics"] and ["group_statistics"] were
    # computed from the *unmodified* V1 scales. Only the "norm" group actually
    # changes here (RMSNorm targets live in that group), but both the global
    # and the norm-group summaries must be recomputed against the doubled
    # scales, otherwise the saved diagnostics silently misreport this run as
    # having V1's scale distribution.
    group_rows: Dict[str, Any] = {
        group: dict(row) for group, row in v1_result.diagnostics["group_statistics"].items()
    }
    norm_group_names = [spec.name for spec in v1_result.specs.values() if spec.group == "norm"]
    group_rows["norm"] = {
        **group_rows["norm"],
        "scale_summary": _finite_summary([scales[name] for name in norm_group_names]),
    }

    diagnostics = {
        **v1_result.diagnostics,
        "method": "recursive_modular_shell_rmsnorm_only",
        "version": "1+rmsnorm_only",
        "description": (
            "V1 shadow-tree scales with only per-block RMSNorm (input_norm, "
            "post_attention_norm) scales multiplied by rmsnorm_multiplier. "
            "Ablation control for recursive_modular_shell_v2: isolates the "
            "RMSNorm-scale-shrink effect from the rest of V2's sensitivity "
            "calibration."
        ),
        "base_method": "recursive_modular_shell",
        "rmsnorm_multiplier": float(rmsnorm_multiplier),
        "rmsnorm_target_roles": list(_RMSNORM_ONLY_TARGET_ROLES),
        "rmsnorm_target_parameter_names": sorted(target_names),
        "num_rmsnorm_parameters_scaled": len(target_names),
        "sensitivity_profile_used": False,
        "scale_statistics": _finite_summary(list(scales.values())),
        "group_statistics": group_rows,
        "parameters": parameter_rows,
    }

    return RecursiveModularResult(
        scales=scales,
        assigned_masses=v1_result.assigned_masses,
        specs=v1_result.specs,
        tree=v1_result.tree,
        diagnostics=diagnostics,
    )


def build_recursive_modular_scales_rmsnorm_only(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    mass_config: Mapping[str, float],
    rmsnorm_multiplier: float = 2.0,
) -> Dict[str, float]:
    return build_recursive_modular_result_rmsnorm_only(
        named_parameters, mass_config, rmsnorm_multiplier=rmsnorm_multiplier
    ).scales
