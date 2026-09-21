"""Strict validation for functional-distance-matched evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.distance_matching import (
    DISTANCE_MATCH_PROTOCOL,
    build_three_matched_targets,
    monotone_log_fit,
)
from utils.official_randopt_protocol import (
    CANDIDATE_SEED_SCHEME,
    PARAMETER_NOISE_SCHEME,
    WEIGHT_RESTORE_SCHEME,
    build_candidate_seeds,
)
from utils.perturbation_norms import sensitivity_profile_fingerprint


MATCHED_METHODS = ("isotropic", "recursive_modular_shell_v2")
MATCHED_TARGETS = ("low", "reference", "high")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {target}")
    return payload


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"Distance-match artifact {name} mismatch: {actual!r} != {expected!r}"
        )


def _require_close(
    name: str,
    actual: float,
    expected: float,
    *,
    relative: float = 1e-10,
    absolute: float = 1e-15,
) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=relative, abs_tol=absolute
    ):
        raise ValueError(
            f"Distance-match artifact {name} mismatch: {actual!r} != {expected!r}"
        )


def _validate_source_files(payload: Mapping[str, Any], repository_root: Path) -> None:
    source = payload.get("source_manifest") or {}
    files = source.get("files") or {}
    if not files:
        raise ValueError("Distance-match artifact has no source manifest")
    normalized: dict[str, str] = {}
    for relative, expected_hash in files.items():
        path = repository_root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Distance-match source file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Distance-match source changed after scale selection: {relative}"
            )
        normalized[str(relative)] = str(expected_hash)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    combined = hashlib.sha256(encoded).hexdigest()
    _require_equal("source combined hash", source.get("combined_sha256"), combined)


def _curve_fits(payload: Mapping[str, Any]) -> dict[str, list[dict[str, float]]]:
    curves = payload.get("curves") or {}
    result: dict[str, list[dict[str, float]]] = {}
    for method in MATCHED_METHODS:
        curve = curves.get(method) or {}
        raw = list(curve.get("raw") or [])
        if len(raw) < 4:
            raise ValueError(f"Distance-match artifact has fewer than four {method} scales")
        scales = [float(row["scale"]) for row in raw]
        distances = [float(row["aggregate"]["median_output_kl"]) for row in raw]
        recomputed = monotone_log_fit(scales, distances)
        stored = list(curve.get("monotone_log_fit") or [])
        if len(stored) != len(recomputed):
            raise ValueError(f"Distance-match fitted curve length is invalid for {method}")
        for index, (actual, expected) in enumerate(zip(stored, recomputed)):
            for key in ("scale", "raw_distance", "fitted_distance"):
                _require_close(
                    f"{method} fit[{index}].{key}",
                    float(actual.get(key, float("nan"))),
                    float(expected[key]),
                )
        result[method] = recomputed
    return result


def _validated_targets(
    payload: Mapping[str, Any],
    fits: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, Mapping[str, Any]]:
    reference_scale = float(payload.get("reference_modular_scale", float("nan")))
    if not math.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Distance-match reference Modular scale is invalid")
    recomputed = build_three_matched_targets(
        fits["isotropic"],
        fits["recursive_modular_shell_v2"],
        reference_modular_scale=reference_scale,
    )
    targets = list(payload.get("targets") or [])
    if [row.get("name") for row in targets] != list(MATCHED_TARGETS):
        raise ValueError("Distance-match targets must be ordered low/reference/high")
    for actual, expected in zip(targets, recomputed):
        for key in (
            "target_median_output_kl",
            "randopt_scale",
            "modular_scale",
            "randopt_fitted_kl",
            "modular_fitted_kl",
        ):
            _require_close(
                f"target {actual.get('name')}.{key}",
                float(actual.get(key, float("nan"))),
                float(expected[key]),
            )
        relative_mismatch = abs(
            float(actual["randopt_fitted_kl"]) - float(actual["modular_fitted_kl"])
        ) / float(actual["target_median_output_kl"])
        if relative_mismatch > 1e-9:
            raise ValueError("Distance-match target fitted KL mismatch is too large")
    return {str(row["name"]): row for row in targets}


def validate_distance_match_artifact(
    artifact_path: str | Path,
    *,
    repository_root: str | Path,
    target_name: str,
    dataset: str,
    model_name: str,
    model_revision: str | None,
    perturbation_method: str,
    radius: float,
    mass_config: Mapping[str, float],
    sensitivity_profile: Mapping[str, Any] | None,
    population_size: int,
    population_prefixes: Sequence[int],
    top_k_values: Sequence[int],
    global_seed: int,
    train_samples: int,
    test_samples: int | None,
    train_data_path: str | Path,
    test_data_path: str | Path,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], str | None]:
    """Validate an artifact against one exact population-evaluation run.

    Returns the parsed artifact and its resolved immutable model revision.
    """

    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing distance-match artifact: {path}")
    payload = read_json_object(path)
    required = {
        "status": "complete",
        "protocol": DISTANCE_MATCH_PROTOCOL,
        "selection_uses_accuracy": False,
        "candidate_seed_scheme": CANDIDATE_SEED_SCHEME,
        "noise_scheme": PARAMETER_NOISE_SCHEME,
        "weight_restore_scheme": WEIGHT_RESTORE_SCHEME,
    }
    for key, expected in required.items():
        _require_equal(key, payload.get(key), expected)
    _require_equal(
        "selection_metric",
        payload.get("selection_metric"),
        "hierarchical median forward output KL",
    )
    if perturbation_method not in MATCHED_METHODS:
        raise ValueError(f"Unsupported matched-control method: {perturbation_method}")
    if target_name not in MATCHED_TARGETS:
        raise ValueError(f"Unknown matched-control target: {target_name}")
    if verify_source_files:
        _validate_source_files(payload, Path(repository_root))

    calibration = payload.get("calibration") or {}
    calibration_pool_size = int(calibration.get("candidate_pool_size", -1))
    calibration_prefix = int(calibration.get("candidate_prefix", -1))
    calibration_seed = int(calibration.get("global_seed", -1))
    calibration_examples = int(calibration.get("num_examples", -1))
    if calibration_examples < 1:
        raise ValueError("Distance-match calibration example count is invalid")
    if calibration_pool_size < 1 or not 1 <= calibration_prefix <= calibration_pool_size:
        raise ValueError("Distance-match calibration candidate sizes are invalid")
    expected_calibration_pool = build_candidate_seeds(
        calibration_seed, calibration_pool_size
    )
    _require_equal(
        "calibration candidate seeds",
        [int(value) for value in calibration.get("candidate_seeds", [])],
        expected_calibration_pool[:calibration_prefix],
    )
    measurement = calibration.get("measurement") or {}
    _require_equal("KL direction", measurement.get("kl_direction"), "KL(base || perturbed)")
    _require_equal("full-vocabulary KL", measurement.get("full_vocabulary_kl"), True)

    sweep = payload.get("source_sweep") or {}
    sweep_path = Path(str(sweep.get("path", "")))
    if not sweep_path.is_file() or sha256_file(sweep_path) != sweep.get("sha256"):
        raise ValueError("Source distance sweep is missing or changed")

    dataset_payload = payload.get("dataset") or {}
    _require_equal("dataset.name", dataset_payload.get("name"), dataset)
    _require_equal("dataset argument", dataset, "gsm8k")
    calibration_data_path = Path(
        str(dataset_payload.get("calibration_data_path", ""))
    )
    calibration_manifest_path = Path(
        str(dataset_payload.get("calibration_manifest_path", ""))
    )
    if (
        not calibration_data_path.is_file()
        or sha256_file(calibration_data_path)
        != dataset_payload.get("calibration_data_sha256")
    ):
        raise ValueError("Held-out distance-calibration data is missing or changed")
    if (
        not calibration_manifest_path.is_file()
        or sha256_file(calibration_manifest_path)
        != dataset_payload.get("calibration_manifest_sha256")
    ):
        raise ValueError("Distance-calibration overlap manifest is missing or changed")
    overlap_manifest = read_json_object(calibration_manifest_path)
    _require_equal(
        "calibration example count",
        calibration_examples,
        int(overlap_manifest.get("num_examples", -1)),
    )
    manifest_calibration = overlap_manifest.get("calibration") or {}
    _require_equal(
        "manifest calibration path",
        Path(str(manifest_calibration.get("path", ""))).resolve(),
        calibration_data_path.resolve(),
    )
    _require_equal(
        "manifest calibration hash",
        manifest_calibration.get("sha256"),
        dataset_payload.get("calibration_data_sha256"),
    )
    model = payload.get("model") or {}
    _require_equal("model.name", model.get("name"), model_name)
    resolved_revision = model.get("resolved_revision")
    if model_revision is not None:
        _require_equal("model revision", model_revision, resolved_revision)

    _require_equal("mass_config", dict(mass_config), payload.get("mass_config"))
    profile = payload.get("fixed_countdown_profile") or {}
    _require_equal("fixed profile dataset", profile.get("dataset"), "countdown")
    profile_path = Path(str(profile.get("path", "")))
    if not profile_path.is_file() or sha256_file(profile_path) != profile.get("sha256"):
        raise ValueError("Fixed Countdown sensitivity profile is missing or changed")
    if perturbation_method == "recursive_modular_shell_v2":
        if sensitivity_profile is None:
            raise ValueError("Modular matched run requires the fixed Countdown profile")
        _require_equal(
            "fixed profile fingerprint",
            sensitivity_profile_fingerprint(sensitivity_profile),
            profile.get("fingerprint"),
        )
    elif sensitivity_profile is not None:
        raise ValueError("RandOpt matched run must not receive a sensitivity profile")

    fits = _curve_fits(payload)
    targets = _validated_targets(payload, fits)
    target = targets[target_name]
    expected_scale = (
        float(target["randopt_scale"])
        if perturbation_method == "isotropic"
        else float(target["modular_scale"])
    )
    _require_close("run radius", float(radius), expected_scale, relative=0.0)

    evaluation = payload.get("matched_evaluation") or {}
    _require_equal("population_size", int(population_size), int(evaluation.get("population_size", -1)))
    _require_equal(
        "population_prefixes",
        [int(value) for value in population_prefixes],
        [int(value) for value in evaluation.get("population_prefixes", [])],
    )
    _require_equal(
        "top_k_values",
        [int(value) for value in top_k_values],
        [int(value) for value in evaluation.get("top_k_values", [])],
    )
    if int(global_seed) not in [int(value) for value in evaluation.get("global_seeds", [])]:
        raise ValueError(f"Seed {global_seed} is outside the matched-evaluation seed set")
    _require_equal("train_samples", int(train_samples), int(evaluation.get("train_samples", -1)))
    expected_test_samples = int(evaluation.get("test_samples", -1))
    if test_samples is not None:
        _require_equal("test_samples", int(test_samples), expected_test_samples)

    for label, actual_path, path_key, hash_key in (
        (
            "selection",
            Path(train_data_path).expanduser().resolve(),
            "selection_data_path",
            "selection_data_sha256",
        ),
        (
            "test",
            Path(test_data_path).expanduser().resolve(),
            "test_data_path",
            "test_data_sha256",
        ),
    ):
        expected_path = Path(str(evaluation.get(path_key, ""))).resolve()
        _require_equal(f"{label} path", actual_path, expected_path)
        if not actual_path.is_file() or sha256_file(actual_path) != evaluation.get(hash_key):
            raise ValueError(f"Matched-control {label} data is missing or changed")
        manifest_row = overlap_manifest.get(label) or {}
        _require_equal(
            f"manifest {label} path",
            Path(str(manifest_row.get("path", ""))).resolve(),
            actual_path,
        )
        _require_equal(
            f"manifest {label} hash",
            manifest_row.get("sha256"),
            evaluation.get(hash_key),
        )
    audit = dataset_payload.get("overlap_audit") or {}
    for key in ("calibration_vs_selection", "calibration_vs_test", "selection_vs_test"):
        _require_equal(f"overlap_audit.{key}", int(audit.get(key, -1)), 0)
        _require_equal(
            f"manifest overlap_audit.{key}",
            int((overlap_manifest.get("overlap_audit") or {}).get(key, -1)),
            0,
        )
    return payload, None if resolved_revision is None else str(resolved_revision)


__all__ = [
    "MATCHED_METHODS",
    "MATCHED_TARGETS",
    "read_json_object",
    "sha256_file",
    "validate_distance_match_artifact",
]
