"""End-to-end experiment runner.

Runs a single job described by a manifest row (CSV) under the v1.1 protocol.

Key invariants enforced:
  - preprocessing and training specs are read from configs/base_protocol.yaml
  - fixed eval indices (and fixed PTQ calibration indices) are used
  - every run produces a JSONSchema-validated receipt

This is intentionally a small, auditable runner rather than a general framework.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from icon_primitive.core.kappa import KappaConfig, compute_kappa
from icon_primitive.data.datasets import (
    DatasetSpec,
    load_dataset,
    make_calib_subset,
    make_dataloader,
    make_eval_subset,
)
from icon_primitive.models.probes import ProbeConfig, create_spatial_probe, create_vector_probe
from icon_primitive.models.quantization import (
    PTQConfig,
    apply_ptq_to_vector_probe,
    cast_model_precision,
    run_ptq_calibration,
)
from icon_primitive.training.scheduler import WarmupCosineConfig
from icon_primitive.training.trainer import TrainConfig, train_classifier
from icon_primitive.utils.hashing import hash_any, hash_state_dict
from icon_primitive.utils.receipt import build_receipt, compute_data_hash, save_receipt
from icon_primitive.utils.seeding import get_seed_config, set_seed


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _canon_concat_projection(v: Any) -> str:
    """Canonicalize concat projection flag to receipt/schema enums.

    Accepts: None / bool / "true"/"false" / "none" / "project_to_width".
    """
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "project_to_width" if v else "none"
    # pandas can give 0/1 as floats
    if isinstance(v, (int, float)):
        return "project_to_width" if float(v) != 0.0 else "none"
    s = str(v).strip().lower()
    if s in {"", "none", "false", "0", "no", "n"}:
        return "none"
    if s in {"true", "1", "yes", "y", "project", "projection", "project_to_width"}:
        return "project_to_width"
    return s


def _canon_linear_budget(v: Any) -> str:
    """Canonicalize linear budget to schema enums."""
    if v is None:
        return "shape_matched"
    s = str(v).strip().lower()
    if s in {"", "none"}:
        return "shape_matched"
    if s in {"shape_matched", "param_matched", "flops_matched"}:
        return s
    # Backwards-compat alias used in some manifests/config packs
    if s == "params_or_flops_matched":
        # Default to param_matched (safe, schema-compliant). The raw value is
        # still preserved in extras.manifest for audit.
        return "param_matched"
    return s


def _canon_section_for_receipt(section: str) -> str:
    """Map manifest section codes to receipt schema enums."""
    s = str(section).strip()
    mapping = {
        "1A": "1A_activation",
        "1B": "1B_linear_type",
        "1B_ext": "1B_linear_type",
        "1C": "1C_normalization",
        "1D": "1D_precision",
        "1E": "1E_skip",
        "1E_ext": "1E_skip",
        "1F": "1F_independence_core",
        "1F_ext": "1F_independence_linear_type",
        "1G": "1G_robustness",
    }
    return mapping.get(s, s)


def _flatten(t: torch.Tensor) -> torch.Tensor:
    if t.dim() > 2:
        return t.view(t.shape[0], -1)
    return t


def _kappa_config_from_protocol(protocol: Dict[str, Any], *, estimator_name: str) -> KappaConfig:
    meas = protocol["measurement_channel"]
    evalp = protocol["evaluation"]
    split = evalp["mi_train_test_split"]

    est_name = estimator_name.lower() if estimator_name else protocol["mi_estimators"]["primary"]["name"]
    if est_name == "ksg32":
        est_name = "ksg"

    est_cfg: Dict[str, Any] = {}
    if est_name == "infonce":
        p = protocol["mi_estimators"]["primary"]
        est_cfg = {
            "hidden_dims": tuple(p["critic"]["hidden_dims"]),
            "activation": p["critic"].get("activation", "relu"),
            "steps": p["training"]["steps"],
            "batch_size": p["training"]["batch_size"],
            "lr": p["training"]["lr"],
            "temperature": p["training"]["temperature"],
            "saturation_margin": p["diagnostics"]["saturation_margin"],
        }
    elif est_name == "mine":
        p = protocol["mi_estimators"]["secondary"]
        est_cfg = {
            "hidden_dims": tuple(p["critic"]["hidden_dims"]),
            "activation": p["critic"].get("activation", "relu"),
            "steps": p["training"]["steps"],
            "batch_size": p["training"]["batch_size"],
            "lr": p["training"]["lr"],
            "ema_decay": p["training"]["ema_decay"],
        }
    elif est_name == "ksg":
        p = protocol["mi_estimators"]["tertiary"]
        est_cfg = {
            "projection_dim": p["projection"]["output_dim"],
            "projection_seed": p["projection"]["seed"],
            "k": p["k"],
        }
    else:
        raise ValueError(f"Unknown estimator_name: {est_name}")

    return KappaConfig(
        sigma=float(meas["sigma_main"]),
        sigma_mode=str(meas["sigma_mode"]),
        n_eval=int(evalp["n_eval"]),
        mi_train=int(split["train"]),
        mi_test=int(split["test"]),
        estimator_name=est_name,
        estimator_config=est_cfg,
        mi_seed=int(protocol["seeds"]["mi_seed"]),
        run_sanity=True,
    )


def _build_vector_model(
    dataset: str,
    row: Dict[str, Any],
    protocol: Dict[str, Any],
    *,
    init_seed: int,
) -> torch.nn.Module:
    cfg = ProbeConfig(
        num_classes=int((row.get('num_classes') or '').strip() or (1000 if dataset=='imagenet_subset' else 10)),
        width=int(row.get("width") or 256),
        activation=str(row.get("activation") or protocol["baselines"]["vector_probe"]["activation"]),
        normalization=str(row.get("normalization") or "none"),
        precision=str(row.get("precision") or "fp32"),
        skip=str(row.get("skip") or "none"),
        skip_scale=str(row.get("skip_scale") or "none"),
        concat_projection=_canon_concat_projection(row.get("concat_projection")),
        linear_type="dense",
        linear_budget=_canon_linear_budget(row.get("linear_budget")),
    )
    stem_seed = int(protocol["seeds"]["stem_seed"])
    return create_vector_probe(dataset, cfg, stem_seed=stem_seed, init_seed=init_seed)


def _build_spatial_model(
    dataset: str,
    row: Dict[str, Any],
    protocol: Dict[str, Any],
    *,
    init_seed: int,
) -> torch.nn.Module:
    cfg = ProbeConfig(
        width=int(row.get("width") or 16),
        activation=str(row.get("activation") or protocol["baselines"]["spatial_probe"]["activation"]),
        normalization=str(row.get("normalization") or protocol["baselines"]["spatial_probe"]["normalization"]),
        precision=str(row.get("precision") or protocol["baselines"]["spatial_probe"]["precision"]),
        skip=str(row.get("skip") or protocol["baselines"]["spatial_probe"]["skip"]),
        skip_scale=str(row.get("skip_scale") or "none"),
        concat_projection=_canon_concat_projection(row.get("concat_projection")),
        linear_type=str(row.get("linear_type") or protocol["baselines"]["spatial_probe"]["linear_type"]),
        linear_budget=_canon_linear_budget(row.get("linear_budget")),
    )
    spatial_cfg = {
        "c_in": 16,
        "c_out": 16,
        "h": 8,
        "w": 8,
    }
    stem_seed = int(protocol["seeds"]["stem_seed"])
    return create_spatial_probe(dataset, cfg, spatial_cfg, stem_seed=stem_seed, init_seed=init_seed)


def _extract_XZ(
    model: torch.nn.Module,
    eval_loader,
    *,
    device: torch.device,
    tap: str,
    n_eval: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval().to(device)
    Xs = []
    Zs = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            xb = xb.to(device)
            _ = model(xb, capture=True, tap=tap)
            X, Z = model.get_XZ_for_kappa(tap=tap)
            Xs.append(_flatten(X).cpu())
            Zs.append(_flatten(Z).cpu())
    X_all = torch.cat(Xs, dim=0)[:n_eval]
    Z_all = torch.cat(Zs, dim=0)[:n_eval]
    if X_all.shape[0] != n_eval:
        raise RuntimeError(f"Eval subset produced {X_all.shape[0]} samples, expected {n_eval}")
    return X_all, Z_all


def _convert_for_eval(
    model: torch.nn.Module,
    *,
    precision: str,
    calib_loader,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    p = precision.lower()
    extras: Dict[str, Any] = {}
    if p in {"fp32", ""}:
        return model, extras
    if p in {"fp16", "bf16"}:
        m2 = copy.deepcopy(model)
        m2 = cast_model_precision(m2, p)
        extras["conversion"] = {"method": "cast", "target": p}
        return m2, extras
    if p in {"int8", "int4"}:
        bits = 8 if p == "int8" else 4
        observer = "minmax" if bits == 8 else "percentile"
        ptq_cfg = PTQConfig(bits=bits, observer=observer, percentile=99.9)
        m2 = copy.deepcopy(model)
        m2, _ = apply_ptq_to_vector_probe(m2, ptq_cfg)
        obs_stats = run_ptq_calibration(m2, calib_loader, device=device)
        extras["conversion"] = {
            "method": "ptq",
            "bits": bits,
            "weight": "per_channel_symmetric",
            "activation": "per_tensor_symmetric",
            "observer": observer,
            "percentile": 99.9 if observer == "percentile" else None,
        }
        extras["observer_stats"] = obs_stats
        return m2, extras
    raise ValueError(f"Unknown precision: {precision}")


def run_manifest_row(
    *,
    row: Dict[str, Any],
    repo_root: Path,
    output_root: Path,
    data_root: Path,
    assets_root: Path,
    protocol_path: Path,
    schema_path: Path,
    imagenet_subset_root: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Path:
    """Run a single experiment job from a manifest row.

    Returns path to saved receipt.json.
    """
    protocol = load_yaml(protocol_path)

    # ------------------------------------------------------------------
    # Determinism: CPU thread count
    # ------------------------------------------------------------------
    # Different thread counts can slightly change floating point reduction
    # orders and lead to tiny-but-real numeric drift. For reproducibility
    # we default to 1 thread and allow an explicit override.
    try:
        num_threads = int(os.getenv("ICON_PRIMITIVE_NUM_THREADS", "1"))
    except Exception:
        num_threads = 1
    if num_threads < 1:
        num_threads = 1
    torch.set_num_threads(num_threads)
    device = device or _default_device()

    run_id = str(row["run_id"])
    section = str(row["section"])
    mode = str(row.get("mode") or "single_model")
    probe = str(row.get("probe") or "vector_probe")

    dataset = str(row.get("dataset") or protocol["data"]["primary_dataset"])
    model_seed = int(row.get("model_seed") or 0)
    data_seed = int(row.get("data_seed") or protocol["seeds"]["data_seed"])
    mi_seed = int(row.get("mi_seed") or protocol["seeds"]["mi_seed"])
    estimator_name = str(row.get("mi_estimator") or protocol["mi_estimators"]["primary"]["name"])
    if estimator_name == "ksg32":
        estimator_name = "ksg"

    set_seed(model_seed=model_seed, data_seed=data_seed, mi_seed=mi_seed, determinism_cfg=protocol["determinism"])

    # Data
    imagenet_resize = int(protocol["data"]["preprocessing"].get("imagenet_resize", 64))
    train_ds = load_dataset(
        DatasetSpec(
            name=dataset,
            split="train",
            root=data_root,
            imagenet_subset_root=imagenet_subset_root,
            imagenet_resize=imagenet_resize,
        )
    )
    test_ds = load_dataset(
        DatasetSpec(
            name=dataset,
            split="test",
            root=data_root,
            imagenet_subset_root=imagenet_subset_root,
            imagenet_resize=imagenet_resize,
        )
    )
    n_eval = int(protocol["evaluation"]["n_eval"])
    eval_subset = make_eval_subset(
        test_ds,
        dataset_name=dataset,
        split="test",
        n_eval=n_eval,
        indices_seed=data_seed,
        assets_eval_dir=assets_root / "eval_indices",
    )

    train_loader = make_dataloader(
        train_ds,
        batch_size=int(protocol["training"]["batch_size"]),
        shuffle=True,
        data_seed=data_seed,
    )
    eval_loader = make_dataloader(
        eval_subset,
        batch_size=int(protocol["training"]["batch_size"]),
        shuffle=False,
        data_seed=data_seed,
    )

    # Models
    init_seed = model_seed  # keep init tied to model_seed

    def _train_and_kappa(act_name: str) -> Tuple[float, Dict[str, Any]]:
        row_local = dict(row)
        row_local["activation"] = act_name
        if probe == "vector_probe":
            model = _build_vector_model(dataset, row_local, protocol, init_seed=init_seed)
        else:
            model = _build_spatial_model(dataset, row_local, protocol, init_seed=init_seed)

        # Stem hash is part of the audit trail (frozen stem provenance).
        stem_hash = None
        try:
            # stem is a FrozenStem wrapper with attribute `.stem` holding the nn.Module
            stem_hash = hash_state_dict(model.stem.stem.state_dict())
        except Exception:
            stem_hash = None

        # Train (always fp32)
        sched = WarmupCosineConfig(
            warmup_epochs=int(protocol["training"]["schedule"]["warmup_epochs"]),
            total_epochs=int(protocol["training"]["schedule"]["total_epochs"]),
        )
        train_cfg = TrainConfig(
            optimizer=str(protocol["training"]["optimizer"]),
            lr=float(protocol["training"]["lr"]),
            weight_decay=float(protocol["training"]["weight_decay"]),
            grad_clip=float(protocol["training"]["grad_clip"]),
            batch_size=int(protocol["training"]["batch_size"]),
            epochs=int(protocol["training"]["schedule"]["total_epochs"]),
            schedule=sched,
        )
        train_metrics = train_classifier(model, train_loader, train_cfg, device=device, dtype_train="fp32")

        # Precision conversion (PTQ) for eval
        precision = str(row_local.get("precision") or "fp32")
        quant_extras: Dict[str, Any] = {}
        calib_loader = None
        if precision in {"int8", "int4"}:
            calib_subset = make_calib_subset(
                train_ds,
                dataset_name=dataset,
                n_calib=1024,
                indices_seed=data_seed,
                assets_calib_dir=assets_root / "calib_indices",
            )
            calib_loader = make_dataloader(
                calib_subset,
                batch_size=int(protocol["training"]["batch_size"]),
                shuffle=False,
                data_seed=data_seed,
            )
        model_eval, quant_extras = _convert_for_eval(model, precision=precision, calib_loader=calib_loader, device=device)

        # Measure kappa
        tap = "post_block"
        X, Z = _extract_XZ(model_eval, eval_loader, device=device, tap=tap, n_eval=n_eval)
        kcfg = _kappa_config_from_protocol(protocol, estimator_name=estimator_name)
        kcfg.mi_seed = mi_seed
        out = compute_kappa(X, Z, kcfg, device=device)
        kappa = float(out.kappa_per_dim)
        meas_meta = {
            "kappa_raw": float(out.kappa_raw),
            "d_z": int(out.d_z),
            "saturation_flag": bool(out.saturation_flag),
            "saturation_margin": float(out.saturation_margin) if out.saturation_margin is not None else None,
            "permuted_mi": float(out.permuted_mi) if out.permuted_mi is not None else None,
            "sanity_passed": bool(out.sanity_passed),
            "train_metrics": train_metrics,
            "quant": quant_extras,
            "stem_hash": stem_hash,
        }
        return kappa, meas_meta

    # Execute per mode
    if mode == "paired_ratio":
        base_act = str(row.get("baseline_activation") or "relu")
        var_act = str(row.get("variant_activation") or "gelu")
        k_base, meta_base = _train_and_kappa(base_act)
        k_var, meta_var = _train_and_kappa(var_act)
        C = float(k_var / max(1e-12, k_base))
        kappa_per_dim = float(k_var)
        extras = {
            "manifest": dict(row),
            "paired": {
                "baseline": meta_base,
                "variant": meta_var,
                "kappa_base": k_base,
                "kappa_variant": k_var,
            },
        }
        ratio = {"base_run_id": f"{run_id}::baseline", "C": C}
        kappa_raw = float(meta_var["kappa_raw"])
        d_z_measured = int(meta_var["d_z"])
    else:
        kappa_per_dim, meta = _train_and_kappa(str(row.get("activation") or "relu"))
        extras = {"manifest": dict(row), "single": meta}
        # Ratio is computed during aggregation once baselines are available.
        ratio = {"base_run_id": "aggregate::baseline", "C": None}
        kappa_raw = float(meta["kappa_raw"])
        d_z_measured = int(meta["d_z"])

    # Record determinism-relevant runtime knobs for audit.
    extras["torch_num_threads"] = int(torch.get_num_threads())
    try:
        extras["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:
        extras["torch_num_interop_threads"] = None

    # Hashes
    config_hash = hash_any(row)
    eval_indices_hash = hash_any(np.asarray(list(eval_subset.indices), dtype=np.int64))
    data_hash = compute_data_hash(
        dataset,
        split_sizes={"train": len(train_ds), "test": len(test_ds)},
        preprocess=protocol["data"]["preprocessing"],
    )

    # Frozen stem hash (from the executed run)
    stem_hash = None
    try:
        if mode == "paired_ratio":
            stem_hash = extras.get("paired", {}).get("variant", {}).get("stem_hash")
        else:
            stem_hash = extras.get("single", {}).get("stem_hash")
    except Exception:
        stem_hash = None

    model_state_hash = None

    # Receipt payload
    # Receipt payload
    probe_info = {
        "type": probe,
        "stem_frozen": True,
        "stem_id": protocol["probes"][probe]["stem_id"] if probe in protocol.get("probes", {}) else "",
        "input": {
            "shape": [n_eval, 256] if probe == "vector_probe" else [n_eval, 16 * 8 * 8],
            "dtype": "fp32",
            "normalization": str(protocol["data"]["preprocessing"]["normalize"][dataset]["mean"]) + "/" + str(protocol["data"]["preprocessing"]["normalize"][dataset]["std"]),
        },
        "output": {
            "d_z": int(d_z_measured),
            "tap": "custom",
            "tap_description": "post_block (after norm/activation/skip)",
        },
    }

    data_info = {
        "dataset": dataset,
        "splits": {"train": "train", "test": "test"},
        "preprocess": protocol["data"]["preprocessing"],
        "eval_indices": {"n_eval": n_eval, "source": str((assets_root / "eval_indices").resolve())},
    }

    act_for_receipt = (
        str(row.get("variant_activation") or "gelu")
        if mode == "paired_ratio"
        else str(row.get("activation") or "relu")
    )
    baseline_cfg = protocol["baselines"][probe]
    width_default = 16 if probe == "spatial_probe" else 256
    model_info = {
        "width": int(row.get("width") or width_default),
        "primitive": {
            "activation": act_for_receipt,
            "normalization": str(row.get("normalization") or baseline_cfg.get("normalization", "none")),
            "precision": str(row.get("precision") or baseline_cfg.get("precision", "fp32")),
            "skip": str(row.get("skip") or baseline_cfg.get("skip", "none")),
            "skip_scale": str(row.get("skip_scale") or "none"),
            "concat_projection": _canon_concat_projection(row.get("concat_projection")),
            "linear_type": str(row.get("linear_type") or baseline_cfg.get("linear_type", "dense")),
            "linear_budget": _canon_linear_budget(row.get("linear_budget") or baseline_cfg.get("linear_budget")),
        },
        "init": protocol["initialization"],
    }

    training_info = {
        "optimizer": protocol["training"]["optimizer"],
        "lr": float(protocol["training"]["lr"]),
        "schedule": protocol["training"]["schedule"]["type"],
        "epochs": int(protocol["training"]["schedule"]["total_epochs"]),
        "batch_size": int(protocol["training"]["batch_size"]),
        "weight_decay": float(protocol["training"]["weight_decay"]),
        "grad_clip": float(protocol["training"]["grad_clip"]),
        "loss": protocol["training"]["loss"],
        "mixed_precision_train": False,
        "seeds": get_seed_config(model_seed, data_seed, mi_seed),
        "determinism": {
            "cudnn_deterministic": bool(protocol["determinism"]["cudnn_deterministic"]),
            "cudnn_benchmark": bool(protocol["determinism"]["cudnn_benchmark"]),
            "torch_deterministic_algorithms": bool(protocol["determinism"].get("torch_deterministic_algorithms", False)),
            "notes": "",
        },
    }

    kcfg = _kappa_config_from_protocol(protocol, estimator_name=estimator_name)
    kcfg.mi_seed = mi_seed
    est_hparams = dict(kcfg.estimator_config)
    diagnostics = {
        "saturation_check": True if kcfg.estimator_name == "infonce" else False,
        "saturation_margin": est_hparams.get("saturation_margin"),
    }
    measurement_info = {
        "kappa_definition": "I(X;Z_tilde)/d_z",
        "noise_channel": {"type": "gaussian", "sigma": float(kcfg.sigma), "sigma_mode": "rms_scaled"},
        "estimators": [
            {
                "name": kcfg.estimator_name,
                "seed": mi_seed,
                "hyperparams": est_hparams,
                "diagnostics": diagnostics,
            }
        ],
        "sanity_checks": {"permuted_Z": True},
    }

    results_info = {
        "kappa": {
            "raw": float(kappa_raw),
            "per_dim": float(kappa_per_dim),
            "by_estimator": {
                kcfg.estimator_name: {"raw": float(kappa_raw), "per_dim": float(kappa_per_dim)}
            },
        },
        "ratio": ratio,
    }

    hashes_info = {
        "config_hash": config_hash,
        "data_hash": data_hash,
        "eval_indices_hash": eval_indices_hash,
    }
    if stem_hash is not None:
        hashes_info["stem_hash"] = stem_hash
    if model_state_hash is not None:
        hashes_info["model_state_hash"] = model_state_hash

    # Save artifacts
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "receipt.json"
    receipt = build_receipt(
        schema_path=schema_path,
        repo_root=repo_root,
        run_id=run_id,
        section=_canon_section_for_receipt(section),
        experiment_name="Icon_primitive",
        probe=probe_info,
        data=data_info,
        model=model_info,
        training=training_info,
        measurement=measurement_info,
        results=results_info,
        hashes=hashes_info,
        artifacts={"run_dir": str(run_dir)},
        extras=extras,
        notes=str(row.get("notes") or ""),
    )
    save_receipt(receipt, receipt_path, schema_path)
    return receipt_path
