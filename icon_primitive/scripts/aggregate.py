#!/usr/bin/env python3
"""Aggregate receipts into the constants YAML (v1.1 template).

Usage:
  python scripts/aggregate.py \
    --receipts outputs/runs \
    --template configs/packs/ICON_Primitive_Constants_Template_v1.1.yaml \
    --output outputs/constants/Icon_primitive_Constants_v1.1.yaml
"""

from __future__ import annotations

# Allow `python scripts/...py` without requiring `pip install -e .`.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from icon_primitive.utils.receipt import load_schema, validate_receipt


def _load_receipts(receipts_root: Path, schema_path: Path) -> List[Dict[str, Any]]:
    receipts: List[Dict[str, Any]] = []
    for p in receipts_root.rglob("receipt.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
            validate_receipt(r, schema_path)
            receipts.append(r)
        except Exception:
            # Ignore invalid receipts.
            continue
    return receipts


def _bootstrap_ci(values: List[float], iters: int = 2000) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    arr = np.asarray(values, dtype=np.float64)
    boots = []
    for _ in range(iters):
        samp = rng.choice(arr, size=arr.shape[0], replace=True)
        boots.append(float(np.mean(samp)))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _seed(r: Dict[str, Any]) -> int:
    return int(r["training"]["seeds"]["model_seed"])


def _kappa_per_dim(r: Dict[str, Any]) -> float:
    return float(r["results"]["kappa"]["per_dim"])


def _ratio_C(r: Dict[str, Any]) -> float:
    c = r["results"]["ratio"]["C"]
    if c is None:
        raise ValueError("ratio.C is null")
    return float(c)


def _section(r: Dict[str, Any]) -> str:
    return str(r["section"])


def _group_stats_by_seed_ratio(
    receipts: List[Dict[str, Any]],
    *,
    baseline_filter,
    variant_filter,
) -> Tuple[List[float], List[str]]:
    base = [r for r in receipts if baseline_filter(r)]
    var = [r for r in receipts if variant_filter(r)]
    base_by_seed: Dict[int, float] = {}
    for r in base:
        base_by_seed[_seed(r)] = _kappa_per_dim(r)
    ratios: List[float] = []
    run_ids: List[str] = []
    for r in var:
        s = _seed(r)
        if s not in base_by_seed:
            continue
        ratios.append(_kappa_per_dim(r) / max(1e-12, base_by_seed[s]))
        run_ids.append(str(r["run_id"]))
    return ratios, run_ids


def _fill_category(
    template: Dict[str, Any],
    receipts: List[Dict[str, Any]],
    *,
    section_name: str,
    category_path: List[str],
    key_name: str,
    baseline_value: str,
    allowed_values: List[str],
) -> None:
    # Navigate template
    node = template
    for k in category_path:
        node = node[k]
    node["baseline"] = baseline_value

    # Compute each value
    for v in allowed_values:

        def is_baseline(r):
            if _section(r) != section_name:
                return False
            prim = r["model"]["primitive"]
            if str(prim[key_name]) != baseline_value:
                return False
            # Skip-specific fairness locks
            if key_name == "skip" and baseline_value == "residual":
                return str(prim.get("skip_scale", "none")) == str(node["values"]["residual"].get("scaling", "variance_preserving"))
            if key_name == "skip" and baseline_value == "dense_concat":
                want_proj = str(node["values"]["dense_concat"].get("projection", "none"))
                return str(prim.get("concat_projection", "none")) == want_proj or (want_proj=="project_to_width" and str(prim.get("concat_projection","none"))=="none")
            return True

        def is_var(r):
            if _section(r) != section_name:
                return False
            prim = r["model"]["primitive"]
            if str(prim[key_name]) != v:
                return False
            # Skip-specific fairness locks
            if key_name == "skip" and v == "residual":
                want_scaling = str(node["values"]["residual"].get("scaling", "variance_preserving"))
                return str(prim.get("skip_scale", "none")) == want_scaling
            if key_name == "skip" and v == "dense_concat":
                want_proj = str(node["values"]["dense_concat"].get("projection", "none"))
                return str(prim.get("concat_projection", "none")) == want_proj or (want_proj=="project_to_width" and str(prim.get("concat_projection","none"))=="none")
            return True

        ratios, run_ids = _group_stats_by_seed_ratio(receipts, baseline_filter=is_baseline, variant_filter=is_var)
        if v == baseline_value:
            ratios = [1.0 for _ in ratios] if ratios else [1.0]
        mean = float(np.mean(ratios)) if ratios else None
        std = float(np.std(ratios)) if ratios else None
        ci = list(_bootstrap_ci(ratios)) if ratios else [None, None]

        node["values"][v]["C"]["mean"] = mean
        node["values"][v]["C"]["std"] = std
        node["values"][v]["C"]["ci95"] = [ci[0], ci[1]]
        node["values"][v]["receipts"] = run_ids


def _get_constant_mean(template: Dict[str, Any], category: str, key: str) -> Optional[float]:
    try:
        val = template["icon_primitive_constants"]["results"][category]["values"][key]["C"]["mean"]
        return float(val) if val is not None else None
    except Exception:
        return None


def _get_linear_type_mean(template: Dict[str, Any], budget: str, op: str) -> Optional[float]:
    try:
        node = template["icon_primitive_constants"]["results"]["linear_type"]["budgets"][budget]["values"][op]["C"]["mean"]
        return float(node) if node is not None else None
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--receipts", required=True, help="Root folder containing runs/*/receipt.json")
    p.add_argument("--template", required=True, help="Constants template YAML")
    p.add_argument("--output", required=True, help="Output constants YAML")
    p.add_argument("--schema", default="schemas/ICON_Primitive_Receipt_Schema_v1.1.json")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    schema_path = (repo_root / args.schema).resolve() if not Path(args.schema).is_absolute() else Path(args.schema)
    receipts_root = (repo_root / args.receipts).resolve() if not Path(args.receipts).is_absolute() else Path(args.receipts)
    tpl_path = (repo_root / args.template).resolve() if not Path(args.template).is_absolute() else Path(args.template)

    with open(tpl_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    receipts = _load_receipts(receipts_root, schema_path)

    # Fill metadata
    croot = template["icon_primitive_constants"]
    croot["created_at_utc"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    # Best effort provenance
    if receipts:
        croot["provenance"]["git_commit"] = receipts[0].get("git", {}).get("commit", "")
        croot["provenance"]["repo"] = receipts[0].get("git", {}).get("repo", "")
    croot["provenance"]["receipt_schema"] = str(schema_path.name)

    # Fill main constants
    _fill_category(
        template,
        receipts,
        section_name="1A_activation",
        category_path=["icon_primitive_constants", "results", "activation"],
        key_name="activation",
        baseline_value="relu",
        allowed_values=["relu", "gelu", "silu", "tanh", "sigmoid", "mish", "identity"],
    )
    _fill_category(
        template,
        receipts,
        section_name="1C_normalization",
        category_path=["icon_primitive_constants", "results", "normalization"],
        key_name="normalization",
        baseline_value="none",
        allowed_values=["none", "layernorm", "rmsnorm", "batchnorm", "groupnorm"],
    )
    _fill_category(
        template,
        receipts,
        section_name="1D_precision",
        category_path=["icon_primitive_constants", "results", "precision"],
        key_name="precision",
        baseline_value="fp32",
        allowed_values=["fp32", "fp16", "bf16", "int8", "int4"],
    )
    _fill_category(
        template,
        receipts,
        section_name="1E_skip",
        category_path=["icon_primitive_constants", "results", "skip"],
        key_name="skip",
        baseline_value="none",
        allowed_values=["none", "residual", "dense_concat"],
    )

    # Linear type: budgets
    lt_node = template["icon_primitive_constants"]["results"]["linear_type"]["budgets"]
    for budget in ["shape_matched", "param_matched"]:
        for op in lt_node[budget]["values"].keys():
            def is_base(r, budget=budget):
                return _section(r) == "1B_linear_type" and str(r["model"]["primitive"].get("linear_budget", "shape_matched")) == budget and str(r["model"]["primitive"]["linear_type"]) == "dense"

            def is_var(r, budget=budget, op=op):
                return _section(r) == "1B_linear_type" and str(r["model"]["primitive"].get("linear_budget", "shape_matched")) == budget and str(r["model"]["primitive"]["linear_type"]) == op

            ratios, run_ids = _group_stats_by_seed_ratio(receipts, baseline_filter=is_base, variant_filter=is_var)
            if op == "dense":
                ratios = [1.0 for _ in ratios] if ratios else [1.0]
            mean = float(np.mean(ratios)) if ratios else None
            std = float(np.std(ratios)) if ratios else None
            ci = list(_bootstrap_ci(ratios)) if ratios else [None, None]
            lt_node[budget]["values"][op]["C"]["mean"] = mean
            lt_node[budget]["values"][op]["C"]["std"] = std
            lt_node[budget]["values"][op]["C"]["ci95"] = [ci[0], ci[1]]
            lt_node[budget]["values"][op]["receipts"] = run_ids

    
    # Verification: independence core (compositional law)
    indep = template["icon_primitive_constants"]["verification"]["independence"]["core"]
    indep_receipts = [r for r in receipts if _section(r) == "1F_independence_core"]

    # Build baseline kappa map from *all* receipts using manifest key (matches verify_1F script logic).
    def _m(r, k, default=""):
        return str(r.get("extras", {}).get("manifest", {}).get(k, default))

    def _base_key(r):
        return (
            _m(r, "probe"),
            _m(r, "dataset"),
            _m(r, "width"),
            _m(r, "mi_estimator"),
            _m(r, "linear_budget"),
            _m(r, "data_seed"),
            _m(r, "mi_seed"),
            _m(r, "model_seed"),
        )

    # baseline primitive definition
    def _is_baseline_primitive(r):
        prim = r["model"]["primitive"]
        return (
            prim.get("activation") == "relu"
            and prim.get("normalization") == "none"
            and prim.get("precision") == "fp32"
            and prim.get("skip") == "none"
            and prim.get("linear_type", "dense") == "dense"
        )

    base_kappa = {}
    for r in receipts:
        if _is_baseline_primitive(r):
            try:
                base_kappa[_base_key(r)] = _kappa_per_dim(r)
            except Exception:
                continue

    errors = []
    used_ids = []

    for r in indep_receipts:
        bk = _base_key(r)
        if bk not in base_kappa:
            continue
        obs = _kappa_per_dim(r) / max(1e-12, base_kappa[bk])
        prim = r["model"]["primitive"]

        # multiplicative predictor: include linear_type factor (shape_matched by default)
        ca = _get_constant_mean(template, "activation", prim["activation"]) or 1.0
        cn = _get_constant_mean(template, "normalization", prim["normalization"]) or 1.0
        cp = _get_constant_mean(template, "precision", prim["precision"]) or 1.0
        cs = _get_constant_mean(template, "skip", prim["skip"]) or 1.0
        budget = str(prim.get("linear_budget", "shape_matched"))
        op = str(prim.get("linear_type", "dense"))
        ct = _get_linear_type_mean(template, budget, op) or 1.0

        pred = float(ct * ca * cn * cp * cs)
        if pred <= 0:
            continue
        err = abs(obs - pred) / abs(obs) if obs != 0 else float("inf")
        errors.append(float(err))
        used_ids.append(str(r["run_id"]))

    if errors:
        indep["mean_error"] = float(np.mean(errors))
        indep["max_error"] = float(np.max(errors))
        indep["receipts"] = used_ids
        indep["passed"] = bool(indep["mean_error"] < 0.05 and indep["max_error"] < 0.15)


    # Verification: independence w.r.t linear_type (optional extension)
    indep_lt = template["icon_primitive_constants"]["verification"]["independence"].get("linear_type_extension")
    lt_receipts = [r for r in receipts if _section(r) == "1F_independence_linear_type"]
    if indep_lt is not None and lt_receipts:
        base_lt = [
            r
            for r in lt_receipts
            if r["model"]["primitive"]["activation"] == "relu"
            and r["model"]["primitive"]["normalization"] == "none"
            and r["model"]["primitive"]["precision"] == "fp32"
            and r["model"]["primitive"]["skip"] == "none"
            and r["model"]["primitive"].get("linear_type", "dense") == "dense"
        ]
        base_by_seed_lt = { _seed(r): _kappa_per_dim(r) for r in base_lt }
        lt_errors: List[float] = []
        lt_used: List[str] = []
        for r in lt_receipts:
            s = _seed(r)
            if s not in base_by_seed_lt:
                continue
            obs = _kappa_per_dim(r) / max(1e-12, base_by_seed_lt[s])
            prim = r["model"]["primitive"]
            ca = _get_constant_mean(template, "activation", prim["activation"]) or 1.0
            cn = _get_constant_mean(template, "normalization", prim["normalization"]) or 1.0
            cp = _get_constant_mean(template, "precision", prim["precision"]) or 1.0
            cs = _get_constant_mean(template, "skip", prim["skip"]) or 1.0
            budget = str(prim.get("linear_budget", "shape_matched"))
            op = str(prim.get("linear_type", "dense"))
            ct = _get_linear_type_mean(template, budget, op) or 1.0
            pred = float(ct * ca * cn * cp * cs)
            if pred <= 0:
                continue
            lt_errors.append(float(abs(obs - pred) / pred))
            lt_used.append(str(r["run_id"]))
        if lt_errors:
            indep_lt["mean_error"] = float(np.mean(lt_errors))
            indep_lt["max_error"] = float(np.max(lt_errors))
            indep_lt["receipts"] = lt_used
            indep_lt["passed"] = bool(indep_lt["mean_error"] < 0.05 and indep_lt["max_error"] < 0.15)

    
    # Verification: robustness (primary measurement channel: InfoNCE)
    rob = template["icon_primitive_constants"]["verification"]["robustness"]
    rob_receipts = [r for r in receipts if _section(r) == "1G_robustness"]

    Cs = []
    rob_ids = []
    datasets=set(); widths=set(); estimators=set()

    for r in rob_receipts:
        # Only InfoNCE as primary channel
        est = str(r.get("extras", {}).get("manifest", {}).get("mi_estimator", ""))
        if est != "infonce":
            continue
        try:
            c = _ratio_C(r)
            if not np.isfinite(c):
                continue
            Cs.append(float(c))
            rob_ids.append(str(r["run_id"]))
            datasets.add(str(r.get("extras", {}).get("manifest", {}).get("dataset","")))
            # width may be float in receipts; normalize to int
            try:
                widths.add(int(float(r.get('extras',{}).get('manifest',{}).get('width',0))))
            except Exception:
                pass
            estimators.add(est)
        except Exception:
            continue

    if Cs:
        rob["receipts"] = rob_ids
        rob["passed"] = bool(float(np.std(Cs)) < 0.03)
        # overwrite tested axes to reflect what we actually tested in primary channel
        rob["tested_axes"]["datasets"] = sorted([d for d in datasets if d and d!="imagenet_subset"])
        rob["tested_axes"]["widths"] = sorted([int(w) for w in widths if str(w).isdigit()])
        rob["tested_axes"]["estimators"] = sorted(list(estimators))

    out_path = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(template, f, sort_keys=False)
    print(f"Saved constants: {out_path}")


if __name__ == "__main__":
    main()
