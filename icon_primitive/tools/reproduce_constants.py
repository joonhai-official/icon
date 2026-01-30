import csv, json, glob, math, statistics
from pathlib import Path
from collections import defaultdict

MANI = "configs/manifests/ICON_Primitive_Run_Manifest_Base153_v1.1.csv"

# load manifest rows (non-imagenet)
rows = {}
with open(MANI, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if row.get("dataset","").strip() == "imagenet_subset":
            continue
        rid = row["run_id"].strip()
        rows[rid] = row

# load receipts + kappa_raw
receipts = {}
for p in glob.glob("outputs/runs/*/receipt.json"):
    rid = Path(p).parent.name
    if rid not in rows:
        continue
    rr = json.load(open(p))
    kraw = rr.get("results",{}).get("kappa",{}).get("raw", None)
    if kraw is None:
        kraw = rr.get("extras",{}).get("single",{}).get("kappa_raw", None)
    receipts[rid] = (rr, kraw)

print("loaded receipts(non-imagenet):", len(receipts))

# section -> (baseline field in manifest, variant field in manifest)
SECTION_FIELDS = {
    "1A_activation": ("baseline_activation", "activation"),
    # manifest uses baseline_activation/variant_activation; some runs may store in activation column, so fallback to variant_activation if present
    "1C_normalization": ("baseline_normalization", "normalization"),
    "1D_precision": ("baseline_precision", "precision"),
    "1E_skip": ("baseline_skip", "skip"),
    "1B_linear_type": ("baseline_linear_type", "linear_type"),
}

# fallbacks: in current manifest, baseline_* may be empty. Use explicit hard baselines.
DEFAULT_BASELINE = {
    "1A_activation": "relu",
    "1C_normalization": "none",
    "1D_precision": "fp32",
    "1E_skip": "none",
    "1B_linear_type": "dense",
}

def section_of(row):
    sec = row.get("section","").strip()
    # manifest sometimes uses short "1A" etc; receipts use full "1A_activation"
    # but your receipts show full names, and manifest has section like '1A', '1B' etc.
    # So map manifest section to receipt section via config_id prefix:
    # we'll use receipt section from receipts when available; else infer.
    return sec

# map run_id to receipt section when possible
rid_to_sec = {}
for rid,(rr,_) in receipts.items():
    rid_to_sec[rid] = rr.get("section")

# define grouping to match base vs variant fairly
def group_key(rid, row, rr):
    # match by probe/dataset/width/estimator/budget and seeds
    return (
        rr.get("section"),
        row.get("probe","").strip(),
        row.get("dataset","").strip(),
        row.get("width","").strip(),
        row.get("mi_estimator","").strip(),
        row.get("linear_budget","").strip(),
        row.get("data_seed","").strip(),
        row.get("mi_seed","").strip(),
        row.get("model_seed","").strip(),
    )

base_map = {}               # group_key -> kappa_base
vars_map = defaultdict(list) # (section, variant, group_key_wo_section?) -> [kappa_var]

for rid, row in rows.items():
    if rid not in receipts:
        continue
    rr, kraw = receipts[rid]
    if kraw is None:
        continue
    sec = rr.get("section")
    if sec not in SECTION_FIELDS:
        continue

    # determine baseline/variant value from manifest row (robust)
    # for activation: sometimes manifest has columns baseline_activation/variant_activation; in your csv it may be baseline_activation/variant_activation, but shown truncated.
    # We'll try several keys.
    if sec == "1A_activation":
        base = (row.get("baseline_activation") or row.get("baseline") or "").strip() or DEFAULT_BASELINE[sec]
        var  = (row.get("variant_activation") or row.get("activation") or "").strip()
        if not var:
            # fallback: receipt primitive activation
            var = rr.get("model",{}).get("primitive",{}).get("activation","").strip()
    else:
        # generic: baseline_* or default baseline
        if sec == "1C_normalization":
            base = (row.get("baseline_normalization") or row.get("baseline") or "").strip() or DEFAULT_BASELINE[sec]
            var  = (row.get("normalization") or "").strip() or rr.get("model",{}).get("primitive",{}).get("normalization","").strip()
        elif sec == "1D_precision":
            base = (row.get("baseline_precision") or row.get("baseline") or "").strip() or DEFAULT_BASELINE[sec]
            var  = (row.get("precision") or "").strip() or rr.get("model",{}).get("primitive",{}).get("precision","").strip()
        elif sec == "1E_skip":
            base = (row.get("baseline_skip") or row.get("baseline") or "").strip() or DEFAULT_BASELINE[sec]
            var  = (row.get("skip") or "").strip() or rr.get("model",{}).get("primitive",{}).get("skip","").strip()
        elif sec == "1B_linear_type":
            base = (row.get("baseline_linear_type") or row.get("baseline") or "").strip() or DEFAULT_BASELINE[sec]
            var  = (row.get("linear_type") or "").strip() or rr.get("model",{}).get("primitive",{}).get("linear_type","").strip()
        else:
            continue

    g = group_key(rid, row, rr)
    if var == base:
        base_map[g] = kraw
    else:
        vars_map[(sec, var, g)].append(kraw)

# compute ratios
ratios = defaultdict(list)  # (sec, var) -> ratios
miss = 0
for (sec, var, g), vals in vars_map.items():
    b = base_map.get(g, None)
    if b is None or b == 0:
        miss += 1
        continue
    for v in vals:
        ratios[(sec, var)].append(v / b)

print("\nmissing base matches:", miss)

print("\n=== C summary (mean/std/CV) ===")
for (sec, var), xs in sorted(ratios.items()):
    if len(xs) < 2:
        continue
    mean = statistics.mean(xs)
    std  = statistics.pstdev(xs)
    cv = std/mean if mean else math.inf
    print(f"{sec:16s} {var:12s} n={len(xs):3d} mean={mean:.4f} std={std:.4f} CV={cv:.4f}")

print("\n=== variants seen per section ===")
for sec in SECTION_FIELDS:
    vs = sorted({v for (s,v) in ratios if s==sec})
    print(sec, vs)
