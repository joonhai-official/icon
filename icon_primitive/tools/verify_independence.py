import csv, json, glob, math, statistics
from pathlib import Path
from collections import defaultdict

MANI="configs/manifests/ICON_Primitive_Run_Manifest_Base153_v1.1.csv"

# ---------- load manifest (non-imagenet only) ----------
rows={}
with open(MANI, newline="", encoding="utf-8") as f:
    r=csv.DictReader(f)
    for row in r:
        if row.get("dataset","").strip()=="imagenet_subset":
            continue
        rows[row["run_id"].strip()] = row

# ---------- load receipts + kappa_raw ----------
receipts={}
for p in glob.glob("outputs/runs/*/receipt.json"):
    rid=Path(p).parent.name
    if rid not in rows: 
        continue
    rr=json.load(open(p))
    kraw = rr.get("results",{}).get("kappa",{}).get("raw", None)
    if kraw is None:
        kraw = rr.get("extras",{}).get("single",{}).get("kappa_raw", None)
    receipts[rid]=(rr,kraw)

# ---------- helper: baseline run finder ----------
BASELINE_PRIM = dict(activation="relu", normalization="none", precision="fp32", skip="none", linear_type="dense")

def base_key(row, rr):
    return (
        row.get("probe","").strip(),
        row.get("dataset","").strip(),
        row.get("width","").strip(),
        row.get("mi_estimator","").strip(),
        row.get("linear_budget","").strip(),
        row.get("data_seed","").strip(),
        row.get("mi_seed","").strip(),
        row.get("model_seed","").strip(),
    )

# find baseline κ for each key
base_kappa={}
for rid,(rr,kraw) in receipts.items():
    if kraw is None: 
        continue
    row=rows[rid]
    prim=rr.get("model",{}).get("primitive",{})
    ok = all(prim.get(k)==v for k,v in BASELINE_PRIM.items())
    if not ok: 
        continue
    base_kappa[base_key(row,rr)] = kraw

print("baseline keys:", len(base_kappa))

# ---------- build primitive C tables from receipts (means across seeds) ----------
def section_C_map(section, field, baseline_val):
    # group by (dataset,width,estimator,probe,linear_budget) ignoring model_seed -> average per-seed already in receipts
    vals=defaultdict(list)  # variant -> [C...]
    for rid,(rr,kraw) in receipts.items():
        if kraw is None: 
            continue
        if rr.get("section")!=section:
            continue
        row=rows[rid]
        bk=base_key(row,rr)
        b=base_kappa.get(bk, None)
        if b is None or b==0:
            continue
        v = rr.get("model",{}).get("primitive",{}).get(field)
        if v is None or v==baseline_val:
            continue
        vals[v].append(kraw/b)
    # mean per variant
    out={}
    for v, xs in vals.items():
        out[v]=statistics.mean(xs) if xs else None
    return out

C_act  = section_C_map("1A_activation","activation","relu")
C_norm = section_C_map("1C_normalization","normalization","none")
C_prec = section_C_map("1D_precision","precision","fp32")
C_skip = section_C_map("1E_skip","skip","none")
C_lin  = section_C_map("1B_linear_type","linear_type","dense")

def getC(table, key):
    return table.get(key, 1.0) if key not in [None,"", "none","dense","relu","fp32"] else 1.0

# ---------- compute 1F prediction vs measured ----------
errs=[]
rows_out=[]
missing_base=0

for rid,(rr,kraw) in receipts.items():
    if rr.get("section")!="1F_independence_core": 
        continue
    if kraw is None:
        continue
    row=rows[rid]
    bk=base_key(row,rr)
    b=base_kappa.get(bk, None)
    if b is None or b==0:
        missing_base += 1
        continue

    prim=rr["model"]["primitive"]
    act=prim.get("activation","relu")
    norm=prim.get("normalization","none")
    prec=prim.get("precision","fp32")
    sk=prim.get("skip","none")
    lt=prim.get("linear_type","dense")

    C_pred = getC(C_act,act) * getC(C_norm,norm) * getC(C_prec,prec) * getC(C_skip,sk) * getC(C_lin,lt)
    C_meas = kraw / b
    rel = abs(C_pred - C_meas) / abs(C_meas) if C_meas!=0 else float("inf")
    errs.append(rel)
    rows_out.append((rid, C_pred, C_meas, rel, act,norm,prec,sk,lt))

print("1F runs:", len([1 for rid,(rr,_) in receipts.items() if rr.get('section')=='1F_independence_core']))
print("missing baseline matches:", missing_base)
if errs:
    mean=statistics.mean(errs)
    mx=max(errs)
    print(f"rel_error mean={mean:.4f}, max={mx:.4f}")
    print("pass(mean<0.05 & max<0.15)?", (mean<0.05 and mx<0.15))
    print("\nWorst 8:")
    for rid,cp,cm,re,act,norm,prec,sk,lt in sorted(rows_out, key=lambda x:x[3], reverse=True)[:8]:
        print(rid, f"pred={cp:.4f}", f"meas={cm:.4f}", f"rel={re:.4f}", f"[{act},{norm},{prec},{sk},{lt}]")
else:
    print("No errors computed. Likely baseline matching failed for all 1F.")
