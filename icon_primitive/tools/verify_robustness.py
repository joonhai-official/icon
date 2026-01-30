import csv, json, glob, math, statistics
from pathlib import Path
from collections import defaultdict

MANI="configs/manifests/ICON_Primitive_Run_Manifest_Base153_v1.1.csv"
mani={}
with open(MANI, newline="", encoding="utf-8") as f:
    r=csv.DictReader(f)
    for row in r:
        mani[row["run_id"].strip()] = row

def get_C(r):
    # preferred: extras.paired baseline/variant raw
    paired = r.get("extras",{}).get("paired",{})
    if isinstance(paired, dict):
        b = paired.get("baseline",{}).get("kappa_raw", None)
        v = paired.get("variant",{}).get("kappa_raw", None)
        if isinstance(b,(int,float)) and isinstance(v,(int,float)) and b!=0:
            C = float(v)/float(b)
            if math.isfinite(C):
                return C
    # fallback: results.ratio.C
    rc = r.get("results",{}).get("ratio",{}).get("C", None)
    if isinstance(rc,(int,float)) and math.isfinite(rc):
        return float(rc)
    return None

paths=sorted(glob.glob("outputs/runs/*/receipt.json"))
vals=[]
bad=[]
for p in paths:
    rid=Path(p).parent.name
    r=json.load(open(p))
    if r.get("section")!="1G_robustness":
        continue
    C=get_C(r)
    m=mani.get(rid,{})
    ds=m.get("dataset","").strip()
    w=m.get("width","").strip()
    est=m.get("mi_estimator","").strip()
    if C is None:
        bad.append((rid,ds,w,est,"C=None"))
        continue
    if not math.isfinite(C):
        bad.append((rid,ds,w,est,f"C={C}"))
        continue
    vals.append((rid,ds,w,est,C))

print("1G total receipts:", len([1 for p in paths if json.load(open(p)).get('section')=='1G_robustness']))
print("usable:", len(vals), "bad:", len(bad))
if bad:
    print("bad examples:", bad[:6])

# summarize by estimator then axis
def summarize(tag, groups):
    print(f"\n== {tag} ==")
    for k,xs in sorted(groups.items()):
        if len(xs)<2:
            print(k, "n=",len(xs),"std=NA")
            continue
        std=statistics.pstdev(xs)
        mean=statistics.mean(xs)
        cv=std/mean if mean else float("inf")
        print(k, f"n={len(xs)} mean={mean:.4f} std={std:.4f} CV={cv:.4f} pass(std<0.03)?", std<0.03)

by_est = defaultdict(list)
for _,ds,w,est,C in vals:
    by_est[est].append(C)
summarize("estimator (global)", by_est)

for est in sorted(by_est):
    sub=[x for x in vals if x[3]==est]
    by_ds=defaultdict(list); by_w=defaultdict(list)
    for _,ds,w,_,C in sub:
        by_ds[ds].append(C)
        by_w[w].append(C)
    summarize(f"dataset (estimator={est})", by_ds)
    summarize(f"width (estimator={est})", by_w)
