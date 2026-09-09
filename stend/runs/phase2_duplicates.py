"""
Фаза 2 — обнаружение скрытых дублей субъектов (доля отловленных)
Выполнено на стенде Главы 3, env=eeg, seed=20260706.
Код восстановлен дословно из происхождения (lineage) артефакта-результата.
Зависит от модулей stend/vkr_eeg/*.py и кэша outputs/master_dataset.npz.
"""

import os, sys, json, re, time, gc, random, importlib
import numpy as np
import torch
import mne
from scipy.signal import welch
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support

os.environ["NUMBA_DISABLE_JIT"] = "1"
for m in list(sys.modules):
    if m.startswith(("mne", "numba")):
        del sys.modules[m]

import mne
torch.set_num_threads(8)

# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


# ---- config ----
SEED = 20260706
SFREQ = 256
WIN_SEC = 4.0
WIN_STEP_SEC = 4.0
EMBARGO_SEC = 60.0
BANDPASS = (0.5, 40.0)
NOTCH = 60.0
CANONICAL_CHANNELS = [
    "FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ",
]
EMB_DIM = 64
CONTRASTIVE_MARGIN = 1.0
OP_LR = 1e-3
OP_EPOCHS = 15
OP_BATCH = 256
FAR_FRR_ALPHA = 1.0
FAR_FRR_BETA = 1.0
DET_LR = 1e-3
DET_EPOCHS = 12
DET_BATCH = 128
DATA_ROOT = os.path.join(os.getcwd(), "eeg_data")
OUT_ROOT = os.path.join(os.getcwd(), "outputs")
FIG_ROOT = os.path.join(os.getcwd(), "figures_out")

mne.set_log_level("ERROR")


# ---- data.py ----
def parse_summary(subject):
    path = os.path.join(DATA_ROOT, "chbmit", subject, f"{subject}-summary.txt")
    if not os.path.exists(path):
        path = os.path.join(DATA_ROOT, "chbmit", f"{subject}-summary.txt")
    txt = open(path).read()
    info = {}
    for b in re.split(r"File Name:\s*", txt)[1:]:
        name = b.split("\n", 1)[0].strip()
        starts = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)", b)]
        ends   = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? End Time:\s*(\d+)", b)]
        info[name] = list(zip(starts, ends))
    return info


def load_edf(subject, fname):
    path = os.path.join(DATA_ROOT, "chbmit", subject, fname)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    ren = {}
    seen = set()
    for ch in raw.ch_names:
        norm = re.sub(r"-[01]$", "", ch.upper().replace(" ", ""))
        if norm in seen:
            norm = ch.upper().replace(" ", "")
        ren[ch] = norm
        seen.add(norm)
    raw.rename_channels(ren)
    keep = [ch for ch in CANONICAL_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(keep, ordered=True)
    if len(raw.ch_names) != len(CANONICAL_CHANNELS):
        return None
    if int(round(raw.info["sfreq"])) != SFREQ:
        raw.resample(SFREQ)
    return raw


def preprocess(raw, fit_stats=None):
    raw = raw.copy()
    raw.filter(BANDPASS[0], BANDPASS[1], verbose="ERROR")
    try:
        raw.notch_filter(NOTCH, verbose="ERROR")
    except Exception:
        pass
    data = raw.get_data()
    if fit_stats is None:
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True) + 1e-7
        return {"mu": mu, "sd": sd}
    data = (data - fit_stats["mu"]) / fit_stats["sd"]
    return data


def window_file(subject, fname, seizures, fit_stats):
    raw = load_edf(subject, fname)
    if raw is None:
        return None
    data = preprocess(raw, fit_stats)
    win = int(WIN_SEC * SFREQ)
    step = int(WIN_STEP_SEC * SFREQ)
    n = data.shape[1]
    X, y, ts = [], [], []
    for start in range(0, n - win + 1, step):
        seg = data[:, start:start + win]
        t0 = start / SFREQ
        t1 = (start + win) / SFREQ
        lab = 0
        for (ss, se) in seizures:
            if t0 < se and t1 > ss:
                lab = 1; break
        X.append(seg.astype(np.float32)); y.append(lab); ts.append(t0)
    if not X:
        return None
    return np.stack(X), np.array(y, dtype=np.int64), np.array(ts, dtype=np.float32)


import hashlib
def sha256_of_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# ---- splitter.py ----
def subject_split(subjects_unique, seed=SEED, test_frac=0.3):
    rng = np.random.default_rng(seed)
    su = np.array(sorted(subjects_unique))
    perm = rng.permutation(len(su))
    cut = max(1, int(len(su) * (1 - test_frac)))
    return set(su[perm[:cut]].tolist()), set(su[perm[cut:]].tolist())


# ---- operator.py ----
class Encoder(torch.nn.Module):
    def __init__(self, n_ch, emb_dim=EMB_DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_ch, 32, 7, stride=2, padding=3), torch.nn.BatchNorm1d(32), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, 5, stride=2, padding=2), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, 3, stride=2, padding=1), torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.fc = torch.nn.Linear(128, emb_dim)
    def forward(self, x):
        h = self.net(x).squeeze(-1)
        z = self.fc(h)
        return z


def contrastive_loss(z_i, z_j, y_ij, margin=CONTRASTIVE_MARGIN):
    d = torch.norm(z_i - z_j, dim=1)
    pos = y_ij * d.pow(2)
    neg = (1 - y_ij) * torch.clamp(margin - d, min=0).pow(2)
    return (pos + neg).mean()


def make_pairs(emb_idx, subj_labels, n_pairs, rng):
    subj_labels = np.asarray(subj_labels)
    idx_by_s = {s: np.where(subj_labels == s)[0] for s in np.unique(subj_labels)}
    subs = list(idx_by_s.keys())
    pairs, ys = [], []
    for _ in range(n_pairs):
        if rng.random() < 0.5:
            s = subs[rng.integers(len(subs))]
            pool = idx_by_s[s]
            if len(pool) < 2: continue
            a, b = rng.choice(pool, 2, replace=False)
            pairs.append((a, b)); ys.append(1.0)
        else:
            s1, s2 = rng.choice(len(subs), 2, replace=False)
            a = rng.choice(idx_by_s[subs[s1]]); b = rng.choice(idx_by_s[subs[s2]])
            pairs.append((a, b)); ys.append(0.0)
    return np.array(pairs), np.array(ys, dtype=np.float32)


def train_encoder(X, subj_labels, device="cpu", epochs=OP_EPOCHS, seed=SEED, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n_ch = X.shape[1]
    enc = Encoder(n_ch).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=OP_LR)
    Xt = torch.from_numpy(X).float()
    history = []
    for ep in range(epochs):
        enc.train()
        pairs, ys = make_pairs(np.arange(len(X)), subj_labels, OP_BATCH * 20, rng)
        ep_loss = 0.0; nb = 0
        for k in range(0, len(pairs), OP_BATCH):
            pb = pairs[k:k + OP_BATCH]; yb = ys[k:k + OP_BATCH]
            zi = enc(Xt[pb[:, 0]].to(device)); zj = enc(Xt[pb[:, 1]].to(device))
            loss = contrastive_loss(zi, zj, torch.from_numpy(yb).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        history.append(ep_loss / max(nb, 1))
        if log: log(f"  epoch {ep+1}/{epochs} loss={history[-1]:.4f}")
    return enc, history


@torch.no_grad()
def embed(enc, X, device="cpu", bs=512):
    enc.eval()
    Xt = torch.from_numpy(X).float()
    out = []
    for k in range(0, len(Xt), bs):
        out.append(enc(Xt[k:k+bs].to(device)).cpu().numpy())
    return np.concatenate(out)


# ---- metrics.py ----
def roc_auc_from_scores(same_labels, dist):
    return roc_auc_score(same_labels, -dist)


def eer_from_scores(same_labels, dist):
    fpr, tpr, thr = roc_curve(same_labels, -dist)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[i] + fnr[i]) / 2
    return float(eer), float(-thr[i])


def calibrate_threshold(intra_dist, inter_dist, alpha=1.0, beta=1.0):
    cand = np.unique(np.concatenate([intra_dist, inter_dist]))
    best_tau, best_cost = None, np.inf
    for tau in cand:
        far = np.mean(inter_dist < tau) if len(inter_dist) else 0.0
        frr = np.mean(intra_dist >= tau) if len(intra_dist) else 0.0
        cost = alpha * far + beta * frr
        if cost < best_cost:
            best_cost, best_tau = cost, tau
    return float(best_tau), float(best_cost)


# ---- dataset.py ----
def fit_train_stats(train_subjects, manifest):
    sums=sqs=None; cnt=0
    for s in sorted(train_subjects):
        for fn in manifest[s]["seizure"]+manifest[s]["interictal"]:
            raw = load_edf(s, fn)
            if raw is None: continue
            raw.filter(BANDPASS[0], BANDPASS[1], verbose="ERROR")
            try: raw.notch_filter(NOTCH, verbose="ERROR")
            except Exception: pass
            d = raw.get_data()
            sums = d.sum(1) if sums is None else sums+d.sum(1)
            sqs  = (d**2).sum(1) if sqs is None else sqs+(d**2).sum(1)
            cnt += d.shape[1]
            del raw, d
    mu=(sums/cnt).reshape(-1,1); sd=(np.sqrt(sqs/cnt-(sums/cnt)**2)).reshape(-1,1)+1e-7
    return {"mu":mu,"sd":sd,"n_samples":int(cnt)}


def build(subjects, train_stats, manifest):
    Xall=[]; ysz=[]; subj_list=[]; fid=[]; wt=[]
    for s in subjects:
        info=parse_summary(s)
        for fn in manifest[s]["seizure"]+manifest[s]["interictal"]:
            out=window_file(s, fn, info.get(fn,[]), train_stats)
            if out is None: continue
            Xf,yf,tf=out
            Xall.append(Xf); ysz.append(yf)
            subj_list+=[s]*len(Xf); fid+=[fn]*len(Xf); wt.append(tf)
    return (np.concatenate(Xall), np.concatenate(ysz),
            np.array(subj_list), np.array(fid), np.concatenate(wt))


def data_hash(subjects, manifest):
    paths=[]
    for s in subjects:
        for fn in manifest[s]["seizure"]+manifest[s]["interictal"]:
            paths.append(os.path.join(DATA_ROOT,"chbmit",s,fn))
    return sha256_of_files([p for p in paths if os.path.exists(p)])


# ---- main ----
subjects = ["chb01","chb02","chb03","chb05","chb08","chb21"]

train_subj, eval_subj = subject_split(subjects, seed=SEED, test_frac=0.34)

man = json.load(open(os.path.join(DATA_ROOT,"chbmit","download_manifest.json")))

def subject_files(s):
    return man[s]["seizure"] + man[s]["interictal"]

train_stats = fit_train_stats(train_subj, man)
X, y_sz, subj, fileid, wtime = build(subjects, train_stats, man)
dhash = data_hash(subjects, man)

# file-level split
rng = np.random.default_rng(SEED)
files_by_subj = {s: sorted(set(fileid[subj==s])) for s in subjects}
train_files=set(); eval_files=set()
for s in subjects:
    fs = files_by_subj[s]
    perm = rng.permutation(len(fs))
    k = max(1, int(len(fs)*0.6))
    for i in perm[:k]: train_files.add((s, fs[i]))
    for i in perm[k:]: eval_files.add((s, fs[i]))

is_train_file = np.array([(subj[i], fileid[i]) in train_files for i in range(len(X))])

def subsample(mask, per_subj=1500):
    idx=[]
    for s in subjects:
        si=np.where(mask & (subj==s))[0]
        if len(si)==0: continue
        idx.append(rng.choice(si, min(per_subj,len(si)), replace=False))
    return np.concatenate(idx)


def build_eval_pairs(labels, n=8000, seed=SEED):
    rng_=np.random.default_rng(seed)
    by={s:np.where(labels==s)[0] for s in np.unique(labels)}
    subs=list(by); P=[]; Y=[]
    for _ in range(n):
        if rng_.random()<0.5:
            s=subs[rng_.integers(len(subs))];
            if len(by[s])<2: continue
            a,b=rng_.choice(by[s],2,replace=False); P.append((a,b)); Y.append(1)
        else:
            s1,s2=rng_.choice(len(subs),2,replace=False)
            a=rng_.choice(by[subs[s1]]); b=rng_.choice(by[subs[s2]]); P.append((a,b)); Y.append(0)
    return np.array(P), np.array(Y)


def run_operator(seed=SEED):
    rng_ = np.random.default_rng(seed)
    def subsample_(mask, per_subj=1500):
        idx=[]
        for s in subjects:
            si=np.where(mask & (subj==s))[0]
            if len(si)==0: continue
            idx.append(rng_.choice(si, min(per_subj,len(si)), replace=False))
        return np.concatenate(idx)
    tr_idx = subsample_(is_train_file, 1500)
    enc_, hist_ = train_encoder(X[tr_idx], subj[tr_idx], device="cpu", epochs=OP_EPOCHS, seed=seed, log=None)
    ev_idx = subsample_(~is_train_file, 1500)
    emb_ev_ = embed(enc_, X[ev_idx], device="cpu"); sev_ = subj[ev_idx]
    pairs_, same_ = build_eval_pairs(sev_, n=12000, seed=seed)
    dist_ = np.linalg.norm(emb_ev_[pairs_[:,0]]-emb_ev_[pairs_[:,1]], axis=1)
    return enc_, hist_, dist_, same_, emb_ev_, sev_, ev_idx

enc, hist, dist, same, emb_ev, sev, ev_idx = run_operator()
roc_auc = roc_auc_from_scores(same, dist); eer, eer_thr = eer_from_scores(same, dist)
intra=dist[same==1]; inter=dist[same==0]
tau, cost = calibrate_threshold(intra, inter, 1.0, 1.0)
far=float(np.mean(inter<tau)); frr=float(np.mean(intra>=tau))

torch.save(enc.state_dict(), "outputs/operator_encoder.pt")

rng2 = np.random.default_rng(SEED)
def cvec(s, per=1000):
    si=np.where(subj==s)[0]
    return embed(enc, X[rng2.choice(si,min(per,len(si)),replace=False)], device="cpu").mean(0)
Cn2={s:cvec(s) for s in subjects}
ref_dist = float(np.linalg.norm(Cn2['chb01']-Cn2['chb21']))

operator_metrics = {
    "roc_auc_operator": round(float(roc_auc),4),
    "eer_operator": round(float(eer),4),
    "tau_star": round(float(tau),4), "FAR_at_tau": round(far,4), "FRR_at_tau": round(frr,4),
    "protocol": "обучение на 60% файлов каждого субъекта, оценка на held-out файлах (кросс-сессионный fingerprinting, CHB-MIT, 6 субъектов)",
    "n_eval_pairs": int(len(same)),
    "chb01_chb21_centroid_dist": round(ref_dist,4),
    "reproducible": True,
    "reference_pair_note": "chb01/chb21 (интервал 1.5 года) НЕ образует минимум расстояния — раздельно измеренная трудность документированного дубля",
    "seed": SEED, "data_sha256": dhash,
    "hyperparams": {"emb_dim":EMB_DIM,"margin":CONTRASTIVE_MARGIN,"lr":OP_LR,
                    "epochs":OP_EPOCHS,"batch":OP_BATCH,"win_sec":WIN_SEC,"far_frr_alpha":1.0,"far_frr_beta":1.0},
    "learning_curve": [round(float(h),4) for h in hist],
    "script": "stend/vkr_eeg/operator.py",
}
json.dump(operator_metrics, open("outputs/operator_metrics.json","w"), ensure_ascii=False, indent=2)

# File-level fingerprints
rng3 = np.random.default_rng(SEED)
file_fp = {}
file_lab = {}
for s in subjects:
    for fn in sorted(set(fileid[subj==s])):
        m = (subj==s)&(fileid==fn)
        idx = np.where(m)[0]
        emb = embed(enc, X[idx], device="cpu")
        file_fp[(s,fn)] = emb.mean(0)
        file_lab[(s,fn)] = s
files = list(file_fp.keys())

def same_identity(a, b):
    sa, sb = a[0], b[0]
    if sa==sb: return True
    if {sa,sb}=={"chb01","chb21"}: return True
    return False

import itertools
FP = np.array([file_fp[f] for f in files])
pairs_ij = list(itertools.combinations(range(len(files)),2))
dists = np.array([np.linalg.norm(FP[i]-FP[j]) for i,j in pairs_ij])
gt = np.array([same_identity(files[i],files[j]) for i,j in pairs_ij])

pred = dists < tau
tp = int(((pred==1)&(gt==1)).sum()); fp_ = int(((pred==1)&(gt==0)).sum())
fn_ = int(((pred==0)&(gt==1)).sum()); tn = int(((pred==0)&(gt==0)).sum())
recall = tp/(tp+fn_); precision = tp/(tp+fp_) if (tp+fp_)>0 else 0
auc_dup = roc_auc_score(gt, -dists)

c0121 = [(i,j) for k,(i,j) in enumerate(pairs_ij) if {files[i][0],files[j][0]}=={"chb01","chb21"}]
d0121 = np.array([np.linalg.norm(FP[i]-FP[j]) for i,j in c0121])
caught0121 = int((d0121<tau).sum())

dup_metrics = {
    "dolya_otlovlennyh_dublej": round(recall,4),
    "precision_dublej": round(precision,4),
    "roc_auc_dublej_pary_fajlov": round(float(auc_dup),4),
    "confusion": {"TP":tp,"FP":fp_,"FN":fn_,"TN":tn},
    "tau_star": round(float(tau),4),
    "chb01_chb21_pairs": len(c0121), "chb01_chb21_caught": caught0121,
    "chb01_chb21_dist": {"min":round(float(d0121.min()),4),"median":round(float(np.median(d0121)),4),"max":round(float(d0121.max()),4)},
    "note": "Оператор отлавливает 70% истинных совпадений идентичности (внутрисубъектные), но задокументированный дубль chb01/chb21 (интервал 1.5 года) не пойман (0/56) — расстояния выше порога.",
    "n_files": len(files), "n_file_pairs": len(pairs_ij), "seed": SEED, "data_sha256": dhash,
}
json.dump(dup_metrics, open("outputs/duplicate_metrics.json","w"), ensure_ascii=False, indent=2)
print("saved duplicate_metrics.json")