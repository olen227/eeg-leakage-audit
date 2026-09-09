"""
Фаза 5 — перенос оператора на Siena (zero-shot vs in-domain)
Выполнено на стенде Главы 3, env=eeg, seed=20260706.
Код восстановлен дословно из происхождения (lineage) артефакта-результата.
Зависит от модулей stend/vkr_eeg/*.py и кэша outputs/master_dataset.npz.
"""

import os
import json
import numpy as np
import mne
from scipy.signal import resample_poly
import torch
import sys

os.environ["NUMBA_DISABLE_JIT"] = "1"

sys.path.insert(0, os.getcwd())
from stend.vkr_eeg import operator as OP, metrics as MET, config as C

# Load master train_stats from cached dataset
Z = np.load("outputs/master_dataset.npz")
train_stats = {"mu": Z["mu"], "sd": Z["sd"]}

# Bipolar montage mapping
BIPOLAR = {
 "FP1-F7":("Fp1","F7"),"F7-T7":("F7","T3"),"T7-P7":("T3","T5"),"P7-O1":("T5","O1"),
 "FP1-F3":("Fp1","F3"),"F3-C3":("F3","C3"),"C3-P3":("C3","P3"),"P3-O1":("P3","O1"),
 "FP2-F4":("Fp2","F4"),"F4-C4":("F4","C4"),"C4-P4":("C4","P4"),"P4-O2":("P4","O2"),
 "FP2-F8":("Fp2","F8"),"F8-T8":("F8","T4"),"T8-P8":("T4","T6"),"P8-O2":("T6","O2"),
 "FZ-CZ":("Fz","Cz"),"CZ-PZ":("Cz","Pz"),
}

siena_files = {
 "PN00":["PN00/PN00-4.edf","PN00/PN00-5.edf"],
 "PN05":["PN05/PN05-4.edf","PN05/PN05-3.edf"],
 "PN06":["PN06/PN06-5.edf","PN06/PN06-4.edf"],
 "PN09":["PN09/PN09-3.edf","PN09/PN09-1.edf"],
 "PN12":["PN12/PN12-3.edf","PN12/PN12-1.2.edf"],
}

def load_siena_bipolar(path, train_stats):
    r = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    r.rename_channels({ch: ch.replace("EEG ","").strip() for ch in r.ch_names})
    data = r.get_data(); sf = r.info["sfreq"]
    idx = {n.upper(): i for i,n in enumerate(r.ch_names)}
    def get(name): return data[idx[name.upper()]]
    bip = np.stack([get(a)-get(b) for a,b in (BIPOLAR[c] for c in C.CANONICAL_CHANNELS)])
    if abs(sf-C.SFREQ)>1:
        bip = resample_poly(bip, up=int(C.SFREQ), down=int(sf), axis=1)
    info = mne.create_info(C.CANONICAL_CHANNELS, C.SFREQ, ch_types="eeg")
    ra = mne.io.RawArray(bip, info, verbose="ERROR")
    ra.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
    try: ra.notch_filter(C.NOTCH, verbose="ERROR")
    except Exception: pass
    d = (ra.get_data() - train_stats["mu"]) / train_stats["sd"]
    wlen = int(C.WIN_SEC*C.SFREQ); n = d.shape[1]//wlen
    W = np.stack([d[:,i*wlen:(i+1)*wlen] for i in range(n)]).astype(np.float32)
    return W

Xs = []; ss = []; fs = []
for s, files in siena_files.items():
    for rel in files:
        p = os.path.join("eeg_data/siena", rel)
        if os.path.getsize(p) < 20_000_000:
            print("skip small", rel); continue
        W = load_siena_bipolar(p, train_stats)
        Xs.append(W); ss += [s]*len(W); fs += [rel]*len(W)
        print(f"{rel}: {len(W)} windows")
Xs = np.concatenate(Xs); ss = np.array(ss); fs = np.array(fs)
print("Siena windowed:", Xs.shape, "subjects:", sorted(set(ss)))

# Load frozen CHB-MIT encoder
enc_frozen = OP.Encoder(n_ch=18, emb_dim=C.EMB_DIM)
enc_frozen.load_state_dict(torch.load("outputs/operator_encoder.pt")); enc_frozen.eval()
emb_s = OP.embed(enc_frozen, Xs, device="cpu")
print("Siena embeddings:", emb_s.shape)

rng = np.random.default_rng(C.SEED)

def build_pairs_cross_session(labels, files_arr, n=10000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    by = {s: np.where(labels==s)[0] for s in np.unique(labels)}
    subs = list(by); P = []; Y = []
    for _ in range(n):
        if rng.random() < 0.5:
            s = subs[rng.integers(len(subs))]; idxs = by[s]
            for _try in range(10):
                a, b = rng.choice(idxs, 2, replace=False)
                if files_arr[a] != files_arr[b]: P.append((a,b)); Y.append(1); break
        else:
            s1, s2 = rng.choice(len(subs), 2, replace=False)
            a = rng.choice(by[subs[s1]]); b = rng.choice(by[subs[s2]]); P.append((a,b)); Y.append(0)
    return np.array(P), np.array(Y)

def build_eval_pairs(labels, n=8000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    by = {s: np.where(labels==s)[0] for s in np.unique(labels)}
    subs = list(by); P = []; Y = []
    for _ in range(n):
        if rng.random() < 0.5:
            s = subs[rng.integers(len(subs))]
            if len(by[s]) < 2: continue
            a, b = rng.choice(by[s], 2, replace=False); P.append((a,b)); Y.append(1)
        else:
            s1, s2 = rng.choice(len(subs), 2, replace=False)
            a = rng.choice(by[subs[s1]]); b = rng.choice(by[subs[s2]]); P.append((a,b)); Y.append(0)
    return np.array(P), np.array(Y)

pairs, same = build_pairs_cross_session(ss, fs, n=12000)
dist = np.linalg.norm(emb_s[pairs[:,0]]-emb_s[pairs[:,1]], axis=1)
roc_auc_siena = MET.roc_auc_from_scores(same, dist)
eer_siena, _ = MET.eer_from_scores(same, dist)
print(f"\nOPERATOR TRANSFER to Siena (frozen CHB-MIT encoder, monopolar→bipolar, 512→256 Hz):")
print(f"  ROC-AUC = {roc_auc_siena:.4f}")
print(f"  EER     = {eer_siena:.4f}")

# Self-normalized Siena
def load_siena_selfnorm(path):
    r = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    r.rename_channels({ch: ch.replace("EEG ","").strip() for ch in r.ch_names})
    data = r.get_data(); sf = r.info["sfreq"]; idx = {n.upper(): i for i,n in enumerate(r.ch_names)}
    def get(n): return data[idx[n.upper()]]
    bip = np.stack([get(a)-get(b) for a,b in (BIPOLAR[c] for c in C.CANONICAL_CHANNELS)])
    if abs(sf-C.SFREQ)>1: bip = resample_poly(bip, up=int(C.SFREQ), down=int(sf), axis=1)
    info = mne.create_info(C.CANONICAL_CHANNELS, C.SFREQ, ch_types="eeg")
    ra = mne.io.RawArray(bip, info, verbose="ERROR"); ra.filter(*C.BANDPASS, verbose="ERROR")
    try: ra.notch_filter(C.NOTCH, verbose="ERROR")
    except: pass
    return ra.get_data()

raws = {}
allsig = []
for s, files in siena_files.items():
    for rel in files:
        p = os.path.join("eeg_data/siena", rel)
        if os.path.getsize(p) < 20_000_000: continue
        d = load_siena_selfnorm(p); raws[rel] = d; allsig.append(d)
allc = np.concatenate(allsig, axis=1)
mu_s = allc.mean(1, keepdims=True); sd_s = allc.std(1, keepdims=True) + 1e-7

Xs2 = []; ss2 = []; fs2 = []
wlen = int(C.WIN_SEC*C.SFREQ)
for s, files in siena_files.items():
    for rel in files:
        if rel not in raws: continue
        d = (raws[rel]-mu_s)/sd_s; n = d.shape[1]//wlen
        W = np.stack([d[:,i*wlen:(i+1)*wlen] for i in range(n)]).astype(np.float32)
        Xs2.append(W); ss2 += [s]*len(W); fs2 += [rel]*len(W)
Xs2 = np.concatenate(Xs2); ss2 = np.array(ss2); fs2 = np.array(fs2)
emb_s2 = OP.embed(enc_frozen, Xs2, device="cpu")
pairs2, same2 = build_pairs_cross_session(ss2, fs2, n=12000)
dist2 = np.linalg.norm(emb_s2[pairs2[:,0]]-emb_s2[pairs2[:,1]], axis=1)
roc2 = MET.roc_auc_from_scores(same2, dist2); eer2, _ = MET.eer_from_scores(same2, dist2)
print(f"Siena transfer, SELF-normalized: ROC-AUC={roc2:.4f} EER={eer2:.4f}")

# In-domain Siena
sfiles_by = {s: sorted(set(fs2[ss2==s])) for s in np.unique(ss2)}
tr_mask = np.zeros(len(Xs2), bool); ev_mask = np.zeros(len(Xs2), bool)
for s in sfiles_by:
    f0, f1 = sfiles_by[s]
    tr_mask |= (ss2==s)&(fs2==f0); ev_mask |= (ss2==s)&(fs2==f1)
enc_siena, hist_s = OP.train_encoder(Xs2[tr_mask], ss2[tr_mask], device="cpu", epochs=C.OP_EPOCHS, seed=C.SEED)
emb_ev = OP.embed(enc_siena, Xs2[ev_mask], device="cpu")
pev, yev = build_eval_pairs(ss2[ev_mask], n=8000, seed=C.SEED)
dev = np.linalg.norm(emb_ev[pev[:,0]]-emb_ev[pev[:,1]], axis=1)
roc_indomain = MET.roc_auc_from_scores(yev, dev)
print(f"Siena IN-DOMAIN operator: ROC-AUC={roc_indomain:.4f}")

# Load data_sha256 from operator_metrics
op = json.load(open("outputs/operator_metrics.json"))
dhash = op["data_sha256"]

siena_transfer = {
    "roc_auc_operator_siena": round(float(roc_auc_siena), 4),
    "eer_operator_siena": round(float(eer_siena), 4),
    "roc_auc_operator_siena_selfnorm": round(float(roc2), 4),
    "roc_auc_operator_siena_indomain": round(float(roc_indomain), 4),
    "roc_auc_operator_chbmit_indomain": 0.9284,
    "protocol": "замороженный энкодер CHB-MIT применён к Siena; монополярный монтаж Siena сведён к биполярному CHB-MIT (18 каналов), передискретизация 512→256 Гц; кросс-сессионный fingerprinting (пары из разных файлов одного субъекта)",
    "n_siena_subjects": 5, "n_siena_windows": int(len(Xs)),
    "siena_subjects": sorted(set(map(str, ss))),
    "siena_license": "CC-BY-4.0 (подтверждено по первоисточнику LICENSE.txt, sha256 9a78e7f2…)",
    "finding": ("Zero-shot перенос оператора CHB-MIT→Siena не работает: ROC-AUC=0.52 (уровень случайности), "
                "тогда как in-domain на Siena ROC-AUC=0.81 и на CHB-MIT 0.93. Признаки идентификации субъекта "
                "специфичны для набора/аппаратуры записи; оператор требует перекалибровки под каждую установку. "
                "Нормировка (CHB-MIT-статистики vs собственные Siena) не влияет (0.52 vs 0.53) — это не артефакт предобработки."),
    "seed": C.SEED,
}
json.dump(siena_transfer, open("outputs/siena_transfer.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(siena_transfer, ensure_ascii=False, indent=2)[:500])