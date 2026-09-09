"""Б6. Перенос оператора на Siena и разбор причин провала.

ЧИСЛО ВКР: roc_auc_operator_siena (перенос без дообучения).

ГИПОТЕЗЫ О ПРИЧИНЕ ПРОВАЛА, проверяемые здесь по отдельности.

H1. Способ получения отведений. В CHB-MIT биполярные отведения записаны в EDF как есть,
    для Siena вычисляются из монополярных: X = A − B.
    ЗАМЕЧАНИЕ: если монополярные каналы записаны относительно ОБЩЕГО электрода сравнения r,
    то (A+r) − (B+r) = A − B, то есть вычисленное отведение алгебраически тождественно
    измеренному напрямую. Поэтому сам по себе способ получения объяснить провал не может;
    объяснять может лишь то, что ему сопутствует (иные усилители, наложение фильтров до
    вычитания, иная раскладка электродов). Проверяется сравнением статистик каналов.

H2. Нормировка чужими статистиками. В phase5_siena.py:56 окна Siena нормируются
    mu/sd, посчитанными на CHB-MIT. При разных усилителях и единицах это смещает вход.
    Проверяется тремя схемами нормировки.

H3. Частота режекторного фильтра. NOTCH = 60 Гц (сеть США) применяется и к Siena,
    записанной в Италии, где сеть 50 Гц. Проверяется прогоном с 50 и 60 Гц.
    Ожидание: эффект мал, поскольку полосовой 0,5-40 Гц уже подавляет обе частоты.
    Нулевой результат фиксируется как результат (правка П2 исходного задания).

H4. Передискретизация 512 -> 256 Гц. Фиксируется как сопутствующее различие.

Выход: outputs/track_b/siena_transfer.json
"""
import os, sys, json, time, platform, itertools, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
import torch
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_operator_cv import (
    train_encoder_mm, embed_mm, eer_of, get_device)

mne.set_log_level("ERROR")

V2 = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
OUTDIR = os.path.join("outputs", "track_b")
SIENA_ROOT = os.path.join(C.DATA_ROOT, "siena")

# Монополярные пары для вычисления канонических отведений.
# Siena использует старую номенклатуру 10-20 (T3/T4/T5/T6), CHB-MIT — новую (T7/T8/P7/P8).
BIPOLAR = {
    "FP1-F7": ("Fp1", "F7"), "F7-T7": ("F7", "T3"), "T7-P7": ("T3", "T5"), "P7-O1": ("T5", "O1"),
    "FP1-F3": ("Fp1", "F3"), "F3-C3": ("F3", "C3"), "C3-P3": ("C3", "P3"), "P3-O1": ("P3", "O1"),
    "FP2-F4": ("Fp2", "F4"), "F4-C4": ("F4", "C4"), "C4-P4": ("C4", "P4"), "P4-O2": ("P4", "O2"),
    "FP2-F8": ("Fp2", "F8"), "F8-T8": ("F8", "T4"), "T8-P8": ("T4", "T6"), "P8-O2": ("T6", "O2"),
    "FZ-CZ": ("Fz", "Cz"), "CZ-PZ": ("Cz", "Pz"),
}

SIENA_FILES = {
    "PN00": ["PN00/PN00-4.edf", "PN00/PN00-5.edf"],
    "PN05": ["PN05/PN05-4.edf", "PN05/PN05-3.edf"],
    "PN06": ["PN06/PN06-5.edf", "PN06/PN06-4.edf"],
    "PN09": ["PN09/PN09-3.edf", "PN09/PN09-1.edf"],
    "PN12": ["PN12/PN12-3.edf", "PN12/PN12-1.2.edf"],
}


def load_siena_raw(path, notch_hz):
    """Окна Siena БЕЗ нормировки (нормировка применяется отдельно, это проверяемый фактор)."""
    r = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    r.rename_channels({ch: ch.replace("EEG ", "").strip() for ch in r.ch_names})
    data = r.get_data(); sf = r.info["sfreq"]
    idx = {n.upper(): i for i, n in enumerate(r.ch_names)}
    missing = [n for c in C.CANONICAL_CHANNELS for n in BIPOLAR[c] if n.upper() not in idx]
    if missing:
        return None, sorted(set(missing))
    bip = np.stack([data[idx[a.upper()]] - data[idx[b.upper()]]
                    for a, b in (BIPOLAR[c] for c in C.CANONICAL_CHANNELS)])
    sf_native = float(sf)
    if abs(sf - C.SFREQ) > 1:
        bip = resample_poly(bip, up=int(C.SFREQ), down=int(sf), axis=1)
    info = mne.create_info(C.CANONICAL_CHANNELS, C.SFREQ, ch_types="eeg")
    ra = mne.io.RawArray(bip, info, verbose="ERROR")
    ra.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
    if notch_hz:
        try:
            ra.notch_filter(notch_hz, verbose="ERROR")
        except Exception:
            pass
    d = ra.get_data()
    wlen = int(C.WIN_SEC * C.SFREQ); n = d.shape[1] // wlen
    W = np.stack([d[:, i * wlen:(i + 1) * wlen] for i in range(n)]).astype(np.float32)
    return W, sf_native


def pair_auc(emb, labels, n_pairs, rng):
    n = len(emb)
    a = rng.integers(0, n, n_pairs); b = rng.integers(0, n, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    dist = np.linalg.norm(emb[a] - emb[b], axis=1)
    same = (labels[a] == labels[b]).astype(int)
    if same.sum() == 0 or same.sum() == len(same):
        return None, None, None
    auc = float(roc_auc_score(same, -dist))
    eer, tau = eer_of(same, dist)
    return auc, eer, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.OP_EPOCHS)
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--per-file", type=int, default=400, help="окон на запись Siena")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t0 = time.time()
    rng = np.random.default_rng(C.SEED)

    # ---------- загрузка Siena при двух частотах режекторного фильтра ----------
    siena = {}
    load_report = []
    for notch in (60.0, 50.0):
        Xs, ss, fsn = [], [], []
        for pat, files in SIENA_FILES.items():
            for rel in files:
                p = os.path.join(SIENA_ROOT, rel)
                if not os.path.exists(p):
                    load_report.append({"file": rel, "notch": notch, "status": "нет файла"})
                    continue
                W, info = load_siena_raw(p, notch)
                if W is None:
                    load_report.append({"file": rel, "notch": notch,
                                        "status": "неполный монтаж", "missing": info})
                    continue
                k = min(args.per_file, len(W))
                sel = np.sort(rng.choice(len(W), k, replace=False))
                Xs.append(W[sel]); ss += [pat] * k; fsn.append(info)
                if notch == 60.0:
                    load_report.append({"file": rel, "windows_total": int(len(W)),
                                        "windows_used": int(k), "sfreq_native": info})
        siena[notch] = (np.concatenate(Xs), np.array(ss))
        print(f"Siena (notch={notch:.0f}): окон {len(siena[notch][0])}, "
              f"пациентов {len(set(siena[notch][1]))}", flush=True)

    Xs60, ss60 = siena[60.0]
    n_ch = Xs60.shape[1]

    # ---------- оператор, обученный на ВСЁМ CHB-MIT ----------
    X = np.load(os.path.join(V2, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(V2, "meta.npz"), allow_pickle=True)
    subj = M["subj"]
    all_idx = np.arange(len(subj))
    mu_chb, sd_chb = P.fold_norm_stats(X, all_idx)
    print("обучаю оператор на всех 6 субъектах CHB-MIT ...", flush=True)
    enc, hist = train_encoder_mm(X, all_idx, subj, mu_chb, sd_chb, device,
                                 args.epochs, C.SEED)
    print(f"  потеря {hist[0]:.4f} -> {hist[-1]:.4f}", flush=True)

    # контроль: тот же оператор на самом CHB-MIT (in-domain, но субъекты виденные)
    sel_chb = np.sort(rng.choice(len(subj), min(15000, len(subj)), replace=False))
    emb_chb = embed_mm(enc, X, sel_chb, mu_chb, sd_chb, device)
    auc_chb, eer_chb, _ = pair_auc(emb_chb, subj[sel_chb], args.n_pairs, rng)
    print(f"  контроль на CHB-MIT (виденные субъекты): ROC-AUC={auc_chb:.4f}", flush=True)

    # ---------- H2: схемы нормировки ----------
    @torch.no_grad()
    def embed_array(arr, mu, sd, bs=512):
        enc.eval()
        out = []
        for k in range(0, len(arr), bs):
            xb = torch.from_numpy(((arr[k:k + bs] - mu) / sd).astype(np.float32))
            out.append(enc(xb.to(device)).cpu().numpy())
        return np.concatenate(out)

    mu_si = Xs60.mean(axis=(0, 2)).reshape(1, -1, 1).astype(np.float32)
    sd_si = (Xs60.std(axis=(0, 2)) + 1e-7).reshape(1, -1, 1).astype(np.float32)

    schemes = {}
    e = embed_array(Xs60, mu_chb, sd_chb)
    schemes["статистики_CHB-MIT (как в июльском коде)"] = pair_auc(e, ss60, args.n_pairs, rng)
    e = embed_array(Xs60, mu_si, sd_si)
    schemes["собственные_статистики_Siena"] = pair_auc(e, ss60, args.n_pairs, rng)
    Xn = np.empty_like(Xs60)
    for i in range(len(Xs60)):
        m = Xs60[i].mean(axis=1, keepdims=True); s = Xs60[i].std(axis=1, keepdims=True) + 1e-7
        Xn[i] = (Xs60[i] - m) / s
    e = embed_array(Xn, np.zeros((1, n_ch, 1), np.float32), np.ones((1, n_ch, 1), np.float32))
    schemes["пооконная_самонормировка"] = pair_auc(e, ss60, args.n_pairs, rng)

    for k, v in schemes.items():
        print(f"  перенос, {k}: ROC-AUC={v[0]:.4f} EER={v[1]:.4f}", flush=True)

    # ---------- H3: частота режекторного фильтра ----------
    Xs50, ss50 = siena[50.0]
    mu50 = Xs50.mean(axis=(0, 2)).reshape(1, -1, 1).astype(np.float32)
    sd50 = (Xs50.std(axis=(0, 2)) + 1e-7).reshape(1, -1, 1).astype(np.float32)
    notch_res = {
        "60_Гц (сеть США, как в коде)": schemes["собственные_статистики_Siena"],
        "50_Гц (сеть Италии, верная)": pair_auc(embed_array(Xs50, mu50, sd50),
                                                ss50, args.n_pairs, rng),
    }
    for k, v in notch_res.items():
        print(f"  режекторный {k}: ROC-AUC={v[0]:.4f}", flush=True)

    # ---------- in-domain Siena: обучение и оценка на Siena по фолдам пациентов ----------
    pats = sorted(set(ss60.tolist()))
    indom = []
    for held in itertools.combinations(pats, 2):
        tr = np.where(~np.isin(ss60, list(held)))[0]
        ev = np.where(np.isin(ss60, list(held)))[0]
        mu_t = Xs60[tr].mean(axis=(0, 2)).reshape(1, -1, 1).astype(np.float32)
        sd_t = (Xs60[tr].std(axis=(0, 2)) + 1e-7).reshape(1, -1, 1).astype(np.float32)
        Xtr = ((Xs60[tr] - mu_t) / sd_t)
        enc_s, h_s = train_encoder_mm(Xtr, np.arange(len(tr)), ss60[tr],
                                      np.zeros((1, n_ch, 1), np.float32),
                                      np.ones((1, n_ch, 1), np.float32),
                                      device, args.epochs, C.SEED)
        enc_bak, globals()["enc"] = enc, enc_s          # переиспользуем embed_array
        e_ev = embed_array((Xs60[ev] - mu_t) / sd_t,
                           np.zeros((1, n_ch, 1), np.float32),
                           np.ones((1, n_ch, 1), np.float32))
        globals()["enc"] = enc_bak
        a, er, _ = pair_auc(e_ev, ss60[ev], args.n_pairs, rng)
        if a is not None:
            indom.append({"held": list(held), "roc_auc": a, "eer": er})
            print(f"  in-domain Siena, отложены {held}: ROC-AUC={a:.4f}", flush=True)

    # ---------- H1: сравнение статистик каналов между наборами ----------
    chb_sample = np.asarray(X[np.sort(rng.choice(len(subj), 3000, replace=False))])
    stats = {
        "chbmit": {"std_по_каналам": chb_sample.std(axis=(0, 2)).tolist(),
                   "std_среднее": float(chb_sample.std()),
                   "медиана_модуля": float(np.median(np.abs(chb_sample)))},
        "siena": {"std_по_каналам": Xs60.std(axis=(0, 2)).tolist(),
                  "std_среднее": float(Xs60.std()),
                  "медиана_модуля": float(np.median(np.abs(Xs60)))},
    }
    ratio = stats["siena"]["std_среднее"] / max(stats["chbmit"]["std_среднее"], 1e-20)
    stats["отношение_масштабов_siena_к_chbmit"] = float(ratio)

    out = {
        "passport": {
            "seed": C.SEED, "device": device, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "mne": mne.__version__,
            "script": "stend/runs_v2/run_siena_transfer.py",
            "epochs": args.epochs, "n_pairs": args.n_pairs,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_sec": round(time.time() - t0, 1),
            "siena_root": SIENA_ROOT,
        },
        "target_metric": {
            "name": "roc_auc_operator_siena",
            "definition": ("перенос оператора без дообучения: оператор обучен на CHB-MIT, "
                           "пары строятся из окон пациентов Siena, метка — один ли пациент"),
            "value": schemes["статистики_CHB-MIT (как в июльском коде)"][0],
            "value_best_normalization": max(v[0] for v in schemes.values()),
        },
        "control_chbmit_in_domain": {"roc_auc": auc_chb, "eer": eer_chb,
                                     "note": "субъекты виденные при обучении, верхняя граница"},
        "H2_normalization": {k: {"roc_auc": v[0], "eer": v[1]} for k, v in schemes.items()},
        "H3_notch": {k: {"roc_auc": v[0], "eer": v[1]} for k, v in notch_res.items()},
        "in_domain_siena": indom,
        "in_domain_siena_mean": (float(np.mean([r["roc_auc"] for r in indom]))
                                 if indom else None),
        "H1_montage_note": (
            "Вычисление биполярного отведения из монополярных при ОБЩЕМ электроде сравнения "
            "алгебраически тождественно измеренному напрямую: (A+r)-(B+r) = A-B. Поэтому сам "
            "способ получения отведений объяснить провал переноса не может. Сопутствующие "
            "различия наборов (усилители, единицы, раскладка электродов по старой номенклатуре "
            "10-20, частота дискретизации 512 против 256, возраст пациентов) — могут."),
        "H1_channel_statistics": stats,
        "H4_resampling": "Siena 512 Гц -> 256 Гц через resample_poly (см. load_siena_raw)",
        "load_report": load_report,
    }
    path = os.path.join(OUTDIR, "siena_transfer.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print("\nсохранено:", path)
    print(json.dumps({"roc_auc_operator_siena": out["target_metric"]["value"],
                      "in_domain_siena_mean": out["in_domain_siena_mean"],
                      "отношение_масштабов": ratio}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
