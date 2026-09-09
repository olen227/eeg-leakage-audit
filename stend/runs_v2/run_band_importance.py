"""Б5. Разбор аномалии дельта-полосы.

Июльский результат (phase6_ablation.py): базовая честная F1 = 0,161, тогда как в
основном прогоне 0,328 — расхождение внутри одного прогона не объяснено; кроме того
удаление ЛЮБОЙ полосы повышало F1, что для «зависимости от низких частот» неправдоподобно.

ПРОВЕРЯЕМАЯ ГИПОТЕЗА: аномалия — артефакт некалиброванного порога 0,5.
При распространённости 0,377 % и обучении с pos_weight выход детектора смещён;
F1@0,5 находится на крутом участке кривой, поэтому ЛЮБОЕ возмущение входа, сдвигающее
распределение оценок, может случайно повысить F1@0,5, не улучшая разделимость.

Проверка: важность полос измеряется тремя способами одновременно —
  (а) F1 при пороге 0,5           (воспроизводит июльскую схему);
  (б) F1 при пороге, калиброванном на обучающей части (утечки нет);
  (в) AUPRC и ROC-AUC             (пороговонезависимые).
Если гипотеза верна, «улучшение от удаления любой полосы» видно в (а) и исчезает в (в).

Выход: outputs/track_b/band_importance_v2.json
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_decomposition import (
    get_device, train_detector_mm, calibrate_threshold_on_train, V2, OUTDIR)

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}


def bandstop(x, lo, hi, fs=C.SFREQ):
    sos = butter(4, [lo, hi], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


@torch.no_grad()
def predict_array(det, arr, mu, sd, device, bs=512):
    det.eval()
    out = np.empty(len(arr), dtype=np.float32)
    for k in range(0, len(arr), bs):
        xb = torch.from_numpy(((arr[k:k+bs] - mu) / sd).astype(np.float32))
        out[k:k+len(xb)] = torch.sigmoid(det(xb.to(device))).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t0 = time.time()

    X = np.load(os.path.join(V2, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(V2, "meta.npz"), allow_pickle=True)
    y = M["y_sz"]; subj = M["subj"]; fid = M["fileid"]; t_abs = M["t_abs"]
    ident_of = P.build_identity_map(sorted(set(subj.tolist())))
    print(f"устройство: {device}", flush=True)

    per_fold = {}
    for I in sorted(set(ident_of.values())):
        fold = P.make_fold(ident_of, subj, fid, t_abs, I, seed=C.SEED,
                           embargo_sec=C.EMBARGO_SEC, y=y)
        ev = fold["eval"]; tr = fold["train"]["P2"]
        if y[ev].sum() == 0:
            continue
        print(f"\n=== фолд {I}: eval={len(ev)} приступных={int(y[ev].sum())} ===", flush=True)
        mu, sd = P.fold_norm_stats(X, tr)
        det = train_detector_mm(X, tr, y, mu, sd, device, args.epochs, C.SEED)
        thr, _ = calibrate_threshold_on_train(det, X, tr, y, mu, sd, device, C.SEED)
        Xe = np.asarray(X[ev])
        yev = y[ev]

        def score(arr):
            p = predict_array(det, arr, mu, sd, device)
            return {"f1_at_0.5": float(f1_score(yev, (p >= 0.5).astype(int), zero_division=0)),
                    "f1_calibrated": float(f1_score(yev, (p >= thr).astype(int), zero_division=0)),
                    "auprc": float(average_precision_score(yev, p)),
                    "roc_auc": float(roc_auc_score(yev, p))}

        base = score(Xe)
        print(f"  база: F1@0.5={base['f1_at_0.5']:.4f} F1(калибр,τ={thr:.3f})={base['f1_calibrated']:.4f} "
              f"AUPRC={base['auprc']:.4f}", flush=True)
        bands = {}
        for bn, (lo, hi) in BANDS.items():
            Xp = bandstop(Xe.reshape(-1, Xe.shape[-1]), lo, hi).reshape(Xe.shape).astype(np.float32)
            s = score(Xp)
            s["delta_vs_base"] = {k: base[k] - s[k] for k in base}   # >0 => полоса важна
            bands[bn] = s
            print(f"  без {bn:6s}: F1@0.5={s['f1_at_0.5']:.4f} F1(калибр)={s['f1_calibrated']:.4f} "
                  f"AUPRC={s['auprc']:.4f} | ΔAUPRC={s['delta_vs_base']['auprc']:+.4f}", flush=True)
        per_fold[I] = {"threshold": float(thr), "eval_n": int(len(ev)),
                       "eval_pos": int(yev.sum()), "base": base, "bands": bands}

    # агрегация: средний вклад полосы по фолдам, по каждому способу измерения
    agg = {}
    for met in ["f1_at_0.5", "f1_calibrated", "auprc", "roc_auc"]:
        agg[met] = {bn: float(np.mean([per_fold[I]["bands"][bn]["delta_vs_base"][met]
                                       for I in per_fold])) for bn in BANDS}
    n_improve_05 = sum(1 for I in per_fold for bn in BANDS
                       if per_fold[I]["bands"][bn]["delta_vs_base"]["f1_at_0.5"] < 0)
    n_improve_auprc = sum(1 for I in per_fold for bn in BANDS
                          if per_fold[I]["bands"][bn]["delta_vs_base"]["auprc"] < 0)
    total = len(per_fold) * len(BANDS)

    out = {"passport": {"seed": C.SEED, "device": device, "python": platform.python_version(),
                        "torch": torch.__version__, "epochs": args.epochs,
                        "script": "stend/runs_v2/run_band_importance.py",
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "elapsed_sec": round(time.time() - t0, 1)},
           "hypothesis": ("июльская аномалия «удаление любой полосы повышает F1» — артефакт "
                          "некалиброванного порога 0,5, а не зависимости от низких частот"),
           "per_fold": per_fold,
           "mean_band_contribution": agg,
           "anomaly_check": {
               "cases_total": total,
               "cases_where_removal_improved_F1_at_0.5": n_improve_05,
               "cases_where_removal_improved_AUPRC": n_improve_auprc,
               "interpretation": ("если доля улучшений велика для F1@0,5 и мала для AUPRC, "
                                  "гипотеза о пороговом артефакте подтверждается")}}
    path = os.path.join(OUTDIR, "band_importance_v2.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print("\nсохранено:", path)
    print(json.dumps(out["anomaly_check"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
