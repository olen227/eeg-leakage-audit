"""Честный детектор на АВТОРСКОМ разбиении TUSZ: обучение на train, оценка на eval.

ЗАЧЕМ. Разложение завышения на TUSZ (run_decomposition --group-folds) сопоставимо с
остальными наборами работы, но не с литературой: статьи по TUSZ отчитываются на
официальных частях train/dev/eval, которые авторы набора разделили по пациентам.
Здесь детектор обучается ровно как честный протокол P2 (те же гиперпараметры, та же
подвыборка межприступных окон, тот же seed) на официальной train-части и оценивается
на официальной eval-части. Пациенты частей не пересекаются (проверено по кэшу: 675
пациентов, каждый ровно в одной части), так что утечки перекрытия пациентов нет по
построению.

ЧТО СЧИТАЕТСЯ. В кэше лежат ВСЕ окна каждой записи подряд, поэтому оценка идёт на
непрерывных записях, и событийный счёт по SzCORE (допуски 30/60 с, склейка гипотез
ближе 90 с, ложные тревоги на реальную длительность) здесь корректен без оговорок.
Приводятся:
  - пооконные метрики при двух порогах, откалиброванных ТОЛЬКО на train: на
    подвыборке (как в разложении) и в естественной пропорции классов;
  - событийные метрики SzCORE для тех же порогов, без сглаживания и со сглаживанием;
  - кривая «чувствительность — ложные тревоги в сутки» по сетке порогов на eval —
    описательная, как ROC; рабочие точки на ней отмечены, а не выбраны по eval.

ЗАПУСК:
    $PY stend/runs_v2/run_official_split_tusz.py --cache $VKR_CACHE_ROOT/tusz \
        --neg-ratio 5 --epochs 12 --ram-train --tag official_split_tusz
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_decomposition import (get_device, subsample_negatives, train_detector_mm,
                                             predict_mm, calibrate_threshold_on_train,
                                             calibrate_threshold_natural, load_rows, V2)
from stend.runs_v2.run_event_scoring import smooth
from stend.runs_v2.run_event_continuous import events_sec, szcore_events

OUTDIR = os.path.join("outputs", "track_b")


def window_metrics(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "f1": float(f1_score(y, pred, zero_division=0)),
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


FILE_GAP_SEC = 86400.0   # разнос файлов на общей шкале: события разных записей не пересекаются


def events_for(y, pred, files, wtime, min_consec, merge_gap):
    """События истины и гипотез по всем файлам на одной шкале времени.

    wtime — время от начала СВОЕГО файла, поэтому без разноса события разных записей
    легли бы на одни и те же координаты и сопоставлялись бы между файлами. Каждый файл
    сдвигается на сутки относительно предыдущего — больше любого допуска SzCORE.
    """
    ref, hyp = [], []
    for k, f in enumerate(np.unique(files)):
        sel = np.where(files == f)[0]
        o = sel[np.argsort(wtime[sel], kind="stable")]
        t = wtime[o] + k * FILE_GAP_SEC
        p = smooth(pred[o], min_consec, merge_gap) if min_consec > 1 else pred[o].astype(bool)
        ref += events_sec(y[o].astype(bool), t)
        hyp += events_sec(np.asarray(p).astype(bool), t)
    return ref, hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(V2, "tusz"))
    ap.add_argument("--tag", default="official_split_tusz")
    ap.add_argument("--eval-split", default="eval", help="eval | dev | dev,eval")
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    ap.add_argument("--neg-ratio", type=float, default=5.0)
    ap.add_argument("--norm-sample", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--ram-train", action="store_true")
    ap.add_argument("--min-consec", type=int, default=3)
    ap.add_argument("--merge-gap", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t_start = time.time()

    X = np.load(os.path.join(args.cache, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(args.cache, "meta.npz"), allow_pickle=True)
    y, subj, fid, wtime, split = M["y_sz"], M["subj"], M["fileid"], M["wtime"], M["split"]
    ev_splits = args.eval_split.split(",")
    tr_full = np.where(split == "train")[0]
    ev = np.where(np.isin(split, ev_splits))[0]
    ev = ev[np.lexsort((wtime[ev], fid[ev], subj[ev]))]
    assert not set(subj[tr_full].tolist()) & set(subj[ev].tolist()), "пациенты train и eval пересекаются"
    print(f"устройство: {device} | train: {len(tr_full)} окон, {int(y[tr_full].sum())} приступных, "
          f"{len(set(subj[tr_full]))} пациентов | eval({args.eval_split}): {len(ev)} окон, "
          f"{int(y[ev].sum())} приступных, {len(set(subj[ev]))} пациентов", flush=True)

    rs = np.random.default_rng(C.SEED)
    tr_norm = np.sort(rs.choice(tr_full, min(args.norm_sample, len(tr_full)), replace=False))
    mu, sd = P.fold_norm_stats(X, tr_norm)
    tr, sub_info = subsample_negatives(tr_full, y, args.neg_ratio, args.seed)
    if args.ram_train:
        Xtr, tr_loc, y_loc = load_rows(X, tr), np.arange(len(tr)), y[tr]
    else:
        Xtr, tr_loc, y_loc = X, tr, y
    t0 = time.time()
    det = train_detector_mm(Xtr, tr_loc, y_loc, mu, sd, device, args.epochs, args.seed,
                            log=lambda s: print(s, flush=True))
    print(f"обучение: {time.time() - t0:.0f} с", flush=True)
    thr_tr, f1_tr = calibrate_threshold_on_train(det, Xtr, tr_loc, y_loc, mu, sd, device, args.seed)
    thr_nat, f1_nat = calibrate_threshold_natural(det, X, tr_full, y, mu, sd, device, args.seed)
    del Xtr
    prob = predict_mm(det, X, ev, mu, sd, device)
    yev = y[ev]
    files = np.array([f"{a}|{b}" for a, b in zip(subj[ev], fid[ev])])
    hours = len(ev) * C.WIN_SEC / 3600.0

    out = {"window_threshold_free": {"roc_auc": float(roc_auc_score(yev, prob)),
                                     "auprc": float(average_precision_score(yev, prob)),
                                     "prevalence_eval": float(yev.mean()),
                                     "prevalence_train_full": float(y[tr_full].mean()),
                                     "prevalence_train_subsample": float(y[tr].mean())},
           "operating_points": {}}
    for name, thr, f1c in (("calib_train_subsample", thr_tr, f1_tr), ("calib_natural", thr_nat, f1_nat)):
        pred = (prob >= thr).astype(int)
        op = {"window": window_metrics(yev, prob, thr), "f1_on_calibration_set": f1c}
        for sm_name, mc, mg in (("raw", 1, 0), ("smoothed", args.min_consec, args.merge_gap)):
            ref, hyp = events_for(yev, pred, files, wtime[ev], mc, mg)
            op[f"event_szcore_{sm_name}"] = szcore_events(ref, hyp, hours)
        out["operating_points"][name] = op
        e = op["event_szcore_smoothed"]
        print(f"{name:22s} τ={thr:.3f} F1окно={op['window']['f1']:.4f} | SzCORE сглаж.: "
              f"чувств.={e['sensitivity']:.3f} точн.={e['precision']:.3f} F1соб={e['f1']:.4f} "
              f"ложных/сут={e['false_alarms_per_24h']}", flush=True)

    # описательная кривая по сетке порогов (как ROC): рабочие точки НЕ выбираются по eval
    curve = []
    for q in np.concatenate([np.linspace(0.50, 0.99, 50), np.linspace(0.99, 0.9999, 40)]):
        t = float(np.quantile(prob, q)); pred = (prob >= t).astype(int)
        ref, hyp = events_for(yev, pred, files, wtime[ev], args.min_consec, args.merge_gap)
        e = szcore_events(ref, hyp, hours)
        curve.append({"threshold": t, "sensitivity": e["sensitivity"], "false_alarms_per_24h": e["false_alarms_per_24h"],
                      "f1_event": e["f1"], "f1_window": float(f1_score(yev, pred, zero_division=0))})
    # чувствительность при заданном числе ложных тревог в сутки — интерполяцией по кривой
    at_fa = {}
    for fa in (1, 5, 10, 20):
        ok = [c for c in curve if c["false_alarms_per_24h"] is not None and c["false_alarms_per_24h"] <= fa]
        at_fa[str(fa)] = max((c["sensitivity"] for c in ok), default=None)
    out["curve_smoothed"] = curve
    out["sensitivity_at_false_alarms_per_24h"] = at_fa
    print("чувствительность при ложных/сут ≤ 1/5/10/20:", at_fa, flush=True)

    out["passport"] = {"script": "stend/runs_v2/run_official_split_tusz.py", "cache": args.cache,
                       "x_dtype": str(X.dtype), "train_split": "train", "eval_split": args.eval_split,
                       "n_train_windows_full": int(len(tr_full)), "train_subsample": sub_info,
                       "n_eval_windows": int(len(ev)), "eval_hours": round(hours, 2),
                       "n_ref_events_eval": out["operating_points"]["calib_natural"]["event_szcore_smoothed"]["n_ref_events"],
                       "epochs": args.epochs, "neg_ratio": args.neg_ratio, "norm_sample": args.norm_sample,
                       "seed": args.seed, "ram_train": args.ram_train, "detector": "v1",
                       "min_consec_windows": args.min_consec, "merge_gap_windows": args.merge_gap,
                       "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
                       "numpy": np.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "elapsed_sec": round(time.time() - t_start, 1)}
    np.savez(os.path.join(OUTDIR, f"{args.tag}_predictions.npz"), idx=ev, y=yev, prob=prob.astype(np.float32))
    p = os.path.join(OUTDIR, f"{args.tag}.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print("сохранено:", p)


if __name__ == "__main__":
    main()
