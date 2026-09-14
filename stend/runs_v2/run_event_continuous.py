"""Событийная оценка по SzCORE на НЕПРЕРЫВНЫХ отложенных записях.

ЗАЧЕМ ОТДЕЛЬНЫЙ ПРОГОН. В run_event_scoring.py события собирались по соседству окон в
оценочной подвыборке (случайная половина окон отложенных файлов). Соседство в подвыборке
не есть соседство по времени: между двумя соседними оценочными окнами могут лежать
невыбранные окна, а «длительность» подвыборки — не длительность записи. Поэтому те
показатели были перерасчётом, а не валидацией по SzCORE.

ЧТО ДЕЛАЕТСЯ ЗДЕСЬ. Для каждого фолда детектор обучается ровно как в run_decomposition.py
(тот же make_fold, тот же seed, та же подвыборка межприступных окон, та же калибровка
порога на обучающей части), после чего предсказания строятся на ВСЕХ окнах отложенных
файлов — непрерывных записях в порядке времени. Далее:
  - событийный счёт «как раньше» (3 подряд окна, склейка разрывов ≤ 2 окон) — для
    сопоставимости с run_event_scoring.py, но уже по времени;
  - событийный счёт по правилам SzCORE: допуск 30 с до и 60 с после размеченного приступа,
    гипотезы ближе 90 с объединяются, гипотезы длиннее 5 мин режутся на 5-минутные
    отрезки при подсчёте ложных тревог; ложные тревоги в сутки — по реальной длительности
    отложенных записей.
Контроль воспроизводимости: пооконная F1 на оценочной подвыборке пересчитывается заново
и сравнивается с decomposition_24.json — расхождение печатается.

Запуск (полный, P0 и P2, ~7–8 ч на M1 Pro):
    python stend/runs_v2/run_event_continuous.py --cache ~/eeg_data/_cache/full24 \
        --neg-ratio 20 --epochs 12 --protocols P0,P2 --tag event_continuous_24
Пробный: добавить --folds ID_chb08 --epochs 1 --tag _smoke_cont
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_decomposition import (get_device, subsample_negatives, train_detector_mm,
                                             predict_mm, calibrate_threshold_on_train,
                                             slice_provenance, V2)
from stend.runs_v2.detector_v2 import train_detector_v2
from stend.runs_v2.run_event_scoring import smooth, to_events

OUTDIR = os.path.join("outputs", "track_b")
PRE_TOL, POST_TOL, MERGE_SEC, SPLIT_SEC = 30.0, 60.0, 90.0, 300.0   # допуски SzCORE


def events_sec(mask, t):
    """События как интервалы времени [start, end) по подряд идущим окнам одного файла."""
    return [(float(t[a]), float(t[b - 1]) + C.WIN_SEC) for a, b in to_events(mask)]


def szcore_events(ref, hyp, total_hours):
    """Событийный счёт по правилам SzCORE для одного набора интервалов."""
    # объединить гипотезы, разделённые менее чем MERGE_SEC
    hyp = sorted(hyp); merged = []
    for a, b in hyp:
        if merged and a - merged[-1][1] < MERGE_SEC:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    # разрезать длинные гипотезы на 5-минутные отрезки
    split = []
    for a, b in merged:
        while b - a > SPLIT_SEC:
            split.append((a, a + SPLIT_SEC)); a += SPLIT_SEC
        split.append((a, b))
    ext = [(a - PRE_TOL, b + POST_TOL) for a, b in ref]
    detected = sum(1 for (a, b) in ext if any(pa < b and pb > a for pa, pb in split))
    fp = sum(1 for (pa, pb) in split if not any(pa < b and pb > a for a, b in ext))
    sens = detected / len(ref) if ref else None
    prec = (len(split) - fp) / len(split) if split else None
    f1 = (2 * sens * prec / (sens + prec)) if (sens and prec) else 0.0
    return {"n_ref_events": len(ref), "n_hyp_events": len(split), "detected": detected,
            "false_alarms": fp, "sensitivity": sens, "precision": prec, "f1": f1,
            "false_alarms_per_24h": round(fp / total_hours * 24, 2) if total_hours else None,
            "hours": round(total_hours, 2)}


def simple_events(ref, hyp, total_hours):
    """Счёт «как в run_event_scoring.py»: обнаружено при любом пересечении, без допусков."""
    detected = sum(1 for (a, b) in ref if any(pa < b and pb > a for pa, pb in hyp))
    matched = sum(1 for (pa, pb) in hyp if any(pa < b and pb > a for a, b in ref))
    fp = len(hyp) - matched
    sens = detected / len(ref) if ref else None
    prec = matched / len(hyp) if hyp else None
    f1 = (2 * sens * prec / (sens + prec)) if (sens and prec) else 0.0
    return {"n_ref_events": len(ref), "n_hyp_events": len(hyp), "detected": detected,
            "false_alarms": fp, "sensitivity": sens, "precision": prec, "f1": f1,
            "false_alarms_per_24h": round(fp / total_hours * 24, 2) if total_hours else None,
            "hours": round(total_hours, 2)}


def score_protocol(y, pred, files, wtime, min_consec, merge_gap):
    """Собрать события по файлам (по времени) и посчитать обе схемы."""
    ref_all, hyp_all, hyp_raw_all = [], [], []
    sm_all = np.zeros_like(pred)
    for f in np.unique(files):
        sel = np.where(files == f)[0]
        o = sel[np.argsort(wtime[sel], kind="stable")]
        t = wtime[o]
        sm = smooth(pred[o], min_consec, merge_gap)
        sm_all[o] = sm
        ref_all += events_sec(y[o].astype(bool), t)
        hyp_all += events_sec(sm.astype(bool), t)
        hyp_raw_all += events_sec(pred[o].astype(bool), t)
    hours = len(y) * C.WIN_SEC / 3600.0
    return {
        "event_simple_smoothed": simple_events(ref_all, hyp_all, hours),
        "event_szcore_smoothed": szcore_events(ref_all, hyp_all, hours),
        "event_szcore_raw": szcore_events(ref_all, hyp_raw_all, hours),
    }, sm_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    ap.add_argument("--tag", default="event_continuous_24")
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    ap.add_argument("--neg-ratio", type=float, default=20.0)
    ap.add_argument("--protocols", default="P0,P2")
    ap.add_argument("--folds", default="")
    ap.add_argument("--detector", choices=["v1", "v2"], default="v1")
    ap.add_argument("--widths", default="")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--min-consec", type=int, default=3)
    ap.add_argument("--merge-gap", type=int, default=2)
    ap.add_argument("--reference", default="decomposition_24",
                    help="прогон, с которым сверяется пооконная F1 на оценочной подвыборке")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(",")] if args.widths else None
    protocols = args.protocols.split(",")

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t_start = time.time()
    cache_dir = args.cache or V2
    slice_note, slice_hash = slice_provenance(cache_dir)
    X = np.load(os.path.join(cache_dir, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(cache_dir, "meta.npz"), allow_pickle=True)
    y, subj, fid, t_abs, wtime = M["y_sz"], M["subj"], M["fileid"], M["t_abs"], M["wtime"]
    key = np.array([f"{a}|{b}" for a, b in zip(subj, fid)])
    ident_of = P.build_identity_map(sorted(set(subj.tolist())))
    identities = sorted(set(ident_of.values()))
    if args.folds:
        identities = [i for i in identities if i in args.folds.split(",")]
    ref = None
    rp = os.path.join(OUTDIR, f"{args.reference}.json")
    if os.path.exists(rp):
        ref = json.load(open(rp))["per_fold"]

    part_path = os.path.join(OUTDIR, f"{args.tag}_partial.json")
    pool_path = os.path.join(OUTDIR, f"{args.tag}_continuous_predictions.npz")
    per_fold, pool = {}, {}
    if args.resume and os.path.exists(part_path) and os.path.exists(pool_path):
        per_fold = json.load(open(part_path))["per_fold"]
        z = np.load(pool_path, allow_pickle=True)
        pool = {k: [z[k]] for k in z.files}
        print(f"возобновление: {len(per_fold)} фолдов", flush=True)

    print(f"устройство: {device} | окон: {len(y)} | фолдов: {len(identities)} | протоколы: {protocols}", flush=True)
    for I in identities:
        if I in per_fold:
            continue
        print(f"\n=== фолд {I} ===", flush=True)
        fold = P.make_fold(ident_of, subj, fid, t_abs, I, seed=C.SEED, embargo_sec=C.EMBARGO_SEC, y=y)
        ev = fold["eval"]
        held = set(key[ev].tolist())
        all_idx = np.where(np.isin(key, list(held)))[0]
        # порядок времени: файл, затем время от начала файла
        all_idx = all_idx[np.lexsort((wtime[all_idx], fid[all_idx]))]
        if y[ev].sum() == 0:
            per_fold[I] = {"skipped": "no positive windows in eval"}; continue
        ev_pos = np.isin(all_idx, ev)
        res = {"n_all": int(len(all_idx)), "n_eval": int(len(ev)), "n_files_held": fold["n_files_held"],
               "hours_all": round(len(all_idx) * C.WIN_SEC / 3600, 3), "pos_all": int(y[all_idx].sum()),
               "protocols": {}}
        pool.setdefault("idx", []).append(all_idx)
        pool.setdefault("y", []).append(y[all_idx])
        pool.setdefault("is_eval", []).append(ev_pos.astype(np.int8))
        pool.setdefault("fold", []).append(np.array([I] * len(all_idx)))
        for pr in protocols:
            t0 = time.time()
            tr_full = fold["train"][pr]
            mu, sd = P.fold_norm_stats(X, tr_full)
            tr, sub_info = subsample_negatives(tr_full, y, args.neg_ratio, args.seed)
            det = (train_detector_v2(X, tr, y, mu, sd, device, args.epochs, args.seed, widths=widths)
                   if args.detector == "v2" else
                   train_detector_mm(X, tr, y, mu, sd, device, args.epochs, args.seed))
            thr, f1_tr = calibrate_threshold_on_train(det, X, tr, y, mu, sd, device, args.seed)
            prob = predict_mm(det, X, all_idx, mu, sd, device)
            pred = (prob >= thr).astype(int)
            f1_eval = float(f1_score(y[all_idx][ev_pos], pred[ev_pos], zero_division=0))
            f1_all = float(f1_score(y[all_idx], pred, zero_division=0))
            ref_f1 = (ref or {}).get(I, {}).get("protocols", {}).get(pr, {}).get("f1")
            res["protocols"][pr] = {"threshold": float(thr), "f1_eval_subset": f1_eval,
                                    "f1_eval_subset_reference": ref_f1, "f1_all_windows": f1_all,
                                    "seconds": round(time.time() - t0, 1)}
            pool.setdefault(f"{pr}_prob", []).append(prob.astype(np.float32))
            pool.setdefault(f"{pr}_pred", []).append(pred.astype(np.int8))
            print(f"  {pr:3s} F1(eval-подвыборка)={f1_eval:.4f} "
                  f"{'ref=%.4f' % ref_f1 if ref_f1 is not None else ''} F1(все окна)={f1_all:.4f} "
                  f"τ={thr:.3f} [{res['protocols'][pr]['seconds']:.0f}s]", flush=True)
        per_fold[I] = res
        json.dump({"in_progress": True, "per_fold": per_fold, "elapsed_sec": round(time.time() - t_start, 1)},
                  open(part_path, "w"), ensure_ascii=False, indent=2)
        np.savez(pool_path, **{k: np.concatenate(v) for k, v in pool.items()})
        print(f"  [сохранено: {len(per_fold)} фолдов]", flush=True)

    # --- событийный счёт на непрерывных записях, объединённо по всем фолдам ---
    z = {k: np.concatenate(v) for k, v in pool.items()}
    idx = z["idx"]; files = np.array([f"{a}|{b}" for a, b in zip(subj[idx], fid[idx])])
    out_p = {}
    for pr in protocols:
        sc, sm = score_protocol(z["y"], z[f"{pr}_pred"].astype(int), files, wtime[idx], args.min_consec, args.merge_gap)
        yy, pp = z["y"], z[f"{pr}_pred"].astype(int)
        sc["sample_all_windows"] = {"f1": float(f1_score(yy, pp, zero_division=0)),
                                    "f1_smoothed": float(f1_score(yy, sm, zero_division=0)),
                                    "auprc": float(average_precision_score(yy, z[f"{pr}_prob"])),
                                    "roc_auc": float(roc_auc_score(yy, z[f"{pr}_prob"]))}
        out_p[pr] = sc
    delta = {}
    if "P0" in out_p and "P2" in out_p:
        for scheme in ("event_szcore_smoothed", "event_simple_smoothed"):
            delta[scheme] = {k: out_p["P0"][scheme][k] - out_p["P2"][scheme][k]
                             for k in ("f1", "sensitivity") if out_p["P0"][scheme][k] is not None}
    out = {"passport": {"script": "stend/runs_v2/run_event_continuous.py", "cache": cache_dir,
                        "data_slice": slice_note, "data_sha256": slice_hash, "seed": args.seed,
                        "epochs": args.epochs, "neg_ratio": args.neg_ratio, "detector": args.detector,
                        "widths": widths, "protocols": protocols, "min_consec_windows": args.min_consec,
                        "merge_gap_windows": args.merge_gap,
                        "szcore": {"pre_tol_s": PRE_TOL, "post_tol_s": POST_TOL, "merge_s": MERGE_SEC, "split_s": SPLIT_SEC},
                        "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "elapsed_sec": round(time.time() - t_start, 1)},
           "note": ("предсказания на всех окнах отложенных файлов; события по времени внутри файла; "
                    "длительность — реальная длительность отложенных записей"),
           "n_windows_scored": int(len(idx)), "hours_scored": round(len(idx) * C.WIN_SEC / 3600, 2),
           "n_ref_events": out_p[protocols[0]]["event_szcore_smoothed"]["n_ref_events"],
           "protocols": out_p, "delta_P0_minus_P2": delta, "per_fold": per_fold}
    p = os.path.join(OUTDIR, f"{args.tag}.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print(f"\nокон {len(idx)}, часов {out['hours_scored']}, размеченных приступов {out['n_ref_events']}")
    for pr in protocols:
        s = out_p[pr]["event_szcore_smoothed"]
        print(f"{pr}: SzCORE чувствительность={s['sensitivity']:.3f} ложных/сут={s['false_alarms_per_24h']} F1соб={s['f1']:.4f} | F1 окна(все)={out_p[pr]['sample_all_windows']['f1']:.4f}")
    print("сохранено:", p)


if __name__ == "__main__":
    main()
