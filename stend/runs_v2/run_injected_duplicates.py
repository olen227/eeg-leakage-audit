"""Внедрённые дубли: доверительный интервал для вклада скрытого дубля.

ЗАЧЕМ. Документированная пара chb01≡chb21 в CHB-MIT ОДНА, поэтому измерение вклада
скрытого дубля на ней получается по единственному фолду, где приступных окон десятки.
Три допустимые схемы измерения дают на этой паре противоречащие знаки и по F1, и по
AUPRC (см. real_duplicate_not_measurable.json), то есть величина не измерима.
Доверительный интервал строится на внедрённых дублях.

ЧТО ДЕЛАЕМ. Записи каждой идентичности разрезаются по абсолютному времени пополам:
ранние файлы объявляются «пациентом A», поздние — «пациентом B». Это имитирует ровно ту
ситуацию, ради которой написана работа: один человек попал в набор под двумя номерами,
и об этом никто не знает.

Далее для каждой идентичности сравниваются два обучения на ОДНОМ И ТОМ ЖЕ eval:
  честно : идентичность исключена целиком          (A и B оба вне обучения)
  наивно : в обучении оставлена половина A          (скрытый дубль не удалён)
Разность метрик — вклад невыявленного дубля. Идентичностей 23, значит и измерений до 23,
и доверительный интервал строится обычным бутстрэпом.

ЧЕСТНАЯ ОГОВОРКА. Внедрённый дубль ЛЕГЧЕ настоящего: половины разделены часами, тогда как
chb01 и chb21 разделены полутора годами. Настоящий дубль поэтому должен давать вклад НЕ МЕНЬШЕ
измеренного, и полученный интервал следует читать как оценку снизу.

Выход: outputs/track_b/injected_duplicates.json
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_decomposition import (
    get_device, train_detector_mm, predict_mm, metrics_of,
    calibrate_threshold_on_train, subsample_negatives, OUTDIR)

V2 = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))


def split_identity_in_time(subj, fid, t_abs, y, mask_I, seed):
    """Разрезать файлы идентичности по абсолютному времени пополам.
    Возвращает (файлы ранней половины A, файлы поздней половины B)."""
    keys = sorted(set(zip(subj[mask_I].tolist(), fid[mask_I].tolist())))
    # время начала файла = минимальное t_abs среди его окон
    tstart = {}
    for (a, b) in keys:
        sel = (subj == a) & (fid == b)
        v = t_abs[sel]
        v = v[~np.isnan(v)]
        tstart[(a, b)] = float(v.min()) if len(v) else np.inf
    # у пары chb01/chb21 шкалы времени независимы: делим по СУБЪЕКТУ, а не по времени
    subs = sorted({a for a, _ in keys})
    if len(subs) > 1:
        A = [k for k in keys if k[0] == subs[0]]
        B = [k for k in keys if k[0] != subs[0]]
        mode = "по субъектам (идентичность уже состоит из разных записей)"
    else:
        order = sorted(keys, key=lambda k: tstart[k])
        half = len(order) // 2
        A, B = order[:half], order[half:]
        mode = "по абсолютному времени пополам"
    return A, B, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    ap.add_argument("--cache", type=str, default="")
    ap.add_argument("--neg-ratio", type=float, default=20.0)
    ap.add_argument("--norm-sample", type=int, default=120000)
    ap.add_argument("--tag", type=str, default="injected_duplicates")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t0 = time.time()
    cache = args.cache or V2
    X = np.load(os.path.join(cache, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(cache, "meta.npz"), allow_pickle=True)
    y = M["y_sz"]; subj = M["subj"]; fid = M["fileid"]; t_abs = M["t_abs"]

    subjects = sorted(set(subj.tolist()))
    ident_of = P.build_identity_map(subjects)
    ident = np.array([ident_of[s] for s in subj])
    identities = sorted(set(ident_of.values()))
    key = np.array([f"{a}|{b}" for a, b in zip(subj, fid)])
    print(f"устройство: {device} | идентичностей: {len(identities)}", flush=True)

    rng = np.random.default_rng(C.SEED)
    per_id = {}
    for I in identities:
        mI = ident == I
        A, B, mode = split_identity_in_time(subj, fid, t_abs, y, mI, C.SEED)
        if not A or not B:
            per_id[I] = {"skipped": "недостаточно файлов для разрезания"}
            print(f"  {I}: пропуск (файлов мало)", flush=True)
            continue

        # eval — стратифицированная часть файлов половины B (нужны приступные окна)
        has_sz = {}
        for (a, b) in B:
            s = (subj == a) & (fid == b)
            has_sz[(a, b)] = bool(y[s].sum() > 0)
        sz_f = [f for f in B if has_sz[f]]
        pl_f = [f for f in B if not has_sz[f]]
        E = set()
        for grp, frac in ((sz_f, 0.5), (pl_f, 0.3)):
            if grp:
                pm = rng.permutation(len(grp))
                E |= {grp[i] for i in pm[:max(1, int(round(len(grp) * frac)))]}
        Ek = {f"{a}|{b}" for a, b in E}
        idxE = np.where(mI & np.isin(key, list(Ek)))[0]
        pe = rng.permutation(len(idxE))
        ev = np.sort(idxE[pe[:int(round(len(idxE) * 0.5))]])
        if len(ev) == 0 or y[ev].sum() < 3:
            per_id[I] = {"skipped": f"в eval приступных окон {int(y[ev].sum()) if len(ev) else 0} (<3)"}
            print(f"  {I}: пропуск (мало приступных в eval)", flush=True)
            continue

        others = np.where(~mI)[0]
        Ak = {f"{a}|{b}" for a, b in A}
        idxA = np.where(mI & np.isin(key, list(Ak)))[0]

        res = {"mode": mode, "n_files_A": len(A), "n_files_B": len(B),
               "eval_n": int(len(ev)), "eval_pos": int(y[ev].sum()),
               "variants": {}}
        for name, tr_full in (("честно_дубль_удалён", others),
                              ("наивно_дубль_в_обучении", np.sort(np.concatenate([others, idxA])))):
            rs = np.random.default_rng(C.SEED)
            tn = (np.sort(rs.choice(tr_full, args.norm_sample, replace=False))
                  if args.norm_sample and len(tr_full) > args.norm_sample else tr_full)
            mu, sd = P.fold_norm_stats(X, tn)
            tr, _ = subsample_negatives(tr_full, y, args.neg_ratio, C.SEED)
            det = train_detector_mm(X, tr, y, mu, sd, device, args.epochs, C.SEED)
            thr, _ = calibrate_threshold_on_train(det, X, tr, y, mu, sd, device, C.SEED)
            prob = predict_mm(det, X, ev, mu, sd, device)
            m = metrics_of(y[ev], prob, thr=thr)
            m["train_n"] = int(len(tr))
            res["variants"][name] = m
        d_f1 = (res["variants"]["наивно_дубль_в_обучении"]["f1"]
                - res["variants"]["честно_дубль_удалён"]["f1"])
        d_ap = (res["variants"]["наивно_дубль_в_обучении"]["auprc"]
                - res["variants"]["честно_дубль_удалён"]["auprc"])
        res["delta_f1"] = d_f1; res["delta_auprc"] = d_ap
        per_id[I] = res
        print(f"  {I}: честно F1={res['variants']['честно_дубль_удалён']['f1']:.4f} "
              f"наивно F1={res['variants']['наивно_дубль_в_обучении']['f1']:.4f} "
              f"-> Δ={d_f1:+.4f} (eval+ {res['eval_pos']})", flush=True)
        json.dump({"in_progress": True, "per_identity": per_id},
                  open(os.path.join(OUTDIR, f"{args.tag}_partial.json"), "w"),
                  ensure_ascii=False, indent=2)

    ok = {k: v for k, v in per_id.items() if "delta_f1" in v}
    agg = {}
    if ok:
        for met in ("f1", "auprc"):
            v = np.array([ok[k][f"delta_{met}"] for k in ok], float)
            rs = np.random.default_rng(C.SEED)
            draws = [float(np.mean(v[rs.integers(0, len(v), len(v))])) for _ in range(C.BOOTSTRAP_N)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            agg[met] = {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
                        "n_identities": int(len(v)),
                        "positive_in": int((v > 0).sum()),
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "per_identity": {k: float(ok[k][f"delta_{met}"]) for k in ok}}

    out = {"passport": {"seed": C.SEED, "device": device, "python": platform.python_version(),
                        "torch": torch.__version__, "epochs": args.epochs,
                        "neg_ratio": args.neg_ratio, "cache": cache,
                        "script": "stend/runs_v2/run_injected_duplicates.py",
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "elapsed_sec": round(time.time() - t0, 1)},
           "scheme": ("записи идентичности разрезаны по времени на половины A и B; "
                      "eval из B фиксирован; сравниваются обучение без A (честно) "
                      "и с A (наивно, скрытый дубль не удалён)"),
           "caveat": ("внедрённый дубль легче настоящего: половины разделены часами, "
                      "а chb01 и chb21 — полутора годами. Величина является ВЕРХНЕЙ оценкой "
                      "вклада скрытого дубля."),
           "reference_real_duplicate": {
               "value_f1": None, "n_folds": 1,
               "source": "outputs/track_b/real_duplicate_not_measurable.json",
               "note": ("на единственной документированной паре chb01=chb21 величина не "
                        "приводится: три допустимые схемы измерения дают противоречащие "
                        "знаки и по F1, и по AUPRC при 17-27 приступных окнах в оценке")},
           "per_identity": per_id,
           "aggregate": agg}
    p = os.path.join(OUTDIR, f"{args.tag}.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print("\nсохранено:", p)
    if agg:
        a = agg["f1"]
        print(f"Δ_дубль по F1: {a['mean']:+.4f} [{a['lo']:+.4f}; {a['hi']:+.4f}], "
              f"идентичностей {a['n_identities']}, положителен в {a['positive_in']}")


if __name__ == "__main__":
    main()
