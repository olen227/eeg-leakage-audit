"""Б1. Ablation по компонентам разделителя — поочерёдное отключение ОДНОГО компонента.

В июльской работе вместо ablation пересказан тот же каскад P0→P1→P2. Здесь
отключается ровно один компонент честного протокола P2, всё остальное неизменно,
и оценка ведётся на ТОМ ЖЕ множестве eval — тогда разность есть вклад компонента.

Компоненты честного протокола:
  1. субъектное (идентичностное) разбиение   -> отключение = P1 (пофайловое, тот же пациент)
  2. временное эмбарго                        -> отключение = P2 без эмбарго
  3. изоляция предобработки                   -> отключение = нормировка по ВСЕМ данным (вкл. eval)
ЧЕТВЁРТЫЙ КОМПОНЕНТ — УДАЛЕНИЕ ДУБЛЕЙ — ЗДЕСЬ НЕ ИЗМЕРЯЕТСЯ. Он требовал бы оставить
партнёра по дублю в обучении, но оценочное множество фолда ID_chb01+chb21 составлено
окнами ОБЕИХ записей пары (6 660 окон chb01 и 4 514 окон chb21). Какую запись ни возьми
партнёром, её собственные окна присутствуют в оценке, и в обучение попадают соседние по
времени окна тех же файлов — то есть измеряется перекрытие почти-дублей окон, а не
скрытая повторная регистрация. Прежняя редакция скрипта этого не учитывала и давала
величину +0,1683, смешивавшую два источника завышения. Корректное измерение требует
сузить оценочное множество до одной записи пары и вынесено в отдельный скрипт
stend/runs_v2/run_hidden_duplicate.py; на общем оценочном множестве, которого требует
настоящий ablation, оно невыполнимо.

Выход: outputs/track_b/ablation_components.json
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
    calibrate_threshold_on_train, V2, OUTDIR)


def global_norm_stats(X, chunk=8192):
    """Нормировка по ВСЕМ окнам, включая оценочные — намеренное нарушение изоляции."""
    n_ch = X.shape[1]
    s1 = np.zeros(n_ch); s2 = np.zeros(n_ch); cnt = 0
    for k in range(0, X.shape[0], chunk):
        b = X[k:k + chunk]
        s1 += b.sum(axis=(0, 2)); s2 += (b.astype(np.float64) ** 2).sum(axis=(0, 2))
        cnt += b.shape[0] * b.shape[2]
    mu = s1 / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 1e-20)) + 1e-7
    return mu.astype(np.float32).reshape(1, -1, 1), sd.astype(np.float32).reshape(1, -1, 1)


def run_variant(X, y, tr, ev, mu, sd, device, epochs, seed):
    det = train_detector_mm(X, tr, y, mu, sd, device, epochs, seed)
    thr, _ = calibrate_threshold_on_train(det, X, tr, y, mu, sd, device, seed)
    prob = predict_mm(det, X, ev, mu, sd, device)
    m = metrics_of(y[ev], prob, thr=thr)
    m["train_n"] = int(len(tr))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t0 = time.time()

    X = np.load(os.path.join(V2, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(V2, "meta.npz"), allow_pickle=True)
    y = M["y_sz"]; subj = M["subj"]; fid = M["fileid"]; t_abs = M["t_abs"]
    subjects = sorted(set(subj.tolist()))
    ident_of = P.build_identity_map(subjects)
    print(f"устройство: {device}", flush=True)

    gmu, gsd = global_norm_stats(X)
    out_folds = {}

    for I in sorted(set(ident_of.values())):
        fold = P.make_fold(ident_of, subj, fid, t_abs, I, seed=C.SEED,
                           embargo_sec=C.EMBARGO_SEC, y=y)
        ev = fold["eval"]
        if y[ev].sum() == 0:
            continue
        print(f"\n=== фолд {I}: eval={len(ev)} приступных={int(y[ev].sum())} ===", flush=True)
        P2 = fold["train"]["P2"]
        P1 = fold["train"]["P1"]

        # эмбарго внутри P2 (отключение компонента №2 относительно P2+эмбарго)
        P2e = P.embargo_mask_abs(t_abs, subj, P2, ev, C.EMBARGO_SEC)

        variants = {}
        mu, sd = P.fold_norm_stats(X, P2)
        variants["полный_P2"] = run_variant(X, y, P2, ev, mu, sd, device, args.epochs, C.SEED)

        # 1. без субъектного разбиения -> пофайловое (тот же пациент в обучении)
        mu1, sd1 = P.fold_norm_stats(X, P1)
        variants["без_субъектного_разбиения"] = run_variant(X, y, P1, ev, mu1, sd1, device, args.epochs, C.SEED)

        # 2. эмбарго: в P2 пациенты не пересекаются, эффект ожидается нулевым — фиксируем фактически
        if len(P2e) != len(P2):
            mu2, sd2 = P.fold_norm_stats(X, P2e)
            variants["P2_с_эмбарго"] = run_variant(X, y, P2e, ev, mu2, sd2, device, args.epochs, C.SEED)
        else:
            variants["P2_с_эмбарго"] = {"identical_to_P2": True,
                                        "removed_windows": 0,
                                        "note": ("эмбарго не удалило ни одного окна: в P2 обучающие "
                                                 "и оценочные пациенты не пересекаются, поэтому "
                                                 "внутрисубъектного эмбарго не существует по построению")}

        # 3. без изоляции предобработки -> нормировка по всем данным, включая eval
        variants["без_изоляции_предобработки"] = run_variant(X, y, P2, ev, gmu, gsd, device, args.epochs, C.SEED)

        # 4. вклад скрытого дубля на общем оценочном множестве неизмерим — см. заголовок
        #    модуля; корректное измерение в stend/runs_v2/run_hidden_duplicate.py
        if I == "ID_chb01+chb21":
            variants["без_удаления_дублей"] = {
                "not_measurable_here": True,
                "eval_subjects": sorted(set(subj[ev].tolist())),
                "note": ("оценочное множество содержит окна ОБЕИХ записей пары, поэтому "
                         "любой партнёр по дублю представлен в оценке своими же файлами; "
                         "измерение вклада скрытого дубля выполняется отдельным скриптом "
                         "run_hidden_duplicate.py на суженном оценочном множестве")}

        for k, v in variants.items():
            if "f1" in v:
                print(f"  {k:34s} F1={v['f1']:.4f} AUPRC={v['auprc']:.4f} train={v['train_n']}", flush=True)
            else:
                print(f"  {k:34s} {v.get('note','')}", flush=True)
        out_folds[I] = {"eval_n": int(len(ev)), "eval_pos": int(y[ev].sum()), "variants": variants}

    # вклад компонента = метрика при отключении − метрика полного P2
    contrib = {}
    for met in ["f1", "auprc"]:
        c = {}
        for comp in ["без_субъектного_разбиения", "без_изоляции_предобработки",
                     "без_удаления_дублей", "P2_с_эмбарго"]:
            vals = []
            for I, f in out_folds.items():
                v = f["variants"].get(comp)
                base = f["variants"]["полный_P2"]
                if v and met in v and met in base:
                    vals.append(v[met] - base[met])
            if vals:
                c[comp] = {"mean": float(np.mean(vals)), "per_fold": vals, "n_folds": len(vals)}
        contrib[met] = c

    out = {"passport": {"seed": C.SEED, "device": device, "python": platform.python_version(),
                        "torch": torch.__version__, "epochs": args.epochs,
                        "script": "stend/runs_v2/run_ablation_components.py",
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "elapsed_sec": round(time.time() - t0, 1)},
           "scheme": "отключается ровно один компонент, оценка на том же eval, что и полный P2",
           "per_fold": out_folds,
           "component_contribution": contrib}
    path = os.path.join(OUTDIR, "ablation_components.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print("\nсохранено:", path)
    print(json.dumps(contrib["f1"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
