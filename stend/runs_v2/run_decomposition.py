"""Аддитивное разложение завышения метрик — корректное измерение (v2).

Каждый протокол оценивается на ОДНОМ И ТОМ ЖЕ множестве eval внутри фолда,
поэтому разность метрик — контролируемое измерение вклада утечки.
Схема протоколов и разложения описана в protocols.py.

Оценочная схема: LOIO-CV — leave-one-identity-out, фактическим перебором фолдов.
Ни одно число не вписывается константой.

Выход: outputs/track_b/<tag>.json, имя задаётся ключом --tag.
Итоговые числа работы получены с --tag decomposition_24 на кэше полного набора.
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

from stend.vkr_eeg import config as C
from stend.vkr_eeg.detector import Detector
from stend.runs_v2 import protocols as P
from stend.runs_v2.detector_v2 import train_detector_v2, DetectorV2, n_params

V2 = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
OUTDIR = os.path.join("outputs", "track_b")


def get_device():
    want = os.environ.get("VKR_DEVICE", "auto")
    if want == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return want


def batched_normalized(X, idx, mu, sd, order, bs):
    """Итератор батчей из memmap с пофолдовой нормировкой.
    Индексы внутри батча сортируются — последовательное чтение с диска."""
    for k in range(0, len(order), bs):
        b = np.sort(idx[order[k:k + bs]])
        xb = (X[b] - mu) / sd
        yield b, torch.from_numpy(xb.astype(np.float32))


def subsample_negatives(train_idx, y, neg_ratio, seed):
    """Сбалансированная подвыборка обучающей части: ВСЕ приступные окна + neg_ratio
    случайных межприступных на каждое приступное.

    ЗАЧЕМ. При распространённости 0,34 % полная обучающая часть на 24 субъектах —
    около 800 тыс. окон (58 ГБ), она не помещается в оперативную память, и каждая
    эпоха читает кэш с диска вразнобой: прогон становится дисково-связанным
    (загрузка процессора 23 %, час на протокол вместо 19 минут).

    ЧТО ЭТО НЕ ЛОМАЕТ. Подвыборка применяется ОДИНАКОВО ко всем четырём протоколам
    внутри фолда, поэтому разности метрик — по-прежнему контролируемое измерение
    вклада утечки. Число приступных окон у протоколов различается (у P0 их больше,
    поскольку доступна часть отложенной идентичности) — это различие сохраняется,
    подвыборке подвергаются только межприступные окна.

    ЧТО ЛОМАЕТ. Абсолютные значения F1 не сопоставимы напрямую с прогоном на
    6 субъектах, где обучение шло на полной выборке. Сопоставимы вклады (разности).
    """
    if not neg_ratio:
        return train_idx, {"applied": False}
    ytr = y[train_idx]
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    k = min(len(neg), int(len(pos) * neg_ratio))
    rng = np.random.default_rng(seed)
    sel = np.sort(np.concatenate([pos, rng.choice(neg, k, replace=False)]))
    return train_idx[sel], {"applied": True, "neg_ratio": neg_ratio,
                            "n_before": int(len(train_idx)), "n_after": int(len(sel)),
                            "n_pos": int(len(pos)), "n_neg": int(k)}


def train_detector_mm(X, train_idx, y, mu, sd, device, epochs, seed, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    det = Detector(X.shape[1]).to(device)
    opt = torch.optim.Adam(det.parameters(), lr=C.DET_LR)
    ytr = y[train_idx]
    pos = max(int(ytr.sum()), 1); neg = max(len(ytr) - pos, 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    rng = np.random.default_rng(seed)
    ymap = {int(i): float(v) for i, v in zip(train_idx, ytr)}
    for ep in range(epochs):
        det.train()
        order = rng.permutation(len(train_idx))
        tot = 0.0; nb = 0
        for b, xb in batched_normalized(X, train_idx, mu, sd, order, C.DET_BATCH):
            yb = torch.tensor([ymap[int(i)] for i in b], dtype=torch.float32, device=device)
            logit = det(xb.to(device))
            loss = lossf(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if log:
            log(f"      эпоха {ep+1}/{epochs} loss={tot/max(nb,1):.4f}")
    return det


@torch.no_grad()
def predict_mm(det, X, idx, mu, sd, device, bs=512):
    det.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for k in range(0, len(idx), bs):
        b = idx[k:k + bs]
        xb = torch.from_numpy(((X[b] - mu) / sd).astype(np.float32))
        out[k:k + len(b)] = torch.sigmoid(det(xb.to(device))).cpu().numpy()
    return out


def calibrate_threshold_on_train(det, X, train_idx, y, mu, sd, device, seed,
                                 max_n=30000):
    """Порог решения подбирается ТОЛЬКО по обучающей части (утечки нет).

    При распространённости приступных окон ~0,4 % и обучении с pos_weight выход
    детектора смещён, поэтому фиксированный порог 0,5 — произвольная рабочая точка
    и как метрика непригоден. Порог выбирается максимизацией F1 на train.
    """
    rng = np.random.default_rng(seed)
    ytr = y[train_idx]
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    n_neg = min(len(neg), max(max_n - len(pos), 1))
    sel = np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
    idx = train_idx[sel]
    prob = predict_mm(det, X, idx, mu, sd, device)
    yy = ytr[sel]
    best_t, best_f1 = 0.5, -1.0
    for t in np.quantile(prob, np.linspace(0.50, 0.9995, 200)):
        f = f1_score(yy, (prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, float(best_f1)


def metrics_of(y_true, prob, thr=0.5):
    yhat = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    # Ложные тревоги в сутки — клинически интерпретируемая величина, принятая
    # в стандарте оценки детекторов приступов SzCORE (Dan et al., Epilepsia, 2025).
    # F1 = 0,07 мало о чём говорит врачу, а «1500 ложных тревог в сутки» — говорит.
    hours = len(y_true) * C.WIN_SEC / 3600.0
    m = {
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "threshold": float(thr),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "n": int(len(y_true)), "n_pos": int(y_true.sum()),
        "eval_hours": round(hours, 2),
        "false_alarms_per_24h": round(fp / hours * 24, 1) if hours > 0 else None,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
    }
    if 0 < y_true.sum() < len(y_true):
        m["roc_auc"] = float(roc_auc_score(y_true, prob))
        m["auprc"] = float(average_precision_score(y_true, prob))
    else:
        m["roc_auc"] = None; m["auprc"] = None
    return m


def slice_provenance(cache_dir):
    """Описание среза и его хеш по фактическому каталогу кэша.

    Хеши срезов вычислены при сборке наборов и записаны в PROVENANCE*.json; здесь они
    только сопоставляются каталогу, а не пересчитываются — чтение всех EDF заняло бы
    минуты на каждый прогон.
    """
    name = os.path.basename(os.path.normpath(cache_dir))
    known = {"full24": ("PROVENANCE_24.json", "все файлы 24 субъектов CHB-MIT, 676 файлов"),
             "siena": (None, "набор Siena, 14 субъектов"),
             "helsinki": (None, "набор Helsinki, 79 новорождённых")}
    prov, note = known.get(name, (None, f"кэш {name}"))
    h = None
    if prov:
        p = os.path.join(OUTDIR, prov)
        if os.path.exists(p):
            h = json.load(open(p)).get("data_sha256")
    return note, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    ap.add_argument("--folds", type=str, default="", help="ограничить список идентичностей (через запятую)")
    ap.add_argument("--tag", type=str, default="decomposition_v2")
    ap.add_argument("--cache", type=str, default="", help="каталог кэша (по умолчанию VKR_CACHE_ROOT)")
    ap.add_argument("--max-folds", type=int, default=0, help="ограничить число фолдов (0 = все)")
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="межприступных окон на одно приступное в обучении (0 = все окна)")
    ap.add_argument("--norm-sample", type=int, default=0,
                    help="окон для оценки статистик нормировки (0 = все окна обучающей части)")
    ap.add_argument("--batch", type=int, default=0,
                    help="размер батча (0 = из config). Больший батч лучше загружает ускоритель")
    ap.add_argument("--detector", choices=["v1", "v2"], default="v1",
                    help="v1 — детектор из июльского кода (базовый уровень); "
                         "v2 — усиленный, см. stend/runs_v2/detector_v2.py")
    ap.add_argument("--widths", type=str, default="",
                    help="ширина четырёх блоков усиленного детектора через запятую, "
                         "например 8,16,32,64; задаёт ёмкость модели при неизменной архитектуре")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить прогон: уже посчитанные фолды взять из "
                         "<tag>_partial.json, объединённые предсказания — из <tag>_pooled.npz")
    ap.add_argument("--seed", type=int, default=C.SEED,
                    help="seed прогона; по умолчанию канонический. Меняется ТОЛЬКО для оценки "
                         "разброса результата от инициализации весов, см. FINDINGS 3.18")
    args = ap.parse_args()
    _widths = [int(x) for x in args.widths.split(",")] if args.widths else None
    _slice_note, _slice_hash = slice_provenance(args.cache or V2)

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device()
    t_start = time.time()

    cache_dir = args.cache or V2
    X = np.load(os.path.join(cache_dir, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(cache_dir, "meta.npz"), allow_pickle=True)
    y = M["y_sz"]; subj = M["subj"]; fid = M["fileid"]; t_abs = M["t_abs"]

    subjects = sorted(set(subj.tolist()))
    ident_of = P.build_identity_map(subjects)
    identities = sorted(set(ident_of.values()))
    if args.folds:
        identities = [i for i in identities if i in args.folds.split(",")]
    if args.max_folds and len(identities) > args.max_folds:
        # детерминированный отбор подмножества идентичностей (seed фиксирован)
        pick = np.random.default_rng(C.SEED).permutation(len(identities))[:args.max_folds]
        identities = [identities[i] for i in sorted(pick)]
        print(f"ограничение: {args.max_folds} фолдов из {len(set(ident_of.values()))}", flush=True)

    print(f"устройство: {device} | окон: {len(y)} | приступных: {int(y.sum())} "
          f"({y.mean()*100:.3f} %)", flush=True)
    print(f"идентичности ({len(identities)}): {identities}", flush=True)
    print(f"группировка: {ident_of}", flush=True)

    protocols = ["P0", "P1", "P1e", "P2"]
    per_fold = {}
    pooled = {}

    part_path = os.path.join(OUTDIR, f"{args.tag}_partial.json")
    pool_path = os.path.join(OUTDIR, f"{args.tag}_pooled.npz")
    if args.resume and os.path.exists(part_path):
        prev = json.load(open(part_path))
        per_fold = prev.get("per_fold", {})
        # объединённые предсказания восстанавливаются из спутникового файла: держать
        # их в JSON нельзя (сотни тысяч чисел), а пересчитать без обучения нельзя.
        if os.path.exists(pool_path):
            z = np.load(pool_path)
            for pr in protocols:
                if f"{pr}_y" in z:
                    pooled[pr] = {"y": [z[f"{pr}_y"]], "prob": [z[f"{pr}_prob"]],
                                  "pred": [z[f"{pr}_pred"]]}
            print(f"возобновление: {len(per_fold)} фолдов из {part_path}, "
                  f"объединённые предсказания восстановлены", flush=True)
        else:
            per_fold = {}
            print(f"ВНИМАНИЕ: {part_path} есть, а {pool_path} нет — объединённые метрики "
                  "восстановить нечем, прогон начинается заново", flush=True)

    for I in identities:
        if I in per_fold:
            print(f"\n=== фолд {I}: уже посчитан, пропуск ===", flush=True)
            continue
        print(f"\n=== фолд: отложена {I} ===", flush=True)
        fold = P.make_fold(ident_of, subj, fid, t_abs, I, seed=C.SEED,
                           embargo_sec=C.EMBARGO_SEC, y=y)
        ev = fold["eval"]
        yev = y[ev]
        print(f"  eval: окон={len(ev)}, приступных={int(yev.sum())} | "
              f"файлов отложено {fold['n_files_held']}/{fold['n_files_total_I']} | "
              f"эмбарго убрало {fold['embargo_removed']} окон", flush=True)
        if fold.get("degenerate_steps"):
            print(f"  ВНИМАНИЕ: ступени каскада совпали ({', '.join(fold['degenerate_steps'])}) — "
                  "соответствующие вклады равны нулю ПО ПОСТРОЕНИЮ и измерением не являются",
                  flush=True)
        if fold.get("n_eval_no_abs_time"):
            print(f"  замечание: у {fold['n_eval_no_abs_time']} оценочных окон нет "
                  "абсолютного времени — эмбарго к ним не применяется (контракт данных)",
                  flush=True)
        if yev.sum() == 0:
            print("  ПРОПУСК: в eval нет приступных окон", flush=True)
            per_fold[I] = {"skipped": "no positive windows in eval"}
            continue

        res = {"eval_n": int(len(ev)), "eval_pos": int(yev.sum()),
               "n_files_held": fold["n_files_held"],
               "n_files_total": fold["n_files_total_I"],
               "embargo_removed_windows": fold["embargo_removed"],
               "eval_windows_without_abs_time": fold.get("n_eval_no_abs_time", 0),
               "degenerate_steps": fold.get("degenerate_steps", []),
               "protocols": {}}
        for pr in protocols:
            tr_full = fold["train"][pr]
            t0 = time.time()
            # статистики нормировки — по подвыборке обучающей части (среднее и дисперсия
            # сходятся быстро; чтение всех 800 тыс. окон заняло бы минуты на каждый протокол)
            if args.norm_sample and len(tr_full) > args.norm_sample:
                rs = np.random.default_rng(C.SEED)
                tr_norm = np.sort(rs.choice(tr_full, args.norm_sample, replace=False))
            else:
                tr_norm = tr_full
            mu, sd = P.fold_norm_stats(X, tr_norm)
            tr, sub_info = subsample_negatives(tr_full, y, args.neg_ratio, args.seed)
            det = (train_detector_v2(X, tr, y, mu, sd, device, args.epochs, args.seed,
                                     batch=args.batch or None,
                                     widths=_widths)
                   if args.detector == "v2"
                   else train_detector_mm(X, tr, y, mu, sd, device, args.epochs, args.seed))
            thr, f1_tr = calibrate_threshold_on_train(det, X, tr, y, mu, sd, device, args.seed)
            prob = predict_mm(det, X, ev, mu, sd, device)
            m = metrics_of(yev, prob, thr=thr)
            m["f1_at_0.5"] = float(f1_score(yev, (prob >= 0.5).astype(int), zero_division=0))
            m["f1_train_at_threshold"] = f1_tr
            m["train_n"] = int(len(tr)); m["train_pos"] = int(y[tr].sum())
            m["train_n_full"] = int(len(tr_full)); m["subsample"] = sub_info
            m["seconds"] = round(time.time() - t0, 1)
            res["protocols"][pr] = m
            pooled.setdefault(pr, {"y": [], "prob": [], "pred": []})
            pooled[pr]["y"].append(yev)
            pooled[pr]["prob"].append(prob)
            pooled[pr]["pred"].append((prob >= thr).astype(int))
            print(f"  {pr:4s} train={len(tr):7d} F1={m['f1']:.4f} (τ={thr:.3f}) "
                  f"F1@0.5={m['f1_at_0.5']:.4f} AUPRC={m['auprc']:.4f} "
                  f"ROC-AUC={m['roc_auc']:.4f} [{m['seconds']:.0f}s]", flush=True)
        per_fold[I] = res
        # Промежуточное сохранение после каждого фолда: прогон на 24 субъектах идёт
        # десятки часов, и результат не должен теряться при обрыве.
        json.dump({"in_progress": True, "identity_map": ident_of,
                   "folds_done": list(per_fold), "folds_planned": identities,
                   "per_fold": per_fold,
                   "elapsed_sec": round(time.time() - t_start, 1)},
                  open(part_path, "w"), ensure_ascii=False, indent=2)
        np.savez(pool_path, **{f"{pr}_{k}": np.concatenate(v[k])
                               for pr, v in pooled.items() for k in ("y", "prob", "pred")})
        print(f"  [сохранено промежуточно: {len(per_fold)}/{len(identities)} фолдов]", flush=True)

    # --- агрегация и разложение ---
    ok = {k: v for k, v in per_fold.items() if "protocols" in v}

    def col(pr, met):
        return np.array([v["protocols"][pr][met] for v in ok.values()], dtype=float)

    summary = {}
    for met in ["f1", "auprc", "roc_auc"]:
        s = {}
        for pr in protocols:
            v = col(pr, met)
            s[pr] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                     "per_fold": {k: float(ok[k]["protocols"][pr][met]) for k in ok}}
        d_okna = col("P0", met) - col("P1", met)
        d_emb = col("P1", met) - col("P1e", met)
        d_subj = col("P1e", met) - col("P2", met)
        d_tot = col("P0", met) - col("P2", met)
        s["decomposition"] = {
            "delta_okna_pochti_dubli": {"mean": float(d_okna.mean()), "per_fold": d_okna.tolist()},
            "delta_embargo": {"mean": float(d_emb.mean()), "per_fold": d_emb.tolist()},
            "delta_subject": {"mean": float(d_subj.mean()), "per_fold": d_subj.tolist()},
            "delta_total": {"mean": float(d_tot.mean()), "per_fold": d_tot.tolist()},
            "additivity_exact": bool(np.allclose(d_okna + d_emb + d_subj, d_tot, atol=1e-12)),
        }
        summary[met] = s

    # --- объединённые по фолдам метрики ---
    # При 0,4 % положительных примеров пофолдовая F1 крайне неустойчива
    # (в отдельных фолдах десятки положительных окон). Объединение предсказаний
    # всех фолдов даёт одну оценку по всем отложенным окнам — основной агрегат.
    pooled_metrics = {}
    for pr in protocols:
        if pr not in pooled:
            continue
        yy = np.concatenate(pooled[pr]["y"])
        pp = np.concatenate(pooled[pr]["prob"])
        pd_ = np.concatenate(pooled[pr]["pred"])
        tn, fp, fn, tp = confusion_matrix(yy, pd_, labels=[0, 1]).ravel()
        pooled_metrics[pr] = {
            "f1": float(f1_score(yy, pd_, zero_division=0)),
            "roc_auc": float(roc_auc_score(yy, pp)),
            "auprc": float(average_precision_score(yy, pp)),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "n": int(len(yy)), "n_pos": int(yy.sum()),
        }
    if pooled_metrics:
        pm = pooled_metrics
        summary["pooled"] = {
            "metrics": pm,
            "decomposition_f1": {
                "delta_okna_pochti_dubli": pm["P0"]["f1"] - pm["P1"]["f1"],
                "delta_embargo": pm["P1"]["f1"] - pm["P1e"]["f1"],
                "delta_subject": pm["P1e"]["f1"] - pm["P2"]["f1"],
                "delta_total": pm["P0"]["f1"] - pm["P2"]["f1"],
            },
            "decomposition_auprc": {
                "delta_okna_pochti_dubli": pm["P0"]["auprc"] - pm["P1"]["auprc"],
                "delta_embargo": pm["P1"]["auprc"] - pm["P1e"]["auprc"],
                "delta_subject": pm["P1e"]["auprc"] - pm["P2"]["auprc"],
                "delta_total": pm["P0"]["auprc"] - pm["P2"]["auprc"],
            },
            "note": ("ROC-AUC/AUPRC на объединённых вероятностях: пороги и шкалы "
                     "детекторов разных фолдов различаются, поэтому объединение "
                     "вероятностей вносит смещение; пофолдовые значения см. в summary."),
        }
        np.savez(os.path.join(OUTDIR, f"{args.tag}_pooled_predictions.npz"),
                 **{f"{pr}_{k}": np.concatenate(pooled[pr][k])
                    for pr in pooled for k in ("y", "prob", "pred")})

    passport = {
        "seed": args.seed,
        "seed_canonical": C.SEED,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "script": "stend/runs_v2/run_decomposition.py",
        "input": os.path.join(V2, "X_raw.npy"),
        # Срез выводится из фактического кэша, а не задан константой: при прогоне на
        # другом наборе зашитая строка сделала бы паспорт недостоверным, и число
        # перестало бы прослеживаться до данных, на которых получено.
        "data_slice": _slice_note,
        "data_sha256": _slice_hash,
        "epochs": args.epochs,
        "detector": args.detector,
        # Ёмкость модели фиксируется в паспорте: по ней строится кривая зависимости
        # завышения от мощности, а точка кривой без числа параметров бесполезна.
        "detector_widths": _widths,
        "detector_n_params": (n_params(DetectorV2(18, **({"widths": tuple(_widths)}
                                                         if _widths else {})))
                              if args.detector == "v2" else None),
        "batch": args.batch or C.DET_BATCH,
        "neg_ratio": args.neg_ratio,
        "norm_sample": args.norm_sample,
        "cache_dir": cache_dir,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    out = {"passport": passport,
           "identity_map": ident_of,
           "protocol_definitions": {
               "P0": "прочие идентичности + все окна отложенной, кроме eval (утечка почти-дублей)",
               "P1": "прочие идентичности + окна отложенной из НЕотложенных файлов",
               "P1e": "P1 + временное эмбарго 60 с по абсолютной шкале",
               "P2": "только прочие идентичности (честный)",
               "eval": "одно и то же множество для всех протоколов внутри фолда",
           },
           "per_fold": per_fold,
           "summary": summary}
    path = os.path.join(OUTDIR, f"{args.tag}.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"\nсохранено: {path}", flush=True)
    print(json.dumps(summary["f1"]["decomposition"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
