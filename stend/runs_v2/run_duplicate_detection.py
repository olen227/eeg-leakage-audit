"""Обнаружение дублей идентичности по сырому сигналу — доля отловленных дублей (Б5).

ВОПРОС
------
Какую долю истинных совпадений идентичности метрический оператор находит
по сырому ЭЭГ-сигналу, БЕЗ опоры на метаданные (без имён субъектов)?

ЧЕСТНОСТЬ СХЕМЫ
---------------
1. Кросс-валидация по идентичностям (та же, что в run_operator_cv.py): в каждом
   фолде ДВЕ идентичности полностью исключены из обучения энкодера; все оценочные
   пары строятся только из их записей. Оценивается способность находить дубли
   среди НЕВИДАННЫХ пациентов, а не запоминание обучающих.
2. Единица сравнения — ПАРА ЗАПИСЕЙ (EDF-файлов), а не пара окон: для каждого
   файла считается центроид эмбеддингов его окон, расстояние между файлами —
   евклидово расстояние центроидов. Это соответствует реальной задаче: дубль —
   свойство записи, а не отдельного окна.
3. Истинная метка пары = 1, если файлы принадлежат одной идентичности
   (внутрисубъектные пары И пары chb01-chb21 — документированный дубль CHB-MIT).
4. Порог tau калибруется ТОЛЬКО на обучающих идентичностях данного фолда
   (metrics.calibrate_threshold, alpha=beta=1, минимум FAR+FRR). Отложенные
   идентичности в калибровке не участвуют.
5. Энкодер обучается на метках subj (как в run_operator_cv.py), то есть на том,
   что «сообщают метаданные»: chb01 и chb21 при обучении считаются РАЗНЫМИ
   субъектами. Оператор не получает подсказки о дубле ни в каком виде.

Отдельно измеряется эталонный дубль chb01-chb21 (те же величины на подмножестве
кросс-пар chb01xchb21) — в тех 4 фолдах, где идентичность ID_chb01+chb21 отложена.

Выход: outputs/track_b/duplicate_detection.json
"""
import os, sys, json, time, platform, itertools, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
from sklearn.metrics import roc_auc_score

from stend.vkr_eeg import config as C
from stend.vkr_eeg.metrics import calibrate_threshold
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_operator_cv import train_encoder_mm, embed_mm, get_device

V2 = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
OUTDIR = os.path.join("outputs", "track_b")



def recall_at_fpr(dist, same, target_fpr):
    """Полнота при фиксированной доле ложных тревог.

    ЗАЧЕМ. Порог, откалиброванный на обучающих идентичностях, на невиданные не
    переносится (раздел 3.10 FINDINGS: шкала расстояний сжимается), из-за чего
    наивная полнота вырождается: 0,9996 при доле ложных тревог 0,88 — правило
    «принимать все пары». В биометрии в таких случаях отчётной величиной служит
    полнота при ФИКСИРОВАННОЙ доле ложных тревог: рабочая точка задаётся явно
    и одинакова для всех фолдов, поэтому величины сопоставимы.
    """
    neg = np.sort(dist[same == 0])
    if len(neg) == 0:
        return None, None
    k = max(1, int(round(target_fpr * len(neg)))) - 1
    tau = float(neg[k])                      # столько отрицательных пар окажется ниже порога
    pos = dist[same == 1]
    if len(pos) == 0:
        return None, tau
    return float(np.mean(pos < tau)), tau


def pair_stats(dist, same, tau):
    """Счётчики и метрики бинарного решения «одна идентичность» при dist < tau."""
    pred = dist < tau
    same = same.astype(bool)
    tp = int(np.sum(pred & same)); fp = int(np.sum(pred & ~same))
    fn = int(np.sum(~pred & same)); tn = int(np.sum(~pred & ~same))
    rec = tp / (tp + fn) if (tp + fn) else None
    pre = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * pre * rec / (pre + rec)) if (pre and rec) else (0.0 if (tp + fn) else None)
    prev = float(same.sum()) / len(same) if len(same) else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": rec, "precision": pre, "f1": f1,
            "fpr": (fp / (fp + tn) if (fp + tn) else None),
            "n_pairs": int(len(dist)), "n_positive": int(same.sum()),
            "n_negative": int((~same).sum()),
            # тривиальный референс «принимать ВСЕ пары за одну идентичность»:
            # его recall=1, а precision равна доле положительных пар
            "prevalence_positive": prev,
            "precision_trivial_accept_all": prev,
            "precision_lift_over_trivial": (pre - prev) if (pre is not None and prev is not None) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.OP_EPOCHS)
    ap.add_argument("--per-file", type=int, default=128,
                    help="окон на файл для оценки центроида (минимум окон в файле = 150)")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t_start = time.time()

    x_path = os.path.join(V2, "X_raw.npy"); m_path = os.path.join(V2, "meta.npz")
    X = np.load(x_path, mmap_mode="r")
    M = np.load(m_path, allow_pickle=True)
    subj = M["subj"]; fileid = M["fileid"]

    subjects = sorted(set(subj.tolist()))
    ident_of = P.build_identity_map(subjects)
    identities = sorted(set(ident_of.values()))
    ident_arr = np.array([ident_of[s] for s in subj])

    # ---- фиксированная (один seed на весь прогон) выборка окон каждого файла ----
    rng = np.random.default_rng(C.SEED)
    fkey = np.array([f"{a}|{b}" for a, b in zip(subj, fileid)])
    files = sorted(set(fkey.tolist()))
    f_subj = np.array([k.split("|")[0] for k in files])
    f_ident = np.array([ident_of[s] for s in f_subj])
    samp_idx, samp_file = [], []
    for j, k in enumerate(files):
        w = np.where(fkey == k)[0]
        n = min(args.per_file, len(w))
        samp_idx.append(rng.choice(w, n, replace=False))
        samp_file.append(np.full(n, j))
    samp_idx = np.concatenate(samp_idx); samp_file = np.concatenate(samp_file)
    order = np.argsort(samp_idx)
    samp_idx = samp_idx[order]; samp_file = samp_file[order]
    n_files = len(files)
    print(f"устройство: {device} | идентичностей: {len(identities)} | файлов: {n_files} | "
          f"окон на оценку: {len(samp_idx)}", flush=True)

    # все неупорядоченные пары файлов
    pi, pj = np.triu_indices(n_files, k=1)
    pair_same = (f_ident[pi] == f_ident[pj]).astype(int)

    folds = []
    for (ia, ib) in itertools.combinations(identities, 2):
        held = {ia, ib}
        tr_win = np.where(~np.isin(ident_arr, list(held)))[0]
        print(f"\n=== фолд: отложены {ia}, {ib} ===", flush=True)

        mu, sd = P.fold_norm_stats(X, tr_win)
        enc, hist = train_encoder_mm(X, tr_win, subj, mu, sd, device, args.epochs, C.SEED)

        emb = embed_mm(enc, X, samp_idx, mu, sd, device)
        cent = np.zeros((n_files, emb.shape[1]), dtype=np.float64)
        for j in range(n_files):
            cent[j] = emb[samp_file == j].mean(axis=0)
        dist = np.linalg.norm(cent[pi] - cent[pj], axis=1)

        held_f = np.isin(f_ident, list(held))
        m_eval = held_f[pi] & held_f[pj]          # обе записи из отложенных идентичностей
        m_cal = (~held_f[pi]) & (~held_f[pj])     # обе записи из обучающих идентичностей

        # ---- калибровка порога ТОЛЬКО на обучающих идентичностях ----
        intra = dist[m_cal & (pair_same == 1)]
        inter = dist[m_cal & (pair_same == 0)]
        tau, cost = calibrate_threshold(intra, inter, C.FAR_FRR_ALPHA, C.FAR_FRR_BETA)

        d_ev = dist[m_eval]; s_ev = pair_same[m_eval]
        st = pair_stats(d_ev, s_ev, tau)
        at_fpr = {}
        for tf in (0.01, 0.05, 0.10):
            r, t_ = recall_at_fpr(d_ev, s_ev, tf)
            at_fpr[f"recall_at_fpr_{tf:.2f}"] = r
            at_fpr[f"tau_at_fpr_{tf:.2f}"] = t_
        auc = float(roc_auc_score(s_ev, -d_ev)) if 0 < s_ev.sum() < len(s_ev) else None

        # СПРАВОЧНАЯ (НЕ честная) верхняя граница: порог, подобранный на самих
        # отложенных парах. Показывает, сколько теряется именно из-за непереноса
        # порога, а не из-за неразделимости эмбеддингов.
        tau_or, cost_or = calibrate_threshold(d_ev[s_ev == 1], d_ev[s_ev == 0],
                                              C.FAR_FRR_ALPHA, C.FAR_FRR_BETA)
        st_or = pair_stats(d_ev, s_ev, tau_or)

        r = {"held": [ia, ib],
             "n_files_held": int(held_f.sum()),
             "tau": tau, "calibration": {
                 "n_intra_pairs_train": int(len(intra)), "n_inter_pairs_train": int(len(inter)),
                 "cost_far_plus_frr": cost,
                 "intra_dist_mean": float(intra.mean()) if len(intra) else None,
                 "inter_dist_mean": float(inter.mean()) if len(inter) else None},
             "recall_at_fixed_fpr": at_fpr,
             "eval": {**st, "roc_auc": auc,
                      "dist_positive_mean": float(d_ev[s_ev == 1].mean()) if (s_ev == 1).any() else None,
                      "dist_negative_mean": float(d_ev[s_ev == 0].mean()) if (s_ev == 0).any() else None},
             "eval_oracle_threshold": {
                 "tau_oracle": tau_or, "cost_far_plus_frr": cost_or, **st_or,
                 "note": ("порог подобран НА ОТЛОЖЕННЫХ парах — это НЕ честная оценка, "
                          "а верхняя граница; служит только для разделения двух причин "
                          "ошибки: неразделимость эмбеддингов vs непереносимость порога")},
             "loss_first": hist[0], "loss_last": hist[-1]}

        # ---- эталонный дубль chb01-chb21 (только когда его идентичность отложена) ----
        if ia == "ID_chb01+chb21" or ib == "ID_chb01+chb21":
            s_i, s_j = f_subj[pi], f_subj[pj]
            m_dup = m_eval & (((s_i == "chb01") & (s_j == "chb21")) |
                              ((s_i == "chb21") & (s_j == "chb01")))
            m_neg = m_eval & (pair_same == 0)
            m_within = m_eval & (pair_same == 1) & (s_i == s_j)
            d_dup, d_neg = dist[m_dup], dist[m_neg]
            tp_d = int(np.sum(d_dup < tau)); fn_d = int(np.sum(d_dup >= tau))
            fp_d = int(np.sum(d_neg < tau))
            rec_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) else None
            pre_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) else None
            auc_d = (float(roc_auc_score(np.r_[np.ones(len(d_dup)), np.zeros(len(d_neg))],
                                         -np.r_[d_dup, d_neg]))
                     if len(d_dup) and len(d_neg) else None)
            tp_do = int(np.sum(d_dup < tau_or)); fn_do = int(np.sum(d_dup >= tau_or))
            r["duplicate_chb01_chb21"] = {
                "n_pairs": int(m_dup.sum()), "TP": tp_d, "FN": fn_d,
                "FP_different_identity": fp_d,
                "recall": rec_d, "precision": pre_d, "roc_auc_vs_different_identity": auc_d,
                "recall_at_oracle_threshold": (tp_do / (tp_do + fn_do)) if (tp_do + fn_do) else None,
                "dist_chb01_chb21_mean": float(d_dup.mean()) if len(d_dup) else None,
                "dist_within_subject_mean": float(dist[m_within].mean()) if m_within.any() else None,
                "dist_different_identity_mean": float(d_neg.mean()) if len(d_neg) else None,
                "note": ("precision здесь = TP(дубль)/(TP(дубль)+FP(разные идентичности)) "
                         "в пределах фолда; roc_auc сравнивает пары chb01xchb21 с парами "
                         "РАЗНЫХ пациентов: 0,5 означает полную неотличимость")}
            print(f"  дубль chb01/chb21: пар={m_dup.sum()} recall={rec_d} "
                  f"d(01,21)={d_dup.mean():.4f} d(внутри)={dist[m_within].mean():.4f} "
                  f"d(разные)={d_neg.mean():.4f} AUC={auc_d}", flush=True)

        print(f"  tau={tau:.4f} | пар оценки={st['n_pairs']} (+{st['n_positive']}/-{st['n_negative']}) "
              f"recall={st['recall']} precision={st['precision']} ROC-AUC={auc}", flush=True)
        folds.append(r)

    # ---------------- агрегация ----------------
    def _agg_fpr(key):
        v = [f["recall_at_fixed_fpr"].get(key) for f in folds
             if f.get("recall_at_fixed_fpr", {}).get(key) is not None]
        return ({"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                 "n_folds": len(v)} if v else None)

    def _agg(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        if len(v) == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {"mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}

    TP = sum(f["eval"]["TP"] for f in folds); FP = sum(f["eval"]["FP"] for f in folds)
    FN = sum(f["eval"]["FN"] for f in folds); TN = sum(f["eval"]["TN"] for f in folds)
    rec_pool = TP / (TP + FN) if (TP + FN) else None
    pre_pool = TP / (TP + FP) if (TP + FP) else None
    f1_pool = (2 * pre_pool * rec_pool / (pre_pool + rec_pool)) if (pre_pool and rec_pool) else 0.0

    oTP = sum(f["eval_oracle_threshold"]["TP"] for f in folds)
    oFP = sum(f["eval_oracle_threshold"]["FP"] for f in folds)
    oFN = sum(f["eval_oracle_threshold"]["FN"] for f in folds)
    oTN = sum(f["eval_oracle_threshold"]["TN"] for f in folds)
    orec = oTP / (oTP + oFN) if (oTP + oFN) else None
    opre = oTP / (oTP + oFP) if (oTP + oFP) else None

    dupf = [f["duplicate_chb01_chb21"] for f in folds if "duplicate_chb01_chb21" in f]
    dTP = sum(d["TP"] for d in dupf); dFN = sum(d["FN"] for d in dupf)
    dFP = sum(d["FP_different_identity"] for d in dupf)
    drec = dTP / (dTP + dFN) if (dTP + dFN) else None
    dpre = dTP / (dTP + dFP) if (dTP + dFP) else None

    aggregate = {
        "pooled": {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
                   "recall": rec_pool, "precision": pre_pool, "f1": f1_pool,
                   "fpr": FP / (FP + TN) if (FP + TN) else None,
                   "n_pairs": TP + FP + FN + TN, "n_positive": TP + FN, "n_negative": FP + TN,
                   "prevalence_positive": (TP + FN) / (TP + FP + FN + TN),
                   "precision_trivial_accept_all": (TP + FN) / (TP + FP + FN + TN),
                   "precision_lift_over_trivial": pre_pool - (TP + FN) / (TP + FP + FN + TN)},
        "pooled_oracle_threshold": {
            "TP": oTP, "FP": oFP, "FN": oFN, "TN": oTN,
            "recall": orec, "precision": opre,
            "fpr": oFP / (oFP + oTN) if (oFP + oTN) else None,
            "note": "НЕ честная оценка (порог подобран на отложенных парах), только верхняя граница"},
        "recall_at_fixed_fpr": {k: _agg_fpr(k) for k in
            ("recall_at_fpr_0.01", "recall_at_fpr_0.05", "recall_at_fpr_0.10")},
        "per_fold": {"recall": _agg([f["eval"]["recall"] for f in folds]),
                     "precision": _agg([f["eval"]["precision"] for f in folds]),
                     "f1": _agg([f["eval"]["f1"] for f in folds]),
                     "roc_auc": _agg([f["eval"]["roc_auc"] for f in folds]),
                     "tau": _agg([f["tau"] for f in folds])},
        "duplicate_chb01_chb21": {
            "n_folds": len(dupf), "TP": dTP, "FN": dFN, "FP_different_identity": dFP,
            "recall_pooled": drec, "precision_pooled": dpre,
            "recall_per_fold": _agg([d["recall"] for d in dupf]),
            "precision_per_fold": _agg([d["precision"] for d in dupf]),
            "roc_auc_per_fold": _agg([d["roc_auc_vs_different_identity"] for d in dupf]),
            "recall_at_oracle_threshold_per_fold": _agg(
                [d["recall_at_oracle_threshold"] for d in dupf])},
    }

    # ---- диагностика вырожденности решения (обязательна для чтения recall) ----
    prev = aggregate["pooled"]["prevalence_positive"]
    diagnostics = {
        "threshold_transfer": {
            "train_intra_dist_mean": _agg([f["calibration"]["intra_dist_mean"] for f in folds])["mean"],
            "train_inter_dist_mean": _agg([f["calibration"]["inter_dist_mean"] for f in folds])["mean"],
            "eval_positive_dist_mean": _agg([f["eval"]["dist_positive_mean"] for f in folds])["mean"],
            "eval_negative_dist_mean": _agg([f["eval"]["dist_negative_mean"] for f in folds])["mean"],
            "tau_mean": aggregate["per_fold"]["tau"]["mean"],
            "share_eval_negative_pairs_below_tau": aggregate["pooled"]["fpr"],
            "verdict": (
                "порог, откалиброванный на ОБУЧАЮЩИХ идентичностях, на отложенные не "
                "переносится: на обучающих идентичностях расстояния разнесены "
                "(внутри ~0,06-0,39 против ~1,0 между), а на невиданных идентичностях "
                "эмбеддинги схлопываются и ВСЕ расстояния оказываются много меньше tau. "
                "Решение вырождается в «принять все пары за одну идентичность»"),
        },
        "is_decision_degenerate": bool(rec_pool is not None and rec_pool > 0.99
                                       and aggregate["pooled"]["fpr"] > 0.5),
        "warning": (
            "recall≈1 здесь НЕ означает, что оператор находит дубли: он принимает почти "
            "любую пару записей. Точность (%.4f) практически совпадает с точностью "
            "тривиального правила «принимать всё» (%.4f), прирост %+.4f. Число "
            "dolya_otlovlennyh_dublej обязано приводиться вместе с precision и FPR, "
            "иначе оно вводит в заблуждение." % (pre_pool, prev, pre_pool - prev)),
        "ranking_ability": (
            "при этом ранжирование не бессмысленно: ROC-AUC по парам записей "
            "%.4f±%.4f (разброс %.4f..%.4f) — разделимость есть, но она не выдерживает "
            "переноса ПОРОГА на невиданные идентичности" % (
                aggregate["per_fold"]["roc_auc"]["mean"], aggregate["per_fold"]["roc_auc"]["std"],
                aggregate["per_fold"]["roc_auc"]["min"], aggregate["per_fold"]["roc_auc"]["max"])),
    }

    out = {
        "passport": {
            "seed": C.SEED, "device": device, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "script": "stend/runs_v2/run_duplicate_detection.py",
            "epochs": args.epochs, "windows_per_file": args.per_file,
            "emb_dim": C.EMB_DIM, "contrastive_margin": C.CONTRASTIVE_MARGIN,
            "far_frr_alpha": C.FAR_FRR_ALPHA, "far_frr_beta": C.FAR_FRR_BETA,
            "data_inputs": [os.path.abspath(x_path), os.path.abspath(m_path)],
            "data_slice": "все 208 файлов 6 субъектов CHB-MIT (5 идентичностей)",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_sec": round(time.time() - t_start, 1)},
        "scheme": (
            "кросс-валидация по идентичностям (10 фолдов = все пары из 5 идентичностей): "
            "в каждом фолде 2 идентичности исключены из обучения энкодера целиком; "
            "единица сравнения — пара ЗАПИСЕЙ (центроид эмбеддингов окон файла); "
            "метка пары = 1, если одна идентичность (внутрисубъектные пары и chb01-chb21); "
            "порог tau = argmin(FAR+FRR) на парах записей ОБУЧАЮЩИХ идентичностей"),
        "target_metric": {
            "name": "dolya_otlovlennyh_dublej",
            "definition": "recall по парам записей отложенных идентичностей = TP/(TP+FN)",
            "value_pooled": rec_pool,
            "value_per_fold_mean": aggregate["per_fold"]["recall"]["mean"],
            "value_per_fold_std": aggregate["per_fold"]["recall"]["std"],
            "value_chb01_chb21_pooled": drec,
            "must_be_reported_with": {
                "precision": pre_pool, "fpr": aggregate["pooled"]["fpr"],
                "prevalence_positive": aggregate["pooled"]["prevalence_positive"],
                "roc_auc_per_fold_mean": aggregate["per_fold"]["roc_auc"]["mean"]},
            "caveat": diagnostics["warning"]},
        "identity_map": ident_of,
        "folds": folds,
        "aggregate": aggregate,
        "diagnostics": diagnostics,
    }
    path = os.path.join(OUTDIR, "duplicate_detection.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"\nсохранено: {path}")
    print(json.dumps(out["target_metric"], ensure_ascii=False, indent=2))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
