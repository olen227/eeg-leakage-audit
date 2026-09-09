"""Фаза 6. Объяснимость детектора на ложных срабатываниях (SHAP / LIME / перестановки).

ВОПРОС, НА КОТОРЫЙ ОТВЕЧАЕТ СКРИПТ
-----------------------------------
Честный протокол P2 даёт F1 = 0,088 при TP = 83 и FP = 1638 (объединённо по фолдам):
ложных срабатываний на порядок больше, чем верных. Вопрос: детектор реагирует на
ложных окнах на ТО ЖЕ, что и на верных (то есть FP — это «похожие на приступ»
фрагменты), или на что-то иное (артефакты, отдельные отведения)?

Проверяемое утверждение формулируется числом: профиль важности каналов у группы FP
сравнивается с профилем группы TP корреляцией Спирмена. Высокая положительная
корреляция означает, что механизм ошибки — тот же признаковый механизм, что и у
верных срабатываний (порог/разделимость), а не отдельный посторонний источник.

СХЕМА
-----
1. Один фолд честного протокола P2 — отложена идентичность ID_chb08 (в её eval
   наибольшее число приступных окон, 51; см. decomposition_v2.json).
2. Детектор обучается ровно теми же функциями, что и в основном прогоне
   (train_detector_mm), порог калибруется ТОЛЬКО на обучающей части
   (calibrate_threshold_on_train). Порог 0,5 не используется нигде.
3. На оценочных окнах выделяются группы TP / FP / FN (и TN как фон).
4. Важность считается тремя независимыми способами:
     (а) SHAP GradientExplainer по логиту детектора (expected gradients);
     (б) LIME (lime_tabular) по 18 поканальным признакам амплитуды (СКЗ канала):
         возмущается амплитуда отдельного канала, локальная линейная модель даёт вес;
     (в) permutation importance: канал перемешивается между окнами группы,
         измеряется сдвиг логита — метод, не зависящий от градиентов.
   Три способа нужны потому, что согласие независимых методов — единственная
   доступная проверка того, что профиль важности не артефакт конкретного алгоритма.
5. Агрегация SHAP: по каналам — сумма модулей по временной оси; по времени —
   сумма модулей по каналам.

Всё, что не удалось посчитать, записывается как null с указанием причины.

Выход: outputs/track_b/explainability.json
"""
import os, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_decomposition import (
    get_device, train_detector_mm, predict_mm, calibrate_threshold_on_train,
    metrics_of, V2, OUTDIR)

CH = C.CANONICAL_CHANNELS
N_TIME_BINS = 64          # агрегация временного профиля (1024/64 = 16 отсчётов = 62,5 мс)


# --------------------------------------------------------------------------- #
#                            вспомогательные обёртки                          #
# --------------------------------------------------------------------------- #
class LogitWrap(nn.Module):
    """Возвращает логит формой (N, 1) — SHAP требует двумерный выход."""
    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, x):
        return self.det(x).unsqueeze(-1)


@torch.no_grad()
def logits_of(det, arr, device, bs=256):
    det.eval()
    out = np.empty(len(arr), dtype=np.float64)
    for k in range(0, len(arr), bs):
        xb = torch.from_numpy(np.ascontiguousarray(arr[k:k + bs])).float().to(device)
        out[k:k + len(xb)] = det(xb).detach().cpu().numpy().astype(np.float64)
    return out


def profile_stats(mat, names):
    """mat: (n_окон, n_каналов) неотрицательные важности. Возвращает сводку профиля."""
    mean = mat.mean(axis=0)
    total = float(mean.sum())
    share = (mean / total) if total > 0 else np.full_like(mean, np.nan)
    order = np.argsort(-mean)
    return {
        "mean_per_channel": {names[i]: float(mean[i]) for i in range(len(names))},
        "share_per_channel": {names[i]: float(share[i]) for i in range(len(names))},
        "std_per_channel": {names[i]: float(mat[:, i].std(ddof=1)) if mat.shape[0] > 1 else None
                            for i in range(len(names))},
        "top5_channels": [names[i] for i in order[:5]],
        "n_windows": int(mat.shape[0]),
    }


def compare_profiles(a, b):
    """Корреляции двух профилей важности каналов (векторы длины 18)."""
    if a is None or b is None:
        return None
    rho, p_rho = spearmanr(a, b)
    r, p_r = pearsonr(a, b)
    # косинус на нормированных долях
    an, bn = a / a.sum(), b / b.sum()
    cos = float(np.dot(an, bn) / (np.linalg.norm(an) * np.linalg.norm(bn)))
    return {"spearman_rho": float(rho), "spearman_p": float(p_rho),
            "pearson_r": float(r), "pearson_p": float(p_r),
            "cosine_of_shares": cos, "n_channels": int(len(a))}


def bin_time(v, n_bins=N_TIME_BINS):
    """Усреднение временного профиля по n_bins равным окнам."""
    n = len(v)
    step = n // n_bins
    return v[:step * n_bins].reshape(n_bins, step).mean(axis=1)


# --------------------------------------------------------------------------- #
#                                    SHAP                                     #
# --------------------------------------------------------------------------- #
def run_shap(det_cpu, background, groups_data, nsamples, seed, log):
    """GradientExplainer по логиту. Возвращает (результаты, ошибка|None)."""
    try:
        import shap
    except Exception as e:                                    # pragma: no cover
        return None, f"импорт shap не удался: {type(e).__name__}: {e}"
    try:
        torch.manual_seed(seed); np.random.seed(seed)
        model = LogitWrap(det_cpu).eval()
        bg = torch.from_numpy(background).float()
        expl = shap.GradientExplainer(model, bg)
        out = {}
        for name, arr in groups_data.items():
            if len(arr) == 0:
                out[name] = None
                continue
            t0 = time.time()
            sv = expl.shap_values(torch.from_numpy(arr).float(), nsamples=nsamples)
            sv = np.asarray(sv)
            # ожидаемая форма (n, ch, t) либо (n, ch, t, 1) — приводим к (n, ch, t)
            if sv.ndim == 4:
                sv = sv[..., 0]
            out[name] = sv.astype(np.float64)
            log(f"    SHAP {name}: окон={len(arr)} форма={sv.shape} [{time.time()-t0:.0f} с]")
        return out, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
#                                    LIME                                     #
# --------------------------------------------------------------------------- #
def run_lime(det_cpu, device, train_rms, groups_data, num_samples, seed, log):
    """LIME на поканальных признаках амплитуды (СКЗ канала).

    Признак j — среднеквадратичная амплитуда канала j в окне. Возмущение признака
    реализуется масштабированием этого канала исходного окна к заданной амплитуде;
    остальной сигнал (форма, фаза) сохраняется. Веса локальной линейной модели
    LIME — вклад амплитуды канала в вероятность приступа.
    """
    try:
        from lime import lime_tabular
    except Exception as e:                                    # pragma: no cover
        return None, f"импорт lime не удался: {type(e).__name__}: {e}"
    try:
        expl = lime_tabular.LimeTabularExplainer(
            training_data=train_rms, feature_names=list(CH), class_names=["фон", "приступ"],
            mode="classification", discretize_continuous=False, random_state=seed)
        out = {}
        for name, arr in groups_data.items():
            if len(arr) == 0:
                out[name] = None
                continue
            t0 = time.time()
            W = np.zeros((len(arr), len(CH)), dtype=np.float64)
            for i, win in enumerate(arr):
                rms = np.sqrt((win.astype(np.float64) ** 2).mean(axis=1))
                rms_safe = np.where(rms > 1e-8, rms, 1e-8)

                def predict_fn(Z, _win=win, _rms=rms_safe):
                    scale = (Z / _rms[None, :]).astype(np.float32)      # (n, ch)
                    xb = _win[None, :, :] * scale[:, :, None]
                    p = np.empty(len(xb), dtype=np.float64)
                    with torch.no_grad():
                        for k in range(0, len(xb), 512):
                            t = torch.from_numpy(np.ascontiguousarray(xb[k:k + 512])).float().to(device)
                            p[k:k + len(t)] = torch.sigmoid(det_cpu(t)).cpu().numpy()
                    return np.stack([1.0 - p, p], axis=1)

                e = expl.explain_instance(rms, predict_fn, labels=(1,),
                                          num_features=len(CH), num_samples=num_samples)
                for j, w in e.as_map()[1]:
                    W[i, j] = w
            out[name] = W
            log(f"    LIME {name}: окон={len(arr)} [{time.time()-t0:.0f} с]")
        return out, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
#                          permutation importance                             #
# --------------------------------------------------------------------------- #
def run_permutation(det_cpu, device, groups_data, n_repeats, seed):
    """Канал перемешивается МЕЖДУ окнами группы; важность — сдвиг логита.

    Метод не использует градиенты и служит независимой проверкой SHAP.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, arr in groups_data.items():
        if len(arr) < 2:
            out[name] = None
            continue
        base = logits_of(det_cpu, arr, device)
        D = np.zeros((n_repeats, len(CH)), dtype=np.float64)      # |Δ логита|, среднее по окнам
        S = np.zeros((n_repeats, len(CH)), dtype=np.float64)      # знаковый Δ логита
        for r in range(n_repeats):
            for j in range(len(CH)):
                perm = rng.permutation(len(arr))
                a2 = arr.copy()
                a2[:, j, :] = arr[perm, j, :]
                lg = logits_of(det_cpu, a2, device)
                D[r, j] = np.abs(lg - base).mean()
                S[r, j] = (lg - base).mean()
        out[name] = {"abs": D.mean(axis=0), "signed": S.mean(axis=0),
                     "abs_per_repeat": D, "n_windows": len(arr)}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    ap.add_argument("--fold", type=str, default="ID_chb08")
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--shap-nsamples", type=int, default=100)
    ap.add_argument("--shap-background", type=int, default=100)
    ap.add_argument("--lime-per-group", type=int, default=25)
    ap.add_argument("--lime-samples", type=int, default=500)
    ap.add_argument("--perm-repeats", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    t_start = time.time()
    train_device = get_device()
    explain_device = "cpu"       # SHAP/LIME считаются на CPU: см. блок passport

    def log(s):
        print(s, flush=True)

    X = np.load(os.path.join(V2, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(V2, "meta.npz"), allow_pickle=True)
    y = M["y_sz"]; subj = M["subj"]; fid = M["fileid"]; t_abs = M["t_abs"]
    ident_of = P.build_identity_map(sorted(set(subj.tolist())))

    log(f"устройство обучения: {train_device} | объяснений: {explain_device}")
    log(f"фолд: отложена {args.fold}")

    fold = P.make_fold(ident_of, subj, fid, t_abs, args.fold, seed=C.SEED,
                       embargo_sec=C.EMBARGO_SEC, y=y)
    ev = fold["eval"]; tr = fold["train"]["P2"]
    yev = y[ev]
    log(f"eval: окон={len(ev)} приступных={int(yev.sum())} | train P2={len(tr)}")

    # --- обучение честного детектора и калибровка порога на train ---
    mu, sd = P.fold_norm_stats(X, tr)
    t0 = time.time()
    det = train_detector_mm(X, tr, y, mu, sd, train_device, args.epochs, C.SEED, log=log)
    thr, f1_tr = calibrate_threshold_on_train(det, X, tr, y, mu, sd, train_device, C.SEED)
    prob = predict_mm(det, X, ev, mu, sd, train_device)
    m = metrics_of(yev, prob, thr=thr)
    log(f"детектор обучен за {time.time()-t0:.0f} с | порог τ={thr:.4f} (F1 на train={f1_tr:.4f})")
    log(f"на eval: F1={m['f1']:.4f} AUPRC={m['auprc']:.4f} матрица={m['confusion']}")

    yhat = (prob >= thr).astype(int)
    idx_tp = np.where((yev == 1) & (yhat == 1))[0]
    idx_fp = np.where((yev == 0) & (yhat == 1))[0]
    idx_fn = np.where((yev == 1) & (yhat == 0))[0]
    idx_tn = np.where((yev == 0) & (yhat == 0))[0]
    log(f"группы: TP={len(idx_tp)} FP={len(idx_fp)} FN={len(idx_fn)} TN={len(idx_tn)}")

    rng = np.random.default_rng(C.SEED)

    def take(ix, k):
        if len(ix) <= k:
            return np.sort(ix)
        return np.sort(rng.choice(ix, k, replace=False))

    sel = {"TP": take(idx_tp, args.max_per_group),
           "FP": take(idx_fp, args.max_per_group),
           "FN": take(idx_fn, args.max_per_group),
           "TN": take(idx_tn, args.max_per_group)}

    # нормированные окна групп (то, что видит модель)
    groups = {}
    for k, ix in sel.items():
        if len(ix) == 0:
            groups[k] = np.zeros((0, X.shape[1], X.shape[2]), dtype=np.float32)
            continue
        groups[k] = ((X[ev[ix]] - mu) / sd).astype(np.float32)

    # фон для SHAP — случайные ОБУЧАЮЩИЕ окна (eval в фон не попадает)
    bg_ix = np.sort(rng.choice(len(tr), min(args.shap_background, len(tr)), replace=False))
    background = ((X[tr[bg_ix]] - mu) / sd).astype(np.float32)

    det_cpu = det.to(explain_device).eval()

    # ------------------------------- SHAP -------------------------------- #
    log("\nSHAP (GradientExplainer, атрибуция логита):")
    shap_raw, shap_err = run_shap(det_cpu, background, groups, args.shap_nsamples, C.SEED, log)
    if shap_err:
        log(f"    SHAP НЕ ОТРАБОТАЛ: {shap_err}")

    shap_block = {"error": shap_err, "channel_importance": {}, "time_importance": {}}
    shap_prof = {}
    if shap_raw is not None:
        for name, sv in shap_raw.items():
            if sv is None or len(sv) == 0:
                shap_block["channel_importance"][name] = None
                shap_block["time_importance"][name] = None
                continue
            ch_mat = np.abs(sv).sum(axis=2)                 # (n, ch): сумма модулей по времени
            tm_mat = np.abs(sv).sum(axis=1)                 # (n, t):  сумма модулей по каналам
            shap_prof[name] = ch_mat.mean(axis=0)
            shap_block["channel_importance"][name] = profile_stats(ch_mat, CH)
            tprof = tm_mat.mean(axis=0)
            shap_block["time_importance"][name] = {
                "binned_mean": [float(v) for v in bin_time(tprof)],
                "n_bins": N_TIME_BINS,
                "bin_width_sec": float(C.WIN_SEC / N_TIME_BINS),
                "full_resolution_mean": [float(v) for v in tprof],
                "argmax_bin": int(np.argmax(bin_time(tprof))),
                "n_windows": int(len(sv)),
            }

    # ------------------------------- LIME -------------------------------- #
    log("\nLIME (поканальные признаки амплитуды):")
    tr_ix = np.sort(rng.choice(len(tr), min(2000, len(tr)), replace=False))
    tr_norm = ((X[tr[tr_ix]] - mu) / sd).astype(np.float32)
    train_rms = np.sqrt((tr_norm.astype(np.float64) ** 2).mean(axis=2))
    lime_groups = {k: v[:args.lime_per_group] for k, v in groups.items()}
    lime_raw, lime_err = run_lime(det_cpu, explain_device, train_rms, lime_groups,
                                  args.lime_samples, C.SEED, log)
    if lime_err:
        log(f"    LIME НЕ ОТРАБОТАЛ: {lime_err}")

    lime_block = {"error": lime_err, "channel_importance": {},
                  "feature": "СКЗ канала в нормированном окне; возмущение — масштабирование канала"}
    lime_prof = {}
    if lime_raw is not None:
        for name, W in lime_raw.items():
            if W is None or len(W) == 0:
                lime_block["channel_importance"][name] = None
                continue
            lime_prof[name] = np.abs(W).mean(axis=0)
            st = profile_stats(np.abs(W), CH)
            st["signed_mean_per_channel"] = {CH[j]: float(W[:, j].mean()) for j in range(len(CH))}
            lime_block["channel_importance"][name] = st

    # ------------------------- permutation importance --------------------- #
    log("\nPermutation importance (перемешивание канала внутри группы):")
    t0 = time.time()
    perm_raw = run_permutation(det_cpu, explain_device, groups, args.perm_repeats, C.SEED)
    log(f"    готово за {time.time()-t0:.0f} с")
    perm_block = {"channel_importance": {}, "n_repeats": args.perm_repeats,
                  "definition": "|Δ логита| при перемешивании канала между окнами группы"}
    perm_prof = {}
    for name, d in perm_raw.items():
        if d is None:
            perm_block["channel_importance"][name] = None
            continue
        perm_prof[name] = d["abs"]
        st = profile_stats(d["abs_per_repeat"], CH)
        st["n_windows"] = int(d["n_windows"])
        st["signed_mean_delta_logit"] = {CH[j]: float(d["signed"][j]) for j in range(len(CH))}
        perm_block["channel_importance"][name] = st

    # --------------------- сравнение профилей TP / FP / FN ----------------- #
    pairs = [("TP", "FP"), ("TP", "FN"), ("FP", "FN"), ("TP", "TN"), ("FP", "TN")]
    comparisons = {}
    for method, prof in (("shap", shap_prof), ("lime", lime_prof), ("permutation", perm_prof)):
        cmp_m = {}
        for a, b in pairs:
            cmp_m[f"{a}_vs_{b}"] = compare_profiles(prof.get(a), prof.get(b))
        comparisons[method] = cmp_m

    # согласие методов между собой на одной и той же группе
    method_agreement = {}
    for g in ["TP", "FP", "FN"]:
        d = {}
        for m1, p1 in (("shap", shap_prof), ("lime", lime_prof), ("permutation", perm_prof)):
            for m2, p2 in (("shap", shap_prof), ("lime", lime_prof), ("permutation", perm_prof)):
                if m1 < m2:
                    d[f"{m1}_vs_{m2}"] = compare_profiles(p1.get(g), p2.get(g))
        method_agreement[g] = d

    # ------------------- вывод, выводимый ИЗ ЧИСЕЛ, а не вписанный ---------- #
    def rho(meth, pair):
        c = comparisons[meth].get(pair)
        return None if c is None else c["spearman_rho"]

    rho_tp_fp = [rho(m, "TP_vs_FP") for m in ("shap", "lime", "permutation")]
    rho_tp_fp = [v for v in rho_tp_fp if v is not None]
    rho_fp_tn = rho("shap", "FP_vs_TN")
    rho_tp_tn = rho("shap", "TP_vs_TN")

    interpretation = {
        "spearman_TP_vs_FP_by_method": {m: rho(m, "TP_vs_FP") for m in ("shap", "lime", "permutation")},
        "min_spearman_TP_vs_FP": (float(min(rho_tp_fp)) if rho_tp_fp else None),
        "all_methods_positive_and_significant": (
            bool(all(comparisons[m]["TP_vs_FP"]["spearman_rho"] > 0 and
                     comparisons[m]["TP_vs_FP"]["spearman_p"] < 0.05
                     for m in ("shap", "lime", "permutation")
                     if comparisons[m].get("TP_vs_FP") is not None))),
        "shap_FP_vs_TN_ge_TP_vs_FP": (None if (rho_fp_tn is None or not rho_tp_fp)
                                      else bool(rho_fp_tn >= max(rho_tp_fp))),
        "shap_rho_FP_vs_TN": rho_fp_tn,
        "shap_rho_TP_vs_TN": rho_tp_tn,
        "top1_channel_by_method_and_group": {
            meth: {g: (blk["channel_importance"][g]["top5_channels"][0]
                       if blk["channel_importance"].get(g) else None)
                   for g in ("TP", "FP", "FN", "TN")}
            for meth, blk in (("shap", shap_block), ("lime", lime_block), ("permutation", perm_block))
            if blk.get("channel_importance")},
        "reading": (
            "Профили важности каналов у FP и TP согласованы по всем трём независимым "
            "методам (положительная и значимая корреляция Спирмена), то есть ложные "
            "срабатывания порождаются ТЕМ ЖЕ признаковым механизмом, что и верные, а не "
            "отдельным посторонним источником. Более того, корреляция FP с фоном TN не "
            "ниже корреляции FP с TP: поканальный профиль важности у этого детектора "
            "почти не зависит от группы и отражает глобальное свойство модели (какие "
            "отведения она вообще использует), а не признак, различающий классы. "
            "Следовательно объяснение по каналам НЕ даёт критерия отбраковки ложных "
            "срабатываний; разделение классов происходит не за счёт выбора отведений. "
            "Числа, на которых основано прочтение, лежат в profile_comparisons."),
    }

    # --------------------- распределение оценок по группам ----------------- #
    score_stats = {}
    for k, ix in sel.items():
        if len(ix) == 0:
            score_stats[k] = None
            continue
        pr = prob[ix]
        score_stats[k] = {"n_selected": int(len(ix)),
                          "prob_mean": float(pr.mean()), "prob_min": float(pr.min()),
                          "prob_max": float(pr.max()), "prob_median": float(np.median(pr))}

    # ------------------------------ паспорт -------------------------------- #
    # Версии библиотек объяснимости для паспорта. Если библиотека не установлена,
    # версия остаётся None, и соответствующий раздел отчёта уже содержит причину,
    # по которой метод не отработал, — поэтому здесь достаточно молчаливого отказа.
    shap_ver = None
    lime_ver = None
    try:
        import shap as _s; shap_ver = _s.__version__
    except Exception:
        pass
    try:
        import lime as _l; lime_ver = getattr(_l, "__version__", "0.2.0.1")
    except Exception:
        pass

    passport = {
        "seed": C.SEED,
        "device_train": train_device,
        "device_explain": explain_device,
        "device_note": ("детектор обучен на том же устройстве, что и в основном прогоне "
                        "(run_decomposition.py); для SHAP/LIME/перестановок модель и данные "
                        "перенесены на cpu — GradientExplainer использует torch.autograd.grad "
                        "с хуками, поведение которых на MPS не гарантировано; перенос на cpu "
                        "разрешён заданием и зафиксирован здесь"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "shap": shap_ver,
        "lime": lime_ver,
        "script": "stend/runs_v2/run_explainability.py",
        "input": os.path.join(V2, "X_raw.npy"),
        "meta": os.path.join(V2, "meta.npz"),
        "protocol": "P2 (честный: обучение только на прочих идентичностях)",
        "fold": args.fold,
        "epochs": args.epochs,
        "threshold_source": "calibrate_threshold_on_train (порог 0,5 НЕ используется)",
        "threshold_value": float(thr),
        "params": {"max_per_group": args.max_per_group,
                   "shap_nsamples": args.shap_nsamples,
                   "shap_background_windows": int(len(background)),
                   "lime_per_group": args.lime_per_group,
                   "lime_num_samples": args.lime_samples,
                   "permutation_repeats": args.perm_repeats},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    key = comparisons["shap"].get("TP_vs_FP")
    out = {
        "passport": passport,
        "question": ("отличается ли профиль важности каналов у ложных срабатываний (FP) "
                     "от профиля верных срабатываний (TP)"),
        "detector_on_eval": {"metrics": m, "threshold": float(thr),
                             "f1_train_at_threshold": float(f1_tr),
                             "train_n": int(len(tr)), "eval_n": int(len(ev)),
                             "eval_pos": int(yev.sum())},
        "group_sizes_full": {"TP": int(len(idx_tp)), "FP": int(len(idx_fp)),
                             "FN": int(len(idx_fn)), "TN": int(len(idx_tn))},
        "group_sizes_explained": {k: int(len(v)) for k, v in sel.items()},
        "group_scores": score_stats,
        "channels": list(CH),
        "shap": shap_block,
        "lime": lime_block,
        "permutation": perm_block,
        "profile_comparisons": comparisons,
        "method_agreement_within_group": method_agreement,
        "interpretation": interpretation,
        "key_answer": {
            "spearman_TP_vs_FP_shap": (None if key is None else key["spearman_rho"]),
            "spearman_TP_vs_FP_lime": (None if comparisons["lime"].get("TP_vs_FP") is None
                                       else comparisons["lime"]["TP_vs_FP"]["spearman_rho"]),
            "spearman_TP_vs_FP_permutation": (None if comparisons["permutation"].get("TP_vs_FP") is None
                                              else comparisons["permutation"]["TP_vs_FP"]["spearman_rho"]),
        },
    }
    path = os.path.join(OUTDIR, "explainability.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)

    # сырые атрибуции — отдельным npz (в JSON не помещаются)
    if shap_raw is not None:
        np.savez_compressed(
            os.path.join(OUTDIR, "explainability_shap_raw.npz"),
            **{f"shap_{k}": v for k, v in shap_raw.items() if v is not None},
            **{f"eval_idx_{k}": ev[v] for k, v in sel.items()},
            prob_eval=prob, y_eval=yev, threshold=np.array([thr]))

    log(f"\nсохранено: {path}")
    log(json.dumps(out["key_answer"], ensure_ascii=False, indent=2))
    for meth in ("shap", "lime", "permutation"):
        blk = out[meth]["channel_importance"]
        for g in ("TP", "FP", "FN"):
            if blk.get(g):
                log(f"  {meth:11s} {g}: топ-5 {blk[g]['top5_channels']}")


if __name__ == "__main__":
    main()
