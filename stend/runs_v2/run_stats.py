"""Статистика по результатам разложения: Б2 (ДИ вкладов) и Б3 (критерии значимости).

Б2. В июльской работе бутстрэп посчитан для честной F1, тогда как формула работы
    определяет доверительные интервалы для ВКЛАДОВ Δ_дубли и Δ_утечка. Здесь
    бутстрэп проводится по идентичностям (единица ресэмплинга — идентичность,
    а не окно: окна внутри пациента зависимы) и даёт ДИ для каждого вклада.

Б3. Перестановочный тест для сравнения P0 и P2 и парный критерий Уилкоксона
    по фолдам. Единица перестановки — идентичность.

Вход:  разложение, путь задаётся переменной VKR_STATS_SRC
Выход: bootstrap_contributions<SUF>.json и significance_tests<SUF>.json,
       суффикс задаётся переменной VKR_STATS_SUFFIX.
Итоговые интервалы работы: VKR_STATS_SRC=decomposition_24_embargofix.json,
VKR_STATS_SUFFIX=_24embargofix.
"""
import os, sys, json, time, platform
import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from stend.vkr_eeg import config as C

OUTDIR = os.path.join("outputs", "track_b")
SRC = os.environ.get("VKR_STATS_SRC", os.path.join(OUTDIR, "decomposition_v2.json"))
SUF = os.environ.get("VKR_STATS_SUFFIX", "")


def bootstrap_contributions(per_fold_vals, n_boot, seed, gamma=0.05):
    """per_fold_vals: dict имя_вклада -> массив значений по фолдам.
    Ресэмплинг идентичностей (фолдов) с возвращением."""
    rng = np.random.default_rng(seed)
    keys = list(per_fold_vals)
    k = len(per_fold_vals[keys[0]])
    out = {}
    draws = {kk: [] for kk in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        for kk in keys:
            draws[kk].append(float(np.mean(np.asarray(per_fold_vals[kk])[pick])))
    for kk in keys:
        v = np.array(draws[kk])
        out[kk] = {
            "mean": float(v.mean()),
            "lo": float(np.percentile(v, 100 * gamma / 2)),
            "hi": float(np.percentile(v, 100 * (1 - gamma / 2))),
            "point_estimate": float(np.mean(per_fold_vals[kk])),
            "excludes_zero": bool(np.percentile(v, 100 * gamma / 2) > 0
                                  or np.percentile(v, 100 * (1 - gamma / 2)) < 0),
        }
    return out


def permutation_test_paired(a, b, n_perm, seed):
    """Перестановочный тест для парных наблюдений (по одной паре на идентичность):
    случайно меняем знак разности внутри каждой пары."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = float(d.mean())
    cnt = 0
    for _ in range(n_perm):
        sign = rng.choice([-1.0, 1.0], size=len(d))
        if abs(float((d * sign).mean())) >= abs(obs) - 1e-15:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    d = json.load(open(SRC))
    folds = [k for k, v in d["per_fold"].items() if "protocols" in v]
    metric_sets = {}
    for met in ["f1", "auprc", "roc_auc"]:
        metric_sets[met] = {pr: np.array([d["per_fold"][f]["protocols"][pr][met] for f in folds], float)
                            for pr in ["P0", "P1", "P1e", "P2"]}

    # ---------- Б2: ДИ вкладов ----------
    boot = {}
    for met, m in metric_sets.items():
        contrib = {
            "delta_okna_pochti_dubli": m["P0"] - m["P1"],
            "delta_embargo": m["P1"] - m["P1e"],
            "delta_subject": m["P1e"] - m["P2"],
            "delta_total": m["P0"] - m["P2"],
        }
        boot[met] = bootstrap_contributions(contrib, C.BOOTSTRAP_N, C.SEED, C.BOOTSTRAP_GAMMA)
        boot[met]["_per_fold"] = {k: v.tolist() for k, v in contrib.items()}

    out_b = {
        "passport": {"seed": C.SEED, "n_bootstrap": C.BOOTSTRAP_N, "gamma": C.BOOTSTRAP_GAMMA,
                     "script": "stend/runs_v2/run_stats.py", "python": platform.python_version(),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "source": SRC},
        "unit_of_resampling": "идентичность (не окно): окна внутри пациента зависимы",
        "n_identities": len(folds),
        "identities": folds,
        "contributions": boot,
        "caveat": (f"Ресэмплинг ведётся всего по {len(folds)} идентичностям, поэтому "
                   "бутстрэп-распределение дискретно и интервалы широки. Это ограничение "
                   "объёма выборки, а не метода."),
    }
    p1 = os.path.join(OUTDIR, f"bootstrap_contributions{SUF}.json")
    json.dump(out_b, open(p1, "w"), ensure_ascii=False, indent=2)
    print("сохранено:", p1)

    # ---------- Б3: значимость ----------
    sig = {}
    for met, m in metric_sets.items():
        obs, p_perm = permutation_test_paired(m["P0"], m["P2"], 20000, C.SEED)
        res = {"P0_vs_P2": {"mean_difference": obs, "permutation_p": p_perm,
                            "n_permutations": 20000}}
        try:
            w = stats.wilcoxon(m["P0"], m["P2"], alternative="two-sided",
                               zero_method="wilcox", method="exact")
            res["P0_vs_P2"]["wilcoxon_stat"] = float(w.statistic)
            res["P0_vs_P2"]["wilcoxon_p"] = float(w.pvalue)
        except Exception as e:
            res["P0_vs_P2"]["wilcoxon_error"] = str(e)
        for a, b in [("P0", "P1"), ("P1", "P1e"), ("P1e", "P2")]:
            o, pv = permutation_test_paired(m[a], m[b], 20000, C.SEED)
            res[f"{a}_vs_{b}"] = {"mean_difference": o, "permutation_p": pv}
        sig[met] = res

    out_s = {
        "passport": {"seed": C.SEED, "script": "stend/runs_v2/run_stats.py",
                     "scipy": stats.__name__ and __import__("scipy").__version__,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "source": SRC},
        "n_identities": len(folds),
        "identities": folds,
        "tests": sig,
        "caveat": (f"При {len(folds)} идентичностях парный критерий Уилкоксона имеет "
                   f"минимально достижимое p = {2/2**len(folds):.4f}; перестановочный тест "
                   "по знакам даёт тот же предел. Отсутствие значимости при таком объёме "
                   "выборки НЕ означает отсутствия эффекта."),
    }
    p2 = os.path.join(OUTDIR, f"significance_tests{SUF}.json")
    json.dump(out_s, open(p2, "w"), ensure_ascii=False, indent=2)
    print("сохранено:", p2)
    print(json.dumps(sig["f1"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
