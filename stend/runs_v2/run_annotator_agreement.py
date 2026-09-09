"""Влияние согласия разметчиков на измерение утечки (набор новорождённых).

ВОПРОС. Метка «приступ» в наборе Helsinki получена тремя экспертами независимо, и они
расходятся: единогласно приступом признаны 9 746 окон, ещё 7 872 окна отмечены одним
или двумя экспертами. Отсюда два вопроса, на которые нельзя ответить ни на CHB-MIT,
ни на Siena, где разметчик один:

  1. Насколько показатели детектора зависят от того, КАК определена метка —
     по единогласию, по большинству или по одному эксперту?
  2. Зависит ли от этого ИЗМЕРЕННОЕ ЗАВЫШЕНИЕ Δ, то есть устойчив ли основной
     результат работы к неоднозначности разметки?

Второй вопрос существеннее. Если Δ меняется вместе с определением метки, то величина
завышения — свойство протокола разметки, а не утечки, и это надо знать. Если не меняется,
основной результат работы устойчив к шуму разметки, что его усиливает.

КАК. Кэш окон не пересобирается: число согласившихся экспертов сохранено пооконно
в meta.npz как expert_agreement. Меняется только определение метки y_sz, массив окон
X_raw.npy общий для всех вариантов (подключается ссылкой, не копируется).

Выход: outputs/track_b/annotator_agreement.json
"""
import os, sys, json, time, platform, subprocess, argparse
import numpy as np

sys.path.insert(0, os.getcwd())

from stend.vkr_eeg import config as C

OUTDIR = os.path.join("outputs", "track_b")
PY = sys.executable
LEVELS = {1: "хотя бы один эксперт", 2: "большинство (не менее двух)", 3: "единогласно"}


def make_variant(base, level):
    """Каталог-вариант с той же матрицей окон и меткой по заданному порогу согласия."""
    dst = f"{base}_agree{level}"
    os.makedirs(dst, exist_ok=True)
    link = os.path.join(dst, "X_raw.npy")
    if not os.path.exists(link):
        os.symlink(os.path.join(base, "X_raw.npy"), link)   # ссылка, а не копия
    M = np.load(os.path.join(base, "meta.npz"), allow_pickle=True)
    agree = M["expert_agreement"]
    y = (agree >= level).astype(np.int64)
    np.savez(os.path.join(dst, "meta.npz"),
             y_sz=y, subj=M["subj"], fileid=M["fileid"],
             wtime=M["wtime"], t_abs=M["t_abs"], expert_agreement=agree)
    meta = json.load(open(os.path.join(base, "cache_meta.json")))
    meta.update({"n_seizure_windows": int(y.sum()),
                 "positive_rate": float(y.mean()),
                 "consensus_required": level,
                 "note": meta.get("note", "") + f" | метка: {LEVELS[level]}"})
    json.dump(meta, open(os.path.join(dst, "cache_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    return dst, int(y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.environ.get("VKR_CACHE_ROOT", "."), "helsinki"))
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    ap.add_argument("--max-folds", type=int, default=15)
    ap.add_argument("--neg-ratio", type=float, default=20.0)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    res = {}

    for level in (3, 2, 1):
        cache, n_pos = make_variant(args.base, level)
        tag = f"decomposition_helsinki_a{level}"
        print(f"\n=== метка: {LEVELS[level]} ({n_pos} приступных окон) ===", flush=True)
        cmd = [PY, "-u", "stend/runs_v2/run_decomposition.py",
               "--cache", cache, "--tag", tag, "--epochs", str(args.epochs),
               "--neg-ratio", str(args.neg_ratio), "--norm-sample", "120000",
               "--max-folds", str(args.max_folds)]
        with open(f"logs/agree_{level}.log", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc:
            print(f"  прогон завершился с кодом {rc}, см. logs/agree_{level}.log", flush=True)
            continue
        d = json.load(open(os.path.join(OUTDIR, f"{tag}.json")))
        folds = [k for k, v in d["per_fold"].items() if "protocols" in v]
        g = lambda f_, p: d["per_fold"][f_]["protocols"][p]["f1"]
        dt = np.array([g(f_, "P0") - g(f_, "P2") for f_ in folds])
        pooled = d["summary"]["pooled"]["metrics"]
        res[str(level)] = {
            "label_definition": LEVELS[level],
            "n_seizure_windows": n_pos,
            "positive_rate": float(n_pos / len(np.load(os.path.join(cache, "meta.npz"))["y_sz"])),
            "n_folds": len(folds),
            "delta_total_mean": float(dt.mean()),
            "delta_total_std": float(dt.std(ddof=1)) if len(dt) > 1 else None,
            "delta_positive_in": int((dt > 0).sum()),
            "P0_f1_pooled": pooled["P0"]["f1"], "P2_f1_pooled": pooled["P2"]["f1"],
            "P0_roc_auc": pooled["P0"]["roc_auc"], "P2_roc_auc": pooled["P2"]["roc_auc"],
            "P2_false_alarms_per_24h": pooled["P2"].get("false_alarms_per_24h"),
        }
        r = res[str(level)]
        print(f"  Δ = {r['delta_total_mean']:+.4f} (положителен в "
              f"{r['delta_positive_in']}/{r['n_folds']}), "
              f"P0 F1={r['P0_f1_pooled']:.4f}, P2 F1={r['P2_f1_pooled']:.4f}", flush=True)

    # устойчив ли основной результат к определению метки
    dts = [v["delta_total_mean"] for v in res.values()]
    stability = {
        "delta_range": (max(dts) - min(dts)) if dts else None,
        "delta_relative_range": ((max(dts) - min(dts)) / abs(np.mean(dts))) if dts else None,
        "all_positive": all(x > 0 for x in dts) if dts else None,
    }
    out = {"passport": {"script": "stend/runs_v2/run_annotator_agreement.py",
                        "seed": C.SEED, "epochs": args.epochs,
                        "max_folds": args.max_folds, "base_cache": args.base,
                        "python": platform.python_version(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "elapsed_sec": round(time.time() - t0, 1)},
           "question": ("зависит ли измеренное завышение Δ от того, как определена метка "
                        "приступа при расхождении трёх экспертов"),
           "by_level": res, "stability": stability}
    p = os.path.join(OUTDIR, "annotator_agreement.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print(f"\n{'определение метки':30s} {'окон+':>7s} {'Δ':>9s} {'P0 F1':>8s} {'P2 F1':>8s}")
    for lvl in ("3", "2", "1"):
        if lvl not in res: continue
        r = res[lvl]
        print(f"{r['label_definition']:30s} {r['n_seizure_windows']:7d} "
              f"{r['delta_total_mean']:+9.4f} {r['P0_f1_pooled']:8.4f} {r['P2_f1_pooled']:8.4f}")
    if stability["delta_range"] is not None:
        print(f"\nразмах Δ между определениями: {stability['delta_range']:.4f} "
              f"({stability['delta_relative_range']:.0%} величины)")
    print("сохранено:", p)


if __name__ == "__main__":
    main()
