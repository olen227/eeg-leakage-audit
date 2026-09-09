"""Аудит утечки данных: единая точка входа программного модуля.

НАЗНАЧЕНИЕ. Принять произвольный набор ЭЭГ, приведённый к контракту
(см. dataset_contract.py), и ответить на вопрос: насколько качество детектора завышено
утечкой данных и из чего это завышение складывается.

ЧТО ДЕЛАЕТ, по порядку:
  1. Проверяет набор на соответствие контракту и оценивает достаточность объёма.
     При нарушении критических условий выдаёт предупреждение, но не останавливается —
     ограничения попадают в отчёт.
  2. Прогоняет каскад протоколов P0 / P1 / P1e / P2 по фолдам идентичностей.
     Внутри фолда оценочное множество ОДНО на все протоколы: только тогда разность
     метрик является измерением вклада утечки, а не разницы оценочных выборок.
  3. Считает аддитивное разложение, доверительные интервалы вкладов (бутстрэп по
     идентичностям) и статистическую значимость (перестановочный тест, Уилкоксон).
  4. Формирует итоговый отчёт с вердиктом и ограничениями применимости.

ЗАПУСК:
    python stend/runs_v2/run_audit.py --cache <каталог набора> --name <имя>

Выход: outputs/track_b/audit_<имя>.json и читаемый отчёт в консоль.
"""
import os, sys, json, time, subprocess, argparse, platform
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from stend.vkr_eeg import config as C
from stend.runs_v2 import protocols as P
from stend.runs_v2.dataset_contract import check_dataset, print_report

OUTDIR = os.path.join("outputs", "track_b")
PY = sys.executable


def run(cmd, log):
    print(f"  $ {' '.join(cmd[-6:])}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description="Аудит утечки данных в наборе ЭЭГ")
    ap.add_argument("--cache", required=True, help="каталог набора по контракту")
    ap.add_argument("--name", required=True, help="краткое имя набора для файлов отчёта")
    ap.add_argument("--epochs", type=int, default=C.DET_EPOCHS)
    ap.add_argument("--neg-ratio", type=float, default=20.0)
    ap.add_argument("--norm-sample", type=int, default=120000)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--check-only", action="store_true",
                    help="только проверить контракт и объём, прогон не запускать")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print(f"АУДИТ УТЕЧКИ ДАННЫХ — набор «{args.name}»")
    print("=" * 70)

    # --- шаг 1: контракт и объём ---
    M = np.load(os.path.join(args.cache, "meta.npz"), allow_pickle=True) \
        if os.path.exists(os.path.join(args.cache, "meta.npz")) else None
    ident_map = (P.build_identity_map(sorted(set(M["subj"].tolist())))
                 if M is not None else None)
    chk = check_dataset(args.cache, ident_map)
    print_report(chk)
    if not chk["ok"]:
        print("\nОСТАНОВ: набор не соответствует контракту.")
        sys.exit(2)
    if args.check_only:
        json.dump({"check": chk}, open(os.path.join(OUTDIR, f"audit_{args.name}_check.json"), "w"),
                  ensure_ascii=False, indent=2)
        return

    tag = f"decomposition_{args.name}"
    # --- шаг 2: каскад протоколов ---
    print("\n[1/3] каскад протоколов P0/P1/P1e/P2 по фолдам идентичностей", flush=True)
    cmd = [PY, "-u", "stend/runs_v2/run_decomposition.py",
           "--cache", args.cache, "--tag", tag,
           "--epochs", str(args.epochs), "--neg-ratio", str(args.neg_ratio),
           "--norm-sample", str(args.norm_sample)]
    if args.max_folds:
        cmd += ["--max-folds", str(args.max_folds)]
    if run(cmd, f"logs/audit_{args.name}_decomp.log"):
        print("ОСТАНОВ: каскад завершился с ошибкой, см. лог"); sys.exit(3)

    # --- шаг 3: статистика ---
    print("[2/3] доверительные интервалы вкладов и проверка значимости", flush=True)
    env = dict(os.environ, VKR_STATS_SRC=os.path.join(OUTDIR, f"{tag}.json"),
               VKR_STATS_SUFFIX=f"_{args.name}")
    with open(f"logs/audit_{args.name}_stats.log", "w") as f:
        subprocess.run([PY, "-u", "stend/runs_v2/run_stats.py"], env=env,
                       stdout=f, stderr=subprocess.STDOUT)

    # --- шаг 4: сборка отчёта ---
    print("[3/3] сборка отчёта", flush=True)
    dec = json.load(open(os.path.join(OUTDIR, f"{tag}.json")))
    boot = json.load(open(os.path.join(OUTDIR, f"bootstrap_contributions_{args.name}.json")))
    sig = json.load(open(os.path.join(OUTDIR, f"significance_tests_{args.name}.json")))

    b = boot["contributions"]["f1"]; s = sig["tests"]["f1"]
    pooled = dec["summary"].get("pooled", {}).get("metrics", {})
    names = {"delta_okna_pochti_dubli": "вклад почти-дублей окон",
             "delta_embargo": "вклад временной смежности",
             "delta_subject": "вклад перекрытия пациентов",
             "delta_total": "полное завышение"}

    verdict = []
    tot = b["delta_total"]
    if tot["excludes_zero"] and tot["point_estimate"] > 0:
        verdict.append(f"Утечка ЗАВЫШАЕТ метрику на {tot['point_estimate']:.4f} по F1 "
                       f"(интервал [{tot['lo']:.4f}; {tot['hi']:.4f}] не накрывает ноль, "
                       f"p = {s['P0_vs_P2']['permutation_p']:.4f}).")
    else:
        verdict.append(f"Завышение оценено в {tot['point_estimate']:.4f}, но интервал "
                       f"[{tot['lo']:.4f}; {tot['hi']:.4f}] накрывает ноль "
                       f"(p = {s['P0_vs_P2']['permutation_p']:.4f}) — подтверждения нет. "
                       "Проверьте раздел о достаточности объёма.")
    parts = sorted(((k, b[k]["point_estimate"]) for k in
                    ("delta_okna_pochti_dubli", "delta_embargo", "delta_subject")),
                   key=lambda x: -abs(x[1]))
    verdict.append("Основной источник: " + names[parts[0][0]] +
                   f" ({parts[0][1]:+.4f}).")

    out = {
        "passport": {"module": "аудит утечки данных, stend/runs_v2/run_audit.py",
                     "dataset_name": args.name, "cache": args.cache,
                     "seed": C.SEED, "epochs": args.epochs,
                     "neg_ratio": args.neg_ratio,
                     "python": platform.python_version(),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "elapsed_sec": round(time.time() - t0, 1)},
        "contract_check": chk,
        "protocols_pooled": pooled,
        "contributions": {k: b[k] for k in names},
        "significance": s,
        "verdict": verdict,
        "artifacts": {"decomposition": f"{tag}.json",
                      "bootstrap": f"bootstrap_contributions_{args.name}.json",
                      "significance": f"significance_tests_{args.name}.json"},
    }
    p = os.path.join(OUTDIR, f"audit_{args.name}.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)

    if pooled:
        print("\n" + "=" * 70)
        print("КЛИНИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ (стандарт SzCORE)")
        print("=" * 70)
        for pr in ("P0", "P2"):
            m = pooled.get(pr)
            if not m: continue
            fa = m.get("false_alarms_per_24h")
            se = m.get("sensitivity")
            print(f"  {pr}: чувствительность {se:.3f}, "
                  f"ложных тревог в сутки {fa:.0f}" if se is not None and fa is not None
                  else f"  {pr}: —")

    print("\n" + "=" * 70)
    print("ЗАКЛЮЧЕНИЕ")
    print("=" * 70)
    for k, n in names.items():
        c = b[k]
        mark = "значим" if c["excludes_zero"] else "не значим"
        print(f"  {n:32s} {c['point_estimate']:+.4f}  "
              f"[{c['lo']:+.4f}; {c['hi']:+.4f}]  {mark}")
    print()
    for v in verdict:
        print("  " + v)
    for v in chk["power"]["verdict"]:
        if not v.startswith("объём достаточен"):
            print("  ОГРАНИЧЕНИЕ: " + v)
    print(f"\nотчёт: {p}")


if __name__ == "__main__":
    main()
