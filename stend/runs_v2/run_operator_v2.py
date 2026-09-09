"""Оператор идентификации, исправленная версия: устранение коллапса эмбеддингов.

ДИАГНОЗ (раздел 3.10 FINDINGS). Расстояния между эмбеддингами на знакомых идентичностях
разнесены (внутри 0,242, между 1,024), а на незнакомых схлопываются (0,149 и 0,374) —
вся шкала сжимается почти втрое. Порог, откалиброванный на обучающих людях, на новых
пропускает всё подряд; отсюда вырожденное число «доля отловленных дублей» = 0,9996
при точности 0,58.

ПРИЧИНА. В контрастивной потере слагаемое для «своих» равно d². Эмбеддинги не нормированы,
поэтому модели дешевле всего уменьшить длину ВСЕХ векторов: расстояния падают, штраф
падает, различать при этом ничему не научившись.

ЛЕЧЕНИЕ, проверяемое здесь:
  1. L2-нормировка эмбеддингов на единичную сферу. Сжиматься некуда физически:
     длина всегда 1, расстояние ограничено отрезком [0, 2]. Порог становится переносимым.
  2. Отбор трудных отрицательных пар: вместо случайных «чужих» берутся те, что сейчас
     ближе всего. К концу обучения случайные пары уже разнесены и градиента не дают.

Оба средства включаются флагами, что позволяет измерить вклад каждого отдельно.

ИЗМЕРЯЕТСЯ ТАКЖЕ САМ КОЛЛАПС: отношение средних расстояний на обучающих и на отложенных
идентичностях. Если лечение работает, отношение приближается к единице.

Выход: outputs/track_b/operator_v2.json
"""
import os, sys, json, time, platform, itertools, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from stend.vkr_eeg import config as C
from stend.vkr_eeg.operator import Encoder
from stend.runs_v2 import protocols as P
from stend.runs_v2.run_operator_cv import eer_of, get_device

OUTDIR = os.path.join("outputs", "track_b")
V2 = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))


def encode(enc, x, l2norm):
    z = enc(x)
    return F.normalize(z, dim=1) if l2norm else z


def contrastive(zi, zj, y, margin):
    d = torch.norm(zi - zj, dim=1)
    return (y * d.pow(2) + (1 - y) * torch.clamp(margin - d, min=0).pow(2)).mean()


def train(X, idx_by_ident, device, epochs, seed, l2norm, hard_neg, mu, sd, batch, margin):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(X.shape[1]).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=C.OP_LR)
    ids = list(idx_by_ident)
    hist = []
    for ep in range(epochs):
        enc.train(); tot = 0.0; nb = 0
        for _ in range(20):
            # позитивные пары: два окна одной идентичности
            A, B, Y = [], [], []
            for _ in range(batch // 2):
                s = ids[rng.integers(len(ids))]
                pool = idx_by_ident[s]
                if len(pool) < 2:
                    continue
                a, b = rng.choice(pool, 2, replace=False)
                A.append(a); B.append(b); Y.append(1.0)
            # негативные пары
            if hard_neg and ep > 0:
                # кандидаты в 4 раза больше нужного, оставляем БЛИЖАЙШИЕ
                cA, cB = [], []
                for _ in range((batch // 2) * 4):
                    i1, i2 = rng.choice(len(ids), 2, replace=False)
                    cA.append(rng.choice(idx_by_ident[ids[i1]]))
                    cB.append(rng.choice(idx_by_ident[ids[i2]]))
                with torch.no_grad():
                    enc.eval()
                    xa = torch.from_numpy(((X[np.array(cA)] - mu) / sd).astype(np.float32)).to(device)
                    xb = torch.from_numpy(((X[np.array(cB)] - mu) / sd).astype(np.float32)).to(device)
                    dd = torch.norm(encode(enc, xa, l2norm) - encode(enc, xb, l2norm), dim=1).cpu().numpy()
                    enc.train()
                keep = np.argsort(dd)[:batch // 2]
                for k in keep:
                    A.append(cA[k]); B.append(cB[k]); Y.append(0.0)
            else:
                for _ in range(batch // 2):
                    i1, i2 = rng.choice(len(ids), 2, replace=False)
                    A.append(rng.choice(idx_by_ident[ids[i1]]))
                    B.append(rng.choice(idx_by_ident[ids[i2]])); Y.append(0.0)
            xa = torch.from_numpy(((X[np.array(A)] - mu) / sd).astype(np.float32)).to(device)
            xb = torch.from_numpy(((X[np.array(B)] - mu) / sd).astype(np.float32)).to(device)
            yv = torch.tensor(Y, dtype=torch.float32, device=device)
            loss = contrastive(encode(enc, xa, l2norm), encode(enc, xb, l2norm), yv, margin)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        hist.append(tot / max(nb, 1))
    return enc, hist


@torch.no_grad()
def embed(enc, X, idx, mu, sd, device, l2norm, bs=512):
    enc.eval(); o = []
    for k in range(0, len(idx), bs):
        xb = torch.from_numpy(((X[idx[k:k + bs]] - mu) / sd).astype(np.float32)).to(device)
        o.append(encode(enc, xb, l2norm).cpu().numpy())
    return np.concatenate(o)


def pair_stats(emb, lab, n_pairs, rng):
    n = len(emb)
    a = rng.integers(0, n, n_pairs); b = rng.integers(0, n, n_pairs)
    k = a != b; a, b = a[k], b[k]
    d = np.linalg.norm(emb[a] - emb[b], axis=1)
    same = (lab[a] == lab[b]).astype(int)
    if same.sum() in (0, len(same)):
        return None
    return {"roc_auc": float(roc_auc_score(same, -d)),
            "eer": eer_of(same, d)[0],
            "d_same": float(d[same == 1].mean()),
            "d_diff": float(d[same == 0].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.OP_EPOCHS)
    ap.add_argument("--cache", type=str, default="")
    ap.add_argument("--per-subject", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--folds", type=int, default=15)
    ap.add_argument("--norm-sample", type=int, default=80000)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t0 = time.time()
    cache = args.cache or V2
    X = np.load(os.path.join(cache, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(cache, "meta.npz"), allow_pickle=True)
    subj = M["subj"]
    subjects = sorted(set(subj.tolist()))
    ident_of = P.build_identity_map(subjects)
    ident = np.array([ident_of[s] for s in subj])
    identities = sorted(set(ident_of.values()))
    print(f"устройство: {device} | идентичностей: {len(identities)}", flush=True)

    rng = np.random.default_rng(C.SEED)
    combos = list(itertools.combinations(identities, 2))
    pick = rng.permutation(len(combos))[:args.folds]
    folds = [combos[i] for i in sorted(pick)]

    configs = [
        ("исходная (как в июле)",              dict(l2norm=False, hard_neg=False)),
        ("L2-нормировка",                      dict(l2norm=True,  hard_neg=False)),
        ("L2-нормировка + трудные пары",       dict(l2norm=True,  hard_neg=True)),
    ]
    results = {name: [] for name, _ in configs}

    for fi, (ia, ib) in enumerate(folds, 1):
        held = {ia, ib}
        tr_mask = ~np.isin(ident, list(held))
        tr_idx = np.where(tr_mask)[0]
        rs = np.random.default_rng(C.SEED)
        tn = (np.sort(rs.choice(tr_idx, args.norm_sample, replace=False))
              if len(tr_idx) > args.norm_sample else tr_idx)
        mu, sd = P.fold_norm_stats(X, tn)

        # обучающие окна: до per-subject на идентичность
        by = {}
        for I in identities:
            if I in held: continue
            w = np.where(ident == I)[0]
            by[I] = rng.choice(w, min(args.per_subject, len(w)), replace=False)
        # оценочные окна: отложенные идентичности
        ev = np.sort(np.concatenate([
            rng.choice(np.where(ident == I)[0],
                       min(args.per_subject, (ident == I).sum()), replace=False) for I in held]))
        # контрольные окна ОБУЧАЮЩИХ идентичностей — для измерения коллапса
        trs = np.sort(np.concatenate([by[I][:600] for I in by]))

        print(f"\n=== фолд {fi}/{len(folds)}: отложены {ia}, {ib} "
              f"(обучающих идентичностей {len(by)}) ===", flush=True)
        for name, cfg in configs:
            enc, hist = train(X, by, device, args.epochs, C.SEED,
                              mu=mu, sd=sd, batch=C.OP_BATCH,
                              margin=C.CONTRASTIVE_MARGIN, **cfg)
            ev_st = pair_stats(embed(enc, X, ev, mu, sd, device, cfg["l2norm"]),
                               ident[ev], args.n_pairs, rng)
            tr_st = pair_stats(embed(enc, X, trs, mu, sd, device, cfg["l2norm"]),
                               ident[trs], args.n_pairs, rng)
            if ev_st is None or tr_st is None:
                continue
            # показатель коллапса: во сколько раз шкала сжимается на незнакомых
            collapse = (tr_st["d_diff"] / ev_st["d_diff"]) if ev_st["d_diff"] > 0 else None
            r = {"held": [ia, ib], "eval": ev_st, "train_ref": tr_st,
                 "collapse_ratio": collapse,
                 "loss_first": hist[0], "loss_last": hist[-1]}
            results[name].append(r)
            print(f"  {name:32s} ROC-AUC={ev_st['roc_auc']:.4f} EER={ev_st['eer']:.4f} "
                  f"| d(свои)={ev_st['d_same']:.3f} d(чужие)={ev_st['d_diff']:.3f} "
                  f"| сжатие ×{collapse:.2f}", flush=True)
        json.dump({"in_progress": True, "results": results},
                  open(os.path.join(OUTDIR, "operator_v2_partial.json"), "w"),
                  ensure_ascii=False, indent=2)

    agg = {}
    for name in results:
        if not results[name]: continue
        auc = np.array([r["eval"]["roc_auc"] for r in results[name]])
        eer = np.array([r["eval"]["eer"] for r in results[name]])
        col = np.array([r["collapse_ratio"] for r in results[name] if r["collapse_ratio"]])
        agg[name] = {"roc_auc_mean": float(auc.mean()), "roc_auc_std": float(auc.std(ddof=1)),
                     "eer_mean": float(eer.mean()), "eer_std": float(eer.std(ddof=1)),
                     "collapse_ratio_mean": float(col.mean()) if len(col) else None,
                     "n_folds": int(len(auc))}

    out = {"passport": {"seed": C.SEED, "device": device, "python": platform.python_version(),
                        "torch": torch.__version__, "epochs": args.epochs, "cache": cache,
                        "script": "stend/runs_v2/run_operator_v2.py",
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "elapsed_sec": round(time.time() - t0, 1)},
           "n_identities_total": len(identities),
           "n_train_identities_per_fold": len(identities) - 2,
           "folds": [list(f) for f in folds],
           "configs": {n: c for n, c in configs},
           "per_config": results,
           "aggregate": agg,
           "collapse_ratio_note": ("отношение среднего межличностного расстояния на ОБУЧАЮЩИХ "
                                   "идентичностях к тому же на ОТЛОЖЕННЫХ; 1,0 означает, что "
                                   "шкала переносится, большие значения — коллапс")}
    p = os.path.join(OUTDIR, "operator_v2.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print("\nсохранено:", p)
    for n, a in agg.items():
        print(f"  {n:32s} ROC-AUC={a['roc_auc_mean']:.4f}±{a['roc_auc_std']:.4f} "
              f"EER={a['eer_mean']:.4f} сжатие×{a['collapse_ratio_mean']:.2f}")


if __name__ == "__main__":
    main()
