"""Метрический оператор идентификации субъекта — субъектная кросс-валидация (Б4)
и честная проверка обнаружения скрытого дубля chb01≡chb21.

ОТЛИЧИЕ ОТ ИЮЛЬСКОГО КОДА
-------------------------
В phase1_operator.py оператор оценивался file-holdout'ом: субъекты оценочных пар
присутствовали и в обучении. Такая оценка измеряет запоминание известных субъектов,
а не способность различать НОВЫХ. Здесь оценка проводится по фолдам идентичностей:
в каждом фолде две идентичности полностью исключены из обучения, и все оценочные
пары строятся только из них.

Проверка дубля: chb01 и chb21 (один пациент, интервал 1,5 года) исключаются из
обучения целиком, после чего измеряется, отличимо ли расстояние между их окнами
от расстояния между окнами разных пациентов.

Выход: outputs/track_b/operator_subject_cv.json
"""
import os, sys, json, time, platform, itertools, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import torch
from sklearn.metrics import roc_auc_score, roc_curve

from stend.vkr_eeg import config as C
from stend.vkr_eeg.operator import Encoder, contrastive_loss
from stend.runs_v2 import protocols as P

V2 = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))
OUTDIR = os.path.join("outputs", "track_b")


def get_device():
    want = os.environ.get("VKR_DEVICE", "auto")
    return ("mps" if torch.backends.mps.is_available() else "cpu") if want == "auto" else want


def load_batch(X, idx, mu, sd):
    return torch.from_numpy(((X[np.sort(idx)] - mu) / sd).astype(np.float32))


def train_encoder_mm(X, train_idx, subj, mu, sd, device, epochs, seed, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(X.shape[1]).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=C.OP_LR)
    lab = subj[train_idx]
    by = {s: np.where(lab == s)[0] for s in np.unique(lab)}
    subs = list(by)
    history = []
    for ep in range(epochs):
        enc.train()
        tot = 0.0; nb = 0
        for _ in range(20):                      # 20 батчей пар на эпоху (как в исходном коде)
            A, B, Y = [], [], []
            for _ in range(C.OP_BATCH):
                if rng.random() < 0.5:
                    s = subs[rng.integers(len(subs))]
                    if len(by[s]) < 2: continue
                    a, b = rng.choice(by[s], 2, replace=False); Y.append(1.0)
                else:
                    i1, i2 = rng.choice(len(subs), 2, replace=False)
                    a = rng.choice(by[subs[i1]]); b = rng.choice(by[subs[i2]]); Y.append(0.0)
                A.append(train_idx[a]); B.append(train_idx[b])
            xa = torch.from_numpy(((X[np.array(A)] - mu) / sd).astype(np.float32)).to(device)
            xb = torch.from_numpy(((X[np.array(B)] - mu) / sd).astype(np.float32)).to(device)
            yv = torch.tensor(Y, dtype=torch.float32, device=device)
            loss = contrastive_loss(enc(xa), enc(xb), yv)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        history.append(tot / max(nb, 1))
        if log: log(f"      эпоха {ep+1}/{epochs} loss={history[-1]:.4f}")
    return enc, history


@torch.no_grad()
def embed_mm(enc, X, idx, mu, sd, device, bs=512):
    enc.eval()
    out = []
    for k in range(0, len(idx), bs):
        b = idx[k:k + bs]
        out.append(enc(torch.from_numpy(((X[b] - mu) / sd).astype(np.float32)).to(device)).cpu().numpy())
    return np.concatenate(out)


def eer_of(same, dist):
    fpr, tpr, thr = roc_curve(same, -dist)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(-thr[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.OP_EPOCHS)
    ap.add_argument("--per-subject", type=int, default=3000, help="окон на субъекта для оценки")
    ap.add_argument("--n-pairs", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t_start = time.time()

    X = np.load(os.path.join(V2, "X_raw.npy"), mmap_mode="r")
    M = np.load(os.path.join(V2, "meta.npz"), allow_pickle=True)
    subj = M["subj"]
    subjects = sorted(set(subj.tolist()))
    ident_of = P.build_identity_map(subjects)
    identities = sorted(set(ident_of.values()))
    ident_arr = np.array([ident_of[s] for s in subj])

    print(f"устройство: {device} | идентичностей: {len(identities)}", flush=True)

    rng = np.random.default_rng(C.SEED)
    folds = []
    # каждый фолд: 2 отложенные идентичности (нужны минимум две, чтобы строить
    # и однопациентные, и межпациентные пары ТОЛЬКО из невиданных данных)
    combos = list(itertools.combinations(identities, 2))
    for a, b in combos:
        folds.append((a, b))

    results = []
    for (ia, ib) in folds:
        held = {ia, ib}
        tr_idx = np.where(~np.isin(ident_arr, list(held)))[0]
        ev_mask = np.isin(ident_arr, list(held))
        print(f"\n=== фолд: отложены {ia}, {ib} ===", flush=True)

        mu, sd = P.fold_norm_stats(X, tr_idx)
        enc, hist = train_encoder_mm(X, tr_idx, subj, mu, sd, device, args.epochs, C.SEED)

        # оценочные окна: до per-subject на каждого СУБЪЕКТА отложенных идентичностей
        ev_subjects = sorted(set(subj[ev_mask].tolist()))
        sel = []
        for s in ev_subjects:
            w = np.where(subj == s)[0]
            sel.append(rng.choice(w, min(args.per_subject, len(w)), replace=False))
        sel = np.sort(np.concatenate(sel))
        emb = embed_mm(enc, X, sel, mu, sd, device)
        lab_s = subj[sel]
        lab_i = ident_arr[sel]

        # пары: same=1, если ОДНА И ТА ЖЕ ИДЕНТИЧНОСТЬ (не субъект!)
        n = len(sel)
        ia_ = rng.integers(0, n, args.n_pairs); ib_ = rng.integers(0, n, args.n_pairs)
        keep = ia_ != ib_
        ia_, ib_ = ia_[keep], ib_[keep]
        dist = np.linalg.norm(emb[ia_] - emb[ib_], axis=1)
        same_ident = (lab_i[ia_] == lab_i[ib_]).astype(int)
        same_subj = (lab_s[ia_] == lab_s[ib_]).astype(int)

        auc_i = float(roc_auc_score(same_ident, -dist))
        eer_i, tau_i = eer_of(same_ident, dist)
        auc_s = float(roc_auc_score(same_subj, -dist))
        eer_s, tau_s = eer_of(same_subj, dist)

        r = {"held": [ia, ib], "eval_subjects": ev_subjects,
             "n_eval_windows": int(n), "n_pairs": int(len(dist)),
             "roc_auc_identity": auc_i, "eer_identity": eer_i, "tau_identity": tau_i,
             "roc_auc_subject": auc_s, "eer_subject": eer_s, "tau_subject": tau_s,
             "loss_first": hist[0], "loss_last": hist[-1]}

        # если отложена пара-дубль — измерить её отдельно
        if "chb01" in ev_subjects and "chb21" in ev_subjects:
            m01 = lab_s[ia_] == "chb01"; m21 = lab_s[ib_] == "chb21"
            cross = (m01 & m21) | ((lab_s[ia_] == "chb21") & (lab_s[ib_] == "chb01"))
            other_cross = (lab_i[ia_] != lab_i[ib_])
            within = (lab_s[ia_] == lab_s[ib_])
            r["duplicate_probe"] = {
                "n_pairs_chb01_chb21": int(cross.sum()),
                "dist_chb01_chb21_mean": float(dist[cross].mean()) if cross.any() else None,
                "dist_within_subject_mean": float(dist[within].mean()) if within.any() else None,
                "dist_different_identity_mean": float(dist[other_cross].mean()) if other_cross.any() else None,
                "auc_chb01chb21_vs_different": (
                    float(roc_auc_score(np.r_[np.ones(cross.sum()), np.zeros(other_cross.sum())],
                                        -np.r_[dist[cross], dist[other_cross]]))
                    if cross.any() and other_cross.any() else None),
                "note": ("auc≈0,5 => пара chb01/chb21 неотличима от пар РАЗНЫХ пациентов, "
                         "то есть скрытый дубль оператором НЕ обнаруживается"),
            }
        print(f"  ROC-AUC(идентичность)={auc_i:.4f} EER={eer_i:.4f} | "
              f"ROC-AUC(субъект)={auc_s:.4f} EER={eer_s:.4f} | "
              f"loss {hist[0]:.3f}->{hist[-1]:.3f}", flush=True)
        if "duplicate_probe" in r:
            dp = r["duplicate_probe"]
            print(f"  дубль chb01/chb21: пар={dp['n_pairs_chb01_chb21']} "
                  f"d(01,21)={dp['dist_chb01_chb21_mean']:.4f} "
                  f"d(внутри)={dp['dist_within_subject_mean']:.4f} "
                  f"d(разные)={dp['dist_different_identity_mean']:.4f} "
                  f"AUC={dp['auc_chb01chb21_vs_different']}", flush=True)
        results.append(r)

    agg = {}
    for k in ["roc_auc_identity", "eer_identity", "roc_auc_subject", "eer_subject"]:
        v = np.array([r[k] for r in results], dtype=float)
        agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                  "min": float(v.min()), "max": float(v.max())}

    out = {
        "passport": {"seed": C.SEED, "device": device, "python": platform.python_version(),
                     "torch": torch.__version__, "script": "stend/runs_v2/run_operator_cv.py",
                     "epochs": args.epochs, "data_slice": "все файлы 6 субъектов CHB-MIT",
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "elapsed_sec": round(time.time() - t_start, 1)},
        "scheme": ("кросс-валидация по идентичностям: в каждом фолде ДВЕ идентичности "
                   "полностью исключены из обучения, оценочные пары строятся только из них"),
        "identity_map": ident_of,
        "folds": results,
        "aggregate": agg,
    }
    path = os.path.join(OUTDIR, "operator_subject_cv.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"\nсохранено: {path}")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
