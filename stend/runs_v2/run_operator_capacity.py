"""Диагностика оператора: зависит ли качество от ЧИСЛА обучающих идентичностей.

ЧТО ПРОВЕРЯЕМ. Оператор на невиданных пациентах CHB-MIT даёт ROC-AUC 0,71 при обучении
всего на 3 идентичностях. Открыт вопрос, который на клинических данных неразрешим:
это предел АРХИТЕКТУРЫ или следствие НЕХВАТКИ ЛЮДЕЙ в обучении?

КАК ПРОВЕРЯЕМ. Набор eegmmidb содержит 109 испытуемых — в пять раз больше, чем все
клинические идентичности вместе. Обучаем один и тот же энкодер на возрастающем числе
людей (4, 8, 16, 32, 64, 96) и смотрим на кривую качества на отложенных людях.
Если кривая растёт и не вышла на насыщение — предел не архитектурный.

РАМКА ПРИМЕНЕНИЯ. Это ДИАГНОСТИЧЕСКИЙ эксперимент, а не часть основного результата ВКР.
Тема работы — обнаружение приступов с предотвращением утечки; оператор в ней служебный
инструмент для группировки записей одного пациента. eegmmidb — здоровые испытуемые
в задаче моторного воображения, приступов там нет и быть не должно: оператору нужны
РАЗНЫЕ ЛЮДИ, а не приступы. Итоговый оператор работы обучается и оценивается на
клинических данных (CHB-MIT, Siena); настоящий прогон отвечает только на вопрос
о причине его ограничений.

Дополнительно проверяется перенос: оператор, обученный на 109 здоровых испытуемых,
оценивается на пациентах CHB-MIT. Ожидание умеренное — мы уже установили, что перенос
между наборами у этой конструкции не работает (раздел 3.13 FINDINGS).

Данные: только записи покоя R01 (глаза открыты) и R02 (глаза закрыты) — без моторных
задач, это ближе всего к фоновой ЭЭГ клинического мониторинга.

Выход: outputs/track_b/operator_capacity.json
"""
import os, re, sys, json, time, platform, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
import torch
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from stend.vkr_eeg import config as C
from stend.runs_v2.run_operator_cv import train_encoder_mm, eer_of, get_device

mne.set_log_level("ERROR")

OUTDIR = os.path.join("outputs", "track_b")
MMIDB = os.path.join(C.DATA_ROOT, "eegmmidb")
V2 = os.environ.get("VKR_CACHE_ROOT", os.path.join("outputs", "v2"))

# те же 18 канонических отведений, вычисленные из монополярных каналов 10-10.
# eegmmidb использует новую номенклатуру (T7/T8/P7/P8), совпадающую с CHB-MIT.
BIPOLAR = {
    "FP1-F7": ("FP1", "F7"), "F7-T7": ("F7", "T7"), "T7-P7": ("T7", "P7"), "P7-O1": ("P7", "O1"),
    "FP1-F3": ("FP1", "F3"), "F3-C3": ("F3", "C3"), "C3-P3": ("C3", "P3"), "P3-O1": ("P3", "O1"),
    "FP2-F4": ("FP2", "F4"), "F4-C4": ("F4", "C4"), "C4-P4": ("C4", "P4"), "P4-O2": ("P4", "O2"),
    "FP2-F8": ("FP2", "F8"), "F8-T8": ("F8", "T8"), "T8-P8": ("T8", "P8"), "P8-O2": ("P8", "O2"),
    "FZ-CZ": ("FZ", "CZ"), "CZ-PZ": ("CZ", "PZ"),
}
WIN = int(C.WIN_SEC * C.SFREQ)


def load_subject(sdir, max_win):
    """Окна одного испытуемого: 18 отведений, 256 Гц, полоса 0,5-40, режекторный 60."""
    out = []
    for f in sorted(os.listdir(sdir)):
        if not f.endswith(".edf"):
            continue
        try:
            r = mne.io.read_raw_edf(os.path.join(sdir, f), preload=True, verbose="ERROR")
        except Exception:
            continue
        ren = {ch: ch.upper().replace(".", "").replace(" ", "") for ch in r.ch_names}
        r.rename_channels(ren)
        idx = {n: i for i, n in enumerate(r.ch_names)}
        if any(a not in idx or b not in idx for a, b in BIPOLAR.values()):
            continue
        d = r.get_data(); sf = float(r.info["sfreq"])
        bip = np.stack([d[idx[a]] - d[idx[b]]
                        for a, b in (BIPOLAR[c] for c in C.CANONICAL_CHANNELS)])
        if abs(sf - C.SFREQ) > 1:
            g = np.gcd(int(round(sf)), C.SFREQ)
            bip = resample_poly(bip, up=C.SFREQ // g, down=int(round(sf)) // g, axis=1)
        info = mne.create_info(C.CANONICAL_CHANNELS, C.SFREQ, ch_types="eeg")
        ra = mne.io.RawArray(bip, info, verbose="ERROR")
        ra.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
        try:
            ra.notch_filter(C.NOTCH, verbose="ERROR")
        except Exception:
            pass
        a = ra.get_data()
        n = a.shape[1] // WIN
        if n:
            out.append(np.stack([a[:, i * WIN:(i + 1) * WIN] for i in range(n)]).astype(np.float32))
        del r, ra, d, a
    if not out:
        return None
    W = np.concatenate(out)
    return W[:max_win] if len(W) > max_win else W


def pair_auc(emb, lab, n_pairs, rng):
    n = len(emb)
    a = rng.integers(0, n, n_pairs); b = rng.integers(0, n, n_pairs)
    k = a != b; a, b = a[k], b[k]
    d = np.linalg.norm(emb[a] - emb[b], axis=1)
    same = (lab[a] == lab[b]).astype(int)
    if same.sum() in (0, len(same)):
        return None, None
    auc = float(roc_auc_score(same, -d))
    eer, _ = eer_of(same, d)
    return auc, eer


@torch.no_grad()
def embed(enc, arr, mu, sd, device, bs=512):
    enc.eval()
    o = []
    for k in range(0, len(arr), bs):
        xb = torch.from_numpy(((arr[k:k + bs] - mu) / sd).astype(np.float32))
        o.append(enc(xb.to(device)).cpu().numpy())
    return np.concatenate(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.OP_EPOCHS)
    ap.add_argument("--max-win", type=int, default=150, help="окон на испытуемого")
    ap.add_argument("--n-eval", type=int, default=12, help="сколько людей в оценке")
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--sizes", type=str, default="4,8,16,32,64,96")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = get_device(); t0 = time.time()
    rng = np.random.default_rng(C.SEED)

    subs = sorted(d for d in os.listdir(MMIDB) if re.fullmatch(r"S\d{3}", d))
    print(f"испытуемых в eegmmidb: {len(subs)}", flush=True)
    print("читаю записи (это займёт несколько минут) ...", flush=True)
    data, keep = {}, []
    for i, s in enumerate(subs, 1):
        W = load_subject(os.path.join(MMIDB, s), args.max_win)
        if W is not None and len(W) >= 20:
            data[s] = W; keep.append(s)
        if i % 25 == 0:
            print(f"  {i}/{len(subs)}, пригодных {len(keep)}", flush=True)
    print(f"пригодных испытуемых: {len(keep)}, окон на человека ~{args.max_win}", flush=True)

    perm = rng.permutation(len(keep))
    ev_subj = [keep[i] for i in perm[:args.n_eval]]
    pool = [keep[i] for i in perm[args.n_eval:]]
    Xev = np.concatenate([data[s] for s in ev_subj])
    lev = np.concatenate([[s] * len(data[s]) for s in ev_subj])
    print(f"оценка: {len(ev_subj)} невиданных людей, {len(Xev)} окон", flush=True)

    sizes = [int(x) for x in args.sizes.split(",") if int(x) <= len(pool)]
    curve = []
    for k in sizes:
        tr_subj = pool[:k]
        Xtr = np.concatenate([data[s] for s in tr_subj])
        ltr = np.concatenate([[s] * len(data[s]) for s in tr_subj])
        mu = Xtr.mean(axis=(0, 2)).reshape(1, -1, 1).astype(np.float32)
        sd = (Xtr.std(axis=(0, 2)) + 1e-7).reshape(1, -1, 1).astype(np.float32)
        enc, hist = train_encoder_mm(Xtr, np.arange(len(Xtr)), ltr, mu, sd,
                                     device, args.epochs, C.SEED)
        auc, eer = pair_auc(embed(enc, Xev, mu, sd, device), lev, args.n_pairs, rng)
        curve.append({"n_train_identities": k, "n_train_windows": int(len(Xtr)),
                      "roc_auc": auc, "eer": eer,
                      "loss_first": hist[0], "loss_last": hist[-1]})
        print(f"  обучено на {k:3d} людях -> ROC-AUC={auc:.4f} EER={eer:.4f} "
              f"(потеря {hist[0]:.3f}->{hist[-1]:.3f})", flush=True)
        if k == max(sizes):
            enc_full, mu_full, sd_full = enc, mu, sd

    # перенос на клинические данные CHB-MIT
    transfer = None
    try:
        Xc = np.load(os.path.join(V2, "X_raw.npy"), mmap_mode="r")
        Mc = np.load(os.path.join(V2, "meta.npz"), allow_pickle=True)
        sc = Mc["subj"]
        sel = np.sort(rng.choice(len(sc), min(12000, len(sc)), replace=False))
        emb = embed(enc_full, np.asarray(Xc[sel]), mu_full, sd_full, device)
        auc_t, eer_t = pair_auc(emb, sc[sel], args.n_pairs, rng)
        transfer = {"roc_auc": auc_t, "eer": eer_t, "n_windows": int(len(sel)),
                    "note": ("оператор обучен на здоровых испытуемых eegmmidb, "
                             "оценён на пациентах CHB-MIT без дообучения")}
        print(f"  перенос на CHB-MIT: ROC-AUC={auc_t:.4f} EER={eer_t:.4f}", flush=True)
    except Exception as e:
        transfer = {"error": str(e)}

    out = {
        "passport": {"seed": C.SEED, "device": device, "python": platform.python_version(),
                     "torch": torch.__version__, "mne": mne.__version__,
                     "script": "stend/runs_v2/run_operator_capacity.py",
                     "epochs": args.epochs, "windows_per_subject": args.max_win,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "elapsed_sec": round(time.time() - t0, 1),
                     "data": MMIDB},
        "scope": ("ДИАГНОСТИЧЕСКИЙ эксперимент, не входит в основной результат ВКР: "
                  "отвечает на вопрос, ограничен ли оператор архитектурой или числом "
                  "обучающих идентичностей. Итоговый оператор работы обучается на "
                  "клинических данных."),
        "n_subjects_available": len(keep),
        "eval_subjects": ev_subj,
        "curve": curve,
        "transfer_to_chbmit": transfer,
        "reference_clinical": {
            "roc_auc_chbmit_unseen_identities": 0.7123,
            "n_train_identities": 3,
            "source": "outputs/track_b/operator_subject_cv.json",
            "note": "для сравнения: клинический оператор при 3 обучающих идентичностях"},
    }
    p = os.path.join(OUTDIR, "operator_capacity.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print("\nсохранено:", p)


if __name__ == "__main__":
    main()
