"""Сборка мастер-датасета окон с провенансом (субъект, файл, время окна, метка приступа).
Предобработка фитится ТОЛЬКО на train-субъектах (изоляция предобработки, инвариант 2)."""
import os
import numpy as np
from . import data, config as C

def fit_train_stats(train_subjects, manifest):
    sums=sqs=None; cnt=0
    for s in sorted(train_subjects):
        for fn in manifest[s]["seizure"]+manifest[s]["interictal"]:
            raw = data.load_edf(s, fn)
            if raw is None: continue
            raw.filter(C.BANDPASS[0], C.BANDPASS[1], verbose="ERROR")
            # Режекция сетевой наводки избыточна: полосовой 0,5-40 Гц уже подавляет
            # и 50, и 60 Гц, поэтому отказ режекторного фильтра на данные не влияет.
            try: raw.notch_filter(C.NOTCH, verbose="ERROR")
            except Exception: pass
            d = raw.get_data()
            sums = d.sum(1) if sums is None else sums+d.sum(1)
            sqs  = (d**2).sum(1) if sqs is None else sqs+(d**2).sum(1)
            cnt += d.shape[1]
            del raw, d
    mu=(sums/cnt).reshape(-1,1); sd=(np.sqrt(sqs/cnt-(sums/cnt)**2)).reshape(-1,1)+1e-7
    return {"mu":mu,"sd":sd,"n_samples":int(cnt)}

def build(subjects, train_stats, manifest):
    Xall=[]; ysz=[]; subj=[]; fid=[]; wt=[]
    for s in subjects:
        info=data.parse_summary(s)
        for fn in manifest[s]["seizure"]+manifest[s]["interictal"]:
            out=data.window_file(s, fn, info.get(fn,[]), train_stats)
            if out is None: continue
            Xf,yf,tf=out
            Xall.append(Xf); ysz.append(yf)
            subj+=[s]*len(Xf); fid+=[fn]*len(Xf); wt.append(tf)
    return (np.concatenate(Xall), np.concatenate(ysz),
            np.array(subj), np.array(fid), np.concatenate(wt))

def data_hash(subjects, manifest):
    paths=[]
    for s in subjects:
        for fn in manifest[s]["seizure"]+manifest[s]["interictal"]:
            paths.append(os.path.join(C.DATA_ROOT,"chbmit",s,fn))
    return data.sha256_of_files([p for p in paths if os.path.exists(p)])
