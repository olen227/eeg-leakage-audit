"""Проштамповать артефакты хешем среза данных (замечание верификатора №5).

Требование инструкции ВКР: каждое число прослеживаемо до среза данных через хеш.
В паспортах артефактов поля data_sha256 не было — оно добавляется здесь.

Хеш ВЫЧИСЛЯЕТСЯ заново по манифесту (не подставляется из отчётов) той же функцией
dataset.data_hash, которой считался при сборке датасета. Факт постфактум-штамповки
фиксируется в самом паспорте (provenance_stamped_by / provenance_stamped_utc),
чтобы происхождение поля было прозрачным.
"""
import os, sys, json, time, glob, platform

sys.path.insert(0, os.getcwd())
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from stend.vkr_eeg import dataset, config as C

OUTDIR = os.path.join("outputs", "track_b")
SUBJECTS = ["chb01", "chb02", "chb03", "chb05", "chb08", "chb21"]


def main():
    man_path = os.path.join(C.DATA_ROOT, "chbmit", "download_manifest.json")
    man = json.load(open(man_path))
    n_files = sum(len(man[s]["seizure"]) + len(man[s]["interictal"]) for s in SUBJECTS)
    print(f"манифест: {man_path}\nфайлов в срезе: {n_files}", flush=True)
    print("вычисляю data_sha256 (чтение всех EDF, это долго) ...", flush=True)
    t0 = time.time()
    h = dataset.data_hash(SUBJECTS, man)
    print(f"data_sha256 = {h}   [{time.time()-t0:.0f} с]", flush=True)

    slice_desc = (f"все файлы {len(SUBJECTS)} субъектов CHB-MIT "
                  f"({', '.join(SUBJECTS)}), {n_files} файлов")
    stamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    changed = []
    for p in sorted(glob.glob(os.path.join(OUTDIR, "*.json"))):
        if os.path.basename(p).startswith("_"):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  ПРОПУСК {p}: {e}", flush=True)
            continue
        if not isinstance(d, dict):
            continue
        pas = d.setdefault("passport", {})
        pas["data_sha256"] = h
        pas.setdefault("data_slice", slice_desc)
        pas.setdefault("manifest", man_path)
        pas.setdefault("numpy", __import__("numpy").__version__)
        pas.setdefault("python", platform.python_version())
        pas["provenance_stamped_by"] = "stend/runs_v2/add_provenance.py"
        pas["provenance_stamped_utc"] = stamp_utc
        json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
        changed.append(os.path.basename(p))
        print(f"  проштампован: {os.path.basename(p)}", flush=True)

    summary = {
        "data_sha256": h,
        "data_slice": slice_desc,
        "n_files": n_files,
        "subjects": SUBJECTS,
        "manifest": man_path,
        "hash_function": "stend/vkr_eeg/dataset.data_hash -> data.sha256_of_files "
                         "(sha256 по именам и сырым байтам файлов манифеста, отсортированным)",
        "stamped_artifacts": changed,
        "stamped_utc": stamp_utc,
        "seed": C.SEED,
    }
    sp = os.path.join(OUTDIR, "PROVENANCE.json")
    json.dump(summary, open(sp, "w"), ensure_ascii=False, indent=2)
    print(f"\nсохранено: {sp}\nпроштамповано артефактов: {len(changed)}")


if __name__ == "__main__":
    main()
