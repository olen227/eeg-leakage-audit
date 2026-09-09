"""Генерация download_manifest.json (ВОССТАНОВЛЕНИЕ потерянного входного артефакта).

Оригинальный download_manifest.json июльского прогона не сохранён (не создаётся
ни одним скриптом стенда, только читается). По согласованию (Этап 2):
  правило = ВСЕ файлы 6 субъектов.
  seizure    = файлы, у которых в summary есть хотя бы один приступ;
  interictal = все прочие присутствующие на диске .edf файлы субъекта.

Хеш и счётчики окон, полученные с этим манифестом, НЕ обязаны совпадать с
июльскими (a6f592d1…, 25298/616) — расхождение фиксируется в REPRODUCTION.md (И7).
Молчаливая подгонка манифеста под эталонный хеш ЗАПРЕЩЕНА (И7).
"""
import os, re, json

DATA_ROOT = os.environ.get("VKR_DATA_ROOT", os.path.join(os.getcwd(), "eeg_data"))
SUBJECTS = [f"chb{i:02d}" for i in range(1, 25)]   # все 24 идентификатора CHB-MIT


def parse_summary_files(subject):
    """Возвращает dict {filename: n_seizures} по summary субъекта."""
    path = os.path.join(DATA_ROOT, "chbmit", subject, f"{subject}-summary.txt")
    txt = open(path).read()
    res = {}
    for b in re.split(r"File Name:\s*", txt)[1:]:
        name = b.split("\n", 1)[0].strip()
        starts = re.findall(r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)", b)
        res[name] = len(starts)
    return res


def main():
    manifest = {}
    report = []
    for s in SUBJECTS:
        subj_dir = os.path.join(DATA_ROOT, "chbmit", s)
        present = sorted(f for f in os.listdir(subj_dir) if f.endswith(".edf"))
        summ = parse_summary_files(s)
        seizure, interictal, not_in_summary = [], [], []
        for f in present:
            if f in summ:
                (seizure if summ[f] > 0 else interictal).append(f)
            else:
                # файл на диске, но не упомянут в summary — метка приступа неизвестна,
                # трактуем как interictal, но фиксируем отдельно
                interictal.append(f)
                not_in_summary.append(f)
        in_summary_missing = sorted(set(summ) - set(present))
        manifest[s] = {"seizure": seizure, "interictal": interictal}
        report.append({
            "subject": s,
            "present_edf": len(present),
            "seizure_files": len(seizure),
            "interictal_files": len(interictal),
            "present_not_in_summary": not_in_summary,
            "in_summary_but_missing_on_disk": in_summary_missing,
        })

    out_path = os.path.join(DATA_ROOT, "chbmit", "download_manifest.json")
    json.dump(manifest, open(out_path, "w"), ensure_ascii=False, indent=2)
    print("saved", out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    tot_files = sum(r["present_edf"] for r in report)
    tot_sz = sum(r["seizure_files"] for r in report)
    print(f"\nИТОГО: файлов={tot_files}, seizure-файлов={tot_sz}, "
          f"interictal-файлов={tot_files - tot_sz}")


if __name__ == "__main__":
    main()
