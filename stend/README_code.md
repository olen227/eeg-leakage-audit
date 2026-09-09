# Код экспериментального стенда Главы 3

## Структура
```
stend/
  vkr_eeg/            # переиспользуемые модули стенда (импортируются прогонами)
    config.py         # SEED=20260706, SFREQ, окна, гиперпараметры, 18 каналов
    data.py           # загрузка EDF, нормализация каналов (T8-P8 суффикс), парсинг summary, окна, sha256
    splitter.py       # window_split (P0), subject_split (P2), embargo_mask
    operator.py       # Encoder (1D-CNN, emb=64), contrastive_loss, make_pairs, train_encoder, embed
    metrics.py        # roc_auc, eer, calibrate_threshold (форм. 2.10), f1_binary
    detector.py       # Detector (1D-CNN), train_detector с балансировкой классов, predict
    dataset.py        # fit_train_stats (только train), build с провенансом, data_hash
    __init__.py       # NUMBA_DISABLE_JIT=1 (обход JIT-кэша mne в песочнице)
  runs/               # код прогонов по фазам (восстановлен дословно из lineage результатов)
    phase0_dataset.py    # сборка master-датасета
    phase1_operator.py   # оператор: ROC-AUC=0.928, EER=0.116, τ*=0.506
    phase2_duplicates.py # дубли: доля отловленных=0.701
    phase3_gap.py        # разрыв F1: P0=0.811, P2=0.328 (identity-aware LOIO)
    phase4_decomp.py     # разложение: Δ_дубли=0.407, Δ_утечка=0.076, бутстрэп-ДИ
    phase5_siena.py      # перенос: zero-shot=0.523, in-domain=0.812
    phase6_ablation.py   # абляция протокола, матрицы ошибок, важность каналов/полос
```

## Воспроизведение
1. Окружение: Python 3.12; pip install numpy scipy pandas scikit-learn matplotlib mne pyedflib torch tqdm pyyaml
2. Данные CHB-MIT: скачать по манифесту (6 субъектов chb01,02,03,05,08,21) с physionet.org в eeg_data/chbmit/
3. Прогоны в порядке: phase0 → phase1 → ... → phase6. phase0 создаёт outputs/master_dataset.npz (кэш), который читают остальные.
4. Все прогоны фиксируют seed=20260706; data SHA-256 = a6f592d1148db987daa393ad199d4bd88f9ef0e36219950c3a059ac3d9c3a108.

## Примечания
- Код прогонов извлечён из происхождения (lineage) артефактов-результатов — это фактически выполненные ячейки, не реконструкция.
- Обучение на CPU (без GPU): компактные модели, подмножество субъектов — см. ограничения в §3.3.
- Гарантия честности чисел: все метрики — из реальных прогонов (инвариант 9).
