"""Тесты ядра методики: построение фолдов, каскад протоколов, эмбарго, нормировка.

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ. Не «работает ли код вообще», а свойства, на которых держится
измерение. Если любое из них нарушится, числа работы перестанут быть измерением
вклада утечки, оставаясь при этом формально вычислимыми, — а такую поломку глазами
в логе не увидеть.

Проверяются четыре группы свойств:

  1. Объединение идентичностей. Один пациент под двумя идентификаторами обязан
     стать одной идентичностью, иначе «честное» разбиение само содержит утечку.
  2. Устройство каскада. Обучающие части вложены друг в друга, оценочное множество
     одно на все протоколы, отложенной идентичности нет в обучающей части честного
     протокола.
  3. Временное эмбарго. Окна ближе порога удаляются, дальше — остаются; окна без
     абсолютного времени сохраняются, как предписывает контракт данных.
  4. Изоляция предобработки. Статистики нормировки считаются только по переданным
     индексам и не зависят от остальной выборки.

Данные синтетические: настоящий кэш для этих проверок не нужен и не должен быть
нужен — свойства структурные.
"""
import numpy as np
import pytest

from stend.runs_v2 import protocols as P


# --------------------------------------------------------------------------- #
#                           синтетический набор
# --------------------------------------------------------------------------- #
def make_toy(n_subj=4, files_per_subj=6, win_per_file=20, seed=0):
    """Игрушечный набор в том же виде, в каком его отдаёт кэш.

    Приступные окна расставлены так, чтобы в каждом субъекте были файлы и с
    приступами, и без них: стратификация отбора отложенных файлов иначе не
    проверяется.
    """
    rng = np.random.default_rng(seed)
    subj, fid, t_abs, y = [], [], [], []
    for s in range(n_subj):
        name = f"s{s:02d}"
        t = 0.0
        for f in range(files_per_subj):
            has_sz = (f % 3 == 0)          # каждый третий файл с приступом
            for w in range(win_per_file):
                subj.append(name)
                fid.append(f"{name}_{f:02d}.edf")
                t_abs.append(t)
                y.append(1 if (has_sz and 5 <= w < 9) else 0)
                t += 4.0
            t += 3.0                        # межфайловый промежуток, как в CHB-MIT
    n = len(subj)
    X = rng.standard_normal((n, 3, 16)).astype(np.float32)
    return (X,
            np.array(y, dtype=np.int64),
            np.array(subj),
            np.array(fid),
            np.array(t_abs, dtype=np.float64))


@pytest.fixture(scope="module")
def toy():
    return make_toy()


# --------------------------------------------------------------------------- #
#                     1. объединение идентичностей
# --------------------------------------------------------------------------- #
def test_известный_дубль_объединяется_в_одну_идентичность():
    """chb01 и chb21 — один пациент; без объединения честный протокол содержит утечку."""
    ident = P.build_identity_map(["chb01", "chb02", "chb21"])
    assert ident["chb01"] == ident["chb21"], "документированный дубль не объединён"
    assert ident["chb02"] != ident["chb01"], "посторонний субъект попал в ту же идентичность"


def test_объединение_транзитивно():
    """Если a≡b и b≡c, то все трое обязаны попасть в одну идентичность."""
    ident = P.build_identity_map(["a", "b", "c", "d"],
                                 extra_groups=[("a", "b"), ("b", "c")])
    assert ident["a"] == ident["b"] == ident["c"]
    assert ident["d"] != ident["a"]


def test_метка_идентичности_не_зависит_от_порядка_субъектов():
    """Порядок входного списка не должен влиять на состав идентичностей."""
    a = P.build_identity_map(["chb21", "chb02", "chb01"])
    b = P.build_identity_map(["chb01", "chb02", "chb21"])
    assert a == b


# --------------------------------------------------------------------------- #
#                     2. устройство каскада протоколов
# --------------------------------------------------------------------------- #
def test_оценочное_множество_одно_на_все_протоколы(toy):
    """Главный инвариант работы: eval один, различается только состав обучения.

    Без этого разность метрик смешивает вклад утечки с различием оценочных
    выборок и измерением вклада утечки не является.
    """
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    fold = P.make_fold(ident, subj, fid, t_abs, "ID_s00", seed=1, y=y)
    # eval возвращается один — сам факт единственности ключа и есть инвариант
    assert "eval" in fold and isinstance(fold["eval"], np.ndarray)
    assert len(fold["eval"]) > 0
    assert set(fold["train"]) == {"P0", "P1", "P1e", "P2"}


def test_обучающие_части_вложены_по_убыванию(toy):
    """P2 ⊆ P1e ⊆ P1 ⊆ P0: каждый следующий протокол убирает канал утечки."""
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    tr = P.make_fold(ident, subj, fid, t_abs, "ID_s00", seed=1, y=y)["train"]
    assert set(tr["P2"]).issubset(set(tr["P1e"])), "P2 не вложен в P1e"
    assert set(tr["P1e"]).issubset(set(tr["P1"])), "P1e не вложен в P1"
    assert set(tr["P1"]).issubset(set(tr["P0"])), "P1 не вложен в P0"


def test_eval_не_пересекается_ни_с_одной_обучающей_частью(toy):
    """Оценочные окна не должны попадать в обучение ни одного протокола."""
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    fold = P.make_fold(ident, subj, fid, t_abs, "ID_s00", seed=1, y=y)
    ev = set(fold["eval"].tolist())
    for name, idx in fold["train"].items():
        assert not (ev & set(idx.tolist())), f"eval пересекается с обучением {name}"


def test_честный_протокол_не_содержит_отложенной_идентичности(toy):
    """В P2 пациента из оценки не должно быть ни одним окном."""
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    fold = P.make_fold(ident, subj, fid, t_abs, "ID_s01", seed=1, y=y)
    held = np.array([ident[s] for s in subj])[fold["train"]["P2"]]
    assert "ID_s01" not in set(held.tolist()), "отложенная идентичность попала в честный протокол"


def test_наивный_протокол_содержит_соседние_окна_оценочных_файлов(toy):
    """P0 обязан содержать вторую половину окон отложенных файлов — иначе измерять нечего."""
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    fold = P.make_fold(ident, subj, fid, t_abs, "ID_s00", seed=1, y=y)
    extra = set(fold["train"]["P0"]) - set(fold["train"]["P1"])
    assert extra, "P0 не отличается от P1: канал почти-дублей не воспроизведён"
    ev_files = {(subj[i], fid[i]) for i in fold["eval"]}
    assert all((subj[i], fid[i]) in ev_files for i in extra), \
        "добавленные в P0 окна взяты не из оценочных файлов"


def test_в_оценке_есть_приступные_окна(toy):
    """Стратификация отбора файлов должна оставлять в eval положительные примеры."""
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    for held in ("ID_s00", "ID_s01", "ID_s02", "ID_s03"):
        fold = P.make_fold(ident, subj, fid, t_abs, held, seed=1, y=y)
        assert y[fold["eval"]].sum() > 0, f"{held}: в оценке нет приступных окон"


def test_фолд_воспроизводится_при_том_же_seed(toy):
    """Один и тот же seed обязан давать тот же фолд: иначе числа невоспроизводимы."""
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    a = P.make_fold(ident, subj, fid, t_abs, "ID_s00", seed=7, y=y)
    b = P.make_fold(ident, subj, fid, t_abs, "ID_s00", seed=7, y=y)
    assert np.array_equal(a["eval"], b["eval"])
    for k in a["train"]:
        assert np.array_equal(a["train"][k], b["train"][k])


def test_вырождение_ступени_каскада_сообщается(toy):
    """Совпадение соседних обучающих частей обязано попадать в degenerate_steps.

    Нулевой вклад по построению неотличим от измеренного нуля, поэтому молчать
    о таком совпадении нельзя.
    """
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj.tolist())))
    # эмбарго нулевой длины ничего не удаляет -> P1 и P1e обязаны совпасть
    fold = P.make_fold(ident, subj, fid, t_abs, "ID_s00", seed=1,
                       embargo_sec=0.0, y=y)
    assert any("P1 == P1e" in s for s in fold["degenerate_steps"]), \
        "совпадение P1 и P1e не сообщено"


# --------------------------------------------------------------------------- #
#                          3. временное эмбарго
# --------------------------------------------------------------------------- #
def test_эмбарго_удаляет_близкие_по_времени_окна():
    """Окно, отстоящее от оценочного меньше порога, обязано быть удалено."""
    subj = np.array(["a"] * 6)
    t = np.array([0.0, 10.0, 20.0, 100.0, 200.0, 300.0])
    ev = np.array([2])                       # оценочное окно во времени 20 с
    tr = np.array([0, 1, 3, 4, 5])
    kept = P.embargo_mask_abs(t, subj, tr, ev, embargo_sec=30.0)
    assert 0 not in kept and 1 not in kept, "близкие окна не удалены"
    assert set(kept.tolist()) == {3, 4, 5}, "удалено лишнее"


def test_эмбарго_не_трогает_другого_субъекта():
    """Эмбарго внутрисубъектное: окна чужого пациента удаляться не должны."""
    subj = np.array(["a", "a", "b", "b"])
    t = np.array([0.0, 10.0, 0.0, 10.0])
    kept = P.embargo_mask_abs(t, subj, np.array([0, 2, 3]), np.array([1]),
                              embargo_sec=30.0)
    assert 0 not in kept, "своё близкое окно не удалено"
    assert 2 in kept and 3 in kept, "удалены окна другого субъекта"


def test_окна_без_абсолютного_времени_остаются():
    """Контракт данных объявляет NaN допустимым и предписывает эмбарго не применять.

    Наивная запись `dist >= embargo_sec` даёт на NaN ложь и удалила бы такие окна
    все: именно эта ошибка обращала вклад перекрытия пациентов в ноль по
    построению у субъекта без разметки времени.
    """
    subj = np.array(["a"] * 4)
    t = np.array([np.nan, np.nan, 0.0, 1000.0])
    kept = P.embargo_mask_abs(t, subj, np.array([0, 1, 3]), np.array([2]),
                              embargo_sec=60.0)
    assert 0 in kept and 1 in kept, "окна без абсолютного времени удалены эмбарго"
    assert 3 in kept, "далёкое окно удалено"


def test_эмбарго_при_полностью_отсутствующем_времени_не_удаляет_ничего():
    """Если времени нет ни у одного окна, эмбарго неприменимо целиком."""
    subj = np.array(["a"] * 3)
    t = np.array([np.nan, np.nan, np.nan])
    tr = np.array([0, 1])
    kept = P.embargo_mask_abs(t, subj, tr, np.array([2]), embargo_sec=60.0)
    assert np.array_equal(kept, tr)


# --------------------------------------------------------------------------- #
#                       4. изоляция предобработки
# --------------------------------------------------------------------------- #
def test_нормировка_считается_только_по_переданным_индексам():
    """mu/sd обязаны зависеть лишь от обучающей части: это измеряемый канал утечки."""
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.standard_normal((50, 2, 8)).astype(np.float32),
                        rng.standard_normal((50, 2, 8)).astype(np.float32) * 100 + 500])
    train = np.arange(50)                    # только первая, «спокойная» половина
    mu, sd = P.fold_norm_stats(X, train)
    assert np.all(np.abs(mu) < 1.0), "среднее подхватило статистику вне обучающей части"
    assert np.all(sd < 5.0), "разброс подхватил статистику вне обучающей части"


def test_нормировка_совпадает_с_прямым_расчётом():
    """Поблочный расчёт обязан совпадать с обычным по тем же окнам."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((37, 3, 11)).astype(np.float32)
    idx = np.arange(37)
    mu, sd = P.fold_norm_stats(X, idx, chunk=5)   # намеренно не кратно длине
    assert np.allclose(mu.ravel(), X.mean(axis=(0, 2)), atol=1e-5)
    assert np.allclose(sd.ravel(), X.std(axis=(0, 2)), atol=1e-4)


def test_нормировка_имеет_форму_для_вещания():
    """Форма (1, каналы, 1) нужна, чтобы (X - mu) / sd применялось поканально."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((10, 4, 6)).astype(np.float32)
    mu, sd = P.fold_norm_stats(X, np.arange(10))
    assert mu.shape == (1, 4, 1) and sd.shape == (1, 4, 1)
    assert np.all(sd > 0), "нулевой разброс привёл бы к делению на ноль"


# --------------------------------------------------------------------------- #
#                 групповые фолды (несколько идентичностей в одном)
# --------------------------------------------------------------------------- #
def test_групповой_фолд_откладывает_все_идентичности_группы(toy):
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj)))
    fold = P.make_fold(ident, subj, fid, t_abs, ["ID_s00", "ID_s02"], seed=1, y=y)
    ev_ident = {ident[s] for s in subj[fold["eval"]]}
    assert ev_ident == {"ID_s00", "ID_s02"}
    tr_ident = {ident[s] for s in subj[fold["train"]["P2"]]}
    assert tr_ident == {"ID_s01", "ID_s03"}
    assert fold["held_identities"] == ["ID_s00", "ID_s02"]


def test_групповой_фолд_каскад_вложен_и_eval_общий(toy):
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj)))
    fold = P.make_fold(ident, subj, fid, t_abs, {"ID_s01", "ID_s03"}, seed=1, y=y)
    tr = fold["train"]
    assert set(tr["P1"]) <= set(tr["P0"])
    assert set(tr["P1e"]) <= set(tr["P1"])
    assert set(tr["P2"]) <= set(tr["P1e"])
    for pr in tr:
        assert not set(tr[pr]) & set(fold["eval"])


def test_группа_из_одной_идентичности_совпадает_со_строкой(toy):
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj)))
    a = P.make_fold(ident, subj, fid, t_abs, "ID_s01", seed=1, y=y)
    b = P.make_fold(ident, subj, fid, t_abs, ["ID_s01"], seed=1, y=y)
    assert np.array_equal(a["eval"], b["eval"])
    for pr in a["train"]:
        assert np.array_equal(a["train"][pr], b["train"][pr])


def test_отбор_файлов_в_группе_не_зависит_от_порядка_перечисления(toy):
    X, y, subj, fid, t_abs = toy
    ident = P.build_identity_map(sorted(set(subj)))
    a = P.make_fold(ident, subj, fid, t_abs, ["ID_s02", "ID_s00"], seed=1, y=y)
    b = P.make_fold(ident, subj, fid, t_abs, ["ID_s00", "ID_s02"], seed=1, y=y)
    assert np.array_equal(a["eval"], b["eval"])


def test_нормировка_из_float16_кэша_совпадает_с_float32():
    rng = np.random.default_rng(3)
    X32 = (rng.standard_normal((50, 3, 16)) * 30).astype(np.float32)
    X16 = X32.astype(np.float16)
    idx = np.arange(50)
    mu32, sd32 = P.fold_norm_stats(X32, idx)
    mu16, sd16 = P.fold_norm_stats(X16, idx)
    assert mu16.dtype == np.float32 and sd16.dtype == np.float32
    assert np.allclose(mu16, mu32, atol=1e-2)
    assert np.allclose(sd16, sd32, rtol=1e-3)


# --------------------------------------------------------------------------- #
#          событийный счёт: записи не должны сопоставляться между собой
# --------------------------------------------------------------------------- #
def test_события_разных_файлов_не_сопоставляются():
    from stend.runs_v2.run_event_continuous import score_protocol
    y = np.zeros(200, int); y[50:60] = 1; y[150:155] = 1        # приступ в каждом файле
    pred = np.zeros(200, int); pred[52:58] = 1; pred[100:104] = 1  # попадание в первом, ложная тревога во втором
    files = np.array(["a|f1"] * 100 + ["a|f2"] * 100)
    wtime = np.concatenate([np.arange(100) * 4.0, np.arange(100) * 4.0])  # время от начала СВОЕГО файла
    sc, _ = score_protocol(y, pred, files, wtime, 1, 0)
    e = sc["event_szcore_smoothed"]
    assert e["detected"] == 1 and e["false_alarms"] == 1 and e["sensitivity"] == 0.5
