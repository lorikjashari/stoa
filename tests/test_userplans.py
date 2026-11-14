"""Fine-tuning on the user's own plans: label I/O, sticky splits, data mixing, val-only tuning. Offline, no weights."""
import json

import pytest

from stoa_qa import cubicasa_userplans as U
from stoa_qa.cubicasa_rules import CLASSES, QUESTION_CLASSES

DOOR, WINDOW, SHOWER, BATHTUB, FLOOR = (CLASSES.index(c) for c in ("door", "window", "shower", "bathtub", "floor"))


# ---------------------------------------------------------------- label I/O

def test_yolo_round_trip_keeps_boxes_and_class_ids():
    boxes = [[DOOR, 0.9, 10, 20, 50, 80], [FLOOR, 0.6, 0, 0, 400, 300], [len(CLASSES) - 1, 0.5, 100.5, 100, 120, 140.5]]
    lines = U.to_yolo_lines(boxes, 400, 300)
    assert [int(line.split()[0]) for line in lines] == [DOOR, FLOOR, len(CLASSES) - 1]
    back = U.from_yolo_text("\n".join(lines), 400, 300)
    for b, r in zip(boxes, back):
        assert r[0] == b[0] and r[1] == 1.0
        assert r[2:] == pytest.approx(b[2:], abs=0.11)


def test_yolo_lines_clip_drop_slivers_and_accept_unscored_boxes():
    lines = U.to_yolo_lines([[WINDOW, -10, 5, 30, 25], [DOOR, 0.9, 50, 50, 51, 90], [DOOR, 120, 90, 80, 60]], 100, 100)
    assert len(lines) == 2  # the 1 px wide door is dropped
    c, cx, cy, w, h = map(float, lines[0].split())
    assert (c, cx, w) == (WINDOW, 0.15, 0.3)  # clipped to x 0..30
    _, cx, cy, w, h = map(float, lines[1].split())
    assert (cx, cy, w, h) == pytest.approx((0.9, 0.75, 0.2, 0.3))  # corners given swapped, x clipped at 100


def test_yolo_rejects_bad_class_and_malformed_lines():
    with pytest.raises(ValueError):
        U.to_yolo_lines([[len(CLASSES), 0, 0, 10, 10]], 100, 100)
    with pytest.raises(ValueError, match="line 2"):
        U.from_yolo_text("0 0.5 0.5 0.1 0.1\n99 0.5 0.5 0.1 0.1", 100, 100)
    with pytest.raises(ValueError, match="line 1"):
        U.from_yolo_text("door 0.5 0.5 0.1 0.1", 100, 100)
    assert U.from_yolo_text("\n\n", 100, 100) == []


def test_write_labels_and_label_counts(tmp_path):
    f = tmp_path / "labels" / "p.txt"
    n = U.write_labels(f, [[DOOR, 1, 1, 20, 20], [DOOR, 30, 30, 50, 50], [SHOWER, 1, 1, 30, 30], [BATHTUB, 40, 40, 90, 60]], 100, 100)
    assert n == 4 and f.read_text().endswith("\n")
    c = U.label_counts(f.read_text())
    assert c["doors"] == 2 and c["baths"] == 2 and c["windows"] == 0
    assert set(c) == set(QUESTION_CLASSES)
    U.write_labels(f, [], 100, 100)
    assert f.read_text() == "" and U.label_counts("")["doors"] == 0


def test_classes_txt_matches_class_ids():
    assert U.classes_txt().splitlines() == CLASSES


def test_labeler_page_uses_the_same_classes():
    from pathlib import Path

    html = (Path(U.__file__).resolve().parent.parent / "tools" / "plan_labeler.html").read_text(encoding="utf-8")
    assert "const CLASSES = " + json.dumps(CLASSES).replace(", ", ",") + ";" in html


# ---------------------------------------------------------------- pseudo-labels

def test_pseudo_label_applies_per_question_thresholds_and_overlap_cut():
    thr = {q: 0.5 for q in QUESTION_CLASSES} | {"windows": 0.3}
    iou = {"baths": 0.5}
    boxes = [[DOOR, 0.45, 0, 0, 10, 10],            # under the door threshold: dropped
             [WINDOW, 0.35, 20, 0, 40, 4],          # windows have a lower threshold: kept
             [SHOWER, 0.9, 50, 50, 70, 70],
             [BATHTUB, 0.8, 51, 51, 70, 70],        # same fixture as the shower (IoU > 0.5): suppressed
             [FLOOR, 0.99, 0, 0, 100, 100]]
    kept = U.pseudo_label(boxes, thr, iou)
    assert sorted(b[0] for b in kept) == sorted([WINDOW, SHOWER, FLOOR])


def test_load_picks_reads_the_committed_file():
    from pathlib import Path

    thr, iou = U.load_picks(Path(U.__file__).with_name("cubicasa_picks.json"))
    assert set(thr) == set(QUESTION_CLASSES)


# ---------------------------------------------------------------- splits

def test_splits_are_deterministic_and_have_val_and_test():
    names = [f"p{i:02d}" for i in range(10)]
    a = U.assign_splits(names, seed=1)
    assert a == U.assign_splits(list(reversed(names)), seed=1)
    c = {s: sum(v == s for v in a.values()) for s in U.SPLITS}
    assert c == {"train": 6, "val": 2, "test": 2}
    assert {s: sum(v == s for v in U.assign_splits(names[:3]).values()) for s in U.SPLITS} == {"train": 1, "val": 1, "test": 1}
    assert set(U.assign_splits(names[:2]).values()) == {"train"}  # too few to hold out: says so in the CLI


def test_splits_never_move_an_assigned_plan():
    first = U.assign_splits([f"p{i}" for i in range(5)], seed=0)
    more = U.assign_splits([f"p{i}" for i in range(20)], existing=first, seed=0)
    assert all(more[n] == s for n, s in first.items())
    c = {s: sum(v == s for v in more.values()) for s in U.SPLITS}
    assert c["test"] == 4 and c["val"] == 4  # new plans topped val/test up to their share
    # plans that disappeared are dropped, unknown split names ignored
    assert U.assign_splits(["p1"], existing={"gone": "test", "p1": "bogus"}) == {"p1": "train"}


def test_split_fractions_are_validated():
    with pytest.raises(ValueError):
        U.assign_splits(["a"], val_frac=0.5, test_frac=0.5)


# ---------------------------------------------------------------- data mixing

CC = {"train": [f"t{i:03d}" for i in range(500)], "val": [f"v{i:03d}" for i in range(50)], "test": [f"x{i:03d}" for i in range(50)]}
USER = {"a": "train", "b": "train", "c": "val", "d": "test", "e": "test"}


def test_mix_repeats_user_train_and_replays_cubicasa():
    e = U.mix_plan(USER, CC, repeat=3, replay_ratio=2.0, replay_val=20, seed=0)
    user_train = [x for x in e if x[0] == "user" and x[2] == "train"]
    assert [x[3] for x in user_train] == ["u_a", "u_a__r1", "u_a__r2", "u_b", "u_b__r1", "u_b__r2"]
    assert sum(x[0] == "cubicasa" and x[2] == "train" for x in e) == 12  # 2 per user training image
    assert sum(x[0] == "cubicasa" and x[2] == "val" for x in e) == 20
    assert len({x[3] for x in e}) == len(e)  # unique file names in the dataset
    assert e == U.mix_plan(USER, CC, repeat=3, replay_ratio=2.0, replay_val=20, seed=0)
    assert e != U.mix_plan(USER, CC, repeat=3, replay_ratio=2.0, replay_val=20, seed=1)


def test_mix_never_leaks_test_plans_into_training():
    e = U.mix_plan(USER, CC, repeat=2, replay_ratio=100.0, replay_val=10_000)
    assert not any(x[0] == "cubicasa" and (x[2] == "test" or x[1] in CC["test"]) for x in e)
    assert {x[1] for x in e if x[2] == "test"} == {"d", "e"}
    assert not ({x[1] for x in e if x[2] in ("train", "val")} & {"d", "e"})
    assert {x[1] for x in e if x[0] == "user" and x[2] == "val"} == {"c"}
    assert sum(x[0] == "cubicasa" and x[2] == "train" for x in e) == 400 and sum(x[2] == "val" and x[0] == "cubicasa" for x in e) == 50


def test_mix_without_replay_or_cubicasa_data():
    e = U.mix_plan(USER, {}, repeat=1, replay_ratio=2.0)
    assert all(x[0] == "user" for x in e) and len(e) == 5
    with pytest.raises(ValueError):
        U.mix_plan(USER, CC, repeat=0)


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG fake")


def test_materialize_lays_out_an_ultralytics_dataset(tmp_path):
    user, cc, out = tmp_path / "myplans", tmp_path / "cc", tmp_path / "out"
    for n in ("a", "d"):
        _img(user / "images" / f"{n}.png")
        (user / "labels").mkdir(parents=True, exist_ok=True)
        (user / "labels" / f"{n}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    _img(user / "images" / "unlabelled.jpg")
    for s, names in (("train", ["t1", "t2"]), ("val", ["v1"])):
        for n in names:
            _img(cc / "images" / s / f"{n}.png")
            (cc / "labels" / s).mkdir(parents=True, exist_ok=True)
            (cc / "labels" / s / f"{n}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    (out / "images" / "train").mkdir(parents=True)
    (out / "images" / "train" / "stale.png").write_bytes(b"old")  # a previous mix must not leak in
    e = U.mix_plan({"a": "train", "d": "test"}, {"train": ["t1", "t2"], "val": ["v1"]}, repeat=2, replay_ratio=1.0, replay_val=5)
    yaml = U.materialize(e, user, cc, out)
    txt = yaml.read_text()
    assert "train: images/train" in txt and "test: images/test" in txt and f"  12: {CLASSES[12]}" in txt
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["cc_t1.png", "cc_t2.png", "u_a.png", "u_a__r1.png"]
    assert sorted(p.name for p in (out / "labels" / "train").iterdir()) == ["cc_t1.txt", "cc_t2.txt", "u_a.txt", "u_a__r1.txt"]
    assert [p.name for p in (out / "images" / "test").iterdir()] == ["u_d.png"]
    assert [p.name for p in (out / "images" / "val").iterdir()] == ["cc_v1.png"]
    assert (out / "labels" / "train" / "u_a__r1.txt").read_text() == "0 0.5 0.5 0.1 0.1\n"
    assert json.loads((out / "manifest.json").read_text())[0] == list(e[0])


def test_images_dir_accepts_a_plain_folder(tmp_path):
    _img(tmp_path / "x.PNG")
    (tmp_path / "notes.txt").write_text("")
    assert list(U.list_images(tmp_path)) == ["x"]
    _img(tmp_path / "images" / "y.jpg")
    assert list(U.list_images(tmp_path)) == ["y"]


def test_finetune_args_are_gentle_and_honour_lr():
    a = U.finetune_args()
    assert a["optimizer"] != "auto" and a["lr0"] <= 1e-3 and a["freeze"] == 10 and a["epochs"] <= 30
    assert U.finetune_args(freeze=0)["freeze"] is None
    assert U.finetune_args(epochs=2)["close_mosaic"] == 2


# ---------------------------------------------------------------- scoring + val-only tuning

def test_threshold_grid_matches_cubicasa_hybrid():
    from stoa_qa import cubicasa_hybrid as H

    assert U.THRESHOLDS == H.THRESHOLDS and U.IOUS == H.IOUS


def _plan(door_scores):
    return [[DOOR, s, i * 20, 0, i * 20 + 10, 10] for i, s in enumerate(door_scores)]


def test_count_accuracy_and_total():
    preds = {"p1": _plan([0.9, 0.6]), "p2": _plan([0.9, 0.2]), "p3": _plan([0.9])}
    truth = {"p1": {"doors": 2}, "p2": {"doors": 2}, "p3": {"doors": 1}, "nopred": {"doors": 1}}
    thr = {q: 0.5 for q in QUESTION_CLASSES}
    acc = U.count_accuracy(preds, truth, thr)
    assert acc["doors"] == (2, 3) and acc["windows"] == (0, 0)
    assert U.count_accuracy(preds, truth, thr, plans=["p2"])["doors"] == (0, 1)
    assert U.total(acc) == (2, 3)


def test_tune_uses_only_the_val_sets_it_is_given_and_weights_them():
    # user val wants a low door threshold (faint symbols), CubiCasa val a high one
    user = ({"u1": _plan([0.9, 0.25]), "u2": _plan([0.3, 0.3])}, {"u1": {"doors": 2}, "u2": {"doors": 2}})
    cc = ({f"c{i}": _plan([0.9, 0.35]) for i in range(10)}, {f"c{i}": {"doors": 1} for i in range(10)})
    thr_u, _ = U.tune([(*user, 1.0)])
    thr_c, _ = U.tune([(*cc, 1.0)])
    assert thr_u["doors"] <= 0.25 and thr_c["doors"] > 0.35
    thr_mix, _ = U.tune([(*user, 0.7), (*cc, 0.3)])
    assert thr_mix["doors"] == thr_u["doors"]  # user weight dominates
    thr_mix, _ = U.tune([(*user, 0.3), (*cc, 0.7)])
    assert thr_mix["doors"] == thr_c["doors"]
    # a question with no truth anywhere falls back to the simplest setting
    assert U.tune([(*user, 1.0)])[1]["saunas"] == 1.0


def test_run_tag_from_weights_path():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(U.__file__).resolve().parent.parent / "modal_jobs"))
    from cubicasa_userplans import _tag

    assert _tag([Path("third_party/weights/m13-1280.pt")]) == "m13-1280"
    assert _tag([Path("third_party/weights/m13-1280/best.pt")]) == "m13-1280"
    assert _tag([Path("eval_out/yolo_runs/ft-mine/weights/best.pt")]) == "ft-mine"
    assert _tag([Path("w/m13-1280.pt"), Path("runs/local-y11s-13c/weights/best.pt")]) == "ens-m13-1280-local-y11s-13c"
