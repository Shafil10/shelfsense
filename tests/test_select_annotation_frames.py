from scripts.select_annotation_frames import pick_frames


def row(i, blur, status="kept"):
    return {"file": f"f{i}", "blur_effect": str(blur), "status": status}


def test_picks_sharpest_frame_in_each_part():
    log = [row(0, 0.4), row(1, 0.3), row(2, 0.45), row(3, 0.35), row(4, 0.9, "dropped_blur"), row(5, 0.2)]
    # kept: f0 f1 | f2 f3 f5  -> sharpest of each part
    assert [r["file"] for r in pick_frames(log, 2)] == ["f1", "f5"]


def test_short_video_returns_all_kept_frames():
    log = [row(0, 0.3), row(1, 0.8, "dropped_blur")]
    assert [r["file"] for r in pick_frames(log, 3)] == ["f0"]
