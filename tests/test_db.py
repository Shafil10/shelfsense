"""Tests for src/server/db.py: agreements over time, geofence check-in/out, route km and fuel.

Run from the project root:  python -m pytest -v
"""
import pytest

from src.server.db import DB, dist_m

CFG = {"utc_offset_hours": 6, "geofence_radius_m": 80, "checkout_factor": 1.5, "gps_max_accuracy_m": 50, "min_step_m": 10,
       "max_speed_kmh": 90, "fuel": {"km_per_litre": 40, "price_per_litre_tk": 130}}
SHOP_A = (23.8069, 90.3687)
SHOP_B = (23.8189, 90.3655)  # about 1.4 km north


@pytest.fixture
def db(tmp_path):
    d = DB(tmp_path / "t.db", CFG)
    d.a = d.add_shop("Shwapno, Mirpur 10", *SHOP_A)
    d.b = d.add_shop("Al-Madina Store", *SHOP_B)
    d.rahim = d.add_staff("Rahim")
    return d


def t(minute, second=0):
    return f"2026-11-09T09:{minute:02d}:{second:02d}+00:00"


def test_distance():
    assert 1300 < dist_m(*SHOP_A, *SHOP_B) < 1400


def test_agreement_history_keeps_old_versions(db):
    db.set_agreement(db.a, 4, shelf=2, zone="front", valid_from="2026-11-01T00:00:00+00:00")
    db.set_agreement(db.a, 6, shelf=1, zone="front", valid_from="2026-11-08T00:00:00+00:00")
    assert db.agreement(db.a, "2026-11-05T12:00:00+00:00")["cups"] == 4
    assert db.agreement(db.a, "2026-11-09T12:00:00+00:00")["cups"] == 6
    assert db.agreement(db.b) is None
    with pytest.raises(ValueError):
        db.set_agreement(db.a, -1)
    with pytest.raises(KeyError):
        db.set_agreement(999, 3)


def test_geofence_check_in_and_out(db):
    far = (23.8000, 90.3687)  # about 770 m south of shop A
    assert db.ping(db.rahim, *far, acc=10, t=t(0)) == []
    ev = db.ping(db.rahim, SHOP_A[0] + 0.0002, SHOP_A[1], acc=10, t=t(5))  # 22 m from the shop
    assert [e["kind"] for e in ev] == ["check_in"] and ev[0]["shop_id"] == db.a
    # 100 m away: outside the 80 m circle but inside 1.5 x 80 = 120 m -> still in the shop (no flicker)
    assert db.ping(db.rahim, SHOP_A[0] + 0.0009, SHOP_A[1], acc=10, t=t(9)) == []
    # a bad fix far away does not end the visit
    assert db.ping(db.rahim, *far, acc=200, t=t(10)) == []
    ev = db.ping(db.rahim, *far, acc=10, t=t(17))
    assert [e["kind"] for e in ev] == ["check_out"]
    v = db.visits("2026-11-09")
    assert len(v) == 1 and v[0]["minutes"] == 12.0 and v[0]["shop"] == "Shwapno, Mirpur 10"


def test_walking_from_one_shop_into_the_next(db):
    db.ping(db.rahim, *SHOP_A, acc=5, t=t(0))
    ev = db.ping(db.rahim, *SHOP_B, acc=5, t=t(8))
    assert [(e["kind"], e["shop_id"]) for e in ev] == [("check_out", db.a), ("check_in", db.b)]


def test_route_km_fuel_and_noise(db):
    db.ping(db.rahim, *SHOP_A, acc=5, t=t(0))
    db.ping(db.rahim, SHOP_A[0] + 0.00003, SHOP_A[1], acc=5, t=t(1))   # 3 m jitter: not travel
    db.ping(db.rahim, 23.70, 90.30, acc=5, t=t(1, 5))                  # 13 km in 5 s: GPS error, ignored
    db.ping(db.rahim, *SHOP_B, acc=5, t=t(10))
    db.ping(db.rahim, 23.83, 90.36, acc=300, t=t(12))                  # poor fix: not counted
    db.ping(db.rahim, *SHOP_A, acc=5, t=t(20))
    r = db.route(db.rahim, "2026-11-09")
    assert 2.6 < r["km"] < 2.8                                         # A -> B -> A
    assert r["fuel_cost_tk"] == round(r["km"] / 40 * 130, 1)
    assert r["visits"] == 3 and r["shops"] == 2 and r["route_minutes"] == 20.0
    assert r["avg_minutes_in_shop"] == 10.0                            # A 0-10, B 10-20 (the last one is still open)


def test_days_follow_bangladesh_time(db):
    db.ping(db.rahim, *SHOP_A, acc=5, t="2026-11-09T19:30:00+00:00")   # 01:30 on 10 Nov in Dhaka
    assert db.day_range("2026-11-10") == ("2026-11-09T18:00:00+00:00", "2026-11-10T18:00:00+00:00")
    assert len(db.visits("2026-11-10")) == 1 and db.visits("2026-11-09") == []
    assert db.route(db.rahim, "2026-11-10")["pings"] == 1


def test_pins_are_hashed_unique_and_replaceable(db):
    office = db.new_pin("office")
    first = db.new_pin("staff", db.rahim)
    assert len(first) == 6 and first.isdigit() and first != office
    assert db.check_pin(first) == {"role": "staff", "staff_id": db.rahim, "name": "Rahim"}
    assert db.check_pin(office)["role"] == "office"
    assert all(first not in str(r) for r in db.q("SELECT * FROM access"))   # only hashes are stored
    second = db.new_pin("staff", db.rahim)                                   # a new PIN replaces the old one
    assert db.check_pin(first) is None or first == second
    assert db.check_pin(second)["staff_id"] == db.rahim
    assert db.check_pin("12ab56") is None and db.check_pin("") is None
    with pytest.raises(ValueError):
        db.new_pin("staff")
    with pytest.raises(KeyError):
        db.new_pin("staff", 999)


def test_fuel_settings_override_the_config(db):
    assert db.fuel() == {"km_per_litre": 40, "price_per_litre_tk": 130}
    assert db.set_fuel(km_per_litre=45) == {"km_per_litre": 45.0, "price_per_litre_tk": 130}
    with pytest.raises(ValueError):
        db.set_fuel(price_per_litre_tk=0)
    db.ping(db.rahim, *SHOP_A, acc=5, t=t(0))
    db.ping(db.rahim, *SHOP_B, acc=5, t=t(10))
    r = db.route(db.rahim, "2026-11-09")
    assert r["fuel_cost_tk"] == round(r["km"] / 45 * 130, 1)


def test_share_links(db):
    s = db.new_share(db.rahim, 2)
    assert db.tracked(s["token"]) == {"name": "Rahim", "expires": s["expires"], "position": None, "at_shop": None}
    from datetime import datetime, timezone
    db.ping(db.rahim, *SHOP_A, acc=5, t=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    shown = db.tracked(s["token"])
    assert shown["position"]["lat"] == SHOP_A[0] and shown["at_shop"] == "Shwapno, Mirpur 10"
    assert [x["name"] for x in db.shares()] == ["Rahim"]
    db.end_share(s["token"])
    assert db.tracked(s["token"]) is None and db.shares() == []
    assert db.tracked("made-up") is None
    old = db.new_share(db.rahim, 1)
    db.x("UPDATE shares SET expires='2020-01-01T00:00:00+00:00' WHERE token=?", (old["token"],))
    assert db.tracked(old["token"]) is None                                   # expired
    with pytest.raises(ValueError):
        db.new_share(db.rahim, 0)
    with pytest.raises(KeyError):
        db.new_share(999, 2)


def test_latest_positions_and_events(db):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.ping(db.rahim, *SHOP_A, acc=5, t=now)
    pos = db.latest_positions(600)
    assert [(p["name"], p["lat"]) for p in pos] == [("Rahim", SHOP_A[0])]
    ev = db.events_since(0)
    assert [(e["kind"], e["staff"], e["shop"]) for e in ev] == [("check_in", "Rahim", "Shwapno, Mirpur 10")]
    assert db.events_since(ev[-1]["id"]) == []
    with pytest.raises(KeyError):
        db.ping(999, *SHOP_A)
