"""Tests for src/server/business.py: production -> dispatch -> delivery, returns, Pandamart POs, reseller orders,
stock and restock alerts.

Run from the project root:  python -m pytest -v
"""
import pytest

from src.server.business import Business
from src.server.db import DB

TRACK = {"utc_offset_hours": 6, "geofence_radius_m": 80, "checkout_factor": 1.5, "min_visit_minutes": 2, "gps_max_accuracy_m": 50, "min_step_m": 10,
         "max_speed_kmh": 90, "fuel": {"km_per_litre": 40, "price_per_litre_tk": 130}}


@pytest.fixture
def b(tmp_path):
    db = DB(tmp_path / "t.db", TRACK)
    biz = Business(db, {"restock_days": 2})
    biz.cup = biz.add_product("Coco Fresh cup", "cup", price_wholesale=40, price_retail=50, price_pandamart=45)
    biz.box = biz.add_product("Coco Fresh carton", "box", price_wholesale=900, price_pandamart=1050)
    biz.shop = db.add_shop("Al-Madina Store", 23.8212, 90.3661, "Mirpur 11", "retail")
    biz.panda = db.add_shop("Pandamart Mirpur", 23.8102, 90.3700, "Mirpur 10", "pandamart")
    biz.reseller = db.add_shop("Mirpur Wholesale", 23.8150, 90.3620, "Mirpur 11", "reseller")
    biz.rahim = db.add_staff("Rahim")
    return biz


def test_goods_flow_and_stock(b):
    b.add_production(b.cup, 500)
    b.add_dispatch(b.rahim, [{"product_id": b.cup, "qty": 120}, {"product_id": b.box, "qty": 0}])   # a 0 line is skipped
    b.add_delivery(b.shop, [{"product_id": b.cup, "delivered": 40, "returned": 3, "return_reason": "expired"}], staff_id=b.rahim)
    s = b.stock()
    cup = next(f for f in s["factory"] if f["product_id"] == b.cup)
    assert (cup["made"], cup["dispatched"], cup["in_factory"], cup["delivered"], cup["returned"]) == (500, 120, 380, 40, 3)
    bike = s["riders"][0]["items"][0]
    assert (bike["loaded"], bike["delivered"], bike["on_bike"], bike["picked_up"]) == (120, 40, 80, 3)
    assert s["returns"] == [{"reason": "expired", "product": "Coco Fresh cup", "qty": 3}]
    assert b.deliveries(shop_id=b.shop)[0]["items"][0]["return_reason"] == "expired"


def test_bad_records_are_refused(b):
    with pytest.raises(ValueError):
        b.add_dispatch(b.rahim, [{"product_id": b.cup, "qty": 0}])                                # nothing loaded
    with pytest.raises(ValueError):
        b.add_dispatch(b.rahim, [{"product_id": b.cup, "qty": -5}])
    with pytest.raises(KeyError):
        b.add_dispatch(999, [{"product_id": b.cup, "qty": 5}])
    with pytest.raises(KeyError):
        b.add_production(999, 10)
    with pytest.raises(ValueError):                                                             # a return needs a reason
        b.add_delivery(b.shop, [{"product_id": b.cup, "delivered": 5, "returned": 2}])
    with pytest.raises(ValueError):
        b.add_delivery(b.shop, [{"product_id": b.cup, "returned": 2, "return_reason": "too hot"}])


def test_pandamart_po_over_and_under_delivery(b):
    po = b.add_po("PM-1001", b.panda, [{"product_id": b.cup, "qty": 100}, {"product_id": b.box, "qty": 2}])
    assert b.pos()[0]["result"] == "not delivered"
    b.add_delivery(b.panda, [{"product_id": b.cup, "delivered": 60}], po_id=po)
    b.add_delivery(b.panda, [{"product_id": b.cup, "delivered": 50, "returned": 4, "return_reason": "damaged"},
                             {"product_id": b.box, "delivered": 1}], po_id=po)
    items = {i["name"]: i for i in b.compare_po(po)}
    assert (items["Coco Fresh cup"]["ordered"], items["Coco Fresh cup"]["delivered"], items["Coco Fresh cup"]["difference"]) == (100, 110, 10)
    assert items["Coco Fresh cup"]["returned"] == 4 and items["Coco Fresh carton"]["difference"] == -1
    assert b.pos()[0]["result"] == "over-delivered"                                            # any line over counts
    po2 = b.add_po("PM-1002", b.panda, [{"product_id": b.cup, "qty": 30}])
    b.add_delivery(b.panda, [{"product_id": b.cup, "delivered": 30}], po_id=po2)
    assert {p["number"]: p["result"] for p in b.pos()}["PM-1002"] == "complete"
    with pytest.raises(KeyError):                                                               # a PO of another shop
        b.add_delivery(b.shop, [{"product_id": b.cup, "delivered": 1}], po_id=po2)
    b.close_po(po)
    assert [p["number"] for p in b.pos("open")] == ["PM-1002"]


def test_reseller_orders_use_wholesale_prices(b):
    oid = b.add_order(b.reseller, [{"product_id": b.cup, "qty": 200}, {"product_id": b.box, "qty": 3}], by="Mirpur Wholesale")
    o = b.orders(b.reseller)[0]
    assert o["id"] == oid and o["status"] == "new" and o["total_tk"] == 200 * 40 + 3 * 900
    b.set_order_status(oid, "confirmed")
    with pytest.raises(ValueError):
        b.set_order_status(oid, "lost")
    b.add_delivery(b.reseller, [{"product_id": b.cup, "delivered": 200}], order_id=oid)      # delivering closes the order
    assert b.orders(b.reseller)[0]["status"] == "delivered"
    pid = b.add_order(b.panda, [{"product_id": b.cup, "qty": 10}])
    assert b.orders(b.panda)[0]["items"][0]["unit_price"] == 45 and pid


def test_restock_alerts(b):
    alerts = {a["shop"]: a["reasons"] for a in b.restock()}
    assert alerts["Al-Madina Store"] == ["no delivery yet"] and "Mirpur Wholesale" not in alerts
    b.add_delivery(b.shop, [{"product_id": b.cup, "delivered": 10}])
    b.add_po("PM-7", b.panda, [{"product_id": b.cup, "qty": 50}])
    b.add_order(b.reseller, [{"product_id": b.cup, "qty": 5}])
    alerts = {a["shop"]: a["reasons"] for a in b.restock()}
    assert "Al-Madina Store" not in alerts
    assert alerts["Pandamart Mirpur"][0].startswith("Pandamart PO PM-7") and alerts["Mirpur Wholesale"] == ["1 order(s) waiting"]
    old = "2026-01-01T00:00:00+00:00"
    b.x("UPDATE deliveries SET t=?", (old,))
    assert {a["shop"]: a["reasons"] for a in b.restock()}["Al-Madina Store"] == ["no delivery for more than 2 days"]
