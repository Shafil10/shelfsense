"""NutriCoco's distribution records (requirements 1, 2, 4, 7, 8), next to the tracking data in the same SQLite file.

The flow of goods, each step a record with who, when and how many:
  production (cups made at the factory)
    -> dispatch (goods loaded on a rider's motorcycle at the factory)
      -> delivery at a shop (delivered quantity; returned quantity + reason, e.g. expired or leaking)
Orders that deliveries can answer:
  - Pandamart purchase orders (POs): Pandamart checks its stock and sends a PO; deliveries against the PO show
    over- or under-delivery per product.
  - reseller orders: wholesale customers (resellers / distributors) order directly in the app with their own PIN.
Stock follows from the records: factory = made - dispatched; a rider's motorcycle today = dispatched - delivered;
returned goods come back to the factory as "returned" (not sellable). Shops are customers with a channel:
retail, supermarket, pandamart or reseller ("wholesale" = resellers, everything else is retail).
"""
from datetime import datetime, timedelta, timezone

from src.server.db import iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, unit TEXT DEFAULT 'cup', price_wholesale REAL, price_retail REAL,
  price_pandamart REAL, active INTEGER DEFAULT 1, sample INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS production (
  id INTEGER PRIMARY KEY, t TEXT NOT NULL, product_id INTEGER NOT NULL REFERENCES products(id), qty INTEGER NOT NULL,
  note TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS dispatches (
  id INTEGER PRIMARY KEY, t TEXT NOT NULL, staff_id INTEGER NOT NULL REFERENCES staff(id), note TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS dispatch_items (
  dispatch_id INTEGER NOT NULL REFERENCES dispatches(id), product_id INTEGER NOT NULL REFERENCES products(id), qty INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS purchase_orders (
  id INTEGER PRIMARY KEY, number TEXT NOT NULL, shop_id INTEGER NOT NULL REFERENCES shops(id), po_date TEXT NOT NULL,
  due_date TEXT, status TEXT DEFAULT 'open', note TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS po_items (
  po_id INTEGER NOT NULL REFERENCES purchase_orders(id), product_id INTEGER NOT NULL REFERENCES products(id), qty INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY, shop_id INTEGER NOT NULL REFERENCES shops(id), t TEXT NOT NULL, wanted_date TEXT,
  status TEXT DEFAULT 'new', note TEXT, placed_by TEXT);
CREATE TABLE IF NOT EXISTS order_items (
  order_id INTEGER NOT NULL REFERENCES orders(id), product_id INTEGER NOT NULL REFERENCES products(id), qty INTEGER NOT NULL,
  unit_price REAL);
CREATE TABLE IF NOT EXISTS deliveries (
  id INTEGER PRIMARY KEY, t TEXT NOT NULL, staff_id INTEGER REFERENCES staff(id), shop_id INTEGER NOT NULL REFERENCES shops(id),
  visit_id INTEGER, po_id INTEGER REFERENCES purchase_orders(id), order_id INTEGER REFERENCES orders(id), note TEXT);
CREATE TABLE IF NOT EXISTS delivery_items (
  delivery_id INTEGER NOT NULL REFERENCES deliveries(id), product_id INTEGER NOT NULL REFERENCES products(id),
  delivered INTEGER DEFAULT 0, returned INTEGER DEFAULT 0, return_reason TEXT);
"""
RETURN_REASONS = ("expired", "damaged", "not_chilled", "unsold", "wrong_product", "other")
ORDER_STATUS = ("new", "confirmed", "dispatched", "delivered", "cancelled")
WHOLESALE = ("reseller",)


def now():
    return iso(datetime.now(timezone.utc))


class Business:
    def __init__(self, db, cfg):
        """db: src.server.db.DB (same connection); cfg: the 'business' section of configs/config.yaml."""
        self.db, self.cfg, self.q, self.x = db, cfg, db.q, db.x
        with db.lock, db.con:
            db.con.executescript(SCHEMA)

    def _items(self, items, field="qty", allow_zero=False):
        """[(product_id, qty), ...] with checked numbers; at least one line."""
        out = []
        for it in items or []:
            pid, qty = int(it["product_id"]), int(it.get(field) or 0)
            if qty < 0:
                raise ValueError("quantities cannot be negative")
            if qty == 0 and not allow_zero:
                continue
            if not self.q("SELECT id FROM products WHERE id=?", (pid,)):
                raise KeyError(f"unknown product {pid}")
            out.append((pid, qty))
        if not out:
            raise ValueError("add at least one product with a quantity")
        return out

    # ---------- products and production ----------
    def products(self, all_=False):
        return self.q("SELECT * FROM products" + ("" if all_ else " WHERE active=1") + " ORDER BY id")

    def add_product(self, name, unit="cup", price_wholesale=None, price_retail=None, price_pandamart=None, sample=False):
        return self.x("INSERT INTO products (name, unit, price_wholesale, price_retail, price_pandamart, sample) VALUES (?,?,?,?,?,?)",
                      (name, unit, price_wholesale, price_retail, price_pandamart, int(sample)))

    def update_product(self, pid, **fields):
        allowed = {k: v for k, v in fields.items() if k in ("name", "unit", "price_wholesale", "price_retail", "price_pandamart", "active")}
        for k, v in allowed.items():
            self.x(f"UPDATE products SET {k}=? WHERE id=?", (v, pid))  # k is from the fixed list above

    def add_production(self, product_id, qty, note="", by="", t=None):
        (pid, qty), = self._items([{"product_id": product_id, "qty": qty}])
        return self.x("INSERT INTO production (t, product_id, qty, note, created_by) VALUES (?,?,?,?,?)", (t or now(), pid, qty, note, by))

    # ---------- dispatch from the factory to a rider ----------
    def add_dispatch(self, staff_id, items, note="", by="", t=None):
        if not self.q("SELECT id FROM staff WHERE id=?", (staff_id,)):
            raise KeyError(f"unknown staff {staff_id}")
        lines = self._items(items)
        did = self.x("INSERT INTO dispatches (t, staff_id, note, created_by) VALUES (?,?,?,?)", (t or now(), staff_id, note, by))
        for pid, qty in lines:
            self.x("INSERT INTO dispatch_items (dispatch_id, product_id, qty) VALUES (?,?,?)", (did, pid, qty))
        return did

    def dispatches(self, day=None):
        sql, args = "SELECT d.*, s.name AS staff FROM dispatches d JOIN staff s ON s.id=d.staff_id", []
        if day:
            sql += " WHERE d.t>=? AND d.t<?"
            args += self.db.day_range(day)
        rows = self.q(sql + " ORDER BY d.t DESC", args)
        for r in rows:
            r["items"] = self.q("SELECT di.product_id, p.name, di.qty FROM dispatch_items di JOIN products p ON p.id=di.product_id "
                                "WHERE di.dispatch_id=?", (r["id"],))
        return rows

    # ---------- deliveries (and returns) at shops ----------
    def add_delivery(self, shop_id, items, staff_id=None, po_id=None, order_id=None, note="", t=None):
        if not self.db.shop(shop_id):
            raise KeyError(f"unknown shop {shop_id}")
        lines = []
        for it in items or []:
            pid, delivered, returned = int(it["product_id"]), int(it.get("delivered") or 0), int(it.get("returned") or 0)
            if delivered < 0 or returned < 0:
                raise ValueError("quantities cannot be negative")
            if not delivered and not returned:
                continue
            reason = it.get("return_reason") if returned else None
            if returned and reason not in RETURN_REASONS:
                raise ValueError(f"choose a return reason: {', '.join(RETURN_REASONS)}")
            if not self.q("SELECT id FROM products WHERE id=?", (pid,)):
                raise KeyError(f"unknown product {pid}")
            lines.append((pid, delivered, returned, reason))
        if not lines:
            raise ValueError("add at least one product delivered or returned")
        for ref, table in ((po_id, "purchase_orders"), (order_id, "orders")):
            if ref is not None and not self.q(f"SELECT id FROM {table} WHERE id=? AND shop_id=?", (ref, shop_id)):
                raise KeyError(f"{table[:-1].replace('_', ' ')} {ref} is not for this shop")
        visit = self.db.open_visit(staff_id) if staff_id else None
        t = t or now()
        did = self.x("INSERT INTO deliveries (t, staff_id, shop_id, visit_id, po_id, order_id, note) VALUES (?,?,?,?,?,?,?)",
                     (t, staff_id, shop_id, visit["id"] if visit and visit["shop_id"] == shop_id else None, po_id, order_id, note))
        for pid, delivered, returned, reason in lines:
            self.x("INSERT INTO delivery_items (delivery_id, product_id, delivered, returned, return_reason) VALUES (?,?,?,?,?)",
                   (did, pid, delivered, returned, reason))
        if order_id is not None:
            self.set_order_status(order_id, "delivered")
        return did

    def deliveries(self, day=None, shop_id=None, staff_id=None):
        sql = ("SELECT d.*, s.name AS shop, s.channel, st.name AS staff FROM deliveries d JOIN shops s ON s.id=d.shop_id "
               "LEFT JOIN staff st ON st.id=d.staff_id WHERE 1=1")
        args = []
        if day:
            sql += " AND d.t>=? AND d.t<?"
            args += self.db.day_range(day)
        for col, val in (("d.shop_id", shop_id), ("d.staff_id", staff_id)):
            if val is not None:
                sql += f" AND {col}=?"
                args.append(val)
        rows = self.q(sql + " ORDER BY d.t DESC", args)
        for r in rows:
            r["items"] = self.q("SELECT di.product_id, p.name, di.delivered, di.returned, di.return_reason FROM delivery_items di "
                                "JOIN products p ON p.id=di.product_id WHERE di.delivery_id=?", (r["id"],))
        return rows

    # ---------- Pandamart purchase orders ----------
    def add_po(self, number, shop_id, items, po_date=None, due_date=None, note="", by=""):
        shop = self.db.shop(shop_id)
        if not shop:
            raise KeyError(f"unknown shop {shop_id}")
        lines = self._items(items)
        pid = self.x("INSERT INTO purchase_orders (number, shop_id, po_date, due_date, note, created_by) VALUES (?,?,?,?,?,?)",
                     (str(number).strip() or "?", shop_id, po_date or self.db.today(), due_date, note, by))
        for prod, qty in lines:
            self.x("INSERT INTO po_items (po_id, product_id, qty) VALUES (?,?,?)", (pid, prod, qty))
        return pid

    def pos(self, status=None):
        sql, args = "SELECT po.*, s.name AS shop FROM purchase_orders po JOIN shops s ON s.id=po.shop_id", []
        if status:
            sql += " WHERE po.status=?"
            args.append(status)
        rows = self.q(sql + " ORDER BY po.po_date DESC, po.id DESC", args)
        for r in rows:
            r["items"] = self.compare_po(r["id"])
            r["result"] = ("not delivered" if all(i["delivered"] == 0 for i in r["items"])
                           else "over-delivered" if any(i["difference"] > 0 for i in r["items"])
                           else "under-delivered" if any(i["difference"] < 0 for i in r["items"]) else "complete")
        return rows

    def compare_po(self, po_id):
        """Ordered vs delivered vs returned per product for one PO (difference > 0 = over-delivered)."""
        rows = self.q("""SELECT p.id AS product_id, p.name, COALESCE(pi.qty, 0) AS ordered,
                COALESCE((SELECT SUM(di.delivered) FROM delivery_items di JOIN deliveries d ON d.id=di.delivery_id
                          WHERE d.po_id=? AND di.product_id=p.id), 0) AS delivered,
                COALESCE((SELECT SUM(di.returned) FROM delivery_items di JOIN deliveries d ON d.id=di.delivery_id
                          WHERE d.po_id=? AND di.product_id=p.id), 0) AS returned
            FROM products p LEFT JOIN po_items pi ON pi.product_id=p.id AND pi.po_id=?
            WHERE pi.po_id IS NOT NULL OR p.id IN (SELECT di.product_id FROM delivery_items di JOIN deliveries d
                                                   ON d.id=di.delivery_id WHERE d.po_id=?)""", (po_id,) * 4)
        for r in rows:
            r["difference"] = r["delivered"] - r["ordered"]
        return rows

    def close_po(self, po_id, status="closed"):
        if status not in ("open", "closed"):
            raise ValueError("status must be open or closed")
        self.x("UPDATE purchase_orders SET status=? WHERE id=?", (status, po_id))

    # ---------- reseller (wholesale) orders ----------
    def add_order(self, shop_id, items, wanted_date=None, note="", by=""):
        shop = self.db.shop(shop_id)
        if not shop:
            raise KeyError(f"unknown customer {shop_id}")
        lines = self._items(items)
        col = "price_wholesale" if shop["channel"] in WHOLESALE else "price_pandamart" if shop["channel"] == "pandamart" else "price_retail"
        oid = self.x("INSERT INTO orders (shop_id, t, wanted_date, note, placed_by) VALUES (?,?,?,?,?)",
                     (shop_id, now(), wanted_date, note, by))
        for pid, qty in lines:
            price = self.q(f"SELECT {col} AS p FROM products WHERE id=?", (pid,))[0]["p"]  # col is one of three fixed names
            self.x("INSERT INTO order_items (order_id, product_id, qty, unit_price) VALUES (?,?,?,?)", (oid, pid, qty, price))
        return oid

    def orders(self, shop_id=None, status=None):
        sql, args = "SELECT o.*, s.name AS shop, s.channel FROM orders o JOIN shops s ON s.id=o.shop_id WHERE 1=1", []
        for col, val in (("o.shop_id", shop_id), ("o.status", status)):
            if val is not None:
                sql += f" AND {col}=?"
                args.append(val)
        rows = self.q(sql + " ORDER BY o.t DESC", args)
        for r in rows:
            r["items"] = self.q("SELECT oi.product_id, p.name, oi.qty, oi.unit_price FROM order_items oi JOIN products p ON p.id=oi.product_id "
                                "WHERE oi.order_id=?", (r["id"],))
            r["total_tk"] = round(sum(i["qty"] * (i["unit_price"] or 0) for i in r["items"]), 2)
        return rows

    def set_order_status(self, order_id, status):
        if status not in ORDER_STATUS:
            raise ValueError(f"status must be one of {', '.join(ORDER_STATUS)}")
        if not self.q("SELECT id FROM orders WHERE id=?", (order_id,)):
            raise KeyError(f"unknown order {order_id}")
        self.x("UPDATE orders SET status=? WHERE id=?", (status, order_id))

    # ---------- stock and restock alerts ----------
    def stock(self, day=None):
        """Factory stock, today's stock on each rider's motorcycle, and returns (by reason)."""
        day = day or self.db.today()
        start, end = self.db.day_range(day)
        factory = self.q("""SELECT p.id AS product_id, p.name,
              COALESCE((SELECT SUM(qty) FROM production WHERE product_id=p.id), 0) AS made,
              COALESCE((SELECT SUM(qty) FROM dispatch_items WHERE product_id=p.id), 0) AS dispatched,
              COALESCE((SELECT SUM(di.delivered) FROM delivery_items di WHERE di.product_id=p.id), 0) AS delivered,
              COALESCE((SELECT SUM(di.returned) FROM delivery_items di WHERE di.product_id=p.id), 0) AS returned
            FROM products p WHERE p.active=1 ORDER BY p.id""")
        for f in factory:
            f["in_factory"] = f["made"] - f["dispatched"]
        riders = []
        for s in self.db.staff():
            lines = self.q("""SELECT p.id AS product_id, p.name,
                  COALESCE((SELECT SUM(di.qty) FROM dispatch_items di JOIN dispatches d ON d.id=di.dispatch_id
                            WHERE di.product_id=p.id AND d.staff_id=? AND d.t>=? AND d.t<?), 0) AS loaded,
                  COALESCE((SELECT SUM(di.delivered) FROM delivery_items di JOIN deliveries d ON d.id=di.delivery_id
                            WHERE di.product_id=p.id AND d.staff_id=? AND d.t>=? AND d.t<?), 0) AS delivered,
                  COALESCE((SELECT SUM(di.returned) FROM delivery_items di JOIN deliveries d ON d.id=di.delivery_id
                            WHERE di.product_id=p.id AND d.staff_id=? AND d.t>=? AND d.t<?), 0) AS picked_up
                FROM products p WHERE p.active=1 ORDER BY p.id""", (s["id"], start, end) * 3)
            for ln in lines:
                ln["on_bike"] = ln["loaded"] - ln["delivered"]
            if any(ln["loaded"] or ln["delivered"] or ln["picked_up"] for ln in lines):
                riders.append({"staff_id": s["id"], "name": s["name"], "items": lines})
        returns = self.q("""SELECT di.return_reason AS reason, p.name AS product, SUM(di.returned) AS qty FROM delivery_items di
              JOIN products p ON p.id=di.product_id WHERE di.returned>0 GROUP BY di.return_reason, p.name ORDER BY qty DESC""")
        return {"day": day, "factory": factory, "riders": riders, "returns": returns}

    def restock(self):
        """Shops that may need goods: open POs, reseller orders waiting, or no delivery for business.restock_days."""
        alerts, limit = [], self.cfg["restock_days"]
        cutoff = iso(datetime.now(timezone.utc) - timedelta(days=limit))
        for s in self.db.shops():
            last = self.q("SELECT MAX(t) AS t FROM deliveries WHERE shop_id=? AND id IN (SELECT delivery_id FROM delivery_items WHERE delivered>0)",
                          (s["id"],))[0]["t"]
            reasons = []
            open_pos = [p for p in self.pos("open") if p["shop_id"] == s["id"] and p["result"] != "complete"]
            if open_pos:
                reasons.append(f"Pandamart PO {', '.join(p['number'] for p in open_pos)} not fully delivered")
            waiting = [o for o in self.orders(s["id"]) if o["status"] in ("new", "confirmed")]
            if waiting:
                reasons.append(f"{len(waiting)} order(s) waiting")
            if s["channel"] != "reseller" and (last is None or last < cutoff):
                reasons.append("no delivery yet" if last is None else f"no delivery for more than {limit} days")
            if reasons:
                alerts.append({"shop_id": s["id"], "shop": s["name"], "channel": s["channel"], "last_delivery": last, "reasons": reasons})
        return alerts
