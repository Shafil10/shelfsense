"""Sprint 6: compare what the camera saw with a shop's NutriCoco agreement (requirement 3), and two visits
of the same shop with each other (before / after).

Input: an audit report (src/report/audit.py) whose detector boxes carry a class name ("cls"), and the agreement
valid on the day of the video (cups, shelf 1 = top, zone). The detector must know NutriCoco's products; until it is
trained on NutriCoco's photos every KPI says "pending" with the reason, instead of a guess.

Per keyframe: NutriCoco cups (brand classes), other products and empty gaps, grouped by shelf. The keyframe that
shows the most cups is the "best view" of the fridge, and the KPIs come from it:
  - availability: cups visible vs agreed cups
  - placement: cups on the agreed shelf and in the agreed zone (left / middle / right third of that shelf;
    "front" and "any" accept every visible cup, since depth cannot be judged from one picture)
  - empty gaps on the agreed shelf
  - shelf share: NutriCoco cups / all products on the agreed shelf
Precision over recall (a false "missing cups" accuses an honest shopkeeper): a shortfall is a "violation" only when
the video passed the coverage checks; otherwise the result is "unclear" and a person decides.
"""
BRAND_CLASSES = ("nutricoco_cup", "nutricoco_box")
GAP_CLASS = "empty_gap"


def _zone_of(box, shelf_boxes):
    """'left' / 'middle' / 'right': which third of the shelf (as wide as all boxes on it) the box centre is in."""
    x0 = min(b["xyxy"][0] for b in shelf_boxes)
    x1 = max(b["xyxy"][2] for b in shelf_boxes)
    c = (box["xyxy"][0] + box["xyxy"][2]) / 2
    third = (x1 - x0) / 3 or 1
    return ("left", "middle", "right")[min(int((c - x0) // third), 2)]


def frame_counts(keyframe, agreement, brand=BRAND_CLASSES):
    """Counts in one keyframe, relative to the agreed shelf and zone."""
    boxes = keyframe.get("boxes", [])
    cups = [b for b in boxes if b.get("cls") in brand]
    gaps = [b for b in boxes if b.get("cls") == GAP_CLASS]
    shelf_no, zone = agreement.get("shelf"), agreement.get("zone") or "any"
    on_shelf = [b for b in boxes if shelf_no is None or b.get("shelf") == shelf_no]
    cups_on_shelf = [b for b in cups if b in on_shelf]
    products_on_shelf = [b for b in on_shelf if b.get("cls") != GAP_CLASS]
    in_zone = [b for b in cups_on_shelf if zone in ("any", "front") or _zone_of(b, products_on_shelf or cups_on_shelf) == zone]
    return {"time_sec": keyframe["time_sec"], "evidence": keyframe.get("evidence"), "cups": len(cups),
            "cups_on_shelf": len(cups_on_shelf), "cups_in_place": len(in_zone),
            "gaps_on_shelf": len([g for g in gaps if g in on_shelf]),
            "shelf_share": round(len(cups_on_shelf) / len(products_on_shelf), 3) if products_on_shelf else None}


def check_agreement(report, agreement, brand=BRAND_CLASSES):
    """KPIs of one fridge video against the agreement. Returns {'status', ...}."""
    if not agreement:
        return {"status": "pending", "reason": "no agreement set for this shop"}
    classes = set(report.get("pipeline", {}).get("classes") or [])
    if not classes & set(brand):
        return {"status": "pending", "reason": "the detector does not know NutriCoco products yet (needs NutriCoco's photos)"}
    frames = [frame_counts(k, agreement, brand) for k in report.get("keyframes", [])]
    if not frames:
        return {"status": "pending", "reason": "no usable keyframe"}
    best = max(frames, key=lambda f: (f["cups"], f["cups_in_place"]))
    agreed = int(agreement["cups"])
    covered = not any(f["check"] == "coverage" for f in report.get("integrity", {}).get("flags", []))
    short = max(agreed - best["cups"], 0)
    misplaced = max(min(agreed, best["cups"]) - best["cups_in_place"], 0)
    if not short and not misplaced:
        result = "compliant"
    else:
        result = "violation" if covered else "unclear"
    findings = []
    if short:
        findings.append(f"{best['cups']} of {agreed} agreed cups visible ({short} missing)")
    if misplaced:
        where = f"shelf {agreement.get('shelf')}" + (f", {agreement.get('zone')}" if agreement.get("zone") not in (None, "any", "front") else "")
        findings.append(f"{misplaced} cup(s) not in the agreed place ({where})")
    if best["gaps_on_shelf"]:
        findings.append(f"{best['gaps_on_shelf']} empty gap(s) on the agreed shelf")
    if result == "unclear":
        findings.append("the video did not cover the fridge well enough to be sure: a person should check")
    return {"status": "measured", "result": result, "agreed_cups": agreed, "agreement_id": agreement.get("id"),
            "cups_visible": best["cups"], "cups_in_place": best["cups_in_place"], "gaps_on_shelf": best["gaps_on_shelf"],
            "shelf_share": best["shelf_share"], "best_view": {"time_sec": best["time_sec"], "evidence": best["evidence"]},
            "findings": findings}


def compare(before, after):
    """Before / after: two agreement checks of the same shop (e.g. last visit and this one). Changes as numbers."""
    if before.get("status") != "measured" or after.get("status") != "measured":
        return {"status": "pending", "reason": "both visits need a measured agreement check"}
    out = {"status": "measured"}
    for key in ("cups_visible", "cups_in_place", "gaps_on_shelf"):
        out[key] = {"before": before[key], "after": after[key], "change": after[key] - before[key]}
    if before["shelf_share"] is not None and after["shelf_share"] is not None:
        out["shelf_share"] = {"before": before["shelf_share"], "after": after["shelf_share"],
                              "change": round(after["shelf_share"] - before["shelf_share"], 3)}
    out["result"] = {"before": before["result"], "after": after["result"]}
    better = out["cups_in_place"]["change"] > 0 or (before["result"] != "compliant" and after["result"] == "compliant")
    worse = out["cups_in_place"]["change"] < 0 or (before["result"] == "compliant" and after["result"] != "compliant")
    out["trend"] = "better" if better and not worse else "worse" if worse and not better else "same"
    return out
