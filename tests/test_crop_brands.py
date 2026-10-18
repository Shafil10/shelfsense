from src.vlm.crop_brands import brand_of, brands_of


def test_brand_of_reads_json_and_null():
    assert brand_of('{"brand": "Coca Cola"}') == "coca-cola"
    assert brand_of('```json\n{"brand": "Sprite Zero"}\n```') == "sprite"
    assert brand_of('{"brand": null}') is None
    assert brand_of("I cannot tell.") is None


def test_brands_of_normalises_and_dedups():
    assert brands_of('{"brands": ["Coke", "Coca-Cola", "Fanta", null, 7]}') == {"coca-cola", "fanta"}
    assert brands_of('{"brands": "Fanta"}') == set()
    assert brands_of("no json") == set()
