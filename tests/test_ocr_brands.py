from src.vlm.ocr_brands import load_catalogue, match_brand

CAT = {"coca-cola": "coca-cola", "coke": "coca-cola", "doi kham": "doi kham", "a&w": "a&w",
       "mug": "mug", "canada dry": "canada dry", "sprite": "sprite"}


def test_exact_and_alias():
    assert match_brand("Coca-Cola Original Taste", CAT)[0] == "coca-cola"
    assert match_brand("COKE zero", CAT)[0] == "coca-cola"


def test_ocr_slip_is_snapped():
    assert match_brand("Sprlte lemon lime", CAT)[0] == "sprite"
    assert match_brand("CANADA DRV ginger ale", CAT)[0] == "canada dry"


def test_multiword_brand_without_space():
    assert match_brand("DOIKHAM tomato juice", CAT)[0] == "doi kham"


def test_short_names_need_an_exact_word():
    assert match_brand("A&W root beer", CAT)[0] == "a&w"
    assert match_brand("mugo", CAT)[0] is None            # no fuzzy match for 3-letter names
    assert match_brand("smug", CAT)[0] is None


def test_unrelated_text_is_unknown():
    assert match_brand("100% natural 250 ml", CAT)[0] is None
    assert match_brand("", CAT)[0] is None


def test_catalogue_file_loads_with_aliases():
    cat = load_catalogue()
    assert cat["coke"] == "coca-cola"
    assert "coco fresh" in cat.values()
    assert len(set(cat.values())) > 150


def test_text_goes_to_the_box_that_holds_it():
    from src.vlm.ocr_brands import assign_texts
    boxes = {"left": (0, 0, 100, 200), "right": (100, 0, 200, 200)}
    texts = [{"text": "Sprite", "box": [10, 50, 90, 70]},       # inside left
             {"text": "Fanta", "box": [120, 50, 190, 70]},      # inside right
             {"text": "mostly right", "box": [80, 10, 180, 20]},  # 80 % in right
             {"text": "price", "box": [0, 250, 50, 270]}]       # below both boxes: nobody's
    got = assign_texts(texts, boxes)
    assert [t["text"] for t in got["left"]] == ["Sprite"]
    assert [t["text"] for t in got["right"]] == ["Fanta", "mostly right"]
