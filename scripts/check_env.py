"""Sprint 1: check that the development environment is ready."""
import platform
import sys
from pathlib import Path


def check(name, fn):
    try:
        print(f"[OK]   {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {e}")


def ram():
    import psutil
    return f"{psutil.virtual_memory().total / 1024**3:.1f} GB total"


def opencv():
    import cv2
    return cv2.__version__


def torch_info():
    import torch
    gpu = "GPU available" if torch.cuda.is_available() else "CPU only (expected on this laptop)"
    return f"{torch.__version__} - {gpu}"


def ultralytics_info():
    import ultralytics
    return ultralytics.__version__


def imagehash_info():
    import imagehash
    return imagehash.__version__


def config():
    import yaml
    cfg = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    return f"loaded sections: {', '.join(yaml.safe_load(cfg.read_text()))}"


def main():
    print("ShelfSense environment check\n" + "-" * 30)
    check("Python", lambda: sys.version.split()[0])
    check("OS", lambda: f"{platform.system()} {platform.release()}")
    check("RAM", ram)
    check("OpenCV", opencv)
    check("PyTorch", torch_info)
    check("Ultralytics", ultralytics_info)
    check("ImageHash", imagehash_info)
    check("Config", config)


if __name__ == "__main__":
    main()
