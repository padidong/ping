
import argparse, torch
from torch.serialization import add_safe_globals

# Try allowlist if needed (safe path)
try:
    from lightning.fabric.wrappers import _FabricModule
    add_safe_globals([_FabricModule])
except Exception:
    pass

try:
    from timm.models.efficientnet import EfficientNet as TIMM_EfficientNet
    add_safe_globals([TIMM_EfficientNet])
except Exception:
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source checkpoint file (.pt/.pth)")
    ap.add_argument("--dst", required=True, help="destination state_dict file (.pt)")
    args = ap.parse_args()

    # Unsafe load (trusted only) to extract state_dict
    obj = torch.load(args.src, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        # Might be already a state_dict
        sd = obj
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    else:
        raise RuntimeError("Cannot extract state_dict from given file.")

    # Clean prefixes
    sd_clean = {}
    for k, v in sd.items():
        nk = k.replace("module.", "").replace("model.", "")
        sd_clean[nk] = v

    torch.save(sd_clean, args.dst)
    print(f"Saved weights-only state_dict to {args.dst}")

if __name__ == "__main__":
    main()
