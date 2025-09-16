import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, accuracy_score

# Local imports (match your train.py)
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.cadgl import CADGL


SCEN_MAP = {
    (1, 0, 0): "{a}",
    (0, 1, 0): "{v}",
    (0, 0, 1): "{t}",
    (1, 1, 0): "{a,v}",
    (1, 0, 1): "{a,t}",
    (0, 1, 1): "{v,t}",
    (1, 1, 1): "{a,v,t}",
}


class NPZDataset(Dataset):
    """Thin wrapper over one or more NPZ files."""
    def __init__(self, npz_files: List[Path]):
        self._index = []  # (file_idx, row_idx)
        self._files = []
        for p in npz_files:
            data = np.load(p)
            # sanity checks (minimal)
            for key in ("audio", "visual", "text", "mask", "label"):
                if key not in data:
                    raise ValueError(f"{p} missing field: {key}")
            n = data["label"].shape[0]
            start = len(self._index)
            self._index.extend([(len(self._files), i) for i in range(n)])
            self._files.append({k: data[k] for k in data.files})
        self._len = len(self._index)

        # dtypes for torch
        self._f32 = np.float32
        self._i64 = np.int64

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        fidx, ridx = self._index[idx]
        f = self._files[fidx]
        a = f["audio"][ridx].astype(self._f32)
        v = f["visual"][ridx].astype(self._f32)
        t = f["text"][ridx].astype(self._f32)
        m = f["mask"][ridx].astype(np.int64)  # shape [3]
        y = f["label"][ridx].astype(self._i64)
        return {
            "audio": torch.from_numpy(a),
            "visual": torch.from_numpy(v),
            "text": torch.from_numpy(t),
            "mask": torch.from_numpy(m),
            "label": torch.tensor(y, dtype=torch.long),
        }


def load_npz_list(data_path: Path) -> List[Path]:
    if data_path.is_file():
        return [data_path]
    files = sorted(list(data_path.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {data_path}")
    return files


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def batch_to_model_inputs(batch):
    # Turn per-item tensors into batched tensors
    audio = batch["audio"]
    visual = batch["visual"]
    text = batch["text"]
    mask = batch["mask"]  # [B,3]
    # If any modality is entirely zero for a sample and mask says missing, pass as is.
    return {
        "audio": audio,
        "visual": visual,
        "text": text,
        "modality_masks": mask,
    }


def scenario_keys(mask_tensor: torch.Tensor) -> List[str]:
    """Return scenario key per sample, e.g., '{a}', '{v,t}', ..."""
    mask_np = mask_tensor.detach().cpu().numpy().astype(int)
    keys = []
    for row in mask_np:
        key = SCEN_MAP.get(tuple(row.tolist()), "{unknown}")
        keys.append(key)
    return keys


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    wavg = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"acc": float(acc), "wavg_f1": float(wavg)}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    model.eval()
    all_y, all_p, all_keys = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            inp = batch_to_model_inputs(batch)
            out = model(**inp)
            logits = out["logits"]  # [B, C]
            pred = logits.argmax(dim=-1)

            all_y.extend(batch["label"].detach().cpu().tolist())
            all_p.extend(pred.detach().cpu().tolist())
            all_keys.extend(scenario_keys(batch["mask"]))

    # overall
    metrics_all = compute_metrics(all_y, all_p, num_classes)

    # by scenario
    metrics_by = {}
    keys_unique = sorted(set(all_keys), key=lambda k: list(SCEN_MAP.values()).index(k) if k in SCEN_MAP.values() else 999)
    for k in keys_unique:
        idxs = [i for i, kk in enumerate(all_keys) if kk == k]
        if not idxs:
            continue
        y_k = [all_y[i] for i in idxs]
        p_k = [all_p[i] for i in idxs]
        metrics_by[k] = compute_metrics(y_k, p_k, num_classes)

    return metrics_all, metrics_by


def load_config(path: Path):
    import yaml
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    class C:  # light dot-access
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    return C(raw)


def main():
    ap = argparse.ArgumentParser(description="Evaluate CADGL on pre-extracted features (paper release).")
    ap.add_argument("--config", type=str, required=True, help="YAML config (dims, num_classes, etc.)")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--data", type=str, required=True, help="NPZ file or directory of NPZ shards")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out_dir", type=str, default="eval_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Eval] device = {device}")

    # config & model
    cfg = load_config(Path(args.config))
    model = CADGL(cfg).to(device)

    # load checkpoint (model_state_dict expected)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print("[Eval] checkpoint loaded.")

    # data
    npz_list = load_npz_list(Path(args.data))
    ds = NPZDataset(npz_list)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # run eval
    overall, by_scen = evaluate(model, dl, device, num_classes=cfg.num_classes)

    # print brief
    print("\n=== Overall ===")
    print(json.dumps(overall, indent=2))
    print("\n=== By scenario ===")
    for k, v in by_scen.items():
        print(f"{k}: {json.dumps(v)}")

    # save artifacts
    (out_dir / "overall.json").write_text(json.dumps(overall, indent=2))
    (out_dir / "by_scenario.json").write_text(json.dumps(by_scen, indent=2))

    # optional CSV for tables
    import csv
    with open(out_dir / "by_scenario.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "acc", "macro_f1", "wavg_f1"])
        for k, v in by_scen.items():
            w.writerow([k, f"{v['acc']:.4f}", f"{v['macro_f1']:.4f}", f"{v['wavg_f1']:.4f}"])
    print(f"\n[Eval] results saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()