"""visual.py -- Phase 2.2

Extracts visual embeddings from QR images using frozen MobileNetV2 (1280-dim
GAP layer), or a raw_pixels fallback (69x69 grayscale flattened, 4761 dims)
for testing without torch/GPU.

Implements checkpointed, resumable extraction to survive crashes on
memory-constrained machines (see REBUILD SPEC section 7 for the real
segfaults this was built to avoid).
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from server.project.scripts.utils import get_logger, set_seed

logger = get_logger("visual")

MOBILENET_DIM = 1280
RAW_PIXELS_DIM = 4761  # 69*69
RAW_PIXELS_SIZE = 69


def _load_mobilenet():
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    torch.set_num_threads(1)  # reduces peak memory from BLAS/MKL thread scratch buffers

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Identity()
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def extract_cnn_features(manifest, qr_dir, checkpoint_path, batch_size=8, checkpoint_every=25):
    import torch

    model, preprocess = _load_mobilenet()
    qr_dir = Path(qr_dir)
    checkpoint_path = Path(checkpoint_path)

    filenames = manifest["filename"].tolist()
    n_total = len(filenames)

    start_idx = 0
    results = []
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path)
        results = [checkpoint]
        start_idx = checkpoint.shape[0]
        logger.info(f"Resuming from checkpoint: {start_idx}/{n_total} already processed")

    remaining_filenames = filenames[start_idx:]
    n_batches = (len(remaining_filenames) + batch_size - 1) // batch_size

    new_embeddings = []
    for b in tqdm(range(n_batches), desc="Extracting MobileNetV2 features"):
        batch_files = remaining_filenames[b * batch_size:(b + 1) * batch_size]
        imgs = []
        for fname in batch_files:
            img = Image.open(qr_dir / fname).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs)

        with torch.no_grad():
            emb = model(x).numpy()
        new_embeddings.append(emb)

        del x, emb, imgs
        if (b + 1) % checkpoint_every == 0 or b == n_batches - 1:
            all_so_far = results + new_embeddings
            combined = np.concatenate(all_so_far, axis=0) if all_so_far else np.zeros((0, MOBILENET_DIM))
            np.save(checkpoint_path, combined)
            results = [combined]
            new_embeddings = []
            gc.collect()

    final = results[0] if results else np.zeros((0, MOBILENET_DIM))
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return final


def extract_raw_pixels(manifest, qr_dir):
    qr_dir = Path(qr_dir)
    feats = []
    for fname in tqdm(manifest["filename"].tolist(), desc="Extracting raw-pixel features"):
        img = Image.open(qr_dir / fname).convert("L").resize(
            (RAW_PIXELS_SIZE, RAW_PIXELS_SIZE), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
        feats.append(arr)
    return np.stack(feats, axis=0)


def run(manifest_path, qr_dir, out_path, backbone="mobilenet_v2", batch_size=8, checkpoint_path=None, seed=42):
    set_seed(seed)
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    if backbone == "mobilenet_v2":
        if checkpoint_path is None:
            checkpoint_path = out_path.with_suffix(".checkpoint.npy")
        features = extract_cnn_features(manifest, qr_dir, checkpoint_path, batch_size=batch_size)
    elif backbone == "raw_pixels":
        features = extract_raw_pixels(manifest, qr_dir)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    np.save(out_path, features)
    logger.info(f"Wrote visual features {features.shape} -> {out_path} (backbone={backbone})")
    return features


def build_parser():
    p = argparse.ArgumentParser(description="Extract visual QR features (Phase 2.2).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--qr-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "raw_pixels"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--checkpoint-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        manifest_path=args.manifest,
        qr_dir=args.qr_dir,
        out_path=args.out,
        backbone=args.backbone,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )
