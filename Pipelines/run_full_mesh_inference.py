#!/usr/bin/env python3
import argparse
import subprocess
import json
import os
import shutil
import yaml
from PIL import Image
import numpy as np

def generate_metadata(image_dir, metadata_path):
    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    samples = [{"image": fname, "mask": fname} for fname in files]
    with open(metadata_path, 'w') as f:
        json.dump(samples, f, indent=4)
    print(f"[Metadata] {len(samples)} entries written to {metadata_path}")

def run_inference(stage_dir, config_overrides):
    cfg_path = os.path.join(stage_dir, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # apply overrides in memory
    for section, updates in config_overrides.items():
        cfg.setdefault(section, {}).update(updates)

    # write temp config
    tmp_cfg = os.path.join(stage_dir, "config.tmp.yaml")
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(cfg, f)

    print(f"\n[Inference] Running in {stage_dir} using {os.path.basename(tmp_cfg)}")
    subprocess.run(
        ["python", "inference.py", "--config", os.path.basename(tmp_cfg)],
        cwd=stage_dir,
        check=True
    )

    os.remove(tmp_cfg)
    out_dir = cfg["inference"]["output_dir"]
    return os.path.join(stage_dir, out_dir), cfg

def filter_regions(mask_dir, filenames, area_threshold):
    selected = []
    for fname in filenames:
        path = os.path.join(mask_dir, fname)
        mask = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        if mask.mean() >= area_threshold:
            selected.append(fname)
    return selected

def main():
    parser = argparse.ArgumentParser(
        description="Run full mesh pipeline: Stage1 → Stage2 with region filtering"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Folder containing raw images for Stage 1"
    )
    parser.add_argument(
        "--area_thresh", type=float, default=0.01,
        help="Minimum fraction of mask area to pass to Stage 2 (default: 0.01)"
    )
    args = parser.parse_args()

    project = os.getcwd()
    s1 = os.path.join(project, "mesh_stage1")
    s2 = os.path.join(project, "mesh_stage2")

    # Stage 1 metadata
    meta1 = os.path.join(s1, "metadata.tmp.json")
    generate_metadata(args.input_dir, meta1)

    # Run Stage 1 inference
    out1_dir, cfg1 = run_inference(s1, {
        "inference": {
            "image_dir": args.input_dir,
            "metadata_path": os.path.basename(meta1)
        }
    })

    # Load inference_results.json
    results1 = os.path.join(out1_dir, "inference_results.json")
    with open(results1) as f:
        entries = json.load(f)
    # Filter by anomaly_flag and region area
    flagged = [e["filename"] for e in entries if e["prediction"] == 1]
    masks_dir = out1_dir  # where masks were saved with same filenames
    selected = filter_regions(masks_dir, flagged, args.area_thresh)
    print(f"\n[Filter] {len(selected)}/{len(flagged)} passed area ≥ {args.area_thresh}")

    # Prepare Stage 2 input
    s2_input = os.path.join(s2, "input_images")
    os.makedirs(s2_input, exist_ok=True)
    for fn in selected:
        src = os.path.join(args.input_dir, fn)
        dst = os.path.join(s2_input, fn)
        if os.path.exists(src):
            shutil.copy(src, dst)

    meta2 = os.path.join(s2, "metadata.tmp.json")
    samples2 = [{"image": fn, "mask": fn} for fn in selected]
    with open(meta2, "w") as f:
        json.dump(samples2, f, indent=4)
    print(f"[Metadata] {len(samples2)} entries written to {meta2}")

    # Run Stage 2 inference
    _, cfg2 = run_inference(s2, {
        "inference": {
            "image_dir": "input_images",
            "metadata_path": os.path.basename(meta2)
        }
    })

    # Cleanup
    os.remove(meta1)
    os.remove(meta2)
    print("\n[Done] Full mesh pipeline completed.")

if __name__ == "__main__":
    main()
