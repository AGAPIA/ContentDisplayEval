#!/usr/bin/env python3
import argparse
import subprocess
import json
import os
import shutil
import yaml

def generate_metadata(image_dir, metadata_path):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    samples = [{"filename":fname} for fname in sorted(files)]
    with open(metadata_path, 'w') as f:
        json.dump(samples, f, indent=4)
    print(f"→ Metadata written ({len(samples)} entries) to {metadata_path}")

def run_inference(stage_dir, config_overrides):
    # load original config
    cfg_path = os.path.join(stage_dir, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # apply overrides
    for section, entries in config_overrides.items():
        cfg_section = cfg.setdefault(section, {})
        cfg_section.update(entries)

    # run inference
    print(f"\n=== Running inference in {stage_dir} ===")
    # pass overridden config via a temp file
    tmp_cfg = os.path.join(stage_dir, "config.tmp.yaml")
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(cfg, f)
    subprocess.run(
        ["python", "inference.py", "--config", os.path.basename(tmp_cfg)],
        cwd=stage_dir, check=True
    )
    os.remove(tmp_cfg)
    # return path to results
    out_dir = cfg["inference"]["output_dir"]
    return os.path.join(stage_dir, out_dir, "inference_results.json"), cfg

def main():
    parser = argparse.ArgumentParser(
        description="Run Stage1→Stage2 inference pipeline with gRPC thresholding.")
    parser.add_argument(
        "--input_dir", required=True,
        help="Folder containing raw images for Stage 1 inference.")
    args = parser.parse_args()

    project_root = os.getcwd()
    stage1_dir = os.path.join(project_root, "texture_stage1")
    stage2_dir = os.path.join(project_root, "texture_stage2")

    # 1) Prepare Stage1 metadata
    stage1_meta = os.path.join(stage1_dir, "metadata.tmp.json")
    generate_metadata(args.input_dir, stage1_meta)

    # 2) Run Stage1 inference
    s1_results, s1_cfg = run_inference(stage1_dir, {
        "inference": {
            "image_dir": args.input_dir,
            "metadata_path": os.path.basename(stage1_meta)
        }
    })

    # 3) Load Stage1 results & filter by threshold
    with open(s1_results) as f:
        entries = json.load(f)
    threshold = s1_cfg["inference"].get("threshold", 0.5)
    flagged = [e["filename"] for e in entries if e["anomaly_score"] >= threshold]
    print(f"\n→ {len(flagged)} / {len(entries)} frames passed threshold ≥ {threshold}")

    # 4) Prepare Stage2 input folder & metadata
    stage2_input = os.path.join(stage2_dir, "input_images")
    os.makedirs(stage2_input, exist_ok=True)
    for fn in flagged:
        src = os.path.join(args.input_dir, fn)
        dst = os.path.join(stage2_input, fn)
        if os.path.exists(src):
            shutil.copy(src, dst)
    stage2_meta = os.path.join(stage2_dir, "metadata.tmp.json")
    generate_metadata(stage2_input, stage2_meta)

    # 5) Run Stage2 inference
    _, s2_cfg = run_inference(stage2_dir, {
        "inference": {
            "image_dir": "input_images",
            "metadata_path": os.path.basename(stage2_meta)
        }
    })

    # 6) Clean up temp metadata files
    os.remove(stage1_meta)
    os.remove(stage2_meta)

    print("\n✅ Full pipeline completed.")

if __name__ == "__main__":
    main()
