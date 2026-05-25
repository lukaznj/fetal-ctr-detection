import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add root directory to sys.path to import metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calculate_all_metrics

from preprocessing import preprocess_image
from detection import detect_edges, detect_ellipses_hough, select_ellipses, get_ellipse_iou
from visualization import visualize, save_graphs, save_results


DATASET_DIR = Path(__file__).parent.parent / "dataset"
SPLIT = "testing"
OUTPUT_DIR = Path("results/classical")


def load_annotation(ann_path: Path) -> dict:
    annotations = {}
    if not ann_path.exists():
        return annotations

    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            cx, cy, a, b, angle, cls = (
                float(parts[0]), float(parts[1]),
                float(parts[2]), float(parts[3]),
                float(parts[4]), parts[5]
            )
            annotations[cls] = {
                "cx": cx, "cy": cy,
                "a": max(a, b),
                "b": min(a, b),
                "kut": angle
            }

    return annotations



def analyze_image(name: str, img_dir: Path, ann_dir: Path,
                  output_dir: Path, save_all_flag: bool = False, error_threshold: float = 0.10) -> dict:
    """
    Complete pipeline for a single image. Returns a dict with results.
    """
    ann_path = ann_dir / f"{name}.txt"
    img_path = img_dir / f"{name}.png"
    if not img_path.exists():
        img_path = img_dir / f"{name}.jpg"

    annotation = load_annotation(ann_path)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  {name}: image not found")
        return {
            "name": name,
            "metrics": None,
            "n_candidates": 0
        }

    gray, preprocessed = preprocess_image(img_bgr)
    edges = detect_edges(preprocessed)
    candidates = detect_ellipses_hough(edges, gray.shape)
    detection = select_ellipses(candidates, gray.shape)

    auto_cardiac = detection.get("cardiac") if detection else None
    auto_thorax = detection.get("thorax") if detection else None
    gt_cardiac = annotation.get("cardiac")
    gt_thorax = annotation.get("thorax")

    iou_heart = get_ellipse_iou(auto_cardiac, gt_cardiac, gray.shape)
    iou_thorax = get_ellipse_iou(auto_thorax, gt_thorax, gray.shape)

    metrics_res = calculate_all_metrics(
        auto_cardiac, auto_thorax,
        gt_cardiac, gt_thorax,
        iou_heart, iou_thorax
    )

    ctr_auto = metrics_res["ctr_auto"]
    ctr_gt = metrics_res["ctr_gt"]
    abs_err = metrics_res["abs_err"]
    combined_err = metrics_res["combined_err"]

    ctr_auto_str = f"{ctr_auto:.3f}" if ctr_auto is not None else "N/A"
    ctr_gt_str = f"{ctr_gt:.3f}" if ctr_gt is not None else "N/A"
    err_str = f"{abs_err:.3f}" if abs_err is not None else "N/A"
    comb_err_str = f"{combined_err:.3f} ({metrics_res['category']})" if combined_err is not None else "N/A"
    
    candidates_str = str(len(candidates)).rjust(3)
    
    print(
        f"  {name}: CTR_auto={ctr_auto_str:>6} | "
        f"CTR_GT={ctr_gt_str:>6} | "
        f"AbsErr={err_str:>6} | CombErr={comb_err_str} | cand={candidates_str}"
    )

    # Save visualization logic (same as deep learning)
    save_dir = output_dir / "visualizations"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    if abs_err is None or abs_err >= error_threshold or save_all_flag:
        visualize(img_bgr, gray, edges, detection, annotation, metrics_res, name, save_dir)

    return {
        "name": name,
        "metrics": metrics_res,
        "n_candidates": len(candidates)
    }



def accuracy_analysis(results: list[dict], output_dir: Path):
    valid_results = [r for r in results if r["metrics"] is not None and r["metrics"]["abs_err"] is not None]
    
    errors = [r["metrics"]["abs_err"] for r in valid_results]
    rel_errors = [r["metrics"]["rel_err"] for r in valid_results if r["metrics"]["rel_err"] is not None]
    combined_errors = [r["metrics"]["combined_err"] for r in valid_results if r["metrics"]["combined_err"] is not None]
    
    ctr_auto_all = [r["metrics"]["ctr_auto"] for r in valid_results]
    ctr_gt_all = [r["metrics"]["ctr_gt"] for r in valid_results]
    
    iou_heart = [r["metrics"]["metrics_dict"]["iou_s"] for r in valid_results if r["metrics"]["metrics_dict"]["iou_s"] is not None]
    iou_thorax = [r["metrics"]["metrics_dict"]["iou_t"] for r in valid_results if r["metrics"]["metrics_dict"]["iou_t"] is not None]

    if not errors:
        print("\nNot enough data for accuracy analysis.")
        return

    print("\n" + "=" * 60)
    print("ACCURACY ANALYSIS")
    print("=" * 60)
    print(f"Number of analyzed images:      {len(results)}")
    print(f"Successful CTR detections:      {len(errors)}/{len(results)}")
    print(f"Mean Absolute Error (MAE):      {np.mean(errors):.4f}")
    print(f"Median Absolute Error:          {np.median(errors):.4f}")
    print(f"Error Standard Deviation:       {np.std(errors):.4f}")
    print(f"Max Error:                      {np.max(errors):.4f}")
    print(f"Min Error:                      {np.min(errors):.4f}")

    if rel_errors:
        print(f"Mean Relative Error (1-P_CTR):  {100 * np.mean(rel_errors):.2f}%")
        
    if combined_errors:
        print(f"Mean Combined Metric Error:     {np.mean(combined_errors):.4f}")

    if iou_heart:
        print(f"Mean IoU Heart:                 {np.mean(iou_heart):.3f}")
    if iou_thorax:
        print(f"Mean IoU Thorax:                {np.mean(iou_thorax):.3f}")

    for tol in [0.05, 0.10, 0.15]:
        within = sum(1 for g in errors if g <= tol)
        print(
            f"Within +/- {tol:.0%} CTR tolerance: "
            f"{within}/{len(errors)} ({100 * within / len(errors):.1f}%)"
        )

    save_graphs(errors, ctr_auto_all, ctr_gt_all, output_dir)
    save_results(valid_results, output_dir)

    summary = {
        "n_images": len(results),
        "n_detections": len(errors),
        "mae": float(np.mean(errors)),
        "medae": float(np.median(errors)),
        "std": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "rel_error_mean": float(np.mean(rel_errors)) if rel_errors else None,
        "combined_error_mean": float(np.mean(combined_errors)) if combined_errors else None,
        "iou_heart_mean": float(np.mean(iou_heart)) if iou_heart else None,
        "iou_thorax_mean": float(np.mean(iou_thorax)) if iou_thorax else None,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved: {output_dir / 'summary.json'}")



def main():
    parser = argparse.ArgumentParser(description="CTR detection on FOCUS dataset")
    parser.add_argument("--image", type=str, default=None,
                        help="Image number (e.g. 001). If not set, processes all.")
    parser.add_argument("--split", type=str, default=SPLIT,
                        choices=["training", "validation", "testing"],
                        help="Dataset split to process.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Manual path to dataset folder.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Folder to save results.")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualizations for all images, not just failed ones")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset) if args.dataset else DATASET_DIR
    split_dir = dataset_dir / args.split
    ann_dir = split_dir / "annfiles_ellipse"
    img_dir = split_dir / "images"
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        print(f"No annotation files in {ann_dir}")
        return

    if args.image:
        name = args.image.zfill(3)
        ann_files = [ann_dir / f"{name}.txt"]
        if not ann_files[0].exists():
            print(f"Annotation for image {name} does not exist.")
            return

    print(f"\nFOCUS CTR Detection - {len(ann_files)} image(s)")
    print(f"Dataset: {split_dir}")
    print(f"Results: {output_dir}")
    print("-" * 60)

    results = []
    threshold = 0.0 if args.visualize else 0.10
    
    for ann_path in ann_files:
        name = ann_path.stem
        res = analyze_image(
            name,
            img_dir=img_dir,
            ann_dir=ann_dir,
            output_dir=output_dir,
            save_all_flag=args.visualize,
            error_threshold=threshold
        )
        results.append(res)

    if len(results) > 1:
        accuracy_analysis(results, output_dir)
    elif results:
        r = results[0]
        if r["metrics"]:
            m = r["metrics"]
            print(f"\nCTR auto:      {m['ctr_auto']:.4f}" if m["ctr_auto"] is not None else "\nCTR auto: N/A")
            print(f"CTR GT:        {m['ctr_gt']:.4f}" if m["ctr_gt"] is not None else "CTR GT:   N/A")
            print(f"Abs. error:    {m['abs_err']:.4f}" if m["abs_err"] is not None else "Error:    N/A")
            print(f"Combined err:  {m['combined_err']:.4f} ({m['category']})" if m['combined_err'] is not None else "Comb. Err: N/A")
        else:
            print("\nAnalysis failed.")


if __name__ == "__main__":
    main()
