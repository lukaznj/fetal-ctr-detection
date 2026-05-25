import os
import sys
import csv
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add root directory to sys.path to import metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calculate_all_metrics

from dataset import load_image_and_masks
from model import bce_dice_loss, dice_coef

def ellipse_to_dict(ellipse):
    """Converts OpenCV ellipse to standard dictionary format."""
    if ellipse is None:
        return None
    (cx, cy), (w, h), kut = ellipse
    if w >= h:
        a, b = w / 2, h / 2
        kut_norm = kut % 180
    else:
        a, b = h / 2, w / 2
        kut_norm = (kut + 90) % 180
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "kut": float(kut_norm)}

def fit_ellipses(mask):
    """Returns the best fit ellipse for a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    return cv2.fitEllipse(contour)

def calculate_iou(pred_mask, true_mask):
    """Calculate IoU between two masks."""
    m1 = pred_mask > 0
    m2 = true_mask > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return None
    return float(intersection / union)

def visualize_and_save(img, true_c_ellipse, true_t_ellipse, pred_c_ellipse, pred_t_ellipse, 
                       image_id, metrics_res, save_dir):
    """Draws the ellipses on the image and saves it."""
    if len(img.shape) == 3 and img.shape[2] == 1:
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = (img * 255).astype(np.uint8)
        img_rgb = np.stack([img_rgb]*3, axis=-1)

    if true_c_ellipse:
        cv2.ellipse(img_rgb, true_c_ellipse, (0, 255, 0), 2)
    if true_t_ellipse:
        cv2.ellipse(img_rgb, true_t_ellipse, (0, 255, 0), 2)
        
    if pred_c_ellipse:
        cv2.ellipse(img_rgb, pred_c_ellipse, (0, 0, 255), 2)
    if pred_t_ellipse:
        cv2.ellipse(img_rgb, pred_t_ellipse, (0, 0, 255), 2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    true_ctr = metrics_res['ctr_gt']
    pred_ctr = metrics_res['ctr_auto']
    error = metrics_res['abs_err'] if metrics_res['abs_err'] is not None else -1
    combined_err = metrics_res['combined_err']
    cat = metrics_res['category']
    
    cv2.putText(img_rgb, f"True CTR: {true_ctr:.3f}" if true_ctr else "True CTR: N/A", 
                (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(img_rgb, f"Pred CTR: {pred_ctr:.3f}" if pred_ctr else "Pred CTR: N/A", 
                (10, 70), font, 1, (0, 0, 255), 2)
    cv2.putText(img_rgb, f"Abs Error: {error:.3f}", 
                (10, 110), font, 1, (255, 255, 0), 2)
    
    if combined_err is not None:
        cv2.putText(img_rgb, f"Comb Err: {combined_err:.3f} ({cat})", 
                    (10, 150), font, 1, (255, 0, 255), 2)
                
    save_path = os.path.join(save_dir, f"{image_id}_error_{error:.3f}.png")
    cv2.imwrite(save_path, img_rgb)


def save_graphs(errors, ctr_auto, ctr_gt, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Accuracy Analysis of CTR Detection (Deep Learning)", fontsize=14)

    axes[0].hist(errors, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(errors), color="red", linestyle="--",
                    label=f"MAE = {np.mean(errors):.3f}")
    axes[0].set_xlabel("Absolute CTR Error")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Error Distribution")
    axes[0].legend()

    if ctr_auto and ctr_gt:
        n = min(len(ctr_auto), len(ctr_gt))
        axes[1].scatter(ctr_gt[:n], ctr_auto[:n], alpha=0.65,
                        color="steelblue", edgecolors="navy")
        maximum = max(max(ctr_gt[:n]), max(ctr_auto[:n])) * 1.1
        lim = [0, maximum]
        axes[1].plot(lim, lim, "r--", label="Ideal match")
        axes[1].set_xlabel("CTR Ground Truth")
        axes[1].set_ylabel("CTR Automatic Detection")
        axes[1].set_title("Auto vs GT CTR")
        axes[1].legend()
        axes[1].set_xlim(lim)
        axes[1].set_ylim(lim)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_analysis.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Accuracy graph saved: {path}")


def save_results(results_list, output_dir):
    path = os.path.join(output_dir, "results_per_image.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["name", "ctr_auto", "ctr_gt", "abs_err", "rel_err",
                  "combined_err", "iou_heart", "iou_thorax", "n_candidates"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results_list:
            writer.writerow(r)
    print(f"Results per image saved: {path}")


def evaluate_model(model_path, test_img_dir, test_mask_dir, target_size=(512, 512), error_threshold=0.10, single_image=None, output_dir=None):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})
    
    test_ids = [f.split('.')[0] for f in os.listdir(test_img_dir) if f.endswith('.png')]
    if single_image:
        test_ids = [t for t in test_ids if t == single_image]
    test_ids.sort()
    
    errors = []
    rel_errors = []
    combined_errors = []
    iou_heart_list = []
    iou_thorax_list = []
    ctr_auto_list = []
    ctr_gt_list = []
    
    all_results = []
    
    # We will save DL results to results/deep_learning
    results_dir = os.path.abspath(output_dir) if output_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "deep_learning"))
    os.makedirs(results_dir, exist_ok=True)
    
    save_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nFOCUS CTR Detection - {len(test_ids)} image(s)")
    print(f"Dataset: {os.path.dirname(test_img_dir)}")
    print(f"Results: {results_dir}")
    print("-" * 60)
    
    for image_id in test_ids:
        try:
            img, mask = load_image_and_masks(image_id, test_img_dir, test_mask_dir, target_size)
            
            true_cardiac = (mask[:, :, 0] * 255).astype(np.uint8)
            true_thorax = (mask[:, :, 1] * 255).astype(np.uint8)
            
            true_c_ellipse = fit_ellipses(true_cardiac)
            true_t_ellipse = fit_ellipses(true_thorax)
            
            img_batch = np.expand_dims(img, axis=0)
            pred = model.predict(img_batch, verbose=0)[0]
            
            pred_cardiac = ((pred[:, :, 0] > 0.5) * 255).astype(np.uint8)
            pred_thorax = ((pred[:, :, 1] > 0.5) * 255).astype(np.uint8)
            
            pred_c_ellipse = fit_ellipses(pred_cardiac)
            pred_t_ellipse = fit_ellipses(pred_thorax)

            auto_c_dict = ellipse_to_dict(pred_c_ellipse)
            auto_t_dict = ellipse_to_dict(pred_t_ellipse)
            gt_c_dict = ellipse_to_dict(true_c_ellipse)
            gt_t_dict = ellipse_to_dict(true_t_ellipse)

            iou_srce = calculate_iou(pred_cardiac, true_cardiac)
            iou_toraks = calculate_iou(pred_thorax, true_thorax)

            metrics_res = calculate_all_metrics(
                auto_c_dict, auto_t_dict, 
                gt_c_dict, gt_t_dict, 
                iou_srce, iou_toraks
            )
            
            true_ctr = metrics_res['ctr_gt']
            pred_ctr = metrics_res['ctr_auto']
            error = metrics_res['abs_err']
            comb_err = metrics_res['combined_err']
            
            ctr_auto_str = f"{pred_ctr:.3f}" if pred_ctr is not None else "N/A"
            ctr_gt_str = f"{true_ctr:.3f}" if true_ctr is not None else "N/A"
            err_str = f"{error:.3f}" if error is not None else "N/A"
            comb_err_str = f"{comb_err:.3f} ({metrics_res['category']})" if comb_err is not None else "N/A"
            
            print(
                f"  {image_id}: CTR_auto={ctr_auto_str:>6} | "
                f"CTR_GT={ctr_gt_str:>6} | "
                f"AbsErr={err_str:>6} | CombErr={comb_err_str} | cand=N/A"
            )
            
            if true_ctr is not None and pred_ctr is not None:
                errors.append(error)
                rel_errors.append(metrics_res['rel_err'])
                ctr_auto_list.append(pred_ctr)
                ctr_gt_list.append(true_ctr)
                
                if comb_err is not None:
                    combined_errors.append(comb_err)
                if iou_srce is not None:
                    iou_heart_list.append(iou_srce)
                if iou_toraks is not None:
                    iou_thorax_list.append(iou_toraks)
                
                all_results.append({
                    "name": image_id,
                    "ctr_auto": pred_ctr,
                    "ctr_gt": true_ctr,
                    "abs_err": error,
                    "rel_err": metrics_res['rel_err'],
                    "combined_err": comb_err,
                    "iou_heart": iou_srce,
                    "iou_thorax": iou_toraks,
                    "n_candidates": 0
                })
                
                if error >= error_threshold:
                    visualize_and_save(img, true_c_ellipse, true_t_ellipse, pred_c_ellipse, pred_t_ellipse, 
                                       image_id, metrics_res, save_dir)
            else:
                visualize_and_save(img, true_c_ellipse, true_t_ellipse, pred_c_ellipse, pred_t_ellipse, 
                                   image_id, metrics_res, save_dir)
                
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            
    if errors:
        print("\n" + "=" * 60)
        print("ACCURACY ANALYSIS")
        print("=" * 60)
        print(f"Number of analyzed images:      {len(test_ids)}")
        print(f"Successful CTR detections:      {len(errors)}/{len(test_ids)}")
        print(f"Mean Absolute Error (MAE):      {np.mean(errors):.4f}")
        print(f"Median Absolute Error:          {np.median(errors):.4f}")
        print(f"Error Standard Deviation:       {np.std(errors):.4f}")
        print(f"Max Error:                      {np.max(errors):.4f}")
        print(f"Min Error:                      {np.min(errors):.4f}")
        print(f"Mean Relative Error (1-P_CTR):  {100 * np.mean(rel_errors):.2f}%")
        
        if combined_errors:
            print(f"Mean Combined Metric Error:     {np.mean(combined_errors):.4f}")

        if iou_heart_list:
            print(f"Mean IoU Heart:                 {np.mean(iou_heart_list):.3f}")
        if iou_thorax_list:
            print(f"Mean IoU Thorax:                {np.mean(iou_thorax_list):.3f}")

        for tol in [0.05, 0.10, 0.15]:
            within = sum(1 for g in errors if g <= tol)
            print(
                f"Within +/- {tol:.0%} CTR tolerance: "
                f"{within}/{len(errors)} ({100 * within / len(errors):.1f}%)"
            )
            
        save_graphs(errors, ctr_auto_list, ctr_gt_list, results_dir)
        save_results(all_results, results_dir)
        
        summary = {
            "n_images": len(test_ids),
            "n_detections": len(errors),
            "mae": float(np.mean(errors)),
            "medae": float(np.median(errors)),
            "std": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "rel_error_mean": float(np.mean(rel_errors)) if rel_errors else None,
            "combined_error_mean": float(np.mean(combined_errors)) if combined_errors else None,
            "iou_heart_mean": float(np.mean(iou_heart_list)) if iou_heart_list else None,
            "iou_thorax_mean": float(np.mean(iou_thorax_list)) if iou_thorax_list else None,
        }
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved: {summary_path}")

    else:
        print("\nCould not calculate MAE. No valid predictions.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate U-Net on testing dataset")
    parser.add_argument("--model", type=str, default="unet_augmented.keras",
                        help="Model filename located in deep_learning/models/")
    parser.add_argument("--image", type=str, default=None,
                        help="Image number (e.g. 001). If not set, processes all.")
    parser.add_argument("--split", type=str, default="testing",
                        choices=["training", "validation", "testing"],
                        help="Dataset split to process.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Manual path to dataset folder.")
    parser.add_argument("--output", type=str, default=None,
                        help="Folder to save results.")
    parser.add_argument("--visualize", action="store_true", 
                        help="Save visualizations for all images, not just failed ones")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dataset) if args.dataset else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    test_img_dir = os.path.join(base_dir, args.split, "images")
    test_mask_dir = os.path.join(base_dir, args.split, "annfiles_mask")
    
    model_file = os.path.join(os.path.dirname(__file__), "models", args.model)
    if os.path.exists(model_file):
        threshold = 0.0 if args.visualize else 0.10
        image_name = args.image.zfill(3) if args.image else None
        evaluate_model(model_file, test_img_dir, test_mask_dir, error_threshold=threshold, single_image=image_name, output_dir=args.output)
    else:
        print(f"Model file '{model_file}' not found. Please train the model first.")
