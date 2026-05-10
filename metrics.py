import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from utils import CustomMRIDataset

def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def calc_class_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

def compute_all_metrics():
    summary_data = []
    time_comparison = []
    
    baseline_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "baseline_predictions.csv"))
    
    # Baseline sizes
    orig_size_bytes = get_dir_size(config.DATASET_ROOT)
    orig_size_mb = orig_size_bytes / (1024 * 1024)
    orig_avg_kb = np.mean([os.path.getsize(p) for p in baseline_df['image_path']]) / 1024.0
    
    # Baseline times
    b_avg_prep = baseline_df["original_preprocessing_time_ms"].mean()
    b_avg_inf = baseline_df["original_inference_time_ms"].mean()
    b_avg_e2e = baseline_df["original_end_to_end_time_ms"].mean()
    
    # Baseline Classification metrics
    b_metrics = calc_class_metrics(baseline_df["true_label"], baseline_df["baseline_pred_label"])
    
    summary_data.append({
        "dataset_version": "original_baseline",
        "total_size_mb": orig_size_mb,
        "avg_file_size_kb": orig_avg_kb,
        "avg_preprocessing_time_ms": b_avg_prep,
        "avg_inference_time_ms": b_avg_inf,
        "avg_end_to_end_time_ms": b_avg_e2e,
        **b_metrics,
        "agreement_with_baseline_percent": 100.0,
        "avg_confidence_difference": 0.0,
        "changed_decision_count": 0,
        "size_reduction_percent": 0.0,
        "time_reduction_percent": 0.0
    })
    
    time_comparison.append({
        "dataset_version": "original_baseline",
        "avg_preprocessing_time_ms": b_avg_prep,
        "avg_inference_time_ms": b_avg_inf,
        "avg_end_to_end_time_ms": b_avg_e2e,
        "preprocessing_time_ratio": 1.0,
        "inference_time_ratio": 1.0,
        "end_to_end_time_ratio": 1.0,
        "preprocessing_time_change_percent": 0.0,
        "inference_time_change_percent": 0.0,
        "end_to_end_time_change_percent": 0.0
    })
    
    for level in config.COMPRESSION_RATIOS.keys():
        level_dir = os.path.join(config.PROCESSED_DATASETS_DIR, level)
        proc_df = pd.read_csv(os.path.join(config.RESULTS_DIR, f"{level}_predictions.csv"))
        comp_df = pd.read_csv(os.path.join(config.RESULTS_DIR, f"{level}_vs_baseline.csv"))
        
        # Proc sizes
        p_size_bytes = get_dir_size(level_dir)
        p_size_mb = p_size_bytes / (1024 * 1024)
        p_avg_kb = np.mean([os.path.getsize(p) for p in proc_df['processed_image_path']]) / 1024.0
        
        # Times
        p_avg_prep = proc_df["fft_clahe_preprocessing_time_ms"].mean()
        p_avg_inf = proc_df["fft_clahe_inference_time_ms"].mean()
        p_avg_e2e = proc_df["fft_clahe_end_to_end_time_ms"].mean()
        
        # Class metrics
        p_metrics = calc_class_metrics(proc_df["true_label"], proc_df["pred_label"])
        
        # Comparisons
        agreement = comp_df["prediction_match"].mean() * 100
        conf_diff = np.abs(comp_df["baseline_confidence"] - comp_df["compressed_confidence"]).mean()
        changed = len(comp_df[comp_df["prediction_match"] == 0])
        
        size_red = ((orig_size_bytes - p_size_bytes) / orig_size_bytes) * 100
        time_red = ((b_avg_e2e - p_avg_e2e) / b_avg_e2e) * 100
        
        summary_data.append({
            "dataset_version": level,
            "total_size_mb": p_size_mb,
            "avg_file_size_kb": p_avg_kb,
            "avg_preprocessing_time_ms": p_avg_prep,
            "avg_inference_time_ms": p_avg_inf,
            "avg_end_to_end_time_ms": p_avg_e2e,
            **p_metrics,
            "agreement_with_baseline_percent": agreement,
            "avg_confidence_difference": conf_diff,
            "changed_decision_count": changed,
            "size_reduction_percent": size_red,
            "time_reduction_percent": time_red
        })
        
        time_comparison.append({
            "dataset_version": level,
            "avg_preprocessing_time_ms": p_avg_prep,
            "avg_inference_time_ms": p_avg_inf,
            "avg_end_to_end_time_ms": p_avg_e2e,
            "preprocessing_time_ratio": p_avg_prep / b_avg_prep,
            "inference_time_ratio": p_avg_inf / b_avg_inf,
            "end_to_end_time_ratio": p_avg_e2e / b_avg_e2e,
            "preprocessing_time_change_percent": ((p_avg_prep - b_avg_prep) / b_avg_prep) * 100,
            "inference_time_change_percent": ((p_avg_inf - b_avg_inf) / b_avg_inf) * 100,
            "end_to_end_time_change_percent": ((p_avg_e2e - b_avg_e2e) / b_avg_e2e) * 100
        })
        
    df_sum = pd.DataFrame(summary_data)
    try:
        df_sum.to_csv(os.path.join(config.RESULTS_DIR, "final_summary_metrics.csv"), index=False)
    except PermissionError:
        print("Warning: Could not write final_summary_metrics.csv (file may be locked). Retrying...")
        import time
        time.sleep(1)
        df_sum.to_csv(os.path.join(config.RESULTS_DIR, "final_summary_metrics.csv"), index=False)
    
    with open(os.path.join(config.RESULTS_DIR, "final_summary_metrics.json"), "w") as f:
        json.dump(summary_data, f, indent=4)
        
    df_time = pd.DataFrame(time_comparison)
    try:
        df_time.to_csv(os.path.join(config.RESULTS_DIR, "time_comparison.csv"), index=False)
    except PermissionError:
        print("Warning: Could not write time_comparison.csv (file may be locked). Retrying...")
        import time
        time.sleep(1)
        df_time.to_csv(os.path.join(config.RESULTS_DIR, "time_comparison.csv"), index=False)
    
    print("Final summary metrics and time comparisons generated.")

if __name__ == "__main__":
    compute_all_metrics()