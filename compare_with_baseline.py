import os
import pandas as pd
import config

def compare_baselines():
    baseline_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "baseline_predictions.csv"))
    baseline_df = baseline_df.rename(columns={'image_path': 'original_image_path'})
    
    for level in config.COMPRESSION_RATIOS.keys():
        proc_df = pd.read_csv(os.path.join(config.RESULTS_DIR, f"{level}_predictions.csv"))
        
        merged = pd.merge(baseline_df, proc_df, on=["original_image_path", "true_label"])
        
        results = []
        for _, r in merged.iterrows():
            b_correct = 1 if r["baseline_pred_label"] == r["true_label"] else 0
            c_correct = 1 if r["pred_label"] == r["true_label"] else 0
            p_match = 1 if r["baseline_pred_label"] == r["pred_label"] else 0
            
            # Use 'inference_time_ms' which we generated, and fallback to whatever if needed
            results.append({
                "original_image_path": r["original_image_path"],
                "processed_image_path": r["processed_image_path"],
                "true_label": r["true_label"],
                "baseline_pred": r["baseline_pred_label"],
                "compressed_pred": r["pred_label"],
                "baseline_correct": b_correct,
                "compressed_correct": c_correct,
                "prediction_match": p_match,
                "baseline_confidence": r["baseline_confidence"],
                "compressed_confidence": r["confidence"],
                "baseline_time_ms": r.get("inference_time_ms_x", 0),
                "compressed_time_ms": r.get("inference_time_ms_y", 0)
            })
            
        out_df = pd.DataFrame(results)
        out_df.to_csv(os.path.join(config.RESULTS_DIR, f"{level}_vs_baseline.csv"), index=False)
        print(f"Comparison generated: {level}_vs_baseline.csv")

if __name__ == "__main__":
    compare_baselines()