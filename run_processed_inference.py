import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config
from ml_core.model import get_trained_model
from utils import CustomMRIDataset, get_inference_transform

def run_processed():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = get_trained_model(config.MODEL_PATH, device)
    transform = get_inference_transform()
    
    # Load mapping of generation times to add to overall preprocessing
    gen_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "generation_times.csv"))
    gen_map = {}
    for _, r in gen_df.iterrows():
        gen_map[r["processed_image_path"]] = r["fft_clahe_processing_time_ms"]
    
    # Get original paths mapping (using stem to avoid extension mismatches)
    orig_dataset = CustomMRIDataset(config.DATASET_ROOT)
    orig_paths_map = {os.path.splitext(os.path.basename(p))[0]: p for p, _ in orig_dataset.samples}
    
    for level in config.COMPRESSION_RATIOS.keys():
        print(f"Running inference on {level} processed dataset...")
        level_dir = os.path.join(config.PROCESSED_DATASETS_DIR, level)
        proc_dataset = CustomMRIDataset(level_dir)
        
        results = []
        
        for proc_path, label in proc_dataset.samples:
            filename_stem = os.path.splitext(os.path.basename(proc_path))[0]
            orig_path = orig_paths_map[filename_stem]
            
            # Load baseline generation prep time
            fft_time_ms = gen_map.get(proc_path, 0.0)
            
            start_prep = time.perf_counter()
            image = Image.open(proc_path).convert("RGB")
            t_img = transform(image).unsqueeze(0).to(device)
            load_tf_time_ms = (time.perf_counter() - start_prep) * 1000
            
            total_prep_time_ms = fft_time_ms + load_tf_time_ms
            
            start_inf = time.perf_counter()
            with torch.no_grad():
                outputs = model(t_img)
                prob = F.softmax(outputs, dim=1)[0][1].item()
                pred = outputs.argmax(1).item()
            inf_time_ms = (time.perf_counter() - start_inf) * 1000
            
            results.append({
                "compression_level": level,
                "processed_image_path": proc_path,
                "original_image_path": orig_path,
                "true_label": label,
                "pred_label": pred,
                "pred_name": "yes" if pred == 1 else "no",
                "confidence": prob,
                "fft_clahe_preprocessing_time_ms": total_prep_time_ms,
                "fft_clahe_inference_time_ms": inf_time_ms,
                "fft_clahe_end_to_end_time_ms": total_prep_time_ms + inf_time_ms
            })
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(config.RESULTS_DIR, f"{level}_predictions.csv"), index=False)
        print(f"Inference on {level} complete.")

if __name__ == "__main__":
    run_processed()