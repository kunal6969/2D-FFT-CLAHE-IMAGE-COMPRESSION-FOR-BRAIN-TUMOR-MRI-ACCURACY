import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config
from ml_core.model import get_trained_model
from utils import CustomMRIDataset, get_inference_transform

def run_baseline():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = get_trained_model(config.MODEL_PATH, device)
    transform = get_inference_transform()
    
    dataset = CustomMRIDataset(config.DATASET_ROOT)
    
    results = []
    print("Running baseline inference...")
    
    for img_path, label in dataset.samples:
        
        # Total Preprocessing time start
        start_prep = time.perf_counter()
        image = Image.open(img_path).convert("RGB")
        t_img = transform(image).unsqueeze(0).to(device)
        prep_time_ms = (time.perf_counter() - start_prep) * 1000
        
        # Inference time start
        start_inf = time.perf_counter()
        with torch.no_grad():
            outputs = model(t_img)
            prob = F.softmax(outputs, dim=1)[0][1].item()
            pred = outputs.argmax(1).item()
        inf_time_ms = (time.perf_counter() - start_inf) * 1000
        
        results.append({
            "image_path": img_path,
            "true_label": label,
            "baseline_pred_label": pred,
            "baseline_pred_name": "yes" if pred == 1 else "no",
            "baseline_confidence": prob,
            "original_preprocessing_time_ms": prep_time_ms,
            "original_inference_time_ms": inf_time_ms,
            "original_end_to_end_time_ms": prep_time_ms + inf_time_ms
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.RESULTS_DIR, "baseline_predictions.csv"), index=False)
    print("Baseline inference complete.")

if __name__ == "__main__":
    run_baseline()
