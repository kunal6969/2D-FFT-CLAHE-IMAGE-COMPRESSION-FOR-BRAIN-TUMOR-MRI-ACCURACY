import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random
import config

def create_visual_panels():
    baseline_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "baseline_predictions.csv"))
    
    # Pick 10 random images
    samples = baseline_df.sample(n=min(10, len(baseline_df)), random_state=42)
    
    for idx, r in samples.iterrows():
        orig_path = r["original_image_path" if "original_image_path" in r else "image_path"]
        orig_img = cv2.imread(orig_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        filename = os.path.basename(orig_path)
        
        paths = {"original": orig_path}
        imgs = {"original": orig_img}
        preds = {"original": r["baseline_pred_name"]}
        
        for level in config.COMPRESSION_RATIOS.keys():
            path_level = os.path.join(config.PROCESSED_DATASETS_DIR, level, "yes" if r["true_label"] == 1 else "no", filename)
            paths[level] = path_level
            img_l = cv2.imread(path_level)
            imgs[level] = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
            
            # Find pred
            df_l = pd.read_csv(os.path.join(config.RESULTS_DIR, f"{level}_predictions.csv"))
            p_name = df_l[df_l["original_image_path"] == orig_path]["pred_name"].values[0]
            preds[level] = p_name
            
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["original", "light", "intermediate", "aggressive"]
        
        for i, title in enumerate(titles):
            axes[i].imshow(imgs[title])
            axes[i].set_title(f"{titles[i]}\nPred: {preds[title]}")
            axes[i].axis('off')
            
        plt.suptitle(f"File: {filename} | True: {'yes' if r['true_label'] == 1 else 'no'}")
        plt.tight_layout()
        plt.savefig(os.path.join(config.VISUALS_DIR, f"panel_{filename}.png"))
        plt.close()
        
    print("Visual panels generated.")

if __name__ == "__main__":
    create_visual_panels()