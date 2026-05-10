import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import config


def apply_fft_only(image_np: np.ndarray, keep_ratio: float) -> np.ndarray:
    f_img = np.float32(image_np)
    f_transform = np.fft.fft2(f_img)
    f_shift = np.fft.fftshift(f_transform)

    h, w = f_img.shape
    cy, cx = h // 2, w // 2

    h_keep = int(h * np.sqrt(keep_ratio))
    w_keep = int(w * np.sqrt(keep_ratio))

    mask = np.zeros((h, w), dtype=np.uint8)
    sy, ey = max(0, cy - h_keep // 2), min(h, cy + h_keep // 2)
    sx, ex = max(0, cx - w_keep // 2), min(w, cx + w_keep // 2)
    mask[sy:ey, sx:ex] = 1

    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_reconstructed = np.fft.ifft2(f_ishift)

    img_reconstructed = np.abs(img_reconstructed)
    return np.clip(img_reconstructed, 0, 255).astype(np.uint8)


def apply_fft_clahe_only(
    image_np: np.ndarray,
    keep_ratio: float,
    clip_limit: float,
    tile_grid_size: tuple,
) -> np.ndarray:
    img_reconstructed = apply_fft_only(image_np, keep_ratio)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_reconstructed)


def create_8_panel_visual(image_name):
    # Find original image
    img_path = None
    for label in ["yes", "no"]:
        check_path = os.path.join(config.DATASET_ROOT, label, image_name)
        if os.path.exists(check_path):
            img_path = check_path
            break

    if not img_path:
        print(f"Error: Image '{image_name}' not found in the dataset.")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error reading image.")
        return

    _, bg_mask = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    bg_mask = (bg_mask / 255).astype(img.dtype)

    ratios = {
        "Light": config.COMPRESSION_RATIOS["light"],
        "Moderate": config.COMPRESSION_RATIOS["intermediate"],
        "Aggressive": config.COMPRESSION_RATIOS["aggressive"],
    }

    panels = [None] * 8
    
    # 1. Original
    panels[0] = {"title": "Original Image", "img": img}

    # 2-4. FFT Only
    for idx, (level, ratio) in enumerate(ratios.items(), start=1):
        fft_img = apply_fft_only(img, ratio) * bg_mask
        panels[idx] = {
            "title": f"FFT {level} Compressed",
            "img": fft_img,
        }

    # 5. Center Heading (Updated to remove the 'yes/no' label)
    # This strips the extension and common label words to keep it clean
    clean_display_name = image_name.lower().replace("yes", "").replace("no", "").replace(".jpg", "").replace(".png", "").strip()
    
    panels[4] = {
        "title": "Analysis Overview",
        "heading": clean_display_name,
    }

    # 6-8. FFT + CLAHE
    for idx, (level, ratio) in enumerate(ratios.items(), start=5):
        fft_clahe_img = (
            apply_fft_clahe_only(
                img,
                ratio,
                config.CLAHE_CLIP_LIMIT,
                config.CLAHE_TILE_GRID,
            )
            * bg_mask
        )
        panels[idx] = {
            "title": f"FFT {level} + CLAHE", # Simplified title
            "img": fft_clahe_img,
        }

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        ax.axis("off")

        if idx == 4:
            ax.text(
                0.5, 0.5,
                panel["heading"],
                fontsize=26,
                fontweight="bold",
                ha="center",
                va="center",
                color="darkblue",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    alpha=0.85,
                ),
            )
            ax.set_title(panel["title"], fontsize=16, fontweight="bold", pad=15)
            continue

        ax.imshow(panel["img"], cmap="gray")
        ax.set_title(
            panel["title"],
            fontsize=14,
            fontweight="bold",
            pad=15,
            color="darkblue",
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.2)

    out_filename = os.path.splitext(image_name)[0] + "_8panel.png"
    out_path = os.path.join(config.VISUALS_DIR, out_filename)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Generated clean visual (no labels) at: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 8-panel visual comparison for an image."
    )
    parser.add_argument("image_name", type=str, help="Filename of the image, e.g. 'Y1.jpg'")
    args = parser.parse_args()

    create_8_panel_visual(args.image_name)

    # & "$env:USERPROFILE\Miniconda3\python.exe" generate_visual.py "4 no.jpg" 