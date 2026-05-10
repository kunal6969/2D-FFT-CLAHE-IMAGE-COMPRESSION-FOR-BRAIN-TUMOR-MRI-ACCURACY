import numpy as np
import cv2

def apply_fft_clahe(image_np: np.ndarray, keep_ratio: float, clip_limit: float, tile_grid_size: tuple) -> np.ndarray:
    """
    Applies 2D FFT, filters keeping only a central percentage of frequencies, 
    applies inverse FFT, and enhances contrast via CLAHE.
    """
    # 1. FFT
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
    
    # 2. Inverse FFT
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_reconstructed = np.fft.ifft2(f_ishift)
    
    img_reconstructed = np.abs(img_reconstructed)
    img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)
    
    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(img_reconstructed)
    
    return result
