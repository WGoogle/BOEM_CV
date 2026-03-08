import cv2, numpy as np
from pathlib import Path

img  = cv2.imread("outputs/patches/images/CameraMosaic_D1_Node3_L8_patch_0376.png")
mask = cv2.imread("outputs/patches/masks/CameraMosaic_D1_Node3_L8_patch_0376.png")

# Green overlay
overlay = img.copy()
overlay[mask[:,:,0] > 0] = (0, 200, 0)
vis = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
cv2.imwrite("outputs/patch_check.png", np.hstack([img, vis, mask]))
