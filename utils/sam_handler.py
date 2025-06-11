import torch
import numpy as np
import cv2
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time

class SamHandler:
    def __init__(self, model_name="facebook/sam2.1-hiera-large", device="cuda"):
        print("[SAMHandler] Initializing SAM model once on device:", device)
        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name, device=device)
        # Remove caching (if any) so new coordinates update the segmentation.
        self.cached_image = None
        self.cached_image_shape = None
        # Internal state for toggling segmentation
        self.segmentation_active = False
        self.last_seg_output = None
        self.last_click = None

    def process_seg(self, pil_img, click_coords=None, max_dim=640):
        """
        Always run SAM at the given point (no toggling).
        """
        if not click_coords:
            return pil_img
        
        print("[SAMHandler] Running segmentation at:", click_coords)
        seg_img = self.run_sam_overlay(pil_img, [click_coords], active_idx=0, max_dim=max_dim)
        return seg_img

    ####################################################################
    # SAM → green overlay  + store Boolean mask
        ####################################################################
    def run_sam_overlay(self, pil_img, coords, active_idx: int = 0, max_dim: int = 640):
        """
        Run SAM and return both the overlay image and the boolean mask at original resolution.
        """
        import time, cv2, numpy as np
        t0 = time.time()

        if not coords or not (0 <= active_idx < len(coords)):
            return pil_img

        np_img  = np.array(pil_img)
        orig_h, orig_w, _ = np_img.shape

        # ---------- down-scale if necessary ----------
        if max(orig_w, orig_h) > max_dim:
            scale   = max_dim / max(orig_w, orig_h)
            new_w   = int(orig_w * scale)
            new_h   = int(orig_h * scale)
            np_small = np.array(pil_img.resize((new_w, new_h), Image.BILINEAR))
        else:
            new_w, new_h = orig_w, orig_h
            np_small     = np_img

        # ---------- active point in small space -------
        nx, ny = coords[active_idx]
        px_s   = int(nx * new_w)
        py_s   = int(ny * new_h)

        # ---------- SAM inference ---------------------
        self.predictor.set_image(np_small)
        with torch.no_grad():
            masks, _, _ = self.predictor.predict(
                point_coords = np.array([[px_s, py_s]], dtype=np.float32),
                point_labels = np.array([1],         dtype=np.int64)
            )
        if masks is None or len(masks) == 0:
            return pil_img

        # ---------- resize mask to ORIGINAL resolution ----
        mask_small     = masks[0].astype(np.uint8)
        mask_bool_full = cv2.resize(
            mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # ---------- connected component analysis ----
        mask_uint8 = mask_bool_full.astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            mask_bool_full = (labels == largest_label)
            print(f"[SAM] Selected largest component with area {areas[largest_label-1]} pixels")

        # ---------- STORE mask at ORIGINAL resolution -----
        self.last_mask_bool = mask_bool_full
        print(f"[SAM] Stored mask with shape {mask_bool_full.shape}, has {np.sum(mask_bool_full)} pixels")

        # ---------- green overlay ---------------------
        cv_img                  = np_img.copy()
        cv_img[ mask_bool_full] = (
            0.5 * cv_img[mask_bool_full] + 0.5 * np.array([0,255,0])
        ).astype(np.uint8)
        cv_img[~mask_bool_full] = (0.8 * cv_img[~mask_bool_full]).astype(np.uint8)

        # Store the mask for other functions to use
        self.last_mask_bool = mask_bool_full
        self.last_overlay_image = Image.fromarray(cv_img)  # Store the overlay image too
        
        print(f"[SAM] Inference+overlay {time.time()-t0:.3f}s")
        return Image.fromarray(cv_img)
            
    def get_last_mask(self):
        """Return the most recent Boolean mask."""
        return getattr(self, "last_mask_bool", None)

    def get_last_overlay_image(self):
        """Return the most recent overlay image."""
        return getattr(self, "last_overlay_image", None)

    def toggle_segmentation(self, pil_img, click_coords=None, max_dim=640):
        """
        Toggle segmentation:
            - If segmentation is off, run SAM with the provided coordinate and cache the output.
            - If segmentation is already active, reset and return the original live frame.
        """
        if not self.segmentation_active:
            if click_coords is None:
                raise ValueError("No coordinate provided for starting segmentation.")
            print("[SAMHandler] Starting segmentation at:", click_coords)
            seg_img = self.run_sam_overlay(pil_img, [click_coords], active_idx=0, max_dim=max_dim)
            self.segmentation_active = True
            self.last_seg_output = seg_img
            self.last_click = click_coords
            return seg_img
        else:
            print("[SAMHandler] Toggling segmentation off, resuming live feed.")
            self.segmentation_active = False
            self.last_seg_output = None
            self.last_click = None
            return pil_img
