# depth_handler.py
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import pipeline

try:
    import pyrealsense2 as rs
    RS_OK = True
except ImportError:
    RS_OK = False


class DepthHandler:
    """
    Handles three things:

      1) "Depth-Anything" monocular depth inference for any RGB image
      2) RealSense D400-series RGB-D capture and helpers
      3) Local webcam capture via OpenCV

    New in this version
    -------------------
    * calculate_object_info(mask_bool, depth_arr)   →  dict
        Compute centre X-Y-Z, Euclidean distance, physical width/height and
        bounding-box pixels for a segmented object, given the Boolean mask
        from SAM and the raw 16-bit depth frame from RealSense.
    """

    def __init__(self, device="cuda"):
        # 1 — Depth-Anything
        self.depth_anything = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-base-hf",
            device=device,
        )

        # 2 — RealSense state
        self.rs_pipeline = None
        self.rs_config   = None
        self.rs_align    = None
        self.rs_active   = False
        self.frame_timeout = 2_000  # ms

        # 3 — Local webcam state
        self.local_cap        = None
        self.local_cam_index  = None
        self.local_cameras_info = self.enumerate_local_cameras()

    # ------------------------------------------------------------------
    # Camera enumeration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def enumerate_local_cameras(max_cams: int = 8):
        cams = []
        for idx in range(max_cams):
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                cams.append(idx)
                cap.release()
        return cams

    def list_realsense_devices(self):
        if not RS_OK:
            return []
        try:
            ctx = rs.context()
            return [
                {
                    "type": "realsense-usb",  # Add this line
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                }
                for dev in ctx.devices
            ]
        except Exception as e:
            print("[DepthHandler] Error listing RealSense devices:", e)
            return []
    # ------------------------------------------------------------------
    # RealSense management
    # ------------------------------------------------------------------
    def start_realsense(self, device_info=None):
        if not RS_OK:
            print("[DepthHandler] pyrealsense2 not installed.")
            return False
        if self.rs_active:
            return True
        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config   = rs.config()
            
            # If device_info provided, use specific device
            if device_info and "serial" in device_info:
                self.rs_config.enable_device(device_info["serial"])
                
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.rs_pipeline.start(self.rs_config)
            self.rs_align   = rs.align(rs.stream.color)
            self.rs_active  = True
            return True
        except Exception as e:
            print("[DepthHandler] Failed to start RealSense:", e)
            return False

    def stop_realsense(self):
        if self.rs_active and self.rs_pipeline:
            self.rs_pipeline.stop()
        self.rs_pipeline = self.rs_config = self.rs_align = None
        self.rs_active = False

    def get_realsense_frames(self):
        if not (self.rs_active and self.rs_pipeline):
            return (None, None)
        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=self.frame_timeout)
            aligned = self.rs_align.process(frames)
            color   = aligned.get_color_frame()
            depth   = aligned.get_depth_frame()
            if not color or not depth:
                return (None, None)
            return (
                np.asanyarray(color.get_data()),
                np.asanyarray(depth.get_data()),  # uint16, millimetres
            )
        except Exception as e:
            print("[DepthHandler] RealSense error:", e)
            return (None, None)

    def realsense_color_to_depthanything(self):
        color_arr, _ = self.get_realsense_frames()
        if color_arr is None:
            return None
        return self.run_depth_anything(Image.fromarray(color_arr))

    def realsense_depth_colormap(self):
        _, depth_arr = self.get_realsense_frames()
        if depth_arr is None:
            return None
        depth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_arr, alpha=0.03), cv2.COLORMAP_JET
        )
        return Image.fromarray(depth_color[..., ::-1])

    # ------------------------------------------------------------------
    # Local webcam management
    # ------------------------------------------------------------------
    def start_local_camera(self, index: int):
        if self.local_cap and self.local_cam_index == index:
            return True
        self.stop_local_camera()
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[DepthHandler] Could not open camera {index}")
            return False
        self.local_cap = cap
        self.local_cam_index = index
        return True

    def stop_local_camera(self):
        if self.local_cap:
            self.local_cap.release()
        self.local_cap = self.local_cam_index = None

    def grab_local_frame(self):
        if not self.local_cap:
            return None
        ok, frame = self.local_cap.read()
        return frame if ok else None

    def run_depth_anything(self, pil_img: Image.Image) -> Image.Image:
        depth_out   = self.depth_anything(pil_img)
        depth_arr   = np.array(depth_out["depth"], dtype=np.float32)
        depth_norm  = cv2.normalize(depth_arr, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return Image.fromarray(depth_color[..., ::-1])
    
    def calculate_object_info(self, mask_bool: np.ndarray, depth_arr: np.ndarray):
        """
        Calculate object information from mask and depth data.
        
        Args:
            mask_bool: Boolean mask from SAM segmentation
            depth_arr: Raw depth array from RealSense (uint16, millimeters)
        
        Returns:
            Dict with object geometry information
        """
        import numpy as np
        import cv2
        import pyrealsense2 as rs

        # 1) Align mask to depth resolution
        h_m, w_m = mask_bool.shape
        h_d, w_d = depth_arr.shape
        if (h_m, w_m) != (h_d, w_d):
            mask_depth = cv2.resize(
                mask_bool.astype(np.uint8),
                (w_d, h_d),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            print(f"[DepthHandler] Resized mask from {mask_bool.shape} to {mask_depth.shape}")
        else:
            mask_depth = mask_bool

        # 2) Find ALL pixels that belong to the mask
        mask_y_coords, mask_x_coords = np.where(mask_depth)
        
        if len(mask_y_coords) == 0:
            print("[DepthHandler] No mask pixels found.")
            return None

        print(f"[DepthHandler] Found {len(mask_y_coords)} mask pixels")

        # 3) Calculate CENTER OF MASS of the actual mask pixels
        cx_centroid = float(np.mean(mask_x_coords))
        cy_centroid = float(np.mean(mask_y_coords))
        
        # Round to integer pixel coordinates
        cx_final = int(round(cx_centroid))
        cy_final = int(round(cy_centroid))
        
        print(f"[DepthHandler] True mask centroid: ({cx_centroid:.2f}, {cy_centroid:.2f})")
        print(f"[DepthHandler] Rounded to pixel: ({cx_final}, {cy_final})")
        
        # 4) VERIFY this point is actually inside the mask
        if not mask_depth[cy_final, cx_final]:
            print(f"[DepthHandler] WARNING: Rounded centroid ({cx_final}, {cy_final}) is outside mask!")
            # Find the closest actual mask pixel
            distances = ((mask_y_coords - cy_final)**2 + (mask_x_coords - cx_final)**2)
            closest_idx = np.argmin(distances)
            cx_final = int(mask_x_coords[closest_idx])
            cy_final = int(mask_y_coords[closest_idx])
            print(f"[DepthHandler] Adjusted to closest mask pixel: ({cx_final}, {cy_final})")
        
        # 5) Get depth at the center point with fallback
        center_depth_mm = depth_arr[cy_final, cx_final]
        if center_depth_mm == 0:
            print(f"[DepthHandler] WARNING: No depth at center point, expanding search...")
            # Get median depth from all mask pixels
            mask_depths = depth_arr[mask_depth & (depth_arr > 0)]
            if len(mask_depths) > 0:
                center_depth_mm = int(np.median(mask_depths))
                print(f"[DepthHandler] Using median depth from mask: {center_depth_mm}mm")
            else:
                print("[DepthHandler] No valid depth found in mask")
                return None
        
        z_m = center_depth_mm / 1000.0
        if z_m <= 0:
            print("[DepthHandler] No valid depth found for center point")
            return None

        print(f"[DepthHandler] Center depth: {center_depth_mm}mm ({z_m:.3f}m)")

        # 6) Calculate bounding box for size estimation
        x_min, x_max = mask_x_coords.min(), mask_x_coords.max()
        y_min, y_max = mask_y_coords.min(), mask_y_coords.max()
        px_w = x_max - x_min + 1
        px_h = y_max - y_min + 1

        # 7) Get oriented bounding box for better size estimation
        oriented_width_px = px_w
        oriented_height_px = px_h
        angle = 0.0
        try:
            mask_uint8 = mask_depth.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                rect_center, (oriented_width_px, oriented_height_px), angle = rect
                
                if oriented_width_px > oriented_height_px:
                    oriented_width_px, oriented_height_px = oriented_height_px, oriented_width_px
                    angle += 90
                print(f"[DepthHandler] Oriented box: {oriented_width_px:.1f}x{oriented_height_px:.1f}px, angle={angle:.1f}°")
        except Exception as e:
            print(f"[DepthHandler] Error computing oriented bounding box: {e}")

        # 8) 🔥 CORRECTED: Check if RealSense is active and get COLOR intrinsics
        if not self.rs_active or not self.rs_pipeline:
            print("[DepthHandler] RealSense not active, cannot get intrinsics")
            return None
        
        try:
            # Get COLOR intrinsics (since depth is aligned to color)
            intr = (
                self.rs_pipeline
                    .get_active_profile()
                    .get_stream(rs.stream.color)  # ✅ Changed from depth to color
                    .as_video_stream_profile()
                    .get_intrinsics()
            )
            print(f"[DepthHandler] Color intrinsics - fx:{intr.fx:.1f}, fy:{intr.fy:.1f}, ppx:{intr.ppx:.1f}, ppy:{intr.ppy:.1f}")
        except Exception as e:
            print(f"[DepthHandler] Error getting color intrinsics: {e}")
            return None

        # 9) Compute physical size using color intrinsics
        width_m = (oriented_width_px * z_m) / intr.fx
        height_m = (oriented_height_px * z_m) / intr.fy

        # 10) Apply scale factor and convert to mm
        fixed_scale = 0.62
        width_m *= fixed_scale
        height_m *= fixed_scale
        width_mm = width_m * 1000
        height_mm = height_m * 1000

        # 11) Adjust orientation
        if width_mm > height_mm:
            corrected_angle = angle % 180
        else:
            corrected_angle = (angle + 90) % 180
            width_mm, height_mm = height_mm, width_mm
        
        if corrected_angle > 90:
            corrected_angle = 180 - corrected_angle

        # 12) 🔥 CORRECTED: Deproject using color intrinsics
        try:
            center_xyz = rs.rs2_deproject_pixel_to_point(
                intr, [float(cx_final), float(cy_final)], z_m
            )
        except Exception as e:
            print(f"[DepthHandler] Error deprojecting point: {e}")
            return None
        
        # 13) 🔥 COORDINATE FRAME TRANSFORMATION
        # RealSense coordinate system: X=right, Y=down, Z=forward
        # Robot coordinate system: X=forward, Y=left, Z=up (adjust as needed)
        
        # Option 1: Keep original RealSense coordinates
        center_xyz_mm = [coord * 1000 for coord in center_xyz]
        
        # Option 2: Transform to robot coordinates (uncomment if needed)
        # center_xyz_robot = [
        #     center_xyz[2],   # Robot X = Camera Z (forward)  
        #     -center_xyz[0],  # Robot Y = -Camera X (left)
        #     -center_xyz[1]   # Robot Z = -Camera Y (up)
        # ]
        # center_xyz_robot_mm = [coord * 1000 for coord in center_xyz_robot]
        
        distance_m = float(np.linalg.norm(center_xyz))

        print(f"[DepthHandler] Final Results:")
        print(f"  - Center pixel: ({cx_final}, {cy_final})")
        print(f"  - Camera coords (X,Y,Z): {[round(c, 1) for c in center_xyz_mm]} mm")
        print(f"  - Distance: {distance_m:.3f} m")
        print(f"  - Object size: {round(height_mm)}×{round(width_mm)} mm")
        print(f"  - Orientation: {round(corrected_angle)}°")

        return {
            "center_xyz_mm": center_xyz_mm,              # Camera coordinates (X=right, Y=down, Z=forward)
            # "center_xyz_robot_mm": center_xyz_robot_mm,  # Uncomment if using robot coords
            "distance_m": distance_m,
            "width_mm": float(width_mm),      # Shorter dimension
            "length_mm": float(height_mm),    # Longer dimension  
            "bbox_px": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "orientation_deg": float(corrected_angle),
            "center_px": [int(cx_final), int(cy_final)],  # TRUE center of mass
            "oriented_bbox_px": {
                "width": float(oriented_width_px),
                "height": float(oriented_height_px)
            },
            "intrinsics": {
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy
            }
        }