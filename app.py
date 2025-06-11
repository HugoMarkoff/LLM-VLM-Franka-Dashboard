import io
import re
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
from flask_cors import CORS
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import requests
import flask
from flask import Response, stream_with_context, request
import httpx, asyncio
from utils.sam_handler import SamHandler
from utils.depth_handler import DepthHandler, RS_OK
from utils.robopoints_handler import RoboPointsHandler
import os
from dotenv import load_dotenv

# Add these imports after your existing imports
import threading
from rutils.config import Config
from rutils.chrome_manager import ChromeDriverManager
from rutils.robot_interface import FrankaRobotInterface
from rutils.robot_commands import FrankaRobotCommands
from rutils.network_manager import NetworkManager
from selenium import webdriver


app = Flask(__name__)
CORS(app)

ROBOPOINT_IMAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOPOINT_IMAGE_COUNTER_FILE = os.path.join(ROBOPOINT_IMAGE_DIR, "robopoint_image_counter.txt")

# Load environment variables
load_dotenv()
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip('/')

# Update endpoint URLs
REMOTE_POINTS_ENDPOINT = f"{NGROK_BASE_URL}/robopoint/predict"
LLM_STREAM = f"{NGROK_BASE_URL}/qwen3/chat-stream"  
LLM_FULL = f"{NGROK_BASE_URL}/qwen3/chat"

print(f"[INFO] Using endpoints:")
print(f"  RoboPoint: {REMOTE_POINTS_ENDPOINT}")
print(f"  LLM Stream: {LLM_STREAM}")
print(f"  LLM Full: {LLM_FULL}")

# top-of-file imports
from utils.llm_handler import LLMHandler

robopoint_results = []
current_user_message = ""
current_image_path = ""

robot_automation = None
robot_status = {
    "state": "disconnected",  # disconnected, initializing, ready, moving, error
    "message": "Robot not initialized",
    "buttons_enabled": False
}
robot_lock = threading.Lock()

llm = LLMHandler(stream_url=LLM_STREAM, full_url=LLM_FULL)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

sam_handler = SamHandler(device=device)
depth_handler = DepthHandler(device=device)
robo_handler = RoboPointsHandler(REMOTE_POINTS_ENDPOINT)

# Global variables to store state
g_state = {
    "mode": "rgb",
    "points_str": "",
    "active_cross_idx": 0,
    "clicked_points": [],
    "last_instruction": "",
    "prev_seg_np_img": None,
    "last_seg_output": None,
    "last_heavy_inference_time": 0.0
}
g_last_raw_frame = None
g_frozen_seg = None  # This will store the frozen candidate overlay image (PIL Image or raw bytes)

ThreadPoolExecutor(max_workers=2)
seg_lock = threading.Lock()
depth_lock = threading.Lock()

POINT_RE = re.compile(r"\(([0-9.+-eE]+)\s*,\s*([0-9.+-eE]+)\)")

# ========== ROBOT CONTROL FUNCTIONS ==========

def initialize_robot_system():
    """Initialize the robot system in a separate thread."""
    global robot_automation, robot_status
    
    try:
        with robot_lock:
            robot_status["state"] = "initializing"
            robot_status["message"] = "Setting up network configuration..."
        
        # Create robot configuration
        config = Config(
            robot_ip="172.16.0.2",
            local_ip="172.16.0.1", 
            network_interface="enp2s0"
        )
        
        # Initialize logger and managers
        from rutils.logger import setup_logging
        logger = setup_logging()
        
        from rutils.network_manager import NetworkManager
        from rutils.chrome_manager import ChromeDriverManager
        
        network_manager = NetworkManager(config, logger)
        chrome_manager = ChromeDriverManager(config, logger)
        
        with robot_lock:
            robot_status["message"] = "Configuring network interface..."
        
        # Setup network
        if not network_manager.configure_network():
            raise Exception("Network configuration failed")
        
        with robot_lock:
            robot_status["message"] = "Testing robot connectivity..."
            
        # Test connectivity
        if not network_manager.test_robot_connectivity():
            print("[WARNING] Robot connectivity test failed, but continuing...")
        
        with robot_lock:
            robot_status["message"] = "Starting browser and connecting to robot..."
        
        # Create browser driver
        driver = chrome_manager.create_driver(headless=True)
        
        # Create robot interface
        robot_interface = FrankaRobotInterface(driver, config, logger)
        robot_commands = FrankaRobotCommands(robot_interface, logger)
        
        with robot_lock:
            robot_status["message"] = "Logging into robot interface..."
        
        # Initialize robot
        robot_interface.navigate_and_login()
        robot_interface.ensure_joints_unlocked()
        
        # Store the automation instance
        class SimpleRobotAutomation:
            def __init__(self, driver, robot, commands, chrome_manager):
                self.driver = driver
                self.robot = robot
                self.commands = commands
                self.chrome_manager = chrome_manager
                self._is_initialized = True
            
            def is_robot_ready(self):
                if not self._is_initialized or not self.driver or not self.robot:
                    return False
                try:
                    _ = self.driver.current_url
                    return True
                except:
                    self._is_initialized = False
                    return False
            
            def stop_robot(self):
                try:
                    if self.robot and self.driver:
                        self.robot.release_control()
                    if self.driver:
                        self.driver.quit()
                    self.chrome_manager.cleanup_all_chrome_processes()
                except Exception as e:
                    print(f"Error stopping robot: {e}")
                self._is_initialized = False
        
        robot_automation = SimpleRobotAutomation(driver, robot_interface, robot_commands, chrome_manager)
        
        with robot_lock:
            robot_status["state"] = "ready"
            robot_status["message"] = "Robot ready for commands"
            robot_status["buttons_enabled"] = True
            
        print("[INFO] Robot initialization completed successfully")
        
    except Exception as e:
        with robot_lock:
            robot_status["state"] = "error"
            robot_status["message"] = f"Initialization error: {str(e)}"
            robot_status["buttons_enabled"] = False
        print(f"Robot initialization error: {e}")
        import traceback
        traceback.print_exc()

def execute_robot_movement(direction):
    """Execute robot movement in specified direction."""
    global robot_automation, robot_status
    
    if not robot_automation or not robot_automation.is_robot_ready():
        return False, "Robot not ready"
    
    # Corrected movement mappings (2cm = 20mm)
    movements = {
        "up": {"z": -20},      # Up = negative Z (works fine)
        "down": {"z": 20},     # Down = positive Z (works fine)
        "left": {"x": -20},    # Left = negative X (was positive Y)
        "right": {"x": 20},    # Right = positive X (was negative Y)
        "forward": {"y": -20},  # Forward = positive Y (was positive X)
        "back": {"y": 20}     # Back = negative Y (was negative X)
    }
    
    if direction not in movements:
        return False, "Invalid direction"
    
    try:
        with robot_lock:
            robot_status["state"] = "moving"
            robot_status["message"] = f"Moving {direction}..."
        
        # Execute movement
        move_params = {"x": 0, "y": 0, "z": 0, "speed": 10, "acceleration": 10}
        move_params.update(movements[direction])
        
        success = robot_automation.commands.move_robot(**move_params)
        
        with robot_lock:
            if success:
                robot_status["state"] = "ready"
                robot_status["message"] = "Robot ready for commands"
            else:
                robot_status["state"] = "error"
                robot_status["message"] = "Movement failed"
                
        return success, robot_status["message"]
        
    except Exception as e:
        with robot_lock:
            robot_status["state"] = "error"
            robot_status["message"] = f"Movement error: {str(e)}"
        print(f"Robot movement error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

####Robot Controll end####

def _points_from_string(raw: str):
    """
    Turn RoboPoint's 'result' string into a list of (x,y) floats in [0,1].

    Accepts:
        '[(0.37, 0.61)]'
        '(0.37,0.61)'
        '0.37,0.61'
        'x=0.37 y=0.61'
        '[(0.37,0.61), (0.12,0.88)]'
    Returns [] if **no** valid pair is found.
    """
    raw = raw.strip()
    # 1) quick normalisation of the simplest '0.37,0.61' case
    if raw.count(',') == 1 and raw.count('(') == 0:
        try:
            x, y = map(float, raw.split(','))
            return [(x, y)]
        except Exception:
            pass
    # 2) general regex search – tolerant of any wrapper text
    matches = POINT_RE.findall(raw)
    return [tuple(map(float, xy)) for xy in matches]

def get_next_image_index():
    """Get and increment the persistent image index."""
    if not os.path.exists(ROBOPOINT_IMAGE_COUNTER_FILE):
        idx = 1
    else:
        with open(ROBOPOINT_IMAGE_COUNTER_FILE, "r") as f:
            idx = int(f.read().strip() or "1")
    with open(ROBOPOINT_IMAGE_COUNTER_FILE, "w") as f:
        f.write(str(idx + 1))
    return idx

# Helper function to reset frame and fetch fresh frame if possible
def reset_frame_and_fetch_fresh():
    global g_last_raw_frame, g_frozen_seg, g_state
    g_last_raw_frame = None
    g_frozen_seg = None
    # Reset relevant state variables
    g_state["mode"] = "rgb"
    g_state["points_str"] = ""
    g_state["active_cross_idx"] = 0
    g_state["clicked_points"] = []
    g_state["prev_seg_np_img"] = None
    g_state["last_seg_output"] = None
    print("[INFO] Resetting stored frame, frozen segmentation, and all state variables.")
    
    # Attempt to fetch a fresh frame if RealSense is available
    if depth_handler.start_realsense():
        frames = depth_handler.get_realsense_frames()
        if frames is not None:
            color_arr, _ = frames  # Get color frame; ignore depth frame
            if color_arr is not None:
                # Convert from BGR to RGB
                rgb_arr = cv2.cvtColor(color_arr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_arr)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                g_last_raw_frame = buf.getvalue()
                print("[INFO] Fetched fresh frame from RealSense after reset.")
            else:
                print("[WARNING] No color frame available from RealSense after reset.")
        else:
            print("[WARNING] RealSense frames not ready after reset.")
    else:
        print("[INFO] RealSense not active, no fresh frame fetched after reset.") 

# ────────────────────────────────────────────────────────────────────
#  /exec_action  – returns placement + full geometry string
# ────────────────────────────────────────────────────────────────────

@app.route("/exec_action", methods=["POST"])
def exec_action():
    """
    Enhanced version that handles single action from multi-step sequence
    """
    global g_last_raw_frame

    # Reset state before processing RoboPoint action to ensure fresh start
    reset_frame_and_fetch_fresh()

    data        = request.json or {}
    block_txt   = data.get("action_block", "")
    frame_b64   = data.get("seg_frame", "")

    # 0) Decode frozen frame
    try:
        g_last_raw_frame = base64.b64decode(frame_b64)
    except Exception as e:
        return jsonify({"error": f"Invalid seg_frame: {e}"}), 400
    pil_img = Image.open(io.BytesIO(g_last_raw_frame)).convert("RGB")
    
    # --- SAVE THE IMAGE ---
    img_idx = get_next_image_index()
    img_path = os.path.join(ROBOPOINT_IMAGE_DIR, f"{img_idx}.jpg")
    pil_img.save(img_path, "JPEG")
    print(f"[INFO] Saved RoboPoint image as {img_path}")

    # 1) Parse action block
    request_match = re.search(r"RoboPoint\s*Request:\s*([^;\n|]+)", block_txt, flags=re.I)
    action_match = re.search(r"Action\s*Request:\s*([^;\n|]+)", block_txt, flags=re.I)
    
    if not request_match or not action_match:
        return jsonify({"error": "Malformed [ACTION] block"}), 400
    
    obj_phrase = request_match.group(1).strip()
    action_type = action_match.group(1).strip().lower()
    
    # Extract object and location
    if " at " in obj_phrase:
        object_name, location = obj_phrase.split(" at ", 1)
        instruction = f"{action_type} the {object_name.strip()} at {location.strip()}"
    else:
        object_name = obj_phrase
        instruction = f"{action_type} the {object_name.strip()}"

    # 2) Call RoboPoint
    rp_result = robo_handler.call_remote_for_points(frame_b64, instruction)
    print(f"[RoboPoint result] {rp_result}")
    pts = _points_from_string(rp_result)

    if not pts:
        err = f"RoboPoint returned no usable coordinates for '{instruction}'."
        return jsonify({"error": err, "rp_result": rp_result}), 502

    # Use the first candidate point
    x0, y0 = pts[0]

    # 3) Determine placement description
    def coarse_placement(nx: float, ny: float) -> str:
        col = "left" if nx < 0.33 else "right" if nx > 0.66 else "centre"
        row = "top"  if ny < 0.33 else "bottom" if ny > 0.66 else "middle"
        return f"{row}-{col}" if (row != "middle" or col != "centre") else "centre"

    placement = coarse_placement(x0, y0)

    # 4) SAM overlay
    seg_img = sam_handler.run_sam_overlay(pil_img, pts, active_idx=0)
    
    # 5) Depth geometry (only for pick operations or when available)
    object_info = None
    geometry_msg = "(No geometry available)"
    
    if depth_handler.start_realsense():
        _, depth_arr = depth_handler.get_realsense_frames()
        mask_bool    = sam_handler.get_last_mask()
        object_info  = depth_handler.calculate_object_info(mask_bool, depth_arr)
        
        if object_info:
            cx, cy, cz = object_info["center_xyz_mm"]
            if "pick" in action_type:
                # Full geometry for pick operations
                geometry_msg = (
                    f"Center: {cx:.0f}, {cy:.0f}, {cz:.0f} mm; "
                    f"distance {object_info['distance_m']:.3f} m; "
                    f"Size {object_info['length_mm']:.0f}×{object_info['width_mm']:.0f} mm; "
                    f"Orientation {object_info['orientation_deg']:.0f}°"
                )
            else:
                # Just center point for place operations
                geometry_msg = f"Center: {cx:.0f}, {cy:.0f}, {cz:.0f} mm"

    # Draw circle at centerpoint if object_info is available
    if object_info and "center_px" in object_info:
        center_x, center_y = object_info["center_px"]
        # Get the mask that was used for center calculation
        mask_bool = sam_handler.get_last_mask()
        # Use the same mask for drawing verification
        seg_img = draw_centerpoint_circle(seg_img, center_x, center_y, mask_bool)
        print(f"[DEBUG] Drew circle using mask-verified coordinates ({center_x}, {center_y})")
    elif object_info and "bbox_px" in object_info:
        # Fallback to bounding box center
        x_min, y_min, x_max, y_max = object_info["bbox_px"]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        mask_bool = sam_handler.get_last_mask()
        seg_img = draw_centerpoint_circle(seg_img, center_x, center_y, mask_bool)
        print(f"[DEBUG] Drew circle using bbox center ({center_x}, {center_y})")
    
    # Convert final image to base64
    out_b64 = pil_to_b64(seg_img)

    # 6) Human-readable found message
    found_msg = f"{instruction} located at {placement} ({x0:.2f}, {y0:.2f})"

    # 7) Do NOT reset state after RoboPoint action; keep the processed image and mask
    print("[INFO] Keeping processed frame and mask visible after RoboPoint action.")

    # 8) Respond
    return jsonify({
        "seg_frame"   : out_b64,  # Fixed: was seg_b64, now out_b64
        "found_msg"   : found_msg,
        "geometry_msg": geometry_msg,
        "placement"   : placement,
        "object_info" : object_info,
        "rp_result"   : rp_result,
        "action_type" : action_type,
        "image_path"  : img_path  # Add the actual saved image path
    })

###########################
# Helper Functions
###########################

def get__current_frame(seg_input, seg_frame, instruction):
    """
    Retrieves the current frame from the segmentation window. If the seg_input dropdown is "off",
    returns an error message; otherwise, if the client has captured a seg_frame (as base64),
    it decodes and updates the global frame (g_last_raw_frame). If no new frame is provided,
    the function falls back on the previously stored frame.
    
    It then encodes the frame and calls the remote endpoint with the instruction.
    
    Parameters:
        seg_input (str): The current dropdown selection ("on" or "off").
        seg_frame (str): The base64-encoded frame from the seg window.
        instruction (str): The instruction (e.g. "cup") for the robot.
        
    Returns:
        str: The result from the remote endpoint (point candidate string) or an error message.
    """
    global g_last_raw_frame
    print(f"[DEBUG] Segmentation input dropdown selected: {seg_input}.")
    if seg_input.lower() == "off":
        msg = "Seg window input is off. Please select an active input to capture its frame."
        print(f"[DEBUG] {msg}")
        return msg

    if seg_frame:
        try:
            frame_bytes = base64.b64decode(seg_frame)
            g_last_raw_frame = frame_bytes
            print("[DEBUG] Received new seg frame from client and updated g_last_raw_frame.")
        except Exception as e:
            error_msg = f"Failed to decode seg_frame: {e}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    else:
        if not g_last_raw_frame:
            error_msg = "No segmentation frame available; g_last_raw_frame is empty."
            print(f"[DEBUG] {error_msg}")
            return error_msg
        else:
            print("[DEBUG] No new seg_frame provided; using stored g_last_raw_frame.")

    try:
        frame_b64 = base64.b64encode(g_last_raw_frame).decode("utf-8")
    except Exception as e:
        error_msg = f"Error encoding seg frame: {e}"
        print(f"[ERROR] {error_msg}")
        return error_msg

    print(f"[DEBUG] Prepared seg frame in base64. Sending to remote endpoint with instruction: {instruction}")
    result = call_remote_for_points(frame_b64, instruction)
    print(f"[DEBUG] Remote endpoint returned: {result}")
    return result


def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_remote_for_points(frame_b64, instruction):
    try:
        print(f"[DEBUG] User input '{instruction}' sent to robot point.")
        resp = requests.post(
            REMOTE_POINTS_ENDPOINT,
            json={"image": frame_b64, "instruction": instruction},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", "")
        print(f"[DEBUG] Coordinates returned from robo point: {result}")
        return result
    except Exception as e:
        print("[ERROR calling remote for points]:", e)
        return ""

def move_active_cross(direction, dist=1):
    matches = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", g_state["points_str"])
    coords = [(float(a), float(b)) for a, b in matches]
    idx = g_state["active_cross_idx"]
    if not coords or idx < 0 or idx >= len(coords):
        return
    nx, ny = coords[idx]
    shift = dist * 0.01
    if direction == "left":
        nx = max(0.0, nx - shift)
    elif direction == "right":
        nx = min(nx + shift, 1.0)
    elif direction == "up":
        ny = max(0.0, ny - shift)
    elif direction == "down":
        ny = min(ny + shift, 1.0)
    coords[idx] = (nx, ny)
    g_state["points_str"] = "[" + ", ".join(f"({c[0]:.3f}, {c[1]:.3f})" for c in coords) + "]"

def draw_points_on_image(pil_img, points_str, active_idx=0, only_active=False):
    """
    Draws circles on pil_img at the normalized points in points_str.
    If only_active=True, only the active_idx point is drawn (in blue).
    """
    matches = re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", points_str)
    coords = [(float(a), float(b)) for a, b in matches]
    if not coords:
        print("[DEBUG] No coordinates found in points_str. Returning original image.")
        return pil_img

    print(f"[DEBUG] Drawing circles overlay (only_active={only_active}) at idx={active_idx} over coords: {coords}")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape

    for i, (nx, ny) in enumerate(coords):
        if only_active and i != active_idx:
            continue
        px = int(nx * w)
        py = int(ny * h)
        # Blue circle in BGR
        color = (255, 0, 0)
        radius = 10  # Radius to approximate the size of the original cross (20 pixels span)
        cv2.circle(cv_img, (px, py), radius, color, 2)  # Draw circle with thickness 2

    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def draw_centerpoint_circle(pil_img, center_x, center_y, mask_bool=None):
    """
    Draws a circle and VERIFIES it's on the actual mask using the same logic as center calculation.
    """
    print(f"[DEBUG] Drawing circle at pixel coordinates ({center_x}, {center_y})")
    
    # Convert PIL to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape
    
    px = int(center_x)
    py = int(center_y)
    
    # If we have the mask, let's verify the center point
    if mask_bool is not None:
        print(f"[DEBUG] Mask shape: {mask_bool.shape}, Image shape: {h}x{w}")
        
        # Resize mask to match image if needed
        if mask_bool.shape != (h, w):
            mask_resized = cv2.resize(
                mask_bool.astype(np.uint8), (w, h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            print(f"[DEBUG] Resized mask from {mask_bool.shape} to {mask_resized.shape}")
        else:
            mask_resized = mask_bool
            
        # RECALCULATE center using the SAME logic as in calculate_object_info
        mask_y_coords, mask_x_coords = np.where(mask_resized)
        
        if len(mask_y_coords) > 0:
            # True center of mass calculation
            cx_true = float(np.mean(mask_x_coords))
            cy_true = float(np.mean(mask_y_coords))
            px_corrected = int(round(cx_true))
            py_corrected = int(round(cy_true))
            
            print(f"[DEBUG] Recalculated center: ({cx_true:.2f}, {cy_true:.2f}) -> ({px_corrected}, {py_corrected})")
            
            # Use the recalculated center instead of the passed coordinates
            px, py = px_corrected, py_corrected
            
            # Verify this point is actually in the mask
            if mask_resized[py, px]:
                print(f"[DEBUG] ✅ Corrected center ({px}, {py}) is confirmed on the mask!")
            else:
                print(f"[DEBUG] ❌ Even corrected center ({px}, {py}) is not on mask!")
                # Find closest mask pixel
                distances = ((mask_y_coords - py)**2 + (mask_x_coords - px)**2)
                closest_idx = np.argmin(distances)
                px = int(mask_x_coords[closest_idx])
                py = int(mask_y_coords[closest_idx])
                print(f"[DEBUG] Using closest mask pixel: ({px}, {py})")
        else:
            print(f"[DEBUG] ❌ No mask pixels found!")
    
    # Ensure coordinates are within image bounds
    if 0 <= px < w and 0 <= py < h:
        # Draw a larger, more visible circle
        color = (255, 0, 0)  # Blue in BGR
        radius = 15  # Larger radius
        thickness = 3  # Thicker line
        
        # Draw the circle
        cv2.circle(cv_img, (px, py), radius, color, thickness)
        
        # Also draw a small cross at the exact center
        cross_size = 8
        cv2.line(cv_img, (px-cross_size, py), (px+cross_size, py), color, 2)
        cv2.line(cv_img, (px, py-cross_size), (px, py+cross_size), color, 2)
        
        print(f"[DEBUG] ✅ Drew circle and cross at ({px}, {py}) within image bounds {w}x{h}")
        
        # VERIFICATION: Check if the pixel at this location is green-ish (part of mask overlay)
        pixel_bgr = cv_img[py, px]
        pixel_rgb = pixel_bgr[::-1]  # Convert BGR to RGB
        print(f"[DEBUG] Pixel color at center: RGB{tuple(pixel_rgb)}")
        
        # Check if it's greenish (from the mask overlay)
        if pixel_rgb[1] > pixel_rgb[0] and pixel_rgb[1] > pixel_rgb[2]:  # Green channel dominant
            print(f"[DEBUG] ✅ Pixel appears to be on the green mask!")
        else:
            print(f"[DEBUG] ⚠️ Pixel doesn't appear to be on the green mask")
            
    else:
        print(f"[DEBUG] ❌ Circle coordinates ({px}, {py}) are outside image bounds {w}x{h}")
    
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

###########################
# Flask Routes
###########################
@app.route("/")
def index():
    vlm_list = ["OpenVLM_1", "OpenVLM_2", "OpenVLM_3"]
    llm_list = ["ChatGPT", "Llama", "Claude"]
    return render_template("dashboard.html", vlm_options=vlm_list, llm_options=llm_list)

@app.route("/camera_info", methods=["GET"])
def camera_info():
    # Get list of connected RealSense devices (if pyrealsense2 is installed)
    rs_devices = depth_handler.list_realsense_devices() if RS_OK else []
    return jsonify({
        "local_cameras": depth_handler.local_cameras_info,
        # Report RealSense as available if we detected any devices
        "realsense_available": True if len(rs_devices) > 0 else False,
        "realsense_devices": rs_devices
    })

def broadcast_to_llm_console(token):
    # TEMP: Just print. You should route to a WebSocket or SSE for true real-time in the browser.
    print(f"[LLM Reasoning] {token}")


# ========== ROBOT CONTROL ROUTES ==========

@app.route("/robot/initialize", methods=["POST"])
def robot_initialize():
    """Initialize the robot system."""
    global robot_status
    
    with robot_lock:
        if robot_status["state"] in ["initializing", "moving"]:
            return jsonify({"error": "Robot is busy"}), 400
        
        if robot_status["state"] == "ready":
            return jsonify({"error": "Robot already initialized"}), 400
    
    # Start initialization in background thread
    init_thread = threading.Thread(target=initialize_robot_system)
    init_thread.daemon = True
    init_thread.start()
    
    return jsonify({"status": "Initialization started"})

@app.route("/robot/status", methods=["GET"])
def robot_status_endpoint():
    """Get current robot status."""
    with robot_lock:
        return jsonify(robot_status.copy())

@app.route("/robot/move/<direction>", methods=["POST"])
def robot_move(direction):
    """Move robot in specified direction."""
    global robot_status
    
    with robot_lock:
        if robot_status["state"] != "ready":
            return jsonify({"error": "Robot not ready"}), 400
    
    # Execute movement in background thread
    def move_thread():
        execute_robot_movement(direction)
    
    thread = threading.Thread(target=move_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": f"Moving {direction}"})

@app.route("/robot/stop", methods=["POST"])
def robot_stop():
    """Stop/reset robot system."""
    global robot_automation, robot_status
    
    try:
        if robot_automation:
            robot_automation.stop_robot()
            robot_automation = None
        
        with robot_lock:
            robot_status["state"] = "disconnected"
            robot_status["message"] = "Robot disconnected"
            robot_status["buttons_enabled"] = False
            
        return jsonify({"status": "Robot stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
#  STREAMING CHAT ENDPOINT  (drop-in replacement)
# ---------------------------------------------------------------------------
# Global variable to store the last raw frame
g_last_raw_frame = None
g_frozen_seg = None  # This will store the frozen candidate overlay image (PIL Image or raw bytes)

# Helper function to reset frame and fetch fresh frame if possible
def reset_frame_and_fetch_fresh():
    global g_last_raw_frame, g_frozen_seg
    g_last_raw_frame = None
    g_frozen_seg = None
    print("[INFO] Resetting stored frame and frozen segmentation.")
    
    # Attempt to fetch a fresh frame if RealSense is available
    if depth_handler.start_realsense():
        frames = depth_handler.get_realsense_frames()
        if frames is not None:
            color_arr, _ = frames  # Get color frame; ignore depth frame
            if color_arr is not None:
                # Convert from BGR to RGB
                rgb_arr = cv2.cvtColor(color_arr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_arr)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                g_last_raw_frame = buf.getvalue()
                print("[INFO] Fetched fresh frame from RealSense after reset.")
            else:
                print("[WARNING] No color frame available from RealSense after reset.")
        else:
            print("[WARNING] RealSense frames not ready after reset.")
    else:
        print("[INFO] RealSense not active, no fresh frame fetched after reset.")


# ---------------------------------------------------------------------------
#  STREAMING CHAT ENDPOINT (drop-in replacement)
# ---------------------------------------------------------------------------
@app.route("/chat-stream", methods=["POST"])
def chat_stream():
    """Fixed streaming with proper token accumulation and tag detection"""
    import httpx, asyncio
    
    # Reset frame and state on new chat message
    reset_frame_and_fetch_fresh()
    
    data = request.json or {}
    text = data.get("text", "").strip()
    
    async def relay():
        # System prompt that FORCES proper line breaks
        custom_system_prompt = r"""
You are the AAU Robot Agent. Your job is to convert user commands into robot actions.

CRITICAL FORMATTING RULES:
- Use <think></think> tags for ALL reasoning
- After </think>, output ONLY the [ACTION] block or a question
- Each line in [ACTION] block MUST be on separate lines
- NEVER put "RoboPoint Request" and "Action Request" on the same line

EXAMPLES:

User: "Pick the white cup and place it on the black case"
You:
<think>User wants to pick up a white cup and place it on a black case. This requires two actions: Pick the cup, then Place it on the case.</think>

[ACTION]
RoboPoint Request: white cup; black case
Action Request: Pick; Place

User: "Pick up the blue bottle"
You:
<think>User wants to pick up a blue bottle. Only one action needed.</think>

[ACTION]
RoboPoint Request: blue bottle
Action Request: Pick

User: "Move something"
You:
<think>User said "move something" but didn't specify what object. I need clarification.</think>

Which object should I move?

CRITICAL: Always use proper line breaks in [ACTION] blocks!
"""

        payload = {
            "instructions": "custom",
            "custom_instructions": custom_system_prompt,
            "message": text,
            "session_id": "default"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        # Accumulate tokens to properly detect tags
        accumulated_content = ""
        in_thinking = False
        
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", LLM_STREAM, json=payload, headers=headers) as r:
                
                async for raw in r.aiter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                        
                    token = raw[5:].lstrip()
                    if token == "[DONE]":
                        # Send any remaining content
                        if accumulated_content.strip():
                            if in_thinking:
                                yield f"data: thinking:{accumulated_content}\n\n"
                            else:
                                yield f"data: response:{accumulated_content}\n\n"
                        break
                    
                    # Accumulate the token
                    accumulated_content += token
                    
                    # Check for complete think tags
                    while "<think>" in accumulated_content and not in_thinking:
                        before_think, after_think = accumulated_content.split("<think>", 1)
                        if before_think.strip():
                            yield f"data: response:{before_think}\n\n"
                        accumulated_content = after_think
                        in_thinking = True
                    
                    while "</think>" in accumulated_content and in_thinking:
                        think_content, after_think = accumulated_content.split("</think>", 1)
                        if think_content.strip():
                            yield f"data: thinking:{think_content}\n\n"
                        accumulated_content = after_think
                        in_thinking = False
                    
                    # Send partial content if we have enough tokens (but preserve tag boundaries)
                    if len(accumulated_content) > 50 and "<" not in accumulated_content[-10:]:
                        if in_thinking:
                            yield f"data: thinking:{accumulated_content}\n\n"
                        else:
                            yield f"data: response:{accumulated_content}\n\n"
                        accumulated_content = ""

    def event_stream():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agen = relay().__aiter__()
        try:
            while True:
                chunk = loop.run_until_complete(agen.__anext__())
                yield chunk
        except StopAsyncIteration:
            pass

    return Response(stream_with_context(event_stream()),
                    mimetype="text/event-stream")

@app.route("/chat", methods=["POST"])
def chat():
    # Reset frame and state on new chat message
    reset_frame_and_fetch_fresh()
    
    data = request.json or {}
    text = data.get("text", "").strip()
    frame_b64 = data.get("seg_frame", "")

    # Call Qwen3 Docker endpoint
    try:
        payload = {
            "instructions": "default",
            "message": text, 
            "session_id": "default"
        }
        
        resp = requests.post(LLM_FULL, json=payload, timeout=60)
        resp.raise_for_status()
        llm_data = resp.json()
        
        answer = llm_data.get("result", "")
        parsed = llm_data.get("parsed")
        
        # Check if it's an action
        if parsed and "[ACTION]" in answer:
            # Execute RoboPoint pipeline
            object_phrase = parsed.get("requests", ["unknown"])[0] 
            action_phrase = parsed.get("actions", ["pick"])[0]
            instruction = f"{action_phrase} the {object_phrase}"
            
            pts_str = robo_handler.call_remote_for_points(frame_b64, instruction)
            
            # Process results...
            # (keep existing segmentation and geometry code)
            
        return jsonify({"reply": answer, "answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process_seg", methods=["POST"])
def process_seg_endpoint():
    """
    FIRST CLICK (freeze):
        • Runs SAM at the click → green mask
        • Computes 3-D centre / distance / W×H immediately (RealSense)
        • Returns overlay image + object_info

    SECOND CLICK (unfreeze):
        • Resets segmentation state (handled client-side with /reset_seg)
    """
    global g_last_raw_frame

    # Reset frame and state on click to ensure fresh data
    reset_frame_and_fetch_fresh()

    # ---------- 0  decode incoming frame ----------
    data      = request.json or {}
    frame_b64 = data.get("frame", "")
    if frame_b64:
        raw_bytes        = base64.b64decode(frame_b64)
        g_last_raw_frame = raw_bytes
        pil_img          = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    elif g_last_raw_frame:
        pil_img          = Image.open(io.BytesIO(g_last_raw_frame)).convert("RGB")
    else:
        return jsonify({"frame": ""})

    clicked_x = data.get("clicked_x")
    clicked_y = data.get("clicked_y")

    # ------------------------------------------------------------------
    # 1  SEGMENTATION PATH (user clicked inside the image)
    # ------------------------------------------------------------------
    if clicked_x is not None and clicked_y is not None:
        click_coords = (float(clicked_x), float(clicked_y))

        # 1-a  SAM mask & overlay
        sam_img = sam_handler.run_sam_overlay(
            pil_img, [click_coords], active_idx=0, max_dim=640
        )

        # 1-b  Depth-based geometry (if RealSense available)
        object_info = None
        if depth_handler.start_realsense():  # no-op if already started
            _, depth_arr = depth_handler.get_realsense_frames()
            mask_bool    = sam_handler.get_last_mask()
            object_info  = depth_handler.calculate_object_info(mask_bool, depth_arr)

        # 1-c  Draw circle at centerpoint if object_info is available
        output_img = sam_img
        if object_info and "center_px" in object_info:
            # Use the TRUE center of mass from the mask calculation
            center_x, center_y = object_info["center_px"]
            mask_bool = sam_handler.get_last_mask()
            output_img = draw_centerpoint_circle(sam_img, center_x, center_y, mask_bool)
            print(f"[DEBUG] Manual click: Drew circle using mask-verified coordinates ({center_x}, {center_y})")
        elif object_info and "bbox_px" in object_info:
            # Fallback to bounding box center only if center_px is not available
            x_min, y_min, x_max, y_max = object_info["bbox_px"]
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            mask_bool = sam_handler.get_last_mask()
            output_img = draw_centerpoint_circle(sam_img, center_x, center_y, mask_bool)
            print(f"[DEBUG] Manual click: Drew circle using bbox center fallback ({center_x}, {center_y})")
        elif g_state["points_str"]:
            output_img = draw_points_on_image(
                sam_img, g_state["points_str"], g_state["active_cross_idx"]
            )

    # ------------------------------------------------------------------
    # 2  DEPTH-ANYTHING PATH (no click, e.g. RealSense RGB stream)
    # ------------------------------------------------------------------
    else:
        object_info     = None
        depth_processed = depth_handler.run_depth_anything(pil_img)
        output_img      = (
            draw_points_on_image(
                depth_processed, g_state["points_str"], g_state["active_cross_idx"]
            )
            if g_state["points_str"]
            else depth_processed
        )

    # ---------- 3  send back frame (+ geometry) ----------
    out_b64 = pil_to_b64(output_img)
    return jsonify({
        "frame"       : out_b64,
        "object_info" : object_info  # may be None
    })
    
@app.route("/process_realsense_seg", methods=["POST"])
def process_realsense_seg():
    with depth_lock:
        if not depth_handler.start_realsense():
            return jsonify({"error": "Failed to start RealSense"}), 500
        frames = depth_handler.get_realsense_frames()
        if frames is None:
            return jsonify({"error": "still loading frames"}), 500
        color_arr, _ = frames  # get color frame; ignore depth frame
        if color_arr is None:
            return jsonify({"error": "No color frame available"}), 500
        # Convert from BGR to RGB.
        rgb_arr = cv2.cvtColor(color_arr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_arr)
        return jsonify({"frame": pil_to_b64(pil_img)})

@app.route("/process_depth", methods=["POST"])
def process_depth():
    """
    Processes an image for depth information using Depth Anything logic and returns a clean output.
    """
    data = request.json
    mode = data.get("camera_mode", "off")
    local_idx = data.get("local_idx", -1)
    frame_b64 = data.get("frame", "")

    print(f"[Depth] Received => mode={mode} local_idx={local_idx}")

    if mode == "off":
        depth_handler.stop_local_camera()
        depth_handler.stop_realsense()
        return jsonify({"frame": ""})

    elif mode in ("default_anything", "sidecam_depth"):
        if not frame_b64:
            return jsonify({"error": "no local frame"}), 400
        raw = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        out_img = depth_handler.run_depth_anything(pil_img)
        return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "local_depth_anything":
        with depth_lock:
            if not depth_handler.start_local_camera(local_idx):
                return jsonify({"error": f"Failed to open local camera {local_idx}"}), 500
            frame = depth_handler.grab_local_frame()
            if frame is None:
                return jsonify({"error": "no local frame captured"}), 500
            pil_img = Image.fromarray(frame[..., ::-1])  # Convert BGR -> RGB
            out_img = depth_handler.run_depth_anything(pil_img)
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "realsense_rgb_anything":
        with depth_lock:
            if not depth_handler.start_realsense():
                return jsonify({"error": "Failed to start RealSense"}), 500
            out_img = depth_handler.realsense_color_to_depthanything()
            if out_img is None:
                return jsonify({"error": "still loading frames"})
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "realsense_depth":
        with depth_lock:
            if not depth_handler.start_realsense():
                return jsonify({"error": "Failed to start RealSense"}), 500
            out_img = depth_handler.realsense_depth_colormap()
            if out_img is None:
                return jsonify({"error": "still loading frames"})
            return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "image_depth":
        if not frame_b64:
            return jsonify({"error": "no frame provided"}), 400
        raw = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        out_img = depth_handler.run_depth_anything(pil_img)
        return jsonify({"frame": pil_to_b64(out_img)})

    elif mode == "other":
        if frame_b64:
            raw = base64.b64decode(frame_b64)
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.array(pil_img)
            out_img = Image.fromarray(255 - arr)
            return jsonify({"frame": pil_to_b64(out_img)})
        else:
            arr = np.full((240, 320, 3), 127, dtype=np.uint8)
            out_img = Image.fromarray(arr)
            return jsonify({"frame": pil_to_b64(out_img)})

    else:
        return jsonify({"error": f"Unknown camera_mode: {mode}"}), 400

@app.route("/reset_seg", methods=["POST"])
def reset_seg():
    """
    Reset segmentation state (abort any pending confirm),
    clear all points, and return to RGB mode.
    """
    global g_state, g_last_raw_frame, g_frozen_seg
    g_state["mode"] = "rgb"
    g_state["points_str"] = ""
    g_state["active_cross_idx"] = 0
    g_state["clicked_points"] = []
    g_state["prev_seg_np_img"] = None
    g_state["last_seg_output"] = None
    g_last_raw_frame = None
    g_frozen_seg = None
    print("[INFO] Full reset of segmentation state, frame, and frozen image.")
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    print("[INFO] Starting on :5000 in multi-threaded mode.")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
