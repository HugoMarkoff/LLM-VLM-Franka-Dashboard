"""utils/robopoints_handler.py - Updated for Docker RoboPoint endpoint"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip('/')

class RoboPointsHandler:
    def __init__(self, endpoint_url=None):
        self.endpoint_url = endpoint_url or f"{NGROK_BASE_URL}/robopoint/predict"
        print(f"[RoboPoints] Using endpoint: {self.endpoint_url}")

    def call_remote_for_points(self, frame_b64, instruction):
        """Call Docker RoboPoint endpoint with new format"""
        try:
            payload = {
                "instructions": "default",
                "message": instruction,
                "image": frame_b64
            }
            
            print(f"[RoboPoints] Sending request: {instruction}")
            resp = requests.post(
                self.endpoint_url,
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", "")
            print(f"[RoboPoints] Received: {result}")
            return result
            
        except Exception as e:
            print(f"[ERROR in call_remote_for_points]: {e}")
            return ""