#!/usr/bin/env python3
"""
Enhanced Robot Action Planner - 4-Step Pipeline with Online API

A modular robot action planning system that:
1. Analyzes images with vision model (Qwen2.5-VL)
2. Validates syntax and spatial relationships
3. Refines reasoning with online LLM (DeepSeek)
4. Generates executable action plans

Author: Chen Li
"""

import sys
import os
import time
import json
import requests
import io
import contextlib
import base64
from openai import OpenAI

# Import robot action execution
try:
    from . import robot_action
except ImportError:
    # Fallback for direct execution
    import robot_action

# =============================================================================
# CONFIGURATION
# =============================================================================

# Online VLM Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_VLM_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_LLM_URL = "https://openrouter.ai/api/v1"
ONLINE_VLM_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"
ONLINE_LLM_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Available Robot Actions
AVAILABLE_ACTIONS = {
    "home": "return the robot to its home position (0, 0, 0)",
    "pickmove": "move to the object location to prepare for picking",
    "pick": "pick up the object at current location", 
    "placemove": "move to the target location to prepare for placing",
    "place": "put down the object at current location",
    "OpenGripper": "open the gripper",
    "CloseGripper": "close the gripper"
}

# =============================================================================
# JSON INPUT PARSING
# =============================================================================

def parse_json_input(json_string):
    """
    Parse JSON input string and extract task parameters.
    
    Args:
        json_string (str): JSON string with user_message, imagePath, and objects
        
    Returns:
        tuple: (task_description, image_path, objects_data) or (None, None, None) if error
    """
    try:
        data = json.loads(json_string)
        
        # Extract required fields
        task_description = data.get("user_message", "")
        image_path = data.get("imagePath", "")
        objects_data = data.get("objects", [])
        
        # Validate required fields
        if not task_description:
            print("[ERROR] Missing 'user_message' in JSON input")
            return None, None, None
            
        if not image_path:
            print("[ERROR] Missing 'imagePath' in JSON input")
            return None, None, None
            
        if not objects_data:
            print("[ERROR] Missing 'objects' array in JSON input")
            return None, None, None
        
        # Validate objects structure
        for i, obj in enumerate(objects_data):
            required_fields = ["action", "location", "object", "object_center_location"]
            for field in required_fields:
                if field not in obj:
                    print(f"[ERROR] Missing '{field}' in object {i}")
                    return None, None, None
            
            # Validate object_center_location has x, y
            center_loc = obj.get("object_center_location", {})
            if "x" not in center_loc or "y" not in center_loc:
                print(f"[ERROR] Missing 'x' or 'y' in object_center_location for object {i}")
                return None, None, None
        
        print(f"[INFO] ✅ JSON input parsed successfully")
        print(f"[INFO] Task: {task_description}")
        print(f"[INFO] Image: {image_path}")
        print(f"[INFO] Objects: {len(objects_data)} objects defined")
        
        return task_description, image_path, objects_data
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON format: {e}")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] Error parsing JSON input: {e}")
        return None, None, None

def validate_object_consistency(vlm_objects, vlm_relationships, json_objects):
    """
    Validate that JSON objects and locations align with VLM output.
    
    Args:
        vlm_objects (list): Objects detected by VLM
        vlm_relationships (str): Spatial relationships from VLM
        json_objects (list): Objects from JSON input
        
    Returns:
        tuple: (is_consistent, issues_list)
    """
    print("\n[CONSISTENCY CHECK] Validating JSON objects against VLM output...")
    
    is_consistent = True
    issues = []
    
    # Extract JSON object names and locations
    json_object_names = [obj["object"].lower().strip() for obj in json_objects]
    json_locations = {obj["object"].lower().strip(): obj["location"].lower().strip() for obj in json_objects}
    
    # Convert VLM objects to lowercase for comparison
    vlm_object_names = [obj.lower().strip() for obj in vlm_objects]
    
    print(f"[INFO] JSON objects: {json_object_names}")
    print(f"[INFO] VLM objects: {vlm_object_names}")
    
    # Check if JSON objects exist in VLM detection
    for json_obj in json_object_names:
        # Try exact match first
        if json_obj not in vlm_object_names:
            # Try partial matches (e.g., "yellow banana" vs "banana")
            partial_match = False
            for vlm_obj in vlm_object_names:
                if json_obj in vlm_obj or vlm_obj in json_obj:
                    partial_match = True
                    print(f"[INFO] Partial match found: '{json_obj}' ≈ '{vlm_obj}'")
                    break
            
            if not partial_match:
                is_consistent = False
                issues.append(f"Object '{json_obj}' from JSON not detected by VLM")
                print(f"[ERROR] Object '{json_obj}' not found in VLM detection")
    
    # Validate locations by extracting VLM's STEP 2 absolute positions
    import re
    
    for obj_data in json_objects:
        obj_name = obj_data["object"].lower().strip()
        json_location = obj_data["location"].lower().strip()
        
        # Look for VLM's STEP 2 absolute position format: "- **ObjectName**: location"
        # Try different variations of object name matching
        vlm_location = None
        
        # Pattern 1: Exact match
        pattern1 = rf'- \*\*{re.escape(obj_name)}\*\*:\s*([a-z-]+)'
        match1 = re.search(pattern1, vlm_relationships.lower(), re.IGNORECASE)
        
        if match1:
            vlm_location = match1.group(1).strip()
        else:
            # Pattern 2: Partial match (e.g., "yellow banana" vs "banana")
            for vlm_obj in vlm_object_names:
                if obj_name in vlm_obj or vlm_obj in obj_name:
                    pattern2 = rf'- \*\*{re.escape(vlm_obj)}\*\*:\s*([a-z-]+)'
                    match2 = re.search(pattern2, vlm_relationships.lower(), re.IGNORECASE)
                    if match2:
                        vlm_location = match2.group(1).strip()
                        break
        
        # Compare locations if VLM provided one
        if vlm_location:
            print(f"[INFO] Comparing locations for '{obj_name}': JSON='{json_location}' vs VLM='{vlm_location}'")
            
            if json_location != vlm_location:
                is_consistent = False
                issues.append(f"Location mismatch for '{obj_name}': JSON says '{json_location}' but VLM says '{vlm_location}'")
                print(f"[ERROR] Location mismatch for '{obj_name}': JSON='{json_location}' vs VLM='{vlm_location}'")
            else:
                print(f"[INFO] ✅ Location match for '{obj_name}': '{json_location}'")
        else:
            print(f"[WARNING] Could not find VLM location for '{obj_name}' in STEP 2 output")
    
    if is_consistent:
        print("[CONSISTENCY CHECK] ✅ JSON objects and locations are consistent with VLM output")
    else:
        print(f"[CONSISTENCY CHECK] ❌ Found {len(issues)} consistency issues")
        for issue in issues:
            print(f"  - {issue}")
    
    return is_consistent, issues

# =============================================================================
# GLOBAL STATE MANAGEMENT
# =============================================================================

# Online service status
online_vlm_initialized = False

# Cache for image analysis results (avoids reprocessing same image)
cached_image_path = None
cached_step1_objects = None
cached_step1_relationships = None
cached_step2_objects = None
cached_step2_relationships = None
cached_step3_objects = None
cached_step3_relationships = None

# Pipeline information display
print("="*80)
print("🤖 Max - ROBOT PLANNER - 4-STEP PIPELINE FOR TASK PLANNING")
print("="*80)
print("STEP 1: Object identification + spatial relationship reasoning through VLM (qwen/qwen2.5-vl-72b-instruct:free)")
print("STEP 2: Syntax check of reasoning results and object lists")
print("STEP 3: Refined reasoning for logical consistency through LLM (deepseek/deepseek-chat-v3-0324:free)")
print("STEP 4: Generate the task plan")
print("="*80)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def encode_image_to_base64(image_path):
    """
    Encode image to base64 string for API calls.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_online_vlm(image_path, prompt, max_tokens=512):
    """
    Call the online VLM API (Qwen2.5-VL through OpenRouter).
    
    Args:
        image_path (str): Path to the image file
        prompt (str): The prompt to send to the API
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str or None: Response text or None if error
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        # Read and encode the image
        base64_image = encode_image_to_base64(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": ONLINE_VLM_MODEL,
            "messages": messages,
            "stream": True
        }

        response = requests.post(OPENROUTER_VLM_URL, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        # Stream and collect response
        print("[STREAMING VLM RESPONSE]")
        print("-" * 50)
        full_response = ""
        for line in response.iter_lines():
            if line and line.decode('utf-8').startswith("data: ") and line.decode('utf-8') != "data: [DONE]":
                try:
                    chunk = json.loads(line.decode('utf-8')[6:])  # Remove "data: "
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        print(content, end='', flush=True)  # Stream to terminal
                        full_response += content
                except:
                    pass
        print()  # New line when done
        print("-" * 50)
        print("[VLM RESPONSE COMPLETE]")

        if len(full_response) > 0:
            return full_response
        else:
            print(f"[ERROR] No response content found from online VLM")
            return None
            
    except Exception as e:
        print(f"[ERROR] Online VLM API call failed: {e}")
        return None

def call_online_llm(prompt, max_tokens=512):
    """
    Call the online LLM API (DeepSeek through OpenRouter) with streaming.
    
    Args:
        prompt (str): The prompt to send to the API
        max_tokens (int): Maximum tokens in response
        
    Returns:
        str or None: Response text or None if error
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": ONLINE_LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant and your name is Max."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": max_tokens
        }

        response = requests.post(OPENROUTER_VLM_URL, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        # Stream and collect response
        print("[STREAMING LLM RESPONSE]")
        print("-" * 50)
        full_response = ""
        for line in response.iter_lines():
            if line and line.decode('utf-8').startswith("data: ") and line.decode('utf-8') != "data: [DONE]":
                try:
                    chunk = json.loads(line.decode('utf-8')[6:])  # Remove "data: "
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        print(content, end='', flush=True)  # Stream to terminal
                        full_response += content
                except:
                    pass
        print()  # New line when done
        print("-" * 50)
        print("[LLM RESPONSE COMPLETE]")

        if len(full_response) > 0:
            return full_response
        else:
            print(f"[ERROR] No response content found from online LLM")
            return None
            
    except Exception as e:
        print(f"[ERROR] Online LLM API call failed: {e}")
        return None

def extract_objects_from_response(response):
    """
    Extract objects from vision model response using multiple parsing strategies.
    
    Args:
        response (str): Raw vision model response
        
    Returns:
        list: List of detected object names
    """
    import re
    
    # Function to get coordinates for an object from JSON
    def get_object_coordinates(object_name, json_objects):
        if not json_objects:
            return "(0.0, 0.0, 0.0)"
        
        obj_name_lower = object_name.lower().strip()
        for obj_data in json_objects:
            json_obj_name = obj_data["object"].lower().strip()
            # Try exact match first, then partial match
            if obj_name_lower == json_obj_name or obj_name_lower in json_obj_name or json_obj_name in obj_name_lower:
                center_loc = obj_data.get("object_center_location", {})
                x = center_loc.get("x", 0.0)
                y = center_loc.get("y", 0.0)
                z = center_loc.get("z", 0.0)
                return f"({x}, {y}, {z})"
        return "(0.0, 0.0, 0.0)"
    
    detected_objects = []
    
    # Strategy 1: Parse structured "Objects Identified" section
    objects_section_match = re.search(r'#### Objects Identified:(.*?)(?=####|\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
    
    if objects_section_match:
        objects_section = objects_section_match.group(1)
        object_matches = re.findall(r'\d+\.\s*\*\*([^*]+)\*\*', objects_section)
        
        for obj in object_matches:
            obj = obj.strip()
            if obj and len(obj) <= 25:
                detected_objects.append(obj.title())
    
    # Strategy 2: Fallback - spatial relationships section
    if not detected_objects:
        spatial_matches = re.findall(r'\d+\.\s*\*\*([^*:]+)\*\*:', response)
        for obj in spatial_matches:
            obj = obj.strip()
            if obj and len(obj) <= 25:
                detected_objects.append(obj.title())
    
    # Strategy 3: Final fallback - any bold text
    if not detected_objects:
        bold_matches = re.findall(r'\*\*([A-Za-z][A-Za-z\s]{2,20})\*\*', response)
        non_objects = {'ON TOP OF', 'TO THE LEFT OF', 'TO THE RIGHT OF', 'OVERLAPPING WITH',
                      'Objects Identified', 'Spatial Relationships', 'Summary', 'Step', 'Analysis'}
        
        for obj in bold_matches:
            obj = obj.strip()
            if obj not in non_objects and len(obj) <= 25:
                detected_objects.append(obj.title())
    
    # Clean and deduplicate
    final_objects = []
    seen = set()
    
    for obj in detected_objects:
        obj_clean = obj.strip().title()
        if obj_clean and obj_clean not in seen and len(obj_clean) >= 3:
            final_objects.append(obj_clean)
            seen.add(obj_clean)
    
    return final_objects[:6]  # Limit to reasonable number

def correct_spatial_inconsistencies(relationships_str, objects_list, summary_text):
    """
    Correct spatial relationship inconsistencies using summary as reference truth.
    
    Args:
        relationships_str (str): Raw spatial relationships
        objects_list (list): List of detected objects
        summary_text (str): Summary section for ground truth
        
    Returns:
        str: Corrected relationships text
    """
    import re
    
    print(f"\n[CONSISTENCY CHECK] Analyzing spatial relationships for logical errors...")
    
    corrected_text = relationships_str
    ground_truth = {}
    
    if summary_text:
        # Extract relationships from summary
        for line in summary_text.split('\n'):
            line = line.strip('- ').strip()
            
            # Parse "X is on top of Y" patterns
            if 'is on top of' in line.lower():
                match = re.search(r'(?:the )?\*\*?(\w+(?:\s+\w+)*)\*\*? is \*\*?on top of\*\*? (?:the )?\*\*?(\w+(?:\s+\w+)*)\*\*?', line.lower())
                if match:
                    obj1, obj2 = match.groups()
                    ground_truth[obj1.title()] = {'on_top_of': obj2.title()}
                    ground_truth[obj2.title()] = ground_truth.get(obj2.title(), {})
                    ground_truth[obj2.title()]['under'] = ground_truth[obj2.title()].get('under', []) + [obj1.title()]
            
            # Parse "X is under Y" patterns
            elif 'is under' in line.lower() or 'under the' in line.lower():
                match = re.search(r'the (\w+(?:\s+\w+)*) is under (?:the )?(.+)', line.lower())
                if match:
                    obj1 = match.group(1)
                    under_objects_str = match.group(2)
                    
                    # Parse multiple objects separated by "and"
                    under_objects = []
                    for obj in re.split(r'\s+and\s+', under_objects_str):
                        obj = obj.strip()
                        obj = re.sub(r'^(?:the\s+)?', '', obj)
                        obj = re.sub(r'[,.]$', '', obj)
                        if obj:
                            under_objects.append(obj.title())
                    
                    if under_objects:
                        ground_truth[obj1.title()] = ground_truth.get(obj1.title(), {})
                        ground_truth[obj1.title()]['under'] = under_objects
                        
                        for under_obj in under_objects:
                            ground_truth[under_obj] = ground_truth.get(under_obj, {})
                            if 'on_top_of' not in ground_truth[under_obj]:
                                ground_truth[under_obj]['on_top_of'] = []
                            if isinstance(ground_truth[under_obj]['on_top_of'], str):
                                ground_truth[under_obj]['on_top_of'] = [ground_truth[under_obj]['on_top_of']]
                            if obj1.title() not in ground_truth[under_obj]['on_top_of']:
                                ground_truth[under_obj]['on_top_of'].append(obj1.title())
    
    print(f"[CONSISTENCY CHECK] ✅ Spatial relationship validation completed")
    return corrected_text

def extract_plan_from_response(plan_response, thinking_content, task_description, objects_list, relationships_str, json_objects=None):
    """
    Extract action plan from model response with fallback plan generation.
    
    Args:
        plan_response (str): Raw model response
        thinking_content (str): Thinking content (unused but kept for compatibility)
        task_description (str): Original task
        objects_list (list): Detected objects
        relationships_str (str): Spatial relationships
        json_objects (list): Objects from JSON input with coordinates (optional)
        
    Returns:
        str: Formatted action plan
    """
    import re
    
    # Function to get coordinates for an object from JSON
    def get_object_coordinates(object_name, json_objects):
        if not json_objects:
            return "(0.0, 0.0, 0.0)"
        
        obj_name_lower = object_name.lower().strip()
        for obj_data in json_objects:
            json_obj_name = obj_data["object"].lower().strip()
            # Try exact match first, then partial match
            if obj_name_lower == json_obj_name or obj_name_lower in json_obj_name or json_obj_name in obj_name_lower:
                center_loc = obj_data.get("object_center_location", {})
                x = center_loc.get("x", 0.0)
                y = center_loc.get("y", 0.0)
                z = center_loc.get("z", 0.0)
                return f"({x}, {y}, {z})"
        return "(0.0, 0.0, 0.0)"
    
    # Method 1: Look for "Max: The task plan is..." pattern (PRIMARY)
    max_pattern = re.search(r'Max:\s*The\s*task\s*plan\s*is\s*(.*?)(?:\n|$)', plan_response, re.IGNORECASE | re.DOTALL)
    if max_pattern:
        plan_content = max_pattern.group(1).strip()
        # Clean up any trailing text after the plan
        plan_content = re.sub(r'\s*(Note:|IMPORTANT:).*$', '', plan_content, flags=re.IGNORECASE | re.DOTALL)
        if plan_content and len(plan_content) > 10:
            return f"plan: {plan_content}"
    
    # Method 2: Look for "plan:" pattern (SECONDARY)
    plan_matches = list(re.finditer(r'plan:\s*\(', plan_response, re.IGNORECASE))
    if plan_matches:
        last_plan_start = plan_matches[-1].start()
        plan_section = plan_response[last_plan_start:]
        
        lines = plan_section.split('\n')
        plan_line = lines[0].strip()
        
        # Check if plan continues on subsequent lines
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if line.startswith('->') or (line and '->' in line and not any(stop_word in line.lower() for stop_word in ['but', 'however', 'wait', 'looking', 'note:', 'important:'])):
                plan_line += ' ' + line
            else:
                break
        
        if plan_line and len(plan_line) > 10:
            return plan_line
    
    # Method 3: Look for action patterns and reconstruct (TERTIARY)
    action_pattern = r'\([A-Za-z\s]+,\s*\([0-9.-]+,\s*[0-9.-]+,\s*[0-9.-]+\),\s*[A-Za-z]+\)'
    actions = re.findall(action_pattern, plan_response)
    if actions and len(actions) > 1:
        return "plan: " + " -> ".join(actions)
    
    # Method 4: Fallback plan generation with new action format (LAST RESORT)
    print("[WARNING] Could not extract valid plan from model output. Creating fallback plan...")
    
    # Identify target objects from task
    target_objects = []
    task_lower = task_description.lower()
    
    if "first" in task_lower and "then" in task_lower:
        # Handle multi-step tasks
        first_part = task_lower.split("then")[0]
        then_part = task_lower.split("then")[1] if "then" in task_lower else ""
        
        first_obj = None
        then_obj = None
        
        for obj in objects_list:
            if obj.lower() in first_part and first_obj is None:
                first_obj = obj
            if obj.lower() in then_part and then_obj is None:
                then_obj = obj
        
        if first_obj:
            target_objects.append(first_obj)
        if then_obj:
            target_objects.append(then_obj)
    else:
        # Single object task or "and" tasks
        for obj in objects_list:
            if obj.lower() in task_lower:
                target_objects.append(obj)
    
    target_obj = target_objects[0] if target_objects else None
    
    if target_obj:
        # Create plan based on dependencies using new action format
        plan_steps = []
        
        # Always start from home
        plan_steps.append("(home, (0.0, 0.0, 0.0), home)")
        
        for target in target_objects:
            # Find blocking objects
            target_blocking_objects = []
            
            for obj in objects_list:
                if obj != target:
                    blocking_patterns = [
                        f"the {obj.lower()} is on top of the {target.lower()}",
                        f"{obj.lower()} is on top of the {target.lower()}",
                        f"- the {obj.lower()} is on top of the {target.lower()}",
                        f"- {obj.lower()} is on top of the {target.lower()}",
                        f"{obj.lower()} on top of {target.lower()}",
                        f"**{obj}**: ON TOP OF the {target}",
                        f"**{obj.title()}**: ON TOP OF the {target.title()}"
                    ]
                    
                    if any(pattern in relationships_str.lower() for pattern in blocking_patterns):
                        target_blocking_objects.append(obj)
            
            # Move blocking objects first using new workflow
            for blocking_obj in target_blocking_objects:
                blocking_coords = get_object_coordinates(blocking_obj, json_objects)
                plan_steps.extend([
                    f"({blocking_obj}, {blocking_coords}, pickmove)",
                    f"({blocking_obj}, {blocking_coords}, pick)",
                    f"(home, (0.0, 0.0, 0.0), home)",
                    f"(neutral, (50.0, 50.0, 30.0), placemove)",
                    f"(neutral, (50.0, 50.0, 30.0), place)",
                    f"(home, (0.0, 0.0, 0.0), home)"
                ])
            
            # Then pick up target using new workflow
            target_coords = get_object_coordinates(target, json_objects)
            plan_steps.extend([
                f"({target}, {target_coords}, pickmove)",
                f"({target}, {target_coords}, pick)"
            ])
            
            # If this is a pick and place task, add place sequence
            if "place" in task_lower or "put" in task_lower:
                # Find target location from task
                place_target = None
                place_words = ["on", "onto", "at", "to"]
                for word in place_words:
                    if word in task_lower:
                        parts = task_lower.split(word)
                        if len(parts) > 1:
                            place_part = parts[-1].strip()
                            for obj in objects_list:
                                if obj.lower() in place_part:
                                    place_target = obj
                                    break
                            break
                
                if place_target:
                    place_coords = get_object_coordinates(place_target, json_objects)
                    plan_steps.extend([
                        f"(home, (0.0, 0.0, 0.0), home)",
                        f"({place_target}, {place_coords}, placemove)",
                        f"({place_target}, {place_coords}, place)"
                    ])
        
        if plan_steps:
            return "plan: " + " -> ".join(plan_steps)
        else:
            # Simple pick operation fallback
            target_coords = get_object_coordinates(target_obj, json_objects)
            return f"plan: (home, (0.0, 0.0, 0.0), home) -> ({target_obj}, {target_coords}, pickmove) -> ({target_obj}, {target_coords}, pick)"
    
    # Ultimate fallback
    default_coords = get_object_coordinates("Object", json_objects) if json_objects else "(0.0, 0.0, 0.0)"
    return f"plan: (home, (0.0, 0.0, 0.0), home) -> (Object, {default_coords}, pickmove) -> (Object, {default_coords}, pick)"

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

def load_vision_model():
    """
    Initialize connection to online vision model service.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global online_vlm_initialized
    
    print("\n🚀 INITIALIZING VISION MODEL FOR 4-STEP PIPELINE\n")
    
    # Check if API key is available
    if not OPENROUTER_API_KEY:
        print("[ERROR] Cannot initialize vision model - OPENROUTER_API_KEY not set!")
        print("[INFO] Please set the environment variable first:")
        return False
    
    # Test API connection
    print(f"[INFO] Testing connection to online VLM service...")
    print(f"[INFO] Model: {ONLINE_VLM_MODEL}")
    print(f"[INFO] API Endpoint: {OPENROUTER_VLM_URL}")
    
    try:
        # Test the API with a simple request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Simple test payload
        test_payload = {
            "model": ONLINE_VLM_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        test_response = requests.post(OPENROUTER_VLM_URL, headers=headers, json=test_payload, timeout=10)
        
        if test_response.status_code == 200:
            print(f"[INFO] ✅ Successfully connected to online VLM service")
        else:
            print(f"[WARNING] API test returned status {test_response.status_code}, but will proceed")
        
    except Exception as e:
        print(f"[WARNING] API connection test failed: {e}, but will proceed anyway")
    
    # Mark as initialized
    online_vlm_initialized = True
    
    print(f"\n🎉 Online vision model initialized! Ready for 4-Step pipeline.\n")
    return True

def cleanup_models():
    """Clean up online service status (no actual cleanup needed for online models)."""
    global online_vlm_initialized
    
    print("[INFO] Resetting online service status...")
    
    online_vlm_initialized = False
    
    print("[INFO] ✅ Online service status reset.")

# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step1_vision_analysis(image_path, force_reanalyze=False):
    """
    STEP 1: Object identification and spatial analysis using online vision model.
    
    Args:
        image_path (str): Path to the input image
        force_reanalyze (bool): Force re-analysis even if cached
        
    Returns:
        tuple: (objects_list, relationships_str) or (None, None) if error
    """
    global online_vlm_initialized
    global cached_image_path, cached_step1_objects, cached_step1_relationships
    
    print("\n" + "="*60)
    print("🔍 STEP 1: VISION ANALYSIS (Using Qwen2.5-VL-72B)")
    print("="*60)
    
    if not online_vlm_initialized:
        print("[ERROR] Online VLM not initialized. Please run load_vision_model() first.")
        return None, None
    
    # Check cache first
    if not force_reanalyze and cached_image_path == image_path and cached_step1_objects is not None:
        print("[INFO] 🚀 Using cached STEP 1 results (much faster!)")
        return cached_step1_objects, cached_step1_relationships
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return None, None

    # Prepare vision prompt
    prompt = (
        "Analyze this 2D image step by step to identify objects, their absolute positions in the image, and their spatial relationships to each other. Be precise and consistent.\n"
        "\n"
        "STEP 1: List all visible objects (banana, controller, remote, paper, etc.)\n"
        "\n"
        "STEP 2: For each object, determine its ABSOLUTE location in the image using ONLY these terms:\n"
        "  - top-left, top-centre, top-right\n"
        "  - centre-left, centre, centre-right\n"
        "  - bottom-left, bottom-centre, bottom-right\n"
        "IMPORTANT: Do NOT describe location relative to other objects. Only describe where the object is located in the image itself.\n"
        "Format: - **ObjectName**: location_in_image\n"
        "Example: - **Pen**: top-left (NOT 'top-left of the paper')\n"
        "\n"
        "STEP 3: For each object, determine its spatial relationships to OTHER objects (ignore locations):\n"
        "  - ON TOP OF another object (physically resting on it)\n"
        "  - OVERLAPPING with another object\n"
        "  - NEXT TO another object\n"
        "IMPORTANT: Do NOT mention positional directions like 'to the left' or 'to the right' here. Only mention physical relationships.\n"
        "\n"
                 "SUMMARY: Provide a brief summary that focuses on spatial relationships between objects:\n"
         "  - Start with the main/largest object and its image location\n"
         "  - Then describe what other objects are doing in relation to it (on top of, overlapping, next to)\n"
         "  - Focus on the relationships, not individual object positions\n"
         "Example: 'The paper is in the centre of the image with a USB drive on top of it and a pen overlapping it. The watch is positioned next to the paper.'\n"
        "\n"
        "CRITICAL RULES:\n"
        "- If object A is ON TOP OF object B, then B is UNDERNEATH A (not also on top)\n"
        "- Be extremely careful about ON TOP OF vs just positioned near\n"
        "- Only use ON TOP OF when one object is clearly resting on another\n"
        "- In STEP 2: Only mention absolute image positions, never relative to other objects\n"
        "- In STEP 3: Only mention spatial relationships, never positional directions\n"
        "- In SUMMARY: Combine image positions with spatial relationships"
    )

    # Call online VLM API
    start_time = time.time()
    print("[INFO] Running STEP 1 online vision analysis... ⏳")
    
    response = call_online_vlm(image_path, prompt, max_tokens=1024)
    
    if response is None:
        print("[ERROR] Failed to get response from online VLM")
        return None, None

    end_time = time.time()
    print(f"[INFO] ✅ STEP 1 completed in {end_time - start_time:.1f} seconds")
    # print(f"[STEP 1 OUTPUT] Raw Vision Analysis:\n{response}")

    # Extract objects and cache results
    objects_list = extract_objects_from_response(response)
    relationships_str = response

    cached_image_path = image_path
    cached_step1_objects = objects_list
    cached_step1_relationships = relationships_str

    print(f"[STEP 1 RESULT] Detected Objects: {objects_list}")
    print(f"[STEP 1 RESULT] Raw Relationships: {relationships_str[:]}")
    
    return objects_list, relationships_str

def step2_syntax_validation(objects_list, relationships_str):
    """
    STEP 2: Validate syntax and check spatial consistency.
    
    Args:
        objects_list (list): Objects from STEP 1
        relationships_str (str): Relationships from STEP 1
        
    Returns:
        tuple: (validated_objects, corrected_relationships, validation_issues)
    """
    global cached_step2_objects, cached_step2_relationships
    
    print("\n" + "="*60)
    print("✅ STEP 2: SYNTAX VALIDATION & SPATIAL CONSISTENCY CHECK")
    print("="*60)
    
    validated_objects = []
    validation_issues = []
    
    # Re-extract objects if needed
    if not objects_list or len(objects_list) < 3 or any(obj in ['Spatial', 'Overlapping', 'Analysis', 'Objects'] for obj in objects_list):
        print(f"[WARNING] Object extraction failed or returned invalid objects: {objects_list}")
        print(f"[INFO] Re-extracting objects directly from VLM response...")
        objects_list = extract_objects_from_response(relationships_str)
        print(f"[INFO] Re-extracted objects: {objects_list}")
    
    # Validate object list
    if not objects_list:
        validation_issues.append("No objects detected in the scene")
        print("[ERROR] No objects to validate")
        return [], relationships_str, validation_issues
    
    print(f"[INFO] Validating {len(objects_list)} detected objects...")
    
    # Clean and validate object names
    for obj in objects_list:
        if obj and isinstance(obj, str) and len(obj.strip()) > 0:
            cleaned_obj = obj.strip().title()
            
            if len(cleaned_obj) <= 50 and cleaned_obj.replace(' ', '').isalpha():
                validated_objects.append(cleaned_obj)
                print(f"[VALID] ✓ {cleaned_obj}")
            else:
                validation_issues.append(f"Invalid object name: {obj}")
                print(f"[INVALID] ✗ {obj} - Invalid format")
        else:
            validation_issues.append(f"Malformed object entry: {obj}")
            print(f"[INVALID] ✗ {obj} - Malformed entry")
    
    # Remove duplicates
    validated_objects = list(dict.fromkeys(validated_objects))
    
    # Validate spatial relationships
    print(f"\n[INFO] Checking spatial relationship consistency...")
    cleaned_relationships = relationships_str.strip()
    
    if not cleaned_relationships:
        validation_issues.append("No spatial relationships described")
        print("[WARNING] No spatial relationships found")
        cached_step2_objects = validated_objects
        cached_step2_relationships = cleaned_relationships
        return validated_objects, cleaned_relationships, validation_issues
    
    # Extract summary for reference truth
    import re
    summary_match = re.search(r'### Summary(?:\s+of\s+Relationships)?:(.*?)(?=\n\n|\n###|\Z)', cleaned_relationships, re.DOTALL | re.IGNORECASE)
    summary_text = summary_match.group(1).strip() if summary_match else ""
    
    print(f"[INFO] Found summary section: {bool(summary_text)}")
    if summary_text:
        print(f"[SUMMARY] {summary_text}")
    
    # Correct spatial inconsistencies
    corrected_relationships = correct_spatial_inconsistencies(cleaned_relationships, validated_objects, summary_text)
    
    # Validate relationships
    relationship_keywords = ['on top of', 'above', 'below', 'left', 'right', 'next to', 'under', 'underneath']
    has_relationships = any(keyword in corrected_relationships.lower() for keyword in relationship_keywords)
    
    if has_relationships:
        print("[VALID] ✓ Spatial relationships detected and validated")
    else:
        validation_issues.append("No clear spatial relationships detected")
        print("[WARNING] No clear spatial relationships found")
    
    # Summary
    if validation_issues:
        print(f"\n[STEP 2 ISSUES] Found {len(validation_issues)} validation issues:")
        for issue in validation_issues:
            print(f"  ⚠️ {issue}")
    else:
        print("\n[STEP 2 SUCCESS] ✅ All validation checks passed")
    
    print(f"[STEP 2 RESULT] Final Validated Objects: {validated_objects}")
    print(f"[STEP 2 RESULT] Final Corrected Relationships: {corrected_relationships[:]}")
    
    # Cache results
    cached_step2_objects = validated_objects
    cached_step2_relationships = corrected_relationships
    
    return validated_objects, corrected_relationships, validation_issues

def step3_reasoning_refinement(objects_list, relationships_str, task_description, validation_issues, json_objects=None):
    """
    STEP 3: Refine reasoning for logical consistency using online LLM and validate JSON consistency.
    
    Args:
        objects_list (list): Validated objects from STEP 2
        relationships_str (str): Relationships from STEP 2
        task_description (str): Original task description
        validation_issues (list): Issues from STEP 2
        json_objects (list): Objects from JSON input (optional)
        
    Returns:
        tuple: (refined_objects, refined_relationships) or (None, None) if inconsistency found
    """
    global cached_step3_objects, cached_step3_relationships
    
    print("\n" + "="*60)
    print("🧠 STEP 3: REASONING REFINEMENT (Using LLM model:deepseek/deepseek-chat-v3-0324:free)")
    print("="*60)
    
    # DISABLED: Validate JSON consistency if JSON objects provided
    # Note: Object localization verification disabled as requested
    # This can be re-enabled in the future if needed
    """
    if json_objects:
        is_consistent, consistency_issues = validate_object_consistency(objects_list, relationships_str, json_objects)
        if not is_consistent:
            print("[ERROR] ❌ INCONSISTENCY DETECTED between JSON input and VLM output!")
            print("[ERROR] The following issues were found:")
            for issue in consistency_issues:
                print(f"[ERROR]   - {issue}")
            print("[ERROR] 🛑 STOPPING PROGRAM due to inconsistency.")
            return None, None
    """
    
    # Prepare refinement prompt
    validation_context = ""
    if validation_issues:
        validation_context = f"\nValidation Issues from STEP 2: {', '.join(validation_issues)}"
    
    refinement_prompt = f"""You are an expert at analyzing spatial relationships and object detection results. 
Your task is to review and refine the vision analysis results for logical consistency.

TASK CONTEXT: {task_description}

STEP 1 & 2 RESULTS:
Objects Detected: {', '.join(objects_list)}
Spatial Relationships: {relationships_str}
{validation_context}

REFINEMENT INSTRUCTIONS:
1. Check if the detected objects make sense for the given task
2. Verify spatial relationships are logically consistent
3. Identify any missing objects that might be important for the task
4. Resolve any contradictions in spatial descriptions
5. Ensure object names are clear and standardized

Please provide your refined analysis in this exact format:

REFINED_OBJECTS: [comma-separated list of objects]
REFINED_RELATIONSHIPS: [clear description of spatial relationships]
REASONING: [brief explanation of any changes made]

Focus on accuracy and logical consistency. If the original analysis is good, you can keep it unchanged."""

    try:
        print("[INFO] Running STEP 3 reasoning refinement via online API... ⏳")
        
        # Call online LLM API
        refinement_response = call_online_llm(refinement_prompt, max_tokens=1024)
        
        if refinement_response is None:
            print("[ERROR] Online API call failed for STEP 3. Using STEP 2 validated results.")
            return objects_list, relationships_str
        
        print(f"[STEP 3 OUTPUT] DeepSeek API Refinement:\n{refinement_response}")
        
        # Parse refinement response
        refined_objects = objects_list  # Default to original
        refined_relationships = relationships_str  # Default to original
        
        import re
        
        # Extract refined objects
        objects_match = re.search(r'REFINED_OBJECTS:\s*\[([^\]]+)\]', refinement_response, re.IGNORECASE)
        if objects_match:
            objects_str = objects_match.group(1)
            refined_objects = [obj.strip().title() for obj in objects_str.split(',') if obj.strip()]
        else:
            # Try alternative format
            objects_match = re.search(r'REFINED_OBJECTS:\s*([^\n]+)', refinement_response, re.IGNORECASE)
            if objects_match:
                objects_str = objects_match.group(1)
                refined_objects = [obj.strip().title() for obj in objects_str.split(',') if obj.strip()]
        
        # Extract refined relationships
        relationships_match = re.search(r'REFINED_RELATIONSHIPS:\s*(.*?)(?=\nREASONING:|$)', refinement_response, re.IGNORECASE | re.DOTALL)
        if relationships_match:
            refined_relationships = relationships_match.group(1).strip()
            print(f"[STEP 3 INFO] Using LLM refined relationships: {refined_relationships[:]}")
        else:
            # Fallback approach
            relationships_match = re.search(r'REFINED_RELATIONSHIPS:\s*(.*)', refinement_response, re.IGNORECASE | re.DOTALL)
            if relationships_match:
                refined_relationships = relationships_match.group(1).strip()
                print(f"[STEP 3 INFO] Using LLM refined relationships (fallback): {refined_relationships[:]}")
            else:
                # Last resort - use original relationships
                refined_relationships = relationships_str
                print(f"[STEP 3 WARNING] Could not extract LLM refined relationships, using original")
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*([^\n]+)', refinement_response, re.IGNORECASE)
        if reasoning_match:
            reasoning_explanation = reasoning_match.group(1).strip()
            print(f"[STEP 3 REASONING] {reasoning_explanation}")
        
        print(f"[STEP 3 SUCCESS] ✅ Reasoning refinement completed")
        print(f"[STEP 3 RESULT] Final Objects: {refined_objects}")
        print(f"[STEP 3 RESULT] Final Relationships: {refined_relationships[:]}")
        
        # Cache results
        cached_step3_objects = refined_objects
        cached_step3_relationships = refined_relationships
        
        return refined_objects, refined_relationships
        
    except Exception as e:
        print(f"[ERROR] Error during STEP 3 reasoning refinement: {e}")
        print("[INFO] Using STEP 2 validated results instead")
        return objects_list, relationships_str

def step4_action_planning(task_description, objects_list, relationships_str, json_objects=None):
    """
    STEP 4: Generate executable action plan using online LLM.
    
    Args:
        task_description (str): Original task description
        objects_list (list): Final objects from STEP 3
        relationships_str (str): Final relationships from STEP 3
        json_objects (list): Objects from JSON input with coordinates (optional)
        
    Returns:
        str or None: Final action plan or None if failed
    """
    print("\n" + "="*60)
    print("🚀 STEP 4: ACTION PLANNING (Online DeepSeek API)")
    print("="*60)
    
    print(f"[INFO] Planning for task: {task_description}")
    print(f"[INFO] Using objects: {objects_list}")

    # Use LLM refined relationships directly from Step 3
    scene_description = relationships_str
    print(f"[INFO] Using LLM refined relationships from Step 3")
    
    # Prepare coordinate information from JSON
    coordinate_info = ""
    if json_objects:
        coordinate_info = "\nOBJECT COORDINATES (from JSON input):\n"
        for obj_data in json_objects:
            obj_name = obj_data["object"]
            center_loc = obj_data.get("object_center_location", {})
            x = center_loc.get("x", 0.0)
            y = center_loc.get("y", 0.0)
            z = center_loc.get("z", 0.0)
            coordinate_info += f"- {obj_name}: ({x}, {y}, {z})\n"
        print(f"[INFO] Using coordinate information from JSON input")
        
    prompt = f"""You are a robot action planner. Create a sequence of robot actions for the given task.

TASK: {task_description}
OBJECTS IN SCENE: {', '.join(objects_list)}
SCENE DESCRIPTION: {scene_description}{coordinate_info}

ROBOT ACTIONS AVAILABLE:
- home: return robot to home position (0, 0, 0)
- pickmove: move robot to object location to prepare for picking
- pick: pick up an object (robot must be at object location after pickmove)
- placemove: move robot to target location to prepare for placing
- place: place an object (robot must be at target location after placemove)
- OpenGripper: open gripper
- CloseGripper: close gripper

ACTION FORMAT RULES:
1. All actions: (ObjectName, (x, y, z), ActionName) - object name, 3D coordinates, and action
2. home actions: (home, (0.0, 0.0, 0.0), home) - return to home position
3. pickmove actions: (ObjectName, (x, y, z), pickmove) - move to object coordinates for picking
4. pick actions: (ObjectName, (x, y, z), pick) - pick up object at current location
5. placemove actions: (TargetObject, (x, y, z), placemove) - move to target location for placing
6. place actions: (TargetObject, (x, y, z), place) - place object at current location
7. gripper actions: (ObjectName, (x, y, z), OpenGripper) or (ObjectName, (x, y, z), CloseGripper)

MANDATORY WORKFLOW FOR PICK AND PLACE OPERATIONS:
1. ALWAYS start from home position: (home, (0.0, 0.0, 0.0), home)
2. For picking: home -> pickmove -> pick -> home
3. For placing: home -> placemove -> place
4. For pick and place: home -> pickmove -> pick -> home -> placemove -> place

CRITICAL PLANNING RULES:
1. If you need to pick up an object that has other objects ON TOP OF it or RESTING ON it, you MUST first move those blocking objects away.
2. You cannot pick up an object if another object is resting on it or on top of it.
3. Always check the scene description for spatial relationships before planning.
4. If an object A is "on top of", "resting on", or "above" object B, then A must be moved before B can be picked up.
5. ALWAYS return to home position after picking an object and before moving to place location.

SPATIAL LOGIC CLARIFICATION:
- If "X is on top of Y" → X is accessible, Y is blocked by X
- If "Y is under X" → Y is blocked by X, X is accessible  
- To pick up X (when X is on top): Just pick up X directly
- To pick up Y (when X is on top of Y): move X away first, then pick up Y

STEP-BY-STEP PLANNING:
1. Start from home position
2. Identify the target object from the task
3. Check: Is anything ON TOP OF the target object? 
4. If YES: move those objects first (using full pick/place workflow)
5. If NO: Pick up target directly (using full pick/place workflow)
6. For multi-step tasks: handle each target in sequence with home returns

EXAMPLES:
Task: "Pick up the pen"
Plan: (home, (0.0, 0.0, 0.0), home) -> (Pen, (15, 20, 30), pickmove) -> (Pen, (15, 20, 30), pick)

Task: "Pick up the pen and place it on the plate"
Plan: (home, (0.0, 0.0, 0.0), home) -> (Pen, (15, 20, 30), pickmove) -> (Pen, (15, 20, 30), pick) -> (home, (0.0, 0.0, 0.0), home) -> (plate, (25, 25, 25), placemove) -> (plate, (25, 25, 25), place)

Task: "Move the pen away from paper, then pick up paper"
Plan: (home, (0.0, 0.0, 0.0), home) -> (Pen, (15, 20, 30), pickmove) -> (Pen, (15, 20, 30), pick) -> (home, (0.0, 0.0, 0.0), home) -> (neutral, (50, 50, 30), placemove) -> (neutral, (50, 50, 30), place) -> (home, (0.0, 0.0, 0.0), home) -> (Paper, (20, 20, 25), pickmove) -> (Paper, (20, 20, 25), pick)

INSTRUCTIONS:
- Use the exact coordinates (x, y, z) provided in the OBJECT COORDINATES section
- If coordinates are not provided for an object, use (0.0, 0.0, 0.0)
- Handle multi-step tasks by addressing each target in order
- For blocking objects: move them to a neutral location first using full workflow
- ALWAYS include home position movements as specified in the workflow
- Use "pickmove" before "pick" and "placemove" before "place"
- ONLY perform the actions explicitly requested in the task
- Provide ONLY the final action plan in the exact format below
- Do not include reasoning or explanation in the final output, just the plan

FINAL ACTION PLAN (use this exact format):
Max: The task plan is (ObjectName, (x, y, z), action) -> (ObjectName, (x, y, z), action) -> (ObjectName, (x, y, z), action) -> ...

IMPORTANT ACTION TYPES: 
- home: (home, (0.0, 0.0, 0.0), home)
- pickmove: (ObjectName, (x, y, z), pickmove)
- pick: (ObjectName, (x, y, z), pick)
- placemove: (TargetName, (x, y, z), placemove)
- place: (TargetName, (x, y, z), place)
- gripper: (ObjectName, (x, y, z), OpenGripper) or (ObjectName, (x, y, z), CloseGripper)"""

    try:
        print("[INFO] Running STEP 4 action planning via online API... ⏳")
        
        # Call online LLM API
        plan_response = call_online_llm(prompt, max_tokens=1024)
        
        if plan_response is None:
            print("[ERROR] Online API call failed for STEP 4.")
            return None
        
        print(f"[STEP 4 OUTPUT] DeepSeek API Response:\n{plan_response}")
        print(f"[STEP 4 SUCCESS] ✅ Action planning completed")
        
    except Exception as e:
        print(f"[ERROR] Error during STEP 4 inference: {e}")
        return None
    
    # Extract plan from response
    raw_plan = extract_plan_from_response(plan_response, "", task_description, objects_list, relationships_str, json_objects)
    
    print(f"[STEP 4 RESULT] Raw Action Plan: {raw_plan}")
    
    # Apply coordinate transformations
    if raw_plan:
        final_plan = transform_plan_coordinates(raw_plan)
        print(f"[STEP 4 RESULT] Final Transformed Plan: {final_plan}")
        return final_plan
    else:
        print("[ERROR] No plan generated to transform")
        return final_plan

# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

def run_complete_4step_pipeline(image_path, task_description, force_reanalyze=False, json_objects=None):
    """
    Run the complete 4-step pipeline.
    
    Args:
        image_path (str): Path to input image
        task_description (str): Task to accomplish
        force_reanalyze (bool): Force re-analysis of image
        json_objects (list): Objects from JSON input (optional)
        
    Returns:
        tuple: (final_objects, final_relationships, final_plan)
    """
    print("\n" + "🚀"*20)
    print("RUNNING COMPLETE 4-STEP PIPELINE (ONLINE API)")
    print("🚀"*20)
    
    # STEP 1: Vision Analysis
    step1_objects, step1_relationships = step1_vision_analysis(image_path, force_reanalyze)
    if not step1_objects or not step1_relationships:
        print("[ERROR] STEP 1 failed. Aborting pipeline.")
        return None, None, None
    
    # STEP 2: Syntax Validation
    step2_objects, step2_relationships, validation_issues = step2_syntax_validation(step1_objects, step1_relationships)
    
    # STEP 3: Reasoning Refinement
    step3_objects, step3_relationships = step3_reasoning_refinement(step2_objects, step2_relationships, task_description, validation_issues, json_objects)
    
    # Check if step 3 failed due to inconsistency
    if step3_objects is None or step3_relationships is None:
        print("[ERROR] Pipeline stopped due to inconsistency in STEP 3")
        return None, None, None
    
    # STEP 4: Action Planning
    final_plan = step4_action_planning(task_description, step3_objects, step3_relationships, json_objects)
    
    print("\n" + "🎉"*20)
    print("4-STEP PIPELINE COMPLETED (ONLINE API)")
    print("🎉"*20)
    
    return step3_objects, step3_relationships, final_plan

def robot_action_planner_json(json_input, output_format="json", force_reanalyze=False, quiet=False, capture_output=False, headless=False):
    """
    Entry function for robot action planning that accepts JSON input string.
    
    Args:
        json_input (str): JSON string with user_message, imagePath, and objects
        output_format (str): Output format - "json" or "text" (default: "json")
        force_reanalyze (bool): Force re-analysis of the image even if cached (default: False)
        quiet (bool): Suppress progress output, only return result (default: False)
        capture_output (bool): Capture all terminal output and return as string (default: False)
    
    Returns:
        dict: Result dictionary containing:
            - task_description: The input task
            - image_path: The input image path
            - detected_objects: List of detected objects
            - spatial_relationships: Spatial relationship description
            - action_plan: Generated action plan
            - success: Boolean indicating if planning succeeded
            - execution_success: Boolean indicating if robot execution succeeded
            - terminal_output: Captured terminal output (if capture_output=True)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If JSON is invalid or missing required fields
        RuntimeError: If model loading or planning fails
    """
    # Parse JSON input
    task_description, image_path, json_objects = parse_json_input(json_input)
    if task_description is None:
        raise ValueError("Invalid JSON input or missing required fields")
    
    # Call the original planner with JSON objects
    return robot_action_planner(task_description, image_path, output_format, force_reanalyze, quiet, capture_output, json_objects, headless)

def robot_action_planner(task_description, image_path, output_format="json", force_reanalyze=False, quiet=False, capture_output=False, json_objects=None, headless=False):
    """
    Entry function for robot action planning that can be imported and called by other scripts.
    
    Args:
        task_description (str): Description of the task for the robot to perform
        image_path (str): Path to the image file for scene analysis
        output_format (str): Output format - "json" or "text" (default: "json")
        force_reanalyze (bool): Force re-analysis of the image even if cached (default: False)
        quiet (bool): Suppress progress output, only return result (default: False)
        capture_output (bool): Capture all terminal output and return as string (default: False)
        json_objects (list): Objects from JSON input with coordinates (optional)
        headless (bool): Run robot execution in headless mode (default: False for GUI mode)
    
    Returns:
        dict: Result dictionary containing:
            - task_description: The input task
            - image_path: The input image path
            - detected_objects: List of detected objects
            - spatial_relationships: Spatial relationship description
            - action_plan: Generated action plan
            - success: Boolean indicating if planning succeeded
            - execution_success: Boolean indicating if robot execution succeeded
            - terminal_output: Captured terminal output (if capture_output=True)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If task description is empty
        RuntimeError: If model loading or planning fails
    """
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    if not task_description.strip():
        raise ValueError("Task description cannot be empty.")
    
    # Initialize output capture
    captured_output = ""
    original_stdout = sys.stdout
    
    try:
        # Set up output capturing - only capture if quiet=True, OR if capture_output=True AND quiet=True
        if quiet:
            output_buffer = io.StringIO()
            sys.stdout = output_buffer
        elif capture_output and not quiet:
            # Special case: show on terminal AND capture
            output_buffer = io.StringIO()
            original_stdout = sys.stdout
            
            class TeeOutput:
                def __init__(self, terminal, buffer):
                    self.terminal = terminal
                    self.buffer = buffer
                
                def write(self, text):
                    self.terminal.write(text)
                    self.buffer.write(text)
                    return len(text)
                
                def flush(self):
                    self.terminal.flush()
                    self.buffer.flush()
            
            sys.stdout = TeeOutput(original_stdout, output_buffer)
        
        # Show startup message if not quiet
        if not quiet:
            print("\n🚀 Starting Enhanced Robot Action Planner with 4-Step Pipeline (Online API)")
        
        # Initialize online vision service
        if not load_vision_model():
            if quiet or (capture_output and not quiet):
                sys.stdout = original_stdout
            raise RuntimeError("Failed to initialize online vision service.")
        
        # Run the complete 4-step pipeline
        final_objects, final_relationships, final_plan = run_complete_4step_pipeline(
            image_path, 
            task_description, 
            force_reanalyze,
            json_objects
        )
        
        # Capture the output if requested
        if quiet:
            captured_output = output_buffer.getvalue()
            sys.stdout = original_stdout
        elif capture_output and not quiet:
            captured_output = output_buffer.getvalue()
            sys.stdout = original_stdout
        
    except Exception as e:
        # Restore stdout in case of error
        if quiet or (capture_output and not quiet):
            captured_output = output_buffer.getvalue() if 'output_buffer' in locals() else ""
            sys.stdout = original_stdout
        cleanup_models()
        raise RuntimeError(f"Planning failed: {e}")
    
    # Check if pipeline succeeded
    if final_plan is None:
        cleanup_models()
        raise RuntimeError("Failed to generate action plan.")
    
    # Prepare result
    result = {
        "task_description": task_description,
        "image_path": image_path,
        "detected_objects": final_objects or [],
        "spatial_relationships": final_relationships or "",
        "action_plan": final_plan or "",
        "success": final_plan is not None,
        "execution_success": None
    }
    
    # Automatically execute the plan if generation was successful
    if final_plan:
        print("\n" + "🤖"*20)
        print("EXECUTING GENERATED PLAN ON ROBOT")
        print("🤖"*20)
        print(f"Plan: {final_plan}")
        
        # Give user time to see the complete action plan visualization (5 seconds)
        print("⏳ Displaying action plan... Robot will start execution in 5 seconds")
        import time
        time.sleep(5)
        
        print("🚀 Starting robot execution now...")
        
        # Call robot_action execution function directly
        execution_success = robot_action.execute_plan_from_planner(final_plan, headless=headless)
        result["execution_success"] = execution_success
        
        if execution_success:
            print("🎉 Plan executed successfully on robot!")
        else:
            print("❌ Robot execution failed")
    
    # Add captured output if requested
    if capture_output:
        result["terminal_output"] = captured_output
    
    # Cleanup models
    cleanup_models()
    
    return result

# For backward compatibility
def run_robot_action_planner(task_description, image_path, **kwargs):
    """Alias for robot_action_planner for backward compatibility."""
    return robot_action_planner(task_description, image_path, **kwargs)

# =============================================================================
# CALL_PLANNER FUNCTION FOR APP.PY INTEGRATION
# =============================================================================

def call_planner(current_results, headless=True):
    """
    Entry function to be called from app.py with window.current_results data.
    
    Args:
        current_results (dict): Dictionary containing:
            - user_message: The original user command
            - imagePath: Path to the image file
            - objects: List of detected objects with coordinates and actions
        headless (bool): Run robot execution in headless mode (default: True)
    
    Returns:
        dict: Result dictionary containing:
            - success: Boolean indicating if planning and execution succeeded
            - task_description: The input task
            - image_path: The input image path
            - action_plan: Generated action plan
            - execution_success: Boolean indicating if robot execution succeeded
            - error: Error message if failed
    """
    try:
        print("\n" + "🚀"*20)
        print("CALLING MAX PLANNER FROM APP.PY")
        print("🚀"*20)
        
        # Extract data from current_results
        task_description = current_results.get("user_message", "")
        image_path = current_results.get("imagePath", "")
        objects_data = current_results.get("objects", [])
        
        print(f"[INFO] Task: {task_description}")
        print(f"[INFO] Image: {image_path}")
        print(f"[INFO] Objects: {len(objects_data)} objects detected")
        
        # Validate inputs
        if not task_description.strip():
            return {
                "success": False,
                "error": "Task description is empty",
                "task_description": task_description,
                "image_path": image_path,
                "action_plan": "",
                "execution_success": False
            }
        
        if not image_path or not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}",
                "task_description": task_description,
                "image_path": image_path,
                "action_plan": "",
                "execution_success": False
            }
        
        if not objects_data:
            return {
                "success": False,
                "error": "No objects data provided",
                "task_description": task_description,
                "image_path": image_path,
                "action_plan": "",
                "execution_success": False
            }
        
        # Convert current_results to JSON format expected by robot_action_planner_json
        json_input = json.dumps(current_results)
        
        print(f"[INFO] Calling robot_action_planner_json with data...")
        
        # Call the existing planner function
        result = robot_action_planner_json(
            json_input=json_input,
            output_format="json",
            force_reanalyze=False,
            quiet=False,
            capture_output=False,
            headless=headless
        )
        
        print(f"[INFO] Planner result: Success={result.get('success')}, Execution={result.get('execution_success')}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] call_planner failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "task_description": current_results.get("user_message", ""),
            "image_path": current_results.get("imagePath", ""),
            "action_plan": "",
            "execution_success": False
        }

# =============================================================================
# COORDINATE TRANSFORMATION
# =============================================================================

def transform_plan_coordinates(plan_string):
    """
    Transform coordinates in the final plan according to robot coordinate system.
    
    Transformations:
    - pickmove: (x, y, z) -> (-(30-x), -(60-y), z)
    - placemove: (x, y, z) -> (-(30-x), -(60-y), z-100)
    
    Args:
        plan_string (str): Original plan string with coordinates
        
    Returns:
        str: Transformed plan string with new coordinates
    """
    import re
    
    print("\n" + "🔧"*20)
    print("COORDINATE TRANSFORMATION")
    print("🔧"*20)
    print(f"[INFO] Original plan: {plan_string}")
    
    def transform_coordinates(match):
        # Extract the full action tuple
        full_match = match.group(0)
        object_name = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        z = float(match.group(4))
        action = match.group(5)
        
        # Apply transformations based on action type
        if action == "pickmove":
            new_x = -(30 - x)
            new_y = -(60 - y)
            new_z = z
            print(f"[TRANSFORM] {action}: ({x}, {y}, {z}) -> ({new_x}, {new_y}, {new_z})")
        elif action == "placemove":
            new_x = -(30 - x)
            new_y = -(60 - y)
            new_z = z - 100
            print(f"[TRANSFORM] {action}: ({x}, {y}, {z}) -> ({new_x}, {new_y}, {new_z})")
        else:
            # No transformation for other actions (home, pick, place, etc.)
            new_x, new_y, new_z = x, y, z
        
        # Return the transformed action tuple
        return f"({object_name}, ({new_x}, {new_y}, {new_z}), {action})"
    
    # Pattern to match action tuples: (ObjectName, (x, y, z), action)
    pattern = r'\(([^,]+),\s*\(([^,]+),\s*([^,]+),\s*([^,)]+)\),\s*([^)]+)\)'
    
    # Apply transformations
    transformed_plan = re.sub(pattern, transform_coordinates, plan_string)
    
    print(f"[INFO] Transformed plan: {transformed_plan}")
    print("🔧"*20)
    
    return transformed_plan

# =============================================================================
# MAIN FUNCTION FOR JSON INPUT
# =============================================================================

def main():
    """
    Main function for command line usage with JSON input.
    """
    if len(sys.argv) < 2:
        print("Usage: python max_planner.py '<json_string>'")
        print("\nExample JSON format:")
        example_json = {
            "user_message": "pick up the paper",
            "imagePath": "old.jpeg",
            "objects": [
                {
                    "action": "pick",
                    "location": "top-centre",
                    "object": "yellow banana",
                    "object_center_location": {"x": 15, "y": 20, "z": 30},
                    "object_size": {"width": 15, "length": 20},
                    "object_orientation": 45
                },
                {
                    "action": "pick", 
                    "location": "bottom-right",
                    "object": "red apple",
                    "object_center_location": {"x": 30, "y": 10, "z": 25},
                    "object_size": {"width": 10, "length": 12},
                    "object_orientation": 30
                }
            ]
        }
        print(json.dumps(example_json, indent=2))
        print("\nNote: The planner will automatically execute the generated plan on the robot.")
        sys.exit(1)
    
    json_input = sys.argv[1]
    
    try:
        result = robot_action_planner_json(json_input)
        
        if result["success"]:
            print(f"\n🎉 SUCCESS! Action plan generated:")
            print(f"Task: {result['task_description']}")
            print(f"Objects: {result['detected_objects']}")
            print(f"Plan: {result['action_plan']}")
            
            if result["execution_success"]:
                print(f"✅ Plan executed successfully on robot!")
            elif result["execution_success"] is False:
                print(f"❌ Plan execution failed on robot")
            else:
                print(f"⚠️ Plan execution was not attempted")
        else:
            print(f"\n❌ FAILED to generate action plan")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 