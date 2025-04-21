import torch
import random
import numpy as np
from env.environment_sim import Environment
from feat_extract import VisionLanguageProcessor
import os
import scripts.utils 
from scripts.grasp_detetor import Graspnet
from PIL import Image
import argparse
from groq import Groq
import yaml

# Set seed for reproducibility
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
counter = 0  # Initialize counter for saving poses with unique file names

torch.set_grad_enabled(False)

def setup(ROOT_PATH,use_som=False):
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit()

    print("CUDA is available.")

    config = {
        "ROOT_PATH": ROOT_PATH,
        "GROUNDING_DINO_CONFIG_PATH": os.path.join(ROOT_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        "GROUNDING_DINO_CHECKPOINT_PATH": os.path.join(ROOT_PATH, "groundingdino_swint_ogc.pth"),
        "SAM_ENCODER_VERSION": "vit_h",
        "SAM_CHECKPOINT_PATH": os.path.join(ROOT_PATH, "sam_vit_h_4b8939.pth"),
        "RAM_CHECKPOINT_PATH": os.path.join(ROOT_PATH, "ram_swin_large_14m.pth"),
    }

    # Initialize VisionLanguageProcessor
    processor = VisionLanguageProcessor(config,use_som)
    return processor

def get_text_feats(in_text, processor):
    text_tokens = processor.clip_tokenizer([in_text]).to(processor.device)
    with torch.no_grad():
        text_feats = processor.clip_model.encode_text(text_tokens).float()
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
    return text_feats

def get_image_feats(image_path, processor):
    image = Image.open(image_path).convert("RGB")
    image = processor.clip_preprocess(image).unsqueeze(0).to(processor.device)
    
    with torch.no_grad():
        image_feats = processor.clip_model.encode_image(image).float()
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
    
    return image_feats

def get_scene(env, testing_case_file):
    reset = False
    while not reset:
        env.reset()
        reset, _ = env.add_object_push_from_file(testing_case_file)
        print(f"Environment reset : {reset}")
    color_image, depth_image, mask_image = scripts.utils.get_true_heightmap(env)
    pcd = scripts.utils.get_fuse_pointcloud(env)
    return color_image, depth_image, pcd

def get_feat(color_image, processor):
    return processor.process_image(color_image)

def query_vision_llm(client, image_base64, user_query):
    # Create the system message
    system_message = (
        "You are an advanced vision assistant for a robotic grasping system. You are shown an image with objects labeled with numbers. "
        "Your task is to analyze the user's query and determine which labeled object best matches their request. "
        "First, carefully reason through which object best matches the query. Consider: "
        "- Spatial relationships (above, below, left, right, corner, center, next to) "
        "- Object properties (color, shape, size, material) "
        "- Object categories (fruit, tool, container, etc.) "
        "- Potential functions of objects "
        "After your analysis, ALWAYS end your response with a line that says 'LABEL: X' where X is the number of the object that best answers the query. "
        "This final line must be in exactly this format for the robotic system to process it correctly."
    )

    try:
        # Make the API call using Groq's format
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_completion_tokens=2048,  # Keep this low as we only need the number
            temperature=0.5
        )

        # Extract the label from the response
        label_text = response.choices[0].message.content.strip()
        # print(f"Raw model response: {label_text}")
        return label_text

    except Exception as e:
        print(f"Error querying vision LLM: {e}")
        return None

def load_config(config_path="config.yaml"):
    """Load API keys from config file"""
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return None
            
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def run_inference_with_vlm(query, color_image, depth_image, pcd, processor, feat, env, client, image_base64, save_poses=False):
    global counter
    
    # Query the vision-language model to identify which label to target
    print(f"Querying VLM with: '{query}'")
    label_text = query_vision_llm(client, image_base64, query)    

    import re
    label = None
    label_match = re.search(r'LABEL:\s*(\d+)', label_text)
    if label_match:
        label = label_match.group(1)
  
    if not label:
        print("No valid label identified from VLM response.")
        return
    
    print(f"VLM identified object with label: {label}")
    
    
    # Check if the label exists in our mask_labels
    if label not in feat["mask_labels"]:
        print(f"Label {label} not found in detected objects.")
        return
    
    mask_labels = feat["mask_labels"]
    if isinstance(mask_labels, np.ndarray):
        mask_labels = mask_labels.tolist()
    # Get the index of the identified label
    label_idx = mask_labels.index(label)
    
    # Get the bounding box for the identified object
    best_bbox = feat["xyxy"][label_idx]
    x1, y1, x2, y2 = map(int, best_bbox)
    
    print(f"Targeting object with bounding box: [{x1}, {y1}, {x2}, {y2}]")
    
    # Crop the point cloud to the object's bounding box
    cropped_pcd = scripts.utils.crop_pointcloud(pcd, (x1, y1, x2, y2), color_image, depth_image)
    
    # Compute grasping poses
    graspnet = Graspnet()
    with torch.no_grad():
        sorted_grasp_pose_set = graspnet.grasp_detection(cropped_pcd, env.get_true_object_poses())
        print(f"Number of grasping poses for object {label}: {len(sorted_grasp_pose_set)}")
    
    counter += 1  # Increment counter each time a query is processed
    
    if save_poses:
        safe_query = query.replace(" ", "_")
        file_name = f"poses_vlm_{safe_query}_{label}_{counter}.npy"
        np.save(file_name, np.array(sorted_grasp_pose_set))
        print(f"Saved poses to {file_name}")
    
    if len(sorted_grasp_pose_set) != 0:
        action = sorted_grasp_pose_set[0]
        reward, done = env.step(action)
        print(f"Action executed for object {label}")
        return True
    else:
        print(f"No valid grasping poses found for object {label}")
        return False

def run_inference(query_type, query_input, color_image, depth_image, pcd, processor, feat, env,save_poses=False):
    global counter
    if query_type == "text":
        query_embedding = get_text_feats(query_input, processor)
    elif query_type == "image":
        query_embedding = get_image_feats(query_input, processor)
    objects = []
    for idx, img_feat in enumerate(feat["image_feats"]):
        object_embedding = torch.tensor(img_feat).to(processor.device)
        similarity = torch.cosine_similarity(query_embedding, object_embedding, dim=-1).item()
        objects.append({"idx": idx, "similarity": similarity})
    objects.sort(key=lambda x: x["similarity"], reverse=True)
    best_match_idx = objects[0]["idx"]
    best_bbox = feat["xyxy"][best_match_idx]
    x1, y1, x2, y2 = map(int, best_bbox)
    cropped_pcd = scripts.utils.crop_pointcloud(pcd, (x1, y1, x2, y2), color_image, depth_image)
    graspnet = Graspnet()
    with torch.no_grad():
        sorted_grasp_pose_set = graspnet.grasp_detection(cropped_pcd, env.get_true_object_poses())
        print("Number of grasping poses:", len(sorted_grasp_pose_set))    
    
    counter += 1  # Increment counter each time a query is processed
    if save_poses:
        def get_unique_filename(query_type, query_input):      
            if query_type == "image":
                safe_query = os.path.basename(query_input).split('.')[0]
            else:
                safe_query = query_input.replace(" ", "_")
            
            return f"poses_{query_type}_{safe_query}_{counter}.npy"

        file_name = get_unique_filename(query_type, query_input)
        np.save(file_name, np.array(sorted_grasp_pose_set))
    
    if len(sorted_grasp_pose_set) != 0:
        action = sorted_grasp_pose_set[0]
        reward, done = env.step(action)
        print("Action Executed")
    else:
        print("No valid grasping poses found")
        

def input_section(use_som):
    """Get input from user based on mode"""
    while True:
        query_type = input("Enter query type ('text' or 'image', or 'exit' to quit): ").strip().lower()
        if query_type in ["text", "image", "exit"]:
            if query_type == "exit":
                return "exit", None
            if use_som and query_type == "image":
                print("Only text query is supported for SoM")
                return "exit", None
            query_input = input(f"Enter your {query_type} query: ").strip()
            return query_type, query_input
        else:
            print("Invalid input! Please enter 'text', 'image', or 'exit'.")



def main(testing_case_file, gui, use_som=False, config_path="config.yaml"):
    # path to Grounded-Segment-Anything folder
    ROOT_PATH = "Grounded-Segment-Anything"
    processor = setup(ROOT_PATH,use_som)
    env = Environment(gui=gui)
    env.seed(1234)
    color_image, depth_image, pcd = get_scene(env, testing_case_file)
    feat, image_base64 = get_feat(color_image, processor)
    
    # Initialize Groq client if using SoM
    client = None
    if use_som:
        # Load API key from config
        config = load_config(config_path)        
        if not config:
            print("Error: Could not load configuration file.")
            return            
        api_key = config.get("api_keys", {}).get("groq")        
        if not api_key:
            print("Error: No Groq API key found in config file.")
            print(f"Please add your API key to {config_path} under api_keys.groq")
            return            
        client = Groq(api_key=api_key)
    
    while True:
        query_type, query_input = input_section(use_som)        
        if query_type == "exit":
            break            
        if use_som:
            # Use VLM-based object selection
            run_inference_with_vlm(query_input, color_image, depth_image, pcd, processor, feat, env, client, image_base64, save_poses=True)
        else:
            # Use embedding-based object selection
            run_inference(query_type, query_input, color_image, depth_image, pcd, processor, feat, env, save_poses=False)
        
        reset = False
        while not reset:
            env.reset()
            reset, _ = env.add_object_push_from_file(testing_case_file)
            print(f"Environment reset: {reset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_file", type=str, required=True, help="Path to the testing file")
    parser.add_argument("--gui", type=lambda x: (str(x).lower() == "true"), default=True, help="Enable GUI (default: True)")
    parser.add_argument("--som", action="store_true", help="Use Set of Mark Prompting mode with VLM")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.testing_file, args.gui, args.som, args.config)
