import torch
import random
import numpy as np
from env.environment_sim import Environment
from feat_extract import VisionLanguageProcessor
import os
import scripts.utils 
from scripts.grasp_detetor import Graspnet

# Set seed for reproducibility
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

torch.set_grad_enabled(False)

def setup(ROOT_PATH):
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
    processor = VisionLanguageProcessor(config)
    return processor

def get_text_feats(in_text, processor):
    text_tokens = processor.clip_tokenizer([in_text]).to(processor.device)
    with torch.no_grad():
        text_feats = processor.clip_model.encode_text(text_tokens).float()
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
    return text_feats

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

def run_inference(query_text, color_image, depth_image, pcd, processor, feat, env):
    query_embedding = get_text_feats(query_text, processor)
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
    file_name = f"poses_{query_text}.npy"
    np.save(file_name, np.array(sorted_grasp_pose_set))

    action = sorted_grasp_pose_set[0]
    reward, done = env.step(action)
    print("Action Executed")

def main():
    # path to testing file
    testing_case_file = "/content/Open-Vocabulary-Multi-Modal-Robotic-Grasping/testing_cases/new_set_224_2.txt"
    # path to Grounded-Segment-Anything folder
    ROOT_PATH = "/content/Open-Vocabulary-Multi-Modal-Robotic-Grasping/Grounded-Segment-Anything"
    processor = setup(ROOT_PATH)
    env = Environment(gui=False)
    env.seed(1234)
    color_image, depth_image, pcd = get_scene(env, testing_case_file)
    feat = get_feat(color_image, processor)
    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ")
        if query_text.lower() == 'exit':
            break
        run_inference(query_text, color_image, depth_image, pcd, processor, feat, env)
        reset = False
        while not reset:
            env.reset()
            reset, _ = env.add_object_push_from_file(testing_case_file)
            print(f"Environment reset : {reset}")

if __name__ == "__main__":
    main()
