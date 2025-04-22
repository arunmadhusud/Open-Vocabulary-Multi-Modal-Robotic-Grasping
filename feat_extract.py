import os
import torch
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import open_clip
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS
import cv2
from PIL import Image
import torchvision
import numpy as np
import supervision as sv
from supervision.draw.color import Color, ColorPalette
import dataclasses  
from typing import List
import pickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import io
import base64


class VisionLanguageProcessor:
    def __init__(self, config, use_som = False):
        self.device = torch.device("cuda")
        self.config = config
        self.use_som = use_som

        torch.set_grad_enabled(False)
        print("Initializing models...")

        # Grounding DINO
        self.grounding_dino_model = self.load_grounding_dino()

        # SAM
        self.sam, self.sam_predictor = self.load_sam()

        # CLIP
        if not self.use_som:
            self.clip_model, self.clip_preprocess, self.clip_tokenizer = self.load_clip()

        # RAM (Tag2Text)
        self.tagging_model, self.tagging_transform = self.load_ram()

        print("All models initialized successfully.")

    def load_grounding_dino(self):
        print("Loading Grounding DINO...")
        return Model(
            model_config_path=self.config["GROUNDING_DINO_CONFIG_PATH"],
            model_checkpoint_path=self.config["GROUNDING_DINO_CHECKPOINT_PATH"],
            device=self.device
        )

    def load_sam(self):
        print("Loading SAM...")
        sam = sam_model_registry[self.config["SAM_ENCODER_VERSION"]](checkpoint=self.config["SAM_CHECKPOINT_PATH"])
        sam.to(self.device)
        return sam, SamPredictor(sam)

    def load_clip(self):
        print("Loading CLIP...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to(self.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        return clip_model, clip_preprocess, clip_tokenizer

    def load_ram(self):
        print("Loading RAM (Tag2Text)...")
        tagging_model = ram(pretrained=self.config["RAM_CHECKPOINT_PATH"], image_size=384, vit='swin_l')
        tagging_model = tagging_model.eval().to(self.device)

        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return tagging_model, tagging_transform

    def process_tag_classes(self,text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
        '''
        Convert a text prompt from Tag2Text to a list of classes. 
        '''
        classes = text_prompt.split(',')
        classes = [obj_class.strip() for obj_class in classes]
        classes = [obj_class for obj_class in classes if obj_class != '']
        
        for c in add_classes:
            if c not in classes:
                classes.append(c)
        
        for c in remove_classes:
            classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
        
        return classes

    def get_sam_segmentation_from_xyxy(self,sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def compute_clip_features(self,image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
        
        image = Image.fromarray(image)
        
        # padding = args.clip_padding  # Adjust the padding amount as needed
        padding = 20  # Adjust the padding amount as needed
        
        image_crops = []
        image_feats = []
        text_feats = []

        
        for idx in range(len(detections.xyxy)):
            # Get the crop of the mask with padding
            x_min, y_min, x_max, y_max = detections.xyxy[idx]

            # Check and adjust padding to avoid going beyond the image borders
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply the adjusted padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            # Get the preprocessed image for clip from the crop 
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

            crop_feat = clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
            
            class_id = detections.class_id[idx]
            tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            
            crop_feat = crop_feat.cpu().numpy()
            text_feat = text_feat.cpu().numpy()

            image_crops.append(cropped_image)
            image_feats.append(crop_feat)
            text_feats.append(text_feat)
            
        # turn the list of feats into np matrices
        image_feats = np.concatenate(image_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)

        return image_crops, image_feats, text_feats

    def vis_result_fast(
        self,
        image: np.ndarray, 
        detections: sv.Detections, 
        classes: list[str], 
        color: Color | ColorPalette = ColorPalette.DEFAULT, 
        instance_random_color: bool = False,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        '''
        Annotate the image with the detection results. 
        This is fast but of the same resolution of the input image, thus can be blurry. 
        '''
        # annotate image with detections
        box_annotator = sv.BoxAnnotator(
            color = color,
        )
        mask_annotator = sv.MaskAnnotator(
            color = color
        )

        if hasattr(detections, 'confidence') and hasattr(detections, 'class_id'):
            confidences = detections.confidence
            class_ids = detections.class_id
            if confidences is not None:
                labels = [
                    f"{classes[class_id]} {confidence:0.2f}"
                    for confidence, class_id in zip(confidences, class_ids)
                ]
            else:
                labels = [f"{classes[class_id]}" for class_id in class_ids]
        else:
            print("Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing.")

        
        if instance_random_color:
            # generate random colors for each segmentation
            # First create a shallow copy of the input detections
            detections = dataclasses.replace(detections)
            detections.class_id = np.arange(len(detections))
            
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        
        if draw_bbox:
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image, labels

    def apply_nms(self, detections, nms_threshold=0.5, min_area=500):
        """Apply Non-Maximum Suppression to remove redundant boxes and small areas."""
        print("Applying NMS...")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # Filter out class_id == -1
        valid_idx = detections.class_id != -1
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]

        # Filter out boxes with small area
        widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        areas = widths * heights
        valid_area_idx = areas >= min_area

        detections.xyxy = detections.xyxy[valid_area_idx]
        detections.confidence = detections.confidence[valid_area_idx]
        detections.class_id = detections.class_id[valid_area_idx]

        return detections


    def draw_number_in_mask(self, binary_mask, text, color="g", label_mode='1'):
        """
        Find proper places to draw text given a binary mask.
        """
        def number_to_string(n):
            chars = []
            while n:
                n, remainder = divmod(n-1, 26)
                chars.append(chr(97 + remainder))
            return ''.join(reversed(chars))
            
        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]
        if label_mode == 'a':
            text = number_to_string(int(text))
        else:
            text = str(text)
        
        return (coords_x[len(coords_x)//2], coords_y[len(coords_y)//2]), text, color

    def process_image(self, image_rgb):
        print(f"Processing image")        
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for Grounding DINO
        image_pil = Image.fromarray(image_rgb)

        # RAM inference
        raw_image = image_pil.resize((384, 384))
        raw_image = self.tagging_transform(raw_image).unsqueeze(0).to(self.device)
        res = inference_ram(raw_image, self.tagging_model)
        text_prompt = res[0].replace(' |', ',')
        classes = self.process_tag_classes(text_prompt, add_classes=["other item"])

        # Object Detection with Grounding DINO
        detections = self.grounding_dino_model.predict_with_classes(
            image=image, classes=classes, box_threshold=0.2, text_threshold=0.2
        )

        if len(detections.class_id) > 0:
            detections = self.apply_nms(detections)

            # Segmentation with SAM
            detections.mask = self.get_sam_segmentation_from_xyxy(
                sam_predictor=self.sam_predictor, image=image_rgb, xyxy=detections.xyxy
            )

            # CLIP feature extraction - only if not using SoM
            if not self.use_som:
                image_crops, image_feats, text_feats = self.compute_clip_features(
                    image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, classes, self.device
                )
            else:
                image_crops, image_feats, text_feats = [], [], []
        else:
            image_crops, image_feats, text_feats = [], [], []

        # Save basic visualization
        annotated_image, labels = self.vis_result_fast(image, detections, classes)
        
        # Only generate mask labels and base64 image if SoM is enabled
        mask_labels = []
        image_base64 = None
        
        if self.use_som:
            # Create a better visualization with matplotlib
            fig, ax = plt.subplots(figsize=(image_rgb.shape[1]/100, image_rgb.shape[0]/100), dpi=100)
            ax.imshow(image_rgb)
            
            # Draw masks and add text labels inside each mask
            for i, mask in enumerate(detections.mask):
                # Get a color for this instance
                color = f"C{i % 10}"  # Cycle through matplotlib color cycle
                
                # Convert color to RGB
                color_rgb = np.array(mplc.to_rgb(color))
                
                # Create a colored mask
                mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
                mask_rgba[:, :, :3] = color_rgb
                mask_rgba[:, :, 3] = mask * 0.4  # 0.4 alpha for the mask
                
               
                # Get position and text for this mask
                position, text, _ = self.draw_number_in_mask(mask, i+1, color=color)

                # Store the label for this mask
                mask_labels.append(text)
                
                # Make the text color more visible
                color_for_text = np.maximum(color_rgb, 0.15)
                color_for_text[np.argmax(color_for_text)] = max(0.8, np.max(color_for_text))
                
                # Determine whether to use black or white text background based on the text color
                def contrasting_color(rgb):
                    R, G, B = rgb
                    Y = 0.299 * R + 0.587 * G + 0.114 * B
                    return 'black' if Y > 0.5 else 'white'
                    
                bbox_color = contrasting_color(color_for_text)
                
                # Draw the text
                ax.text(
                    position[0],
                    position[1],
                    text,
                    size=max(np.sqrt(fig.get_figheight() * fig.get_figwidth()) * 2, 10),
                    family="sans-serif",
                    bbox={"facecolor": bbox_color, "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                    verticalalignment="center",
                    horizontalalignment="center",
                    color=color_for_text,
                    zorder=10,
                )
            
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            mask_labels = np.array(mask_labels)
            
            # Save the visualization
            vis_save_path = os.path.join(self.config["ROOT_PATH"], "vis.png")
            plt.savefig(vis_save_path, bbox_inches="tight", pad_inches=0)

            # Create base64 encoding of the image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            print("Saved visualization at:", vis_save_path)
        
        # Create results dictionary
        results = {
            "xyxy": detections.xyxy, 
            "confidence": detections.confidence, 
            "class_id": detections.class_id,
            "mask": detections.mask, 
            "classes": classes, 
            "image_crops": image_crops,
            "image_feats": image_feats, 
            "text_feats": text_feats, 
            "tagging_text_prompt": text_prompt
        }
        
        # Add mask_labels only if SoM is enabled
        if self.use_som and len(mask_labels) > 0:
            results["mask_labels"] = mask_labels
        
        # Save results
        detect_save_path = os.path.join(self.config["ROOT_PATH"], "detect.pkl.gz")
        with gzip.open(detect_save_path, "wb") as f:
            pickle.dump(results, f)
        print("Saved results at:", detect_save_path)

        return results, image_base64


if __name__ == "__main__":
    ROOT_PATH = "/content/Open-Vocabulary-Multi-Modal-Robotic-Grasping/Grounded-Segment-Anything"

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

    processor = VisionLanguageProcessor(config,use_som=False)
    image_path = "/content/color_image_new_2.png"
    image = cv2.imread(image_path)  # BGR format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feat,_= processor.process_image(image_rgb)
