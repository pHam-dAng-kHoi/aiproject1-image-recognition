import cv2
import os
import torch
import numpy as np
from model import build_model

# 1. SETUP CONFIGURATION
MODEL_NAME = "fasterrcnn_resnet50_fpn" 
MODEL_WEIGHTS = 'sessions/best_model.pth' 
IMAGE_DIR = 'testimages/'
OUTPUT_DIR = 'testresults/'  # New folder for labeled images

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CLASSES = ["Banana", "Apple", "Orange", "Lemon", "Lime"]

def load_trained_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(MODEL_NAME, num_classes=6)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def run_test():
    model, device = load_trained_model(MODEL_WEIGHTS)
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {len(images)} images...")

    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        display_img = original_img.copy()
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        output = outputs[0]
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] > 0.5: 
                box = boxes[i].astype(int)
                class_id = int(labels[i])
                fruit_idx = class_id - 1
                
                if 0 <= fruit_idx < len(CLASSES):
                    fruit_name = CLASSES[fruit_idx]
                    label_text = f"{fruit_name}: {scores[i]:.2f}"
                    
                    # 1. Define styling
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2  # Much larger
                    thickness = 3     # Thicker lines
                    color = (0, 255, 0) # Bright Green
                    
                    # 2. Draw the bounding box
                    cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), color, 4)

                    # 3. Add a background label box for readability
                    (label_width, label_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    cv2.rectangle(display_img, 
                                  (box[0], box[1] - label_height - 10), 
                                  (box[0] + label_width, box[1]), 
                                  color, 
                                  -1) # Filled rectangle

                    # 4. Draw the text in black over the green background
                    cv2.putText(display_img, label_text, (box[0], box[1] - 10),
                                font, font_scale, (0, 0, 0), thickness)

        # SAVE THE IMAGE INSTEAD OF SHOWING IT
        output_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
        cv2.imwrite(output_path, display_img)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    run_test()