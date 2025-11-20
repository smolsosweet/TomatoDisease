# predict.py
import os
import json
import torch
import torchvision
import xgboost as xgb
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

class TomatoModel:
    def __init__(self, model_dir="model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        # Load class mapping
        with open(os.path.join(model_dir, "class_mapping.json")) as f:
            mapping = json.load(f)
        self.classes = mapping["classes"]
        self.idx_to_class = mapping["idx_to_class"]

        # Load EfficientNet
        self.effnet = torchvision.models.efficientnet_b0()
        self.effnet.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(1280, len(self.classes))
        )
        eff_path = os.path.join(model_dir, "efficientnet.pt")
        state_dict = torch.load(eff_path, map_location=self.device)

        # Xử lý nếu file cũ có tiền tố "model."
        if any(k.startswith("model.") for k in state_dict.keys()):
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith("model.") and not k.startswith("model.loss_fn"):
                    new_sd[k[6:]] = v
                elif not k.startswith("loss_fn"):
                    new_sd[k] = v
            state_dict = new_sd

        self.effnet.load_state_dict(state_dict)
        self.effnet.to(self.device)
        self.effnet.eval()

        # Load XGBoost
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(os.path.join(model_dir, "xgboost_model.json"))

        # Load best weight
        with open(os.path.join(model_dir, "config.json")) as f:
            self.best_weight = json.load(f)["best_weight"]

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_features(self, img_tensor):
        with torch.no_grad():
            x = self.effnet.features(img_tensor)
            x = self.effnet.avgpool(x)
            x = torch.flatten(x, 1)
            return x.cpu().numpy()

    def predict(self, image):
        """image: PIL Image hoặc đường dẫn"""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # EfficientNet
        with torch.no_grad():
            logits = self.effnet(img_tensor)
            eff_prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # XGBoost
        feats = self._extract_features(img_tensor)
        xgb_prob = self.xgb.predict_proba(feats)[0]

        # Ensemble
        ensemble_prob = self.best_weight * eff_prob + (1 - self.best_weight) * xgb_prob
        pred_idx = int(np.argmax(ensemble_prob))
        confidence = float(ensemble_prob[pred_idx])
        pred_class = self.idx_to_class[str(pred_idx)]

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": {self.idx_to_class[str(i)]: float(p) for i, p in enumerate(ensemble_prob)}
        }
    # DÁN HÀM MỚI TẠI ĐÂY
    def predict_from_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return self.predict(pil_image)

# Hàm tiện dùng
def load_model(model_dir="model"):
    return TomatoModel(model_dir)