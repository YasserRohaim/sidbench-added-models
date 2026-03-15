
from networks.clip import clip 

import torch.nn as nn 
import torch


class UnivFD(nn.Module):
    def __init__(self, num_classes=1):
        super(UnivFD, self).__init__()
        self.model, _ = clip.load("ViT-L/14", device="cpu")
        self.fc = nn.Linear(768, num_classes)
        self.gradient_flow_mode = False

    def forward(self, x):
        features = self.model.encode_image(x) 
        return self.fc(features)
    
    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.fc.load_state_dict(state_dict)

    def set_gradient_flow(self, enabled=True):
        self.gradient_flow_mode = enabled

    def score(self, img, apply_sigmoid=False):
        logits = self.forward(img).flatten()
        if apply_sigmoid:
            return logits.sigmoid()
        return logits

    def predict(self, img):
        with torch.no_grad():
            return self.score(img, apply_sigmoid=True).tolist()
