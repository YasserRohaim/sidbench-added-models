import torch.nn as nn 
import torch

from networks.resnet_npr import resnet50


class NPR(nn.Module):
    def __init__(self):
        super(NPR, self).__init__()
        self.model = resnet50(num_classes=1)
        self.gradient_flow_mode = False

    def forward(self, x):
        return self.model(x)

    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        try:
            self.model.load_state_dict(state_dict['model'], strict=False)
        except:
            print('Loading failed, trying to load model without module')
            self.model.load_state_dict(state_dict)

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
        
