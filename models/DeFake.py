import torch.nn as nn 
import torch

import torch.nn.functional as F
import networks.clip as clip

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
    
class DeFake(nn.Module):
    def __init__(self, blip, encoder, num_classes=2, input_size=1024, hidden_size_list=[512,256]):
        super(DeFake, self).__init__()
        self.gradient_flow_mode = False

        self.net = NeuralNet(input_size, hidden_size_list, num_classes)

        self.blip = blip
        self.encoder = encoder

    def forward(self, x):
        caption = self.blip.generate(x, sample=True, num_beams=1, max_length=60, min_length=5) 
        text = clip.tokenize(list(caption)).to(x.get_device())

        image_features = self.encoder.encode_image(x)
        text_features =self.encoder.encode_text(text)
        
        emb = torch.cat((image_features, text_features), 1) 
        return self.net(emb.float())
    
    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        self.net.load_state_dict(state_dict)

    def set_gradient_flow(self, enabled=True):
        self.gradient_flow_mode = enabled

    def score(self, img, apply_sigmoid=False):
        if self.gradient_flow_mode and torch.is_grad_enabled():
            raise RuntimeError(
                "DeFake does not support end-to-end gradient flow: BLIP caption generation "
                "and text tokenization are non-differentiable."
            )
        output = self.forward(img)
        logits = output[:, 1] - output[:, 0]
        if apply_sigmoid:
            return logits.sigmoid()
        return logits

    def predict(self, img):
        with torch.no_grad():
            return self.score(img, apply_sigmoid=True).tolist()
