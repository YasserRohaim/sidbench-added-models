import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models import get_model
from options import TestOptions
from utils.gradient_flow import detector_score


class TinyGenerator(nn.Module):
    def __init__(self, latent_dim=256, out_size=256):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * out_size * out_size),
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(z.size(0), 3, self.out_size, self.out_size)
        return torch.sigmoid(x)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = TestOptions().initialize(parser)
    parser.add_argument('--latentDim', type=int, default=256, help='latent size for demo generator')
    parser.add_argument('--lr', type=float, default=1e-4, help='generator learning rate')
    opt = parser.parse_args()
    opt.gradientFlowMode = True
    if opt.batchSize == 64:
        opt.batchSize = 2

    device = torch.device('cuda:0' if torch.cuda.is_available() and opt.gpus != '-1' else 'cpu')

    detector = get_model(opt).to(device)
    for p in detector.parameters():
        p.requires_grad = False
    detector.eval()

    generator = TinyGenerator(latent_dim=opt.latentDim, out_size=max(opt.loadSize or opt.cropSize, opt.cropSize)).to(device)
    gen_optim = optim.Adam(generator.parameters(), lr=opt.lr)

    z = torch.randn(opt.batchSize, opt.latentDim, device=device)
    fake_images = generator(z)

    # Maximize detector confusion by pushing fake scores down.
    fake_logits = detector_score(detector, fake_images, opt, apply_sigmoid=False)
    gen_loss = fake_logits.mean()

    gen_optim.zero_grad(set_to_none=True)
    gen_loss.backward()
    gen_optim.step()

    print(f'Demo generator step complete. loss={gen_loss.item():.6f}')


if __name__ == '__main__':
    main()
