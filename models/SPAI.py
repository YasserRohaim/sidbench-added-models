import pathlib
import sys
from typing import Optional

import torch
import torch.nn as nn


class SPAI(nn.Module):
    def __init__(self, opt: Optional[object] = None):
        super().__init__()
        self.gradient_flow_mode = False
        self.opt = opt
        self.feature_extraction_batch = None
        self.model = self._build_model(opt)

    def _build_model(self, opt: Optional[object]) -> nn.Module:
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        spai_root = repo_root / "spai"
        networks_root = repo_root / "networks"

        # Allow importing SPAI as `spai.*` and OpenAI-CLIP-compatible `clip` from networks.
        for path in (spai_root, networks_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        try:
            from spai.config import get_custom_config
            from spai.models import build_cls_model
        except ImportError as exc:
            raise ImportError(
                "SPAI dependencies are missing. Install SPAI requirements "
                "(see ./spai/requirements.txt)."
            ) from exc

        cfg_path = getattr(opt, "spaiConfigPath", "./spai/configs/spai.yaml") if opt else "./spai/configs/spai.yaml"
        cfg_path = pathlib.Path(cfg_path)
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"SPAI config not found: {cfg_path}")

        config = get_custom_config(str(cfg_path))
        config.defrost()
        if opt is not None and getattr(opt, "spaiFeatureExtractionBatch", None) is not None:
            config.MODEL.FEATURE_EXTRACTION_BATCH = int(opt.spaiFeatureExtractionBatch)
        config.freeze()

        self.feature_extraction_batch = config.MODEL.FEATURE_EXTRACTION_BATCH
        return build_cls_model(config)

    def forward(self, x):
        if isinstance(x, list):
            return self.model(x, self.feature_extraction_batch)
        return self.model(x)

    def load_weights(self, ckpt):
        ckpt_path = pathlib.Path(ckpt)
        if not ckpt_path.is_absolute():
            repo_root = pathlib.Path(__file__).resolve().parent.parent
            ckpt_path = (repo_root / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SPAI checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                checkpoint_model = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint_model = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint_model = checkpoint["model_state_dict"]
            else:
                checkpoint_model = checkpoint
        else:
            checkpoint_model = checkpoint

        if isinstance(checkpoint_model, dict) and any(k.startswith("encoder.") for k in checkpoint_model.keys()):
            checkpoint_model = {
                k.replace("encoder.", ""): v
                for k, v in checkpoint_model.items()
                if k.startswith("encoder.")
            }

        self.model.load_state_dict(checkpoint_model, strict=False)

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
