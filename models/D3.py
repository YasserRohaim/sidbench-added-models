import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.clip import clip


class _TransformerAttention(nn.Module):
    def __init__(self, input_dim: int, token_num: int, last_dim: int = 1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * token_num, last_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(
            torch.tensor(k.size(-1), dtype=torch.float32, device=x.device)
        )
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.reshape(output.shape[0], -1)
        return self.fc(output)


class D3(nn.Module):
    """
    Native D3-style detector:
      frozen CLIP backbone + discrepancy branch via shuffled patches + attention head.
    """
    def __init__(
        self,
        arch: str = "ViT-L/14",
        shuffle_times: int = 1,
        original_times: int = 1,
        patch_size: int = 14,
    ):
        super().__init__()

        self.gradient_flow_mode = False
        self.arch = arch
        self.shuffle_times = int(shuffle_times)
        self.original_times = int(original_times)
        self.patch_size = int(patch_size)

        self.model, _ = clip.load(arch, device="cpu")
        for p in self.model.parameters():
            p.requires_grad = False

        self.features = None
        self._register_hook()

        feature_dim = int(self.model.visual.ln_post.weight.shape[0])
        token_num = self.shuffle_times + self.original_times
        self.attention_head = _TransformerAttention(feature_dim, token_num, last_dim=1)
        self.fc_head = nn.Linear(feature_dim, 1)
        self.use_fc_head = False

    def _register_hook(self):
        def hook(_module, _inputs, output):
            self.features = output.clone()

        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)
                return
        raise RuntimeError("D3 could not register CLIP ln_post hook.")

    @staticmethod
    def _unwrap_state(state):
        if not isinstance(state, dict):
            return state
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
        return state

    @staticmethod
    def _strip_prefix(state, prefix):
        out = {}
        for k, v in state.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    @staticmethod
    def _is_attention_head_state(state):
        required = {"query.weight", "query.bias", "key.weight", "key.bias",
                    "value.weight", "value.bias", "fc.weight", "fc.bias"}
        return required.issubset(set(state.keys()))

    @staticmethod
    def _is_fc_state(state):
        return set(state.keys()) == {"weight", "bias"}

    def load_weights(self, ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = self._unwrap_state(state)
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported D3 checkpoint format at {ckpt}")

        if "attention_head.query.weight" in state:
            state = self._strip_prefix(state, "attention_head.")
        elif "module.attention_head.query.weight" in state:
            state = self._strip_prefix(state, "module.attention_head.")

        if self._is_attention_head_state(state):
            self.attention_head.load_state_dict(state, strict=True)
            self.use_fc_head = False
            return

        if self._is_fc_state(state):
            self.fc_head.load_state_dict(state, strict=True)
            self.use_fc_head = True
            return

        # Fallback for whole-model checkpoints.
        missing, unexpected = self.load_state_dict(state, strict=False)
        if len(unexpected) == 0 and len(missing) < len(self.state_dict()):
            return
        raise RuntimeError(
            f"Could not load D3 checkpoint {ckpt}. "
            f"Missing keys: {missing[:8]} Unexpected keys: {unexpected[:8]}"
        )

    def _shuffle_patches(self, x: torch.Tensor) -> torch.Tensor:
        patch_size = self.patch_size
        b, c, h, w = x.size()
        if h < patch_size or w < patch_size:
            return x

        # Make H/W divisible by patch size (center crop), then resize back if needed.
        h_fit = h - (h % patch_size)
        w_fit = w - (w % patch_size)
        top = (h - h_fit) // 2
        left = (w - w_fit) // 2
        x_fit = x[:, :, top:top + h_fit, left:left + w_fit]

        patches = F.unfold(x_fit, kernel_size=patch_size, stride=patch_size)
        perm = torch.randperm(patches.size(-1), device=patches.device)
        shuffled = patches[:, :, perm]
        shuffled_img = F.fold(
            shuffled,
            output_size=(h_fit, w_fit),
            kernel_size=patch_size,
            stride=patch_size,
        )
        if h_fit != h or w_fit != w:
            shuffled_img = F.interpolate(
                shuffled_img, size=(h, w), mode="bilinear", align_corners=False, antialias=True
            )
        return shuffled_img

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model.encode_image(x)
        if self.features is None:
            raise RuntimeError("D3 CLIP hook did not capture features.")
        return self.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_grad = self.gradient_flow_mode and torch.is_grad_enabled()
        ctx = contextlib.nullcontext() if use_grad else torch.no_grad()

        with ctx:
            if self.use_fc_head:
                features = self._backbone_features(x)
                return self.fc_head(features)

            features = []
            for _ in range(self.shuffle_times):
                shuffled = self._shuffle_patches(x)
                features.append(self._backbone_features(shuffled))

            original = self._backbone_features(x)
            for _ in range(self.original_times):
                features.append(original.clone())

        tokens = torch.stack(features, dim=1)
        return self.attention_head(tokens)

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
