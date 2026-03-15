import math

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

RESNET_RGB_MODELS = {'CNNDetect', 'DIMD', 'NPR', 'GramNet'}
CLIP_RGB_MODELS = {'UnivFD', 'Rine'}

NON_DIFFERENTIABLE_RGB_MODELS = {
    'FreqDetect': 'Requires NumPy/SciPy DCT preprocessing.',
    'LGrad': 'Requires non-differentiable OpenCV/PIL gradient-image conversion.',
    'Dire': 'Requires non-differentiable OpenCV/PIL diffusion reconstruction conversion.',
    'RPTC': 'Uses non-differentiable patch sorting in preprocessing.',
    'DeFake': 'Uses non-differentiable caption generation/tokenization.',
}


def _ensure_nchw(images):
    if not torch.is_tensor(images):
        raise TypeError('Expected a torch.Tensor for images.')
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError('Expected images in CHW or NCHW format.')
    if images.shape[1] != 3:
        raise ValueError('Expected RGB images with shape (N, 3, H, W).')
    return images


def _normalize(images, mean, std):
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_t) / std_t


def _resize_square(images, size):
    if size is None:
        return images
    return F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False, antialias=True)


def _center_crop(images, crop_size):
    if crop_size is None:
        return images

    height, width = images.shape[-2:]
    if height < crop_size or width < crop_size:
        scale = max(crop_size / float(height), crop_size / float(width))
        new_h = int(math.ceil(height * scale))
        new_w = int(math.ceil(width * scale))
        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)
        height, width = images.shape[-2:]

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, :, top:top + crop_size, left:left + crop_size]


def prepare_model_inputs(images, opt):
    """
    Prepare detector inputs directly from RGB tensors without PIL/NumPy conversion.
    Expects images in [0, 1], shape (N, 3, H, W) or (3, H, W).
    """
    images = _ensure_nchw(images)
    model_name = getattr(opt, 'modelName')

    if model_name in NON_DIFFERENTIABLE_RGB_MODELS:
        reason = NON_DIFFERENTIABLE_RGB_MODELS[model_name]
        raise ValueError(f'{model_name} cannot be used for end-to-end RGB gradients: {reason}')

    if model_name in RESNET_RGB_MODELS:
        processed = _resize_square(images, getattr(opt, 'loadSize', None))
        processed = _center_crop(processed, getattr(opt, 'cropSize', None))
        return _normalize(processed, IMAGENET_MEAN, IMAGENET_STD)

    if model_name in CLIP_RGB_MODELS:
        processed = _resize_square(images, getattr(opt, 'loadSize', None))
        processed = _center_crop(processed, getattr(opt, 'cropSize', None))
        return _normalize(processed, CLIP_MEAN, CLIP_STD)

    if model_name == 'Fusing':
        input_img = _normalize(images, IMAGENET_MEAN, IMAGENET_STD)
        cropped_img = _resize_square(images, getattr(opt, 'loadSize', None))
        cropped_img = _center_crop(cropped_img, getattr(opt, 'cropSize', None))
        cropped_img = _normalize(cropped_img, IMAGENET_MEAN, IMAGENET_STD)
        height, width = images.shape[-2:]
        scale = torch.tensor([height, width], device=images.device, dtype=torch.long).repeat(images.shape[0], 1)
        return input_img, cropped_img, scale

    raise ValueError(f'Gradient-flow preprocessing is not implemented for model {model_name}.')


def detector_score(model, images, opt, apply_sigmoid=False):
    """
    End-to-end differentiable detector score from RGB tensor inputs.
    """
    if hasattr(model, 'set_gradient_flow'):
        model.set_gradient_flow(True)
    else:
        model.gradient_flow_mode = True

    model_inputs = prepare_model_inputs(images, opt)
    if isinstance(model_inputs, tuple):
        return model.score(*model_inputs, apply_sigmoid=apply_sigmoid)
    return model.score(model_inputs, apply_sigmoid=apply_sigmoid)
