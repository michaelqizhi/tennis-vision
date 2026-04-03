"""Shadow removal preprocessing using ShadowFormer (Guo et al., AAAI 2023).

Provides optional shadow removal for outdoor tennis court images where shadows
interfere with court color detection and line filtering.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.shadowformer import ShadowFormer

# Default weight paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_WEIGHTS = _PROJECT_ROOT / "weights" / "shadowformer_istd_plus.pth"
_GDRIVE_FILE_ID = "10pBsJenoWGriZ9kjWOcE4l4Kzg-F1TFd"

# Model config matching the ISTD+ pretrained checkpoint
_MODEL_CONFIG = dict(
    img_size=320,
    embed_dim=32,
    win_size=10,
    token_projection="linear",
    token_mlp="leff",
)

# Inference size (model was trained at 256 but ISTD+ checkpoint uses 320)
_INFER_SIZE = 320
# Padding must be a multiple of 8 * win_size = 80
_PAD_MULTIPLE = 80


class ShadowRemover:
    """Removes shadows from court images using ShadowFormer.

    Uses lazy loading — model is not loaded until first call to remove_shadows().
    Weights are auto-downloaded from Google Drive if missing.
    """

    def __init__(self, weights_path: str | Path | None = None, device: str = "auto"):
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._model: ShadowFormer | None = None

    def _ensure_loaded(self) -> None:
        """Load model and weights on first use. Auto-downloads weights if missing."""
        if self._model is not None:
            return

        # Auto-download weights if missing
        if not self.weights_path.exists():
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [ShadowRemover] Downloading weights to {self.weights_path} ...")
            try:
                import gdown
                gdown.download(id=_GDRIVE_FILE_ID, output=str(self.weights_path), quiet=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download ShadowFormer weights: {e}\n"
                    f"Please manually download from Google Drive (file ID: {_GDRIVE_FILE_ID}) "
                    f"and place at: {self.weights_path}"
                ) from e

        # Build model
        model = ShadowFormer(**_MODEL_CONFIG)

        # Load state dict (handle DataParallel 'module.' prefix)
        checkpoint = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        model.to(self.device)
        model.eval()
        self._model = model
        print(f"  [ShadowRemover] Model loaded on {self.device}")

    @staticmethod
    def _generate_shadow_mask(image_bgr: np.ndarray) -> np.ndarray:
        """Auto-generate a shadow mask using LAB colorspace thresholding.

        Since we don't have ground-truth shadow masks, we estimate them from
        the L channel of the LAB colorspace. Shadows are darker (low L) but
        maintain similar chrominance.

        Args:
            image_bgr: Input BGR uint8 image.

        Returns:
            Binary mask [H, W] uint8 where 255=shadow, 0=non-shadow.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Otsu threshold on L channel to find dark regions
        _, shadow_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup: remove small noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

        return shadow_mask

    def remove_shadows(self, image_bgr: np.ndarray) -> np.ndarray:
        """Remove shadows from a BGR uint8 image.

        Strategy: resize to 320×320 for inference, compute a ratio map
        (deshadowed / original), resize ratio back to full resolution,
        and apply to original. This preserves full-resolution detail.

        Args:
            image_bgr: Input BGR uint8 image (e.g. near-half crop ~560×1920).

        Returns:
            Deshadowed BGR uint8 image, same size as input.
        """
        self._ensure_loaded()

        orig_h, orig_w = image_bgr.shape[:2]

        # 1. Generate shadow mask at original resolution
        shadow_mask = self._generate_shadow_mask(image_bgr)

        # 2. Resize image and mask to inference size
        img_resized = cv2.resize(image_bgr, (_INFER_SIZE, _INFER_SIZE), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(shadow_mask, (_INFER_SIZE, _INFER_SIZE), interpolation=cv2.INTER_NEAREST)

        # 3. Convert to float32 tensors normalized to [0, 1]
        #    Image: BGR → RGB, [H,W,3] → [1,3,H,W]
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        #    Mask: [H,W] → [1,1,H,W], binary 0/1
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0) / 255.0

        img_tensor = img_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)

        # 4. Pad to multiple of 80 if needed (320 is already 4*80, so no padding needed)
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        pad_h = (_PAD_MULTIPLE - h % _PAD_MULTIPLE) % _PAD_MULTIPLE
        pad_w = (_PAD_MULTIPLE - w % _PAD_MULTIPLE) % _PAD_MULTIPLE
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
            mask_tensor = torch.nn.functional.pad(mask_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # 5. Run model inference
        with torch.no_grad():
            output = self._model(img_tensor, mask_tensor)

        # Remove padding if applied
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]
            img_tensor = img_tensor[:, :, :h, :w]

        # 6. Compute ratio map: deshadowed / (input + epsilon)
        eps = 1e-6
        ratio_map = output / (img_tensor + eps)
        ratio_map = ratio_map.clamp(0.5, 2.0)  # Clamp to reasonable range

        # 7. Resize ratio map to original image size (bilinear interpolation)
        ratio_map = torch.nn.functional.interpolate(
            ratio_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

        # 8. Apply ratio to original image
        #    Convert original to RGB float tensor
        orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_tensor = torch.from_numpy(orig_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        orig_tensor = orig_tensor.to(self.device)

        result = orig_tensor * ratio_map
        result = result.clamp(0.0, 1.0)

        # 9. Convert back to uint8 BGR
        result_np = (result[0].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return result_bgr
