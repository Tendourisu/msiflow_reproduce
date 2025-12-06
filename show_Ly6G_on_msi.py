from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay a Ly6G fluorescence image in red on top of a UMAP-reduced MSI background."
        )
    )
    parser.add_argument(
        "umap_path",
        type=Path,
        help="Path to the grayscale UMAP image produced from MSI data.",
    )
    parser.add_argument(
        "ly6g_path",
        type=Path,
        help="Path to the Ly6G fluorescence image that should be highlighted in red.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional output file name (PNG). The file is always saved next to the UMAP image."
        ),
    )
    parser.add_argument(
        "--overlay-strength",
        type=float,
        default=1,
        help="Maximum contribution from the Ly6G signal (0-1) when blending with the background.",
    )
    parser.add_argument(
        "--overlay-low-percentile",
        type=float,
        default=0.0,
        help="Percentile used as the lower bound for Ly6G intensity normalization.",
    )
    parser.add_argument(
        "--overlay-high-percentile",
        type=float,
        default=99.5,
        help="Percentile used as the upper bound for Ly6G intensity normalization.",
    )
    parser.add_argument(
        "--resize-overlay",
        action="store_true",
        help="Automatically resize the Ly6G image to match the UMAP dimensions if needed.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=float,
        default=None,
        help="Override Pillow's MAX_IMAGE_PIXELS limit to handle large TIFFs safely.",
    )
    parser.add_argument(
        "--allow-large-images",
        action="store_true",
        help="Disable Pillow's decompression bomb check (use only when you trust the input).",
    )
    parser.add_argument(
        "--background-strength",
        type=float,
        default=0.5,
        help="Scale factor (0-1) applied to the UMAP background intensity before blending.",
    )
    return parser.parse_args()


def _configure_image_pixel_limit(allow_large: bool, max_pixels: float | None) -> None:
    if allow_large:
        Image.MAX_IMAGE_PIXELS = None
        print("Warning: Disabled Pillow decompression bomb check; ensure inputs are trusted.")
        return
    if max_pixels is not None:
        if max_pixels <= 0:
            raise ValueError("--max-image-pixels must be positive.")
        Image.MAX_IMAGE_PIXELS = max_pixels


def _load_background(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"UMAP image not found: {path}")
    image = Image.open(path).convert("RGB")
    data = np.asarray(image, dtype=np.float32) / 255.0
    return data


def _apply_background_strength(background: np.ndarray, strength: float) -> np.ndarray:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("--background-strength must be between 0 and 1.")
    if np.isclose(strength, 1.0):
        return background
    return np.clip(background * strength, 0.0, 1.0)


def _component_output_paths(
    umap_path: Path, overlay_path: Path, output_name: str | None
) -> tuple[Path, Path]:
    base_name = Path(output_name).stem if output_name else umap_path.stem
    parent = overlay_path.parent
    msi_path = parent / f"{base_name}_msi.png"
    ly6g_path = parent / f"{base_name}_Ly6G.png"
    return msi_path, ly6g_path


def _create_red_overlay_image(overlay: np.ndarray) -> np.ndarray:
    red_overlay = np.zeros((overlay.shape[0], overlay.shape[1], 3), dtype=np.float32)
    red_overlay[..., 0] = overlay
    return red_overlay


def _save_png_image(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def _load_overlay(path: Path, target_size: tuple[int, int], resize: bool) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Ly6G image not found: {path}")
    try:
        overlay = Image.open(path).convert("F")
    except Image.DecompressionBombError as exc:
        raise RuntimeError(
            "Ly6G image exceeds Pillow's safety limit. "
            "Rerun with --allow-large-images or an appropriate --max-image-pixels value."
        ) from exc
    if overlay.size != target_size:
        if not resize:
            raise ValueError(
                "Ly6G image size does not match the UMAP background. "
                "Use --resize-overlay if resizing is acceptable."
            )
        overlay = overlay.resize(target_size, Image.BILINEAR)
    data = np.asarray(overlay, dtype=np.float32)
    return data


def _normalize_overlay(
    overlay: np.ndarray, low_percentile: float, high_percentile: float
) -> np.ndarray:
    low = np.percentile(overlay, low_percentile)
    high = np.percentile(overlay, high_percentile)
    if high <= low:
        high = overlay.max()
        low = overlay.min()
    if high <= low:
        return np.zeros_like(overlay, dtype=np.float32)
    normalized = np.clip((overlay - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float32)


def _blend_images(background: np.ndarray, overlay: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    overlay_alpha = np.clip(overlay * strength, 0.0, 1.0)
    overlay_rgb = np.zeros_like(background)
    overlay_rgb[..., 0] = 1.0  # red
    alpha = overlay_alpha[..., None]
    blended = background * (1.0 - alpha) + overlay_rgb * alpha
    return np.clip(blended, 0.0, 1.0)


def _resolve_output_path(umap_path: Path, output_name: str | None) -> Path:
    if output_name:
        candidate = Path(output_name)
        if candidate.name != output_name:
            raise ValueError("Only a file name is allowed for --output-name.")
        filename = output_name
    else:
        filename = f"{umap_path.stem}_Ly6G_overlay.png"
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    return umap_path.with_name(filename)


def main() -> None:
    args = parse_args()
    _configure_image_pixel_limit(args.allow_large_images, args.max_image_pixels)
    background = _load_background(args.umap_path)
    background = _apply_background_strength(background, args.background_strength)
    height, width = background.shape[:2]
    overlay = _load_overlay(args.ly6g_path, (width, height), args.resize_overlay)
    normalized_overlay = _normalize_overlay(
        overlay, args.overlay_low_percentile, args.overlay_high_percentile
    )
    blended = _blend_images(background, normalized_overlay, args.overlay_strength)
    output_path = _resolve_output_path(args.umap_path, args.output_name)
    msi_path, ly6g_path = _component_output_paths(args.umap_path, output_path, args.output_name)
    output_image = Image.fromarray((blended * 255).astype(np.uint8))
    output_image.save(output_path)
    _save_png_image(msi_path, background)
    ly6g_image = _create_red_overlay_image(normalized_overlay)
    _save_png_image(ly6g_path, ly6g_image)
    print(f"Saved overlay image to: {output_path}")
    print(f"Saved MSI background to: {msi_path}")
    print(f"Saved Ly6G highlight map to: {ly6g_path}")


if __name__ == "__main__":
    main()
