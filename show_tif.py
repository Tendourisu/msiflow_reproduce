"""将 tif 文件渲染为可视化图片。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 tif 文件渲染为位于同目录下的 PNG 或 SVG 预览图"
    )
    parser.add_argument("tif_path", type=Path, help="待渲染的 tif 文件路径")
    parser.add_argument(
        "-f",
        "--format",
        choices=("png", "svg"),
        default="png",
        help="输出图片格式，默认为 png",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="灰度数据使用的 matplotlib colormap",
    )
    return parser.parse_args()


def _prepare_image(data: np.ndarray) -> np.ndarray:
    """将 tif 数据压缩为可绘制的 2D 或 3D 数组。"""

    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[-1] in {3, 4}:
            return arr
        if arr.shape[0] in {3, 4}:
            return np.moveaxis(arr, 0, -1)

    # 对于 stack 数据，使用最大强度投影
    while arr.ndim > 2:
        arr = arr.max(axis=0)

    return arr


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """将数据缩放到 [0, 1] 区间以便渲染。"""

    arr = arr.astype(np.float32)
    min_val = np.nanmin(arr)
    arr -= min_val
    max_val = np.nanmax(arr)
    if max_val > 0:
        arr /= max_val
    return arr


def render_image(image: np.ndarray, out_path: Path, cmap: str) -> None:
    """使用 matplotlib 保存图像。"""

    fig, ax = plt.subplots(figsize=_figure_size(image.shape))
    ax.axis("off")
    if image.ndim == 2:
        ax.imshow(_normalize_image(image), cmap=cmap)
    else:
        ax.imshow(_normalize_image(image))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _figure_size(shape: Tuple[int, ...]) -> Tuple[float, float]:
    """动态调整图像大小以避免超小或超大的画布。"""

    height, width = shape[:2]
    if height == 0 or width == 0:
        return 4.0, 4.0

    aspect = width / height
    max_inch = 10.0
    min_inch = 2.0

    if aspect >= 1:  # landscape
        width_inch = max_inch
        height_inch = max(max_inch / aspect, min_inch)
    else:  # portrait
        height_inch = max_inch
        width_inch = max(max_inch * aspect, min_inch)

    return width_inch, height_inch


def main() -> None:
    args = parse_args()
    tif_path: Path = args.tif_path

    if not tif_path.exists():
        raise FileNotFoundError(f"找不到指定的文件: {tif_path}")

    data = tifffile.imread(tif_path)
    prepped = _prepare_image(data)

    out_path = tif_path.with_name(f"{tif_path.stem}_preview.{args.format}")
    render_image(prepped, out_path, args.cmap)
    print(f"已生成可视化文件: {out_path}")


if __name__ == "__main__":
    main()
