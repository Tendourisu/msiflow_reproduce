#!/usr/bin/env python3
"""Generate Markdown reports for molecular_signatures_flow results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageSequence

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

LOGGER = logging.getLogger("molecular_signatures_report")
DEFAULT_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"


def slugify(value: str) -> str:
    """Return a filesystem friendly slug."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "asset"


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration as dictionary."""
    if not config_path.exists():
        LOGGER.warning("Config file %s not found. Proceeding without metadata.", config_path)
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to read config files but is not installed.")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return data


def format_float(value: Optional[str], precision: int = 4) -> str:
    """Format numeric strings with consistent precision."""
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value) if value is not None else ""


def convert_tif_to_png(src: Path, dest: Path) -> None:
    """Convert TIF/TIFF image into PNG format."""
    with Image.open(src) as img:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(img)]
        if not frames:
            raise ValueError(f"Unable to extract frames from {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(dest)
        if len(frames) > 1:
            LOGGER.info("Converted only the first frame of %s which contained %s frames.", src, len(frames))


class AssetManager:
    """Handle copying and converting artifacts into the report folder."""

    def __init__(self, assets_dir: Path) -> None:
        self.assets_dir = assets_dir
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[Path, Path] = {}

    def add(self, src: Path, preferred_name: Optional[str] = None) -> str:
        """Copy or convert a file into the assets directory and return the relative path."""
        src = src.resolve()
        if not src.exists():
            raise FileNotFoundError(f"Asset {src} does not exist.")
        if src in self.cache:
            return str(self.cache[src])

        stem = slugify(preferred_name or src.stem)
        suffix = src.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            suffix = ".png"
        dest_name = self._unique_name(stem, suffix)
        dest = self.assets_dir / dest_name
        if src.suffix.lower() in {".tif", ".tiff"}:
            convert_tif_to_png(src, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        rel_path = dest.relative_to(self.assets_dir.parent)
        self.cache[src] = rel_path
        return str(rel_path)

    def _unique_name(self, stem: str, suffix: str) -> str:
        """Return a unique file name in the assets directory."""
        stem = stem or "asset"
        candidate = f"{stem}{suffix}"
        counter = 1
        while (self.assets_dir / candidate).exists():
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        return candidate


class ReportGenerator:
    """Build Markdown report from molecular_signatures_flow outputs."""

    def __init__(
        self,
        result_dir: Path,
        output_dir: Path,
        timestamp: Optional[str],
        max_feature_rows: int,
        max_spatial_rows: int,
        max_masks: int,
        max_ion_imgs: int,
    ) -> None:
        self.result_dir = result_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.timestamp = timestamp or dt.datetime.now().strftime(DEFAULT_TIMESTAMP_FMT)
        self.max_feature_rows = max_feature_rows
        self.max_spatial_rows = max_spatial_rows
        self.max_masks = max_masks
        self.max_ion_imgs = max_ion_imgs
        self.dataset_name = self.result_dir.name
        self.bin_img_dir = self.result_dir / "bin_imgs"
        self.reduced_dir = self.bin_img_dir / "reduced_to_meas_pixels"
        self.config = load_config(self.result_dir / "config.yaml")
        self.class_dirs = self._discover_class_dirs()
        self.subdir = self._create_output_subdir()
        self.assets = AssetManager(self.subdir / "assets")

    def run(self) -> Path:
        """Generate the report and return its path."""
        lines: List[str] = []
        lines.extend(self._build_header_section())
        for class_dir in self.class_dirs:
            lines.extend(self._build_class_section(class_dir))
        report_path = self.subdir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        LOGGER.info("Report written to %s", report_path)
        return report_path

    def _create_output_subdir(self) -> Path:
        folder_name = f"{slugify(self.dataset_name)}-{self.timestamp}"
        destination = self.output_dir / folder_name
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _discover_class_dirs(self) -> List[Path]:
        if not self.reduced_dir.exists():
            raise FileNotFoundError(f"{self.reduced_dir} not found. Please provide a valid result folder.")
        class_dirs = [
            child
            for child in sorted(self.reduced_dir.iterdir())
            if child.is_dir() and (child / "combined_rankings").exists()
        ]
        if not class_dirs:
            raise RuntimeError(
                f"No class folders with combined_rankings found inside {self.reduced_dir}. "
                "Ensure the molecular_signatures_flow finished successfully."
            )
        return class_dirs

    def _build_header_section(self) -> List[str]:
        classification_cfg = self.config.get("classification", {})
        lines = [
            f"# Molecular signatures 报告 · {self.dataset_name}",
            "",
            f"*生成时间：{self.timestamp}*",
            "",
            "本报告聚焦 molecular_signatures_flow，通过 ROI、基于机器学习的特征筛选以及空间相似度分析，"
            "串联出与组织表型高度相关的 m/z 特征。",
            "",
        ]
        if classification_cfg:
            lines.append("**Workflow 参数**")
            params = [
                f"- Model: `{classification_cfg.get('model', 'n/a')}`",
                f"- Class balancing: `{classification_cfg.get('class_balancing_method', 'n/a')}`",
                f"- Top features: {classification_cfg.get('num_top_feat', 'n/a')}",
                f"- Ion image export: {classification_cfg.get('save_ion_imgs', 'n/a')}",
                f"- Ion UMAP export: {classification_cfg.get('save_umap_imgs', 'n/a')}",
            ]
            lines.extend(params)
            lines.append("")
        lines.append(
            "整体流程先将 ROI 映射至测量到的 MSI 像素，再生成正负掩膜训练分类模型，随后结合 SHAP 与 Pearson/Cosine 排名，"
            "以多视角验证潜在分子特征的可靠性。"
        )
        lines.append("")
        return lines

    def _build_class_section(self, class_dir: Path) -> List[str]:
        class_name = class_dir.name
        lines = [f"## Class {class_name} 概览", ""]
        samples = self._collect_samples(class_dir)
        if samples:
            sample_line = ", ".join(sorted(samples))
            lines.append(
                f"在该 class 中找到 **{len(samples)}** 个样本的正类 ROI：{sample_line}。"
                "负类掩膜通过测得像素扣除 ROI 获得，以提供局部背景并提升模型对富集/排斥信号的敏感度。"
            )
            lines.append("")
        lines.extend(self._add_mask_gallery(class_dir, samples))
        lines.extend(self._add_combined_ranking(class_dir))
        lines.extend(self._add_spatial_similarity(class_dir))
        lines.extend(self._add_model_sections(class_dir))
        return lines

    def _collect_samples(self, class_dir: Path) -> List[str]:
        samples = {tif.stem.replace("pos_", "", 1) for tif in class_dir.glob("pos_*.tif")}
        return sorted(samples)

    def _add_mask_gallery(self, class_dir: Path, samples: Sequence[str]) -> List[str]:
        lines: List[str] = []
        mask_entries: List[Tuple[str, str, str]] = []
        limited_samples = samples[: self.max_masks] if samples else []
        for sample_name in limited_samples:
            pos = class_dir / f"pos_{sample_name}.tif"
            neg = class_dir / f"neg_{sample_name}.tif"
            if pos.exists():
                rel = self.assets.add(pos, f"{class_dir.name}_{sample_name}_pos")
                mask_entries.append(("正类 ROI", sample_name, rel))
            if neg.exists():
                rel = self.assets.add(neg, f"{class_dir.name}_{sample_name}_neg")
                mask_entries.append(("负类 ROI", sample_name, rel))
        if mask_entries:
            lines.append("### ROI alignment 质量")
            lines.append(
                "正类掩膜展示 segmentation 标注区域，负类掩膜近似周边背景组织，方便直观评估 ROI 是否与测量像素吻合。"
            )
            lines.append("")
            for label, sample, rel in mask_entries:
                lines.append(f"![{label} · {sample}]({rel})")
            lines.append("")
        return lines

    def _add_combined_ranking(self, class_dir: Path) -> List[str]:
        lines: List[str] = []
        combined_csv = class_dir / "combined_rankings" / "top_features.csv"
        svg = class_dir / "combined_rankings" / "top_features.svg"
        rows = self._read_csv_rows(combined_csv, self.max_feature_rows)
        if rows:
            lines.append("### Combined ranking 综述")
            lines.append(
                "该表整合机器学习 Feature importance 与 Pearson 趋势，筛出同时具备判别力与空间一致性的 m/z。"
            )
            lines.append("")
            table_rows = [
                [row.get("m/z", ""), format_float(row.get("feature importance")), format_float(row.get("mean"))]
                for row in rows
            ]
            lines.append(
                render_table(
                    ["m/z", "特征重要性 (Feature importance)", "平均 Pearson 相关 (Mean Pearson corr)"],
                    table_rows,
                )
            )
            lines.append("")
        if svg.exists():
            rel = self.assets.add(svg, f"{class_dir.name}_combined_top_features")
            lines.append(f"![Combined ranking 热图]({rel})")
            lines.append("")
        return lines

    def _add_spatial_similarity(self, class_dir: Path) -> List[str]:
        lines: List[str] = []
        spatial_dir = class_dir / "spatial_similarity"
        if not spatial_dir.exists():
            return lines
        pearson_rows = self._summarize_spatial_csv(spatial_dir / "overall_spatial_ranking_pearson.csv")
        cosine_rows = self._summarize_spatial_csv(spatial_dir / "overall_spatial_ranking_cosine.csv")
        if pearson_rows or cosine_rows:
            lines.append("### Spatial similarity 指标")
            lines.append(
                "Pearson 与 Cosine 描述 m/z 与 ROI 的空间耦合程度：高值意味着富集，低值/负值提示潜在排斥区域。"
            )
            lines.append("")
        if pearson_rows:
            lines.append("**Pearson ranking（Mean corr）**")
            lines.append(
                render_table(
                    ["m/z", "平均相关 (Mean corr)", "最高样本 (Top sample)"],
                    pearson_rows,
                )
            )
            lines.append("")
        if cosine_rows:
            lines.append("**Cosine ranking（Mean sim）**")
            lines.append(
                render_table(
                    ["m/z", "平均相似度 (Mean sim)", "最高样本 (Top sample)"],
                    cosine_rows,
                )
            )
            lines.append("")
        caption_map = {
            "barplot_pearson_corr": "Pearson 柱状图",
            "barplot_cosine_sim": "Cosine 柱状图",
            "violinplot_pearson_corr": "Pearson 小提琴图",
            "violinplot_cosine_sim": "Cosine 小提琴图",
            "venn_similarity_measures": "Pearson-Cosine Venn 图",
        }
        for stem, caption in caption_map.items():
            path = spatial_dir / f"{stem}.svg"
            if not path.exists():
                continue
            rel = self.assets.add(path, f"{class_dir.name}_{path.stem}")
            lines.append(f"![{caption}]({rel})")
        for tif in spatial_dir.glob("top_mz_*.tif"):
            rel = self.assets.add(tif, tif.stem)
            lines.append(f"![top m/z 空间分布 {tif.stem}]({rel})")
        if lines and lines[-1] != "":
            lines.append("")
        return lines

    def _add_model_sections(self, class_dir: Path) -> List[str]:
        lines: List[str] = []
        skip_names = {"combined_rankings", "spatial_similarity"}
        model_dirs = [
            child for child in sorted(class_dir.iterdir()) if child.is_dir() and child.name not in skip_names
        ]
        for model_dir in model_dirs:
            lines.extend(self._build_model_section(class_dir.name, model_dir))
        return lines

    def _build_model_section(self, class_name: str, model_dir: Path) -> List[str]:
        lines: List[str] = []
        model_label = model_dir.name.replace("_", " ")
        lines.append(f"### ML model · {model_label}")
        lines.append(
            "该分类器区分正负像素，并通过 Feature importance 与 SHAP 概述每个 m/z 的贡献，帮助定位关键离子。"
        )
        lines.append("")
        feat_csv = next(model_dir.glob("*_feature_importance.csv"), None)
        feat_svg = next(model_dir.glob("*_feature_importance.svg"), None)
        shap_svgs = sorted(model_dir.glob("*_shap*.svg"))
        if feat_csv and feat_csv.exists():
            rows = self._read_csv_rows(feat_csv, self.max_feature_rows)
            table = [[row.get("m/z", ""), format_float(row.get("feature importance"))] for row in rows]
            if table:
                lines.append(render_table(["m/z", "特征重要性 (Feature importance)"], table))
                lines.append("")
        if feat_svg and feat_svg.exists():
            rel = self.assets.add(feat_svg, f"{class_name}_{feat_svg.stem}")
            lines.append(f"![Feature importance 图]({rel})")
            lines.append("")
        for shap_svg in shap_svgs:
            rel = self.assets.add(shap_svg, f"{class_name}_{shap_svg.stem}")
            lines.append(f"![{shap_svg.stem.replace('_', ' ')}]({rel})")
            lines.append("")
        ion_imgs = self._collect_ion_images(model_dir)
        if ion_imgs:
            lines.append("**Representative ion images（代表性离子图）**")
            for caption, rel in ion_imgs:
                lines.append(f"![{caption}]({rel})")
            lines.append("")
        ion_umaps = model_dir / "ion_umaps"
        if ion_umaps.exists():
            for img_path in sorted(ion_umaps.glob("*.*")):
                if not img_path.is_file():
                    continue
                rel = self.assets.add(img_path, f"{class_name}_{img_path.stem}")
                lines.append(f"![UMAP 投影 {img_path.stem}]({rel})")
            lines.append("")
        return lines

    def _collect_ion_images(self, model_dir: Path) -> List[Tuple[str, str]]:
        ion_dir = model_dir / "ion_imgs"
        if not ion_dir.exists():
            return []
        collected: List[Tuple[str, str]] = []
        for sample_dir in sorted(ion_dir.glob("*")):
            if not sample_dir.is_dir():
                continue
            for tif in sorted(sample_dir.glob("*.tif")):
                if len(collected) >= self.max_ion_imgs:
                    break
                sample_name, mz_value = parse_ion_filename(tif.stem)
                caption = f"{sample_name} · m/z {mz_value} 离子图"
                rel = self.assets.add(tif, f"{sample_dir.name}_{mz_value}")
                collected.append((caption, rel))
            if len(collected) >= self.max_ion_imgs:
                break
        return collected

    def _summarize_spatial_csv(self, csv_path: Path) -> List[List[str]]:
        rows = self._read_csv_rows(csv_path, self.max_spatial_rows)
        summarized: List[List[str]] = []
        for row in rows:
            mz = row.get("m/z") or row.get("")
            mean = row.get("mean")
            candidates = {
                key: value
                for key, value in row.items()
                if key not in ("m/z", "", "mean") and value not in (None, "")
            }
            top = ""
            if candidates:
                top_key = max(candidates, key=lambda key: float(candidates[key]))
                top = f"{top_key} ({format_float(candidates[top_key])})"
            summarized.append([str(mz), format_float(mean), top])
        return summarized

    def _read_csv_rows(self, csv_path: Path, limit: int) -> List[Dict[str, str]]:
        if not csv_path.exists():
            LOGGER.warning("Missing CSV file %s", csv_path)
            return []
        rows: List[Dict[str, str]] = []
        with csv_path.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                rows.append(row)
                if idx + 1 >= limit:
                    break
        return rows


def parse_ion_filename(stem: str) -> Tuple[str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, stem
    mz = f"{parts[-2]}.{parts[-1]}"
    sample = "_".join(parts[:-2])
    return sample, mz


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header_row = "|" + " | ".join(headers) + "|"
    separator_row = "|" + "|".join(["---"] * len(headers)) + "|"
    data_rows = ["|" + " | ".join(row) + "|" for row in rows]
    return "\n".join([header_row, separator_row, *data_rows])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown reports for molecular_signatures_flow results."
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Path to the processed result folder (contains bin_imgs, msi, config.yaml, ...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report"),
        help="Directory where the report folders will be stored (default: ./report).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional timestamp override (default: current time).",
    )
    parser.add_argument("--max-feature-rows", type=int, default=5, help="Rows to keep in feature tables.")
    parser.add_argument("--max-spatial-rows", type=int, default=5, help="Rows to keep in spatial tables.")
    parser.add_argument("--max-mask-images", type=int, default=4, help="Maximum number of mask samples to display.")
    parser.add_argument("--max-ion-images", type=int, default=6, help="Maximum ion images to embed.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    generator = ReportGenerator(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        max_feature_rows=args.max_feature_rows,
        max_spatial_rows=args.max_spatial_rows,
        max_masks=args.max_mask_images,
        max_ion_imgs=args.max_ion_images,
    )
    return generator.run()


if __name__ == "__main__":
    main()
