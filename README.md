## msiflow 复现说明

### 环境配置

使用 conda 创建并激活环境：

```bash
conda env create --file msiflow_env.yaml
conda activate msiflow
```

### 代码复现

#### 下载代码

```bash
git clone https://github.com/Immunodynamics-Engel-Lab/msiflow
cd msiflow
```

#### 下载数据集

数据集地址：https://zenodo.org/records/11913042  
官方 demo 仅提供了经 `msi_preprocessing` 处理且截取部分区域的 UPEC_12 数据，无法直接复现全流程。为补齐数据并覆盖完整流程：
- 复现 `msi_preprocessing`：下载 `maldi-2-control-02.zip` 并跑预处理 pipeline。源数据较大，.ibd 与 imzML 文件未上传仓库，仅保留预处理结果。
- 复现完整 msiflow：下载 `msi_if_registration.zip`，跑 `msi_if_registration`、`if_segmentation`、`molecular_signatures`。

#### 运行脚本

官方 demo 运行方式：

```bash
bash run_demo.sh
```

为同时覆盖预处理和完整数据，我们额外提供：

```bash
bash run_demo_control.sh
bash run_demo_full.sh
```

这里的control是指我们下载的是control_02数据集进行预处理流程复现。在复现过程终我们发现在官方提供的数据集终control组只有msi原始数据没有ifm原始数据，（推测是control组没有进行中性粒细胞染色，故没有进行ifm实验）因此无法进行msi_if_registration与后续流程的复现。full是指我们下载了完整的msi_if_registration数据集，可以复现完整的msi_if_registration、if_segmentation与molecular_signatures流程。

为验证即插即用能力并优化关键环节（MSI-IFM 配准、MSI 分割），我们在原流程上做了多项改进。

#### 多模态配准优化

- 重构 `antspy_registration.py`，新增参数以便选择 **Rigid**、**Affine**、**SyNRA** 三种配准模式，以及 **Mattes** 或 **CC** 相似度度量。实测在当前 demo 上，简单的 Rigid 配准取得最佳 IoU（0.7755），说明样本不存在明显弹性扭曲，刚性配准可避免过拟合。
- 增加质量评估脚本，自动生成 **Checkerboard**、**Overlay** 可视化，并基于 Ground Truth Mask 计算 Jaccard Index，实现定性+定量评估。
- 额外尝试 **ORB**、**SIFT** 特征点匹配并估计单应性，但在两种模态下均未得到可靠对应关系，说明互信息驱动的配准更适合当前数据。

示例：对比 SyNRA 与 Rigid 配准

```python
# SyNRA 配准
snakemake -F --snakefile msi_if_registration_flow/Snakefile \
  --configfile demo/data/msi_if_registration/config.yaml \
  --config data='demo/data/msi_if_registration' \
  --cores 1

echo "Baseline (SyNRA + Preprocess) Score:"
grep "Jaccard" demo/data/msi_if_registration/registered/mask_overlay_UPEC_12.svg

# Rigid 配准
python msi_if_registration_flow/scripts/antspy_registration.py \
  demo/data/msi_if_registration/fixed/umap/umap_grayscale_UPEC_12.tif \
  demo/data/msi_if_registration/moving/preprocessed/UPEC_12.tif \
  -af_chan 1 \
  -transform_type Rigid \
  -out_file demo/data/msi_if_registration/registered/UPEC_12_rigid_pre.tif

# 计算 IoU
python msi_if_registration_flow/scripts/binary_mask_comparison.py \
  demo/data/msi_if_registration/fixed/mask/UPEC_12.tif \
  demo/data/msi_if_registration/registered/UPEC_12_rigid_pre.tif \
  visualization_results/score_rigid_pre.svg \
  -mask_chan 3

echo "Ours (Rigid + Preprocess) Score:"
grep "Jaccard" visualization_results/score_rigid_pre.svg
```

特征点配准示例：

```python
python msi_if_registration_flow/scripts/feature_based_registration.py \
  <fixed_image_path> \
  <moving_image_path> \
  -af_chan 1 \
  -out_file <output_path> \
  -method ORB \
  -plot  # 可选：输出匹配连线图
```

#### 图像分割优化

- IF 分割：原先 Otsu 仅按亮度二值化，易把黏连细胞合并。引入 Watershed，代码位于 `if_segmentation_flow/scripts/segment_watershed.py`：

```bash
python if_segmentation_flow/scripts/segment_watershed.py \
  <input_image_path> \
  -output <output_path> \
  -chan_to_seg_list 2 \
  -sigma 1 \
  -min_distance 5
```

- MSI 分割：K-Means 假设簇形状近似凸球体，不适合复杂代谢区域。改用 Spectral Clustering，在低维嵌入上聚类以捕获任意形状簇，实现在 `msi_segmentation_flow/scripts/single_sample_segmentation.py`：

```bash
python msi_segmentation_flow/scripts/single_sample_segmentation.py \
  <input_imzML_path> \
  -result_dir <output_dir> \
  -cluster spectral \
  -n_clusters 3 \
  -method umap \
  -n_components 2
```

更多实验、参数与可视化详见 `docs/registration_experiment_report.md` 与 `docs/segmentation_optimization_report.md`。

### 自动报告

自研 `report.py` 支持生成 Markdown 报告并转换 tif 为 png，示例：

```bash
python report.py demo_full/data/ly6g_molecular_signatures
python report.py demo/data/Ly6G_signatures
```

脚本会在 `./report` 下生成对应报告并复制必要资源。

### 辅助 python 文件

部分 tif 过大，vscode 插件预览受限，可用 `show_tif.py` 转换预览：

```bash
python show_tif.py ./demo_full/data/msi_if_registration/moving/UPEC_15/uromask_UPEC_15.tif
```

需要同时查看 msi 与 ifm 叠加效果，可用 `show_Ly6G_on_msi.py`：

```bash
python show_Ly6G_on_msi.py \
  ./demo_full/data/msi_if_registration/fixed/umap/umap_grayscale_UPEC_14.tif \
  ./demo_full/data/msi_if_registration/moving/UPEC_14/Ly6G_UPEC_14.tif \
  --allow-large-images --resize-overlay
```

脚本会在 msi的tif文件 所在目录下输出 msi、ifm 及叠加的 png。

#### 组内分工
- 代码流程复现： 张章、朱昊东
- `registration`与`ifm`分割优化：张章
- 使用contro_02数据集复现`msi_preprocessing流程`与使用完整的msi_if_registration数据集复现`msi_if_registration`、`if_segmentation`、`molecular_signatures`流程： 朱昊东
- 自动报告脚本与辅助`tif`文件脚本开发： 朱昊东
- 论文汇报大纲梳理： 彭梓浥
- ppt撰写： 彭梓浥、张章、朱昊东
