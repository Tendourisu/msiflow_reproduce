## msiflow 复现说明

### 环境配置

使用conda环境

```bash
conda env create --file msiflow_env.yaml
conda activate msiflow
```
### 代码复现
1. 下载代码

```bash
git clone https://github.com/your-repo/msiflow.git 
cd msiflow
```
2. 下载数据集

数据集网站：https://zenodo.org/records/11913042
由于在官方demo文件夹中提供的数据仅包含UPEC_12数据在经过msi_preprocessing处理后并截取了一部分区域，而非完整的UPEC_12数据
- 为了复现msi_preprocessing的结果，我们下载了maldi-2-control-02.zip并进行预处理pipeline复现。由于数据较大，大体积的.ibd与imzML文件未上传至github仓库中，仅保留