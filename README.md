# 赛题九 A、B、C 从零训练

上传评测用的 Linux/amd64 镜像在 `delivery/zhixun-contest-abc-retrain-v1.tar`（本地交付文件），文件摘要见 [SHA256SUMS.txt](delivery/SHA256SUMS.txt)。源码、训练记录和预检记录都在本目录中。

## 从零训练结果

三个任务各训练一个 NumPy CPU 模型，随机种子均为 42。训练读取官方包的 `public/train.jsonl`，并按 `training.json` 核对训练集、验证集的 SHA-256 和行数。训练过程从随机权重和空 Adam 状态开始。公开验证数据用于选配置和训练轮数。

| 任务 | 官方训练／验证样本 | 整份公开验证集 | 最终方案 |
| --- | ---: | --- | --- |
| A 气体分类 | 445／1,244 | Macro-F1 0.8809，准确率 94.69% | 64 单元 MLP，反双曲正弦压缩、类别平衡采样、乙醛／乙醇径向基分类头 |
| B 轴承 RUL | 7,032／1,679 | RMSE 431.38 分钟，MAE 357.85 分钟 | 48 单元 MLP，压缩振动特征，降低长轴承记录在采样中的权重 |
| C 发动机 RUL | 113,089／23,049 | RMSE 46.60 循环，MAE 32.58 循环 | 64 单元 MLP，按运行工况归一化传感器特征 |

A 的主模型单独使用时，乙醛为 3/109，Macro-F1 为 0.7415。加入分类头后，乙醛为 81/109，精确率 82.65%，召回率 74.31%，乙醛 F1 为 0.7826。仍有 27 条乙醛被判为乙醇，15 条乙醇被判为乙醛。分类头用官方训练集拟合，采用每个传感器的上升与衰减响应组合。两个公开验证区段均参与分类头筛选，A 的结果属于公开开发验证。 [UCI 数据集与特征顺序](https://doi.org/10.24432/C5RP6W)

B、C 按设备分割其公开验证集，选择配置时预留一部分完整设备。B 的预留部分只包含一根轴承、42 条快照，RMSE 362.05 分钟。C 的预留部分有 11,375 条记录，RMSE 48.50 循环。评测平台反馈用于判断隐藏集效果。

## 镜像的两个入口

- 从零训练得到的三套模型内置在 `/opt/retrain/models/A`、`/opt/retrain/models/B`、`/opt/retrain/models/C`。每套有 `checkpoint.npz` 和可直接运行的 `predict.py`，也可从 [results/release](results/release/) 单独取用。支持逐行单条请求或有序批量请求。
- SDK 1.0.4 的 `/app/harness.py train` 每轮必须收到父检查点。镜像会核验并实际使用平台下发的父权重和 Adam 状态，在当轮公开训练数据上再训练 1–2 个 epoch。候选主指标有改善时才选候选，否则保留父模型。A 会测试由当轮训练数据拟合的乙醛分类头。SDK 网关仅提供小范围的训练轮数决策，本地从零训练和预检不调用真实模型服务。

## 复现训练与预检

在 `赛题九交接` 目录执行。复跑前给训练结果与预检分别指定新的输出目录。

```powershell
docker load --input retrain/delivery/zhixun-contest-abc-retrain-v1.tar
$official = (Resolve-Path 'zhixun-engine-v2/vendor/baselines').Path
$output = (New-Item -ItemType Directory 'retrain/replay-run').FullName
docker run --rm --platform linux/amd64 --cpus 4 --memory 4g --memory-swap 4g --network none `
  --mount "type=bind,source=$official,target=/official,readonly" `
  --mount "type=bind,source=$output,target=/replay" `
  --entrypoint python3 zhixun-contest:abc-retrain-v1 /app/train_models.py /official /replay A B C
python -B retrain/src/preflight.py zhixun-contest:abc-retrain-v1 `
  zhixun-engine-v2/vendor/baselines retrain/results/release retrain/preflight/replay-run
```

预检在 4 核、4 GiB、交换空间为 0 的 Docker cgroup 中运行，并读取 OpenBLAS 的实际线程数，确认始终使用 1 个线程。训练重复使用归一化后的矩阵，A 的预测缓存分类头的支持向量和系数。每个任务预检两轮 SDK 协议，验证父检查点摘要、归一化状态和 Adam 更新，再验证只读挂载下的独立预测、Harness 预测以及批量结果的顺序。记录见 [预检汇总](preflight/release/summary.json) 和 [资源检查](preflight/release/resources.txt)。训练配置、每轮指标和混淆矩阵见 [A](results/release/A.json)、[B](results/release/B.json)、[C](results/release/C.json)。

