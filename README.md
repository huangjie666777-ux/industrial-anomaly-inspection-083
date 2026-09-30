# Industrial anomaly inspection

面向工厂质检员的工业表面异常定位 Web 应用：上传正常产品图建立参考记忆库，用**独立**校准正常图确定阈值，检测新图中的划痕或异物，并在原图上叠加固定标度的距离热图。纯 CPU、离线运行，不调用任何云服务。

## 运行

```sh
.venv/bin/python -m uvicorn inspection.app:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

运行测试：`.venv/bin/python -m pytest -q`
生成可下载的合成示例文件：`.venv/bin/python scripts/generate_examples.py`（页面内也可直接调用 `/api/examples`）。

## 方法

- 预处理：完整图双线性缩放到 224×224（不中心裁剪），按 ImageNet 均值/方差归一化。
- 特征：冻结的 ResNet18（IMAGENET1K_V1，权重在 `models/`），取 layer2(128 维, 28×28) 双线性对齐到 14×14，与 layer3(256 维, 14×14) 拼接，得到 196 个 384 维局部描述子。
- 记忆库：对参考局部描述子做确定性贪心最远点选择，至多 256 项；每步只计算新点到池的距离，不构造全样本距离矩阵。
- 检测：每个局部到记忆库的最近欧氏距离构成距离图，最大值为图像分数。
- 阈值：仅由独立校准正常图的分数计算 95% 线性分位数，分数**严格大于**阈值才判异常；参考图绝不用于校准。
- 热图：14×14 距离图双线性还原到原图尺寸，颜色上限固定为阈值的两倍；阈值为 0 时改用校准图最高分（带 `1e-6` 下限），跨图使用同一标度，不逐图拉伸。
- 持久化：记忆库、阈值、校准分数和预处理信息原子写入 `data/state.pt`，重启恢复；重建失败保留旧库；单次检测在锁内快照同一版本的库与阈值，不混用两版状态。
- 去重：按缩放后 RGB 像素 SHA256 判定，参考组与校准组不可含相同图像内容。

## 模块组织

- `src/inspection/preprocessing.py`：图像校验、解码、归一化、距离图上采样
- `src/inspection/features.py`：冻结 ResNet18 的 layer2/layer3 局部描述子
- `src/inspection/memorybank.py`：贪心最远点选择、最近邻距离、分位数、热图上限
- `src/inspection/store.py`：样本与状态存储、原子持久化、版本一致性
- `src/inspection/synth.py`：正常/划痕/异物合成示例（仅演示与测试用）
- `src/inspection/app.py`：FastAPI 接口
- `web/`：原生 JavaScript 单页（预览、删除、透明度调节、热图渲染）

## 上传限制

PNG/JPEG，单文件 ≤ 10MB，边长 32–8192 像素；超限或格式错误会在页面与 API 返回中文错误信息。

Python 3.10.12 is available in WSL. Use `.venv/bin/python` for this project. PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, FastAPI 0.115.12 and other exact dependencies are installed in the local `.venv`; see requirements.lock. Browser code may use native JavaScript without a Node build toolchain.

The pretrained ResNet18 IMAGENET1K_V1 weights are already present at `models/resnet18-f37072fd.pth`. Source and checksum are in models/manifest.json. No cloud API, credentials or GPU are required. The binary is omitted from Git and included in the prepared local directory.

To recreate the dependency environment after a fresh Git clone:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth -o models/resnet18-f37072fd.pth
```

Verify the downloaded file against the SHA256 in models/manifest.json. Package and weight downloads require network access; application inference can use the prepared local assets offline. The intended image preprocessing differs from the classification weight's default center crop: follow the requested whole-image preprocessing when implementing the application.
