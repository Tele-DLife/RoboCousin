# 第三方声明

[English](./THIRD_PARTY_NOTICES.md)

RoboCousin 使用、改编或链接第三方软件、模型与资产，并与部分第三方组件进行集成。
RoboCousin 的原创贡献依据仓库根目录的 MIT License 许可；下列第三方内容分别
适用其自身的许可条款。

## RoboTwin

- 项目：https://github.com/RoboTwin-Platform/RoboTwin
- 许可证：MIT
- 版权：Copyright (c) 2025 Tianxing Chen（陈天行）
- 许可证文本：[licenses/RoboTwin-MIT.txt](licenses/RoboTwin-MIT.txt)

RoboCousin 的部分代码与项目结构衍生自 RoboTwin，相关范围包括仿真器与任务代码、
配置结构，以及 `assets/files/50_tasks.gif`、
`assets/files/domain_randomization.png` 等媒体文件。RoboCousin 依照 MIT License
的要求保留 RoboTwin 的版权声明与许可声明。

## Digital Cousins

- 项目：https://github.com/cremebrule/digital-cousins
- 论文：https://arxiv.org/abs/2410.07408
- 许可证：Apache License 2.0
- 许可证文本：[licenses/Digital-Cousins-Apache-2.0.txt](licenses/Digital-Cousins-Apache-2.0.txt)

`cousin_layout/` 中的部分代码源自或改编自 Digital Cousins 的真实场景提取流程，
以及相关模型和工具实现。

保留 Digital Cousins 代码的文件包括：

- `cousin_layout/models/perspective_fields.py`
- `cousin_layout/models/visual_encoder.py`
- `cousin_layout/utils/transform_utils.py`

为 RoboCousin 改编或修改的文件包括：

- `cousin_layout/configs/default.yaml`
- `cousin_layout/models/clip.py`
- `cousin_layout/models/depth_anything_v2.py`
- `cousin_layout/models/dino_v2.py`
- `cousin_layout/models/feature_matcher.py`
- `cousin_layout/models/gpt.py`
- `cousin_layout/models/grounded_sam_v2.py`
- `cousin_layout/modified_pipeline/extraction.py`
- `cousin_layout/pipeline/extraction.py`
- `cousin_layout/utils/processing_utils.py`
- `cousin_layout/utils/scene_utils.py`

RoboCousin 的修改主要包括调整模型访问方式、增加 RoboTwin 资产匹配、结合相机
视角选择物体朝向、导出相对布局，以及建立显式的 `ontop` 支撑关系。相关修改文件
均保留来源说明，并注明修改由 RoboCousin 贡献者完成。围绕上述组件、专为
RoboCousin 开发的代码不属于 Digital Cousins 上游代码，亦不会作此标示。

## EmbodiedGen

- 项目：https://github.com/HorizonRobotics/EmbodiedGen
- 来源文件：https://github.com/HorizonRobotics/EmbodiedGen/blob/master/embodied_gen/utils/gpt_clients.py
- 许可证：Apache License 2.0
- 版权：Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
- 许可证文本：[licenses/EmbodiedGen-Apache-2.0.txt](licenses/EmbodiedGen-Apache-2.0.txt)

`envs/utils/qwen_clients.py` 改编自 EmbodiedGen 的
`embodied_gen/utils/gpt_clients.py`。RoboCousin 对客户端进行了重命名和精简，
删除本项目未使用的特定提供方功能，并接入 RoboCousin 的房间布局配置。修改后的
文件继续保留上游版权声明和 Apache-2.0 许可声明。

## Git 子模块

以下项目以 Git 子模块形式引入。初始化子模块时，Git 将从相应上游仓库检出源代码；
该等源代码分别适用各上游项目自身的许可证。

| 组件 | 上游项目 | 许可证／重要限制 |
| --- | --- | --- |
| PerspectiveFields | https://github.com/jinlinyi/PerspectiveFields | Adobe Research License Terms：仅限非商业用途；不得再分发 Research Materials。RoboCousin 仅保存子模块引用，未在本仓库中复制该等材料。 |
| Depth Anything V2 | https://github.com/DepthAnything/Depth-Anything-V2 | 代码采用 Apache-2.0。RoboCousin 使用需单独下载的 Metric Hypersim Large 权重；其官方模型仓库同样采用 Apache-2.0。 |
| GroundingDINO | https://github.com/IDEA-Research/GroundingDINO | Apache-2.0。 |
| SAM 2 | https://github.com/facebookresearch/segment-anything-2 | Apache-2.0。 |
| PyTorch3D | https://github.com/facebookresearch/pytorch3d | BSD License。 |

初始化后，各子模块均保留其自身的许可证文件。用户应审阅实际检出版本所附的
许可证文件及相关条款。

## 单独下载的模型与资产

RoboCousin 不就用户单独下载或在运行时获取的模型权重、API 服务、数据集或资产
授予任何许可或其他权利。PerspectiveFields 研究材料、DINOv2 权重、语言或视觉
模型服务，以及 RoboTwin 运行时资产，仍分别适用各提供方规定的条款。

RoboCousin 主要面向非商业研究用途。即使仅用于非商业研究，用户仍须履行第三方
条款规定的署名、声明保留、再分发限制及其他义务。

## 不构成认可或重新许可

对第三方项目的引用仅用于说明技术来源，不应被解释为相关第三方对 RoboCousin 的
认可、赞助或背书。仓库根目录的 MIT License 仅适用于 RoboCousin 的原创贡献，
不会将任何第三方组件、模型、权重或资产重新许可为 MIT License。
