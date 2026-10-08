# Third-Party Notices

[简体中文](./THIRD_PARTY_NOTICES.zh-CN.md)

RoboCousin uses, adapts, or links to third-party software, models, and assets,
and integrates with certain third-party components. RoboCousin's original
contributions are licensed under the repository's root MIT License. The
third-party materials identified below remain subject to their respective
terms.

## RoboTwin

- Project: https://github.com/RoboTwin-Platform/RoboTwin
- License: MIT
- Copyright: Copyright (c) 2025 Tianxing Chen (陈天行)
- License text: [licenses/RoboTwin-MIT.txt](licenses/RoboTwin-MIT.txt)

Portions of RoboCousin's code and project structure are derived from RoboTwin.
This includes simulator and task code, configuration structure, and media files
such as `assets/files/50_tasks.gif` and
`assets/files/domain_randomization.png`. RoboCousin retains RoboTwin's copyright
and permission notices as required by the MIT License.

## Digital Cousins

- Project: https://github.com/cremebrule/digital-cousins
- Paper: https://arxiv.org/abs/2410.07408
- License: Apache License 2.0
- License text: [licenses/Digital-Cousins-Apache-2.0.txt](licenses/Digital-Cousins-Apache-2.0.txt)

Parts of `cousin_layout/` originate from or are adapted from the Digital Cousins
real-world scene extraction pipeline and related model and utility
implementations.

Files retaining Digital Cousins code include:

- `cousin_layout/models/perspective_fields.py`
- `cousin_layout/models/visual_encoder.py`
- `cousin_layout/utils/transform_utils.py`

Files adapted or modified for RoboCousin include:

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

RoboCousin's changes include adapting model access, adding RoboTwin asset
matching, selecting object orientation with reference to the camera view,
exporting relative layouts, and establishing explicit `ontop` support
relationships. Modified files retain source notices and identify the changes as
work by RoboCousin contributors. Code developed specifically for RoboCousin
around these components is not presented as upstream Digital Cousins code.

## EmbodiedGen

- Project: https://github.com/HorizonRobotics/EmbodiedGen
- Source file: https://github.com/HorizonRobotics/EmbodiedGen/blob/master/embodied_gen/utils/gpt_clients.py
- License: Apache License 2.0
- Copyright: Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
- License text: [licenses/EmbodiedGen-Apache-2.0.txt](licenses/EmbodiedGen-Apache-2.0.txt)

`envs/utils/qwen_clients.py` is adapted from EmbodiedGen's
`embodied_gen/utils/gpt_clients.py`. RoboCousin renames and narrows the client,
removes provider-specific functionality that is not used here, and integrates
the client with RoboCousin's room-layout configuration. The upstream copyright
and Apache-2.0 license notice are retained in the modified file.

## Git submodules

The following projects are included as Git submodules. When initialized, their
source code is checked out from the respective upstream repositories and remains
subject to each upstream project's license.

| Component | Upstream | License / important restriction |
| --- | --- | --- |
| PerspectiveFields | https://github.com/jinlinyi/PerspectiveFields | Adobe Research License Terms: noncommercial use only; redistribution of the Research Materials is not permitted. RoboCousin stores only a submodule reference and does not copy those materials into this repository. |
| Depth Anything V2 | https://github.com/DepthAnything/Depth-Anything-V2 | Code is Apache-2.0. RoboCousin uses the separately downloaded Metric Hypersim Large checkpoint, whose official model repository is also licensed under Apache-2.0. |
| GroundingDINO | https://github.com/IDEA-Research/GroundingDINO | Apache-2.0. |
| SAM 2 | https://github.com/facebookresearch/segment-anything-2 | Apache-2.0. |
| PyTorch3D | https://github.com/facebookresearch/pytorch3d | BSD license. |

Each initialized submodule retains its own license files. Users should review
the license and related terms included with the exact revision checked out.

## Separately downloaded models and assets

RoboCousin grants no license or other rights to model checkpoints, API services,
datasets, or assets that users download separately or obtain at runtime.
PerspectiveFields research materials, DINOv2 weights, language or vision model
services, and RoboTwin runtime assets remain subject to the terms specified by
their respective providers.

RoboCousin is primarily developed for noncommercial research. Noncommercial
research use does not waive any attribution, notice-preservation,
redistribution, or other obligations imposed by third-party terms.

## No endorsement or relicensing

References to third-party projects identify technical provenance only and must
not be construed as endorsement, sponsorship, or approval of RoboCousin by any
third party. The root MIT License applies only to RoboCousin's original
contributions and does not relicense any third-party components, models,
weights, or assets.
