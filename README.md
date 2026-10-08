# RoboCousin

[简体中文](./README.zh-CN.md)

RoboCousin extends [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) with tools for building tabletop manipulation datasets from custom objects and real-world scene layouts.

The repository focuses on three workflows:

- preparing custom 3D assets and generating grasp contact points;
- generating room layouts and collecting manipulation trajectories in simulation;
- reconstructing a relative desktop layout from an RGB image and replaying it in RoboTwin.

RoboCousin currently focuses on upper-body tabletop manipulation tasks such as pick and place. Support for mobile-base navigation and policy training/evaluation workflows is coming soon.

<p align="center">
  <img src="docs/images/RC_thumbnail.jpg" alt="RoboCousin overview" width="100%">
</p>

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Custom asset preparation](#custom-asset-preparation)
- [Room-based data collection](#room-based-data-collection)
- [Desk Cousin: image-to-layout](#desk-cousin-image-to-layout)
- [Optional 3DGENERATION integration](#optional-3dgeneration-integration)
- [Repository structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

- Linux with an NVIDIA GPU and a CUDA/PyTorch environment compatible with RoboTwin
- Conda or another Python environment manager
- Python 3.10
- Git with submodule support
- `ffmpeg` for video export
- `xdotool` for optional viewer-window placement on X11

For the underlying simulator requirements, see the [RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

### 1. Clone the repository and initialize submodules

```bash
git clone --recurse-submodules https://github.com/Tele-DLife/RoboCousin.git
cd RoboCousin

# Safe to run if the repository was cloned without --recurse-submodules.
git submodule update --init --recursive
```

The optional image-to-layout workflow requires the PerspectiveFields,
Depth-Anything-V2, GroundingDINO, SAM 2, and PyTorch3D submodules.

### Third-party access and licensing

Initializing submodules and running installation or download scripts may cause
the user's local environment to retrieve components directly from third-party
repositories and services. Users obtain these components from the respective
third parties and should review and comply with the applicable upstream terms.
If a component requires account registration, license acceptance, or access
approval, complete that process directly with the relevant third party.

**PerspectiveFields** is governed by the Adobe Research License Terms. It may be
used only for noncommercial purposes, such as research, teaching, and testing,
and the Research Materials may not be redistributed. Initialize or obtain it
directly from the upstream project only after confirming that the intended use
meets those terms.

RoboCousin's original contributions are MIT-licensed. Third-party components
remain subject to their own terms and are not relicensed by RoboCousin. See the
[Third-Party Notices](./THIRD_PARTY_NOTICES.md) for details.

The default depth-estimation module uses the separately downloaded
[Depth Anything V2 Metric Hypersim Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large)
checkpoint. Its official model repository is licensed under Apache-2.0.

### 2. Create the Python environment

```bash
conda create -n robocousin python=3.10 -y
conda activate robocousin
```

### 3. Install dependencies

```bash
bash script/_install.sh
```

The installer:

- installs `script/requirements.txt`;
- installs the matching and rendering dependencies;
- installs or updates PyTorch3D and cuRobo;
- applies RoboCousin's required SAPIEN and mplib runtime patches;
- reports missing optional system tools and submodules.

Because the installer patches files in the active Python environment, use a dedicated environment for RoboCousin.

On Ubuntu, the optional system packages can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg xdotool
```

### 4. Download simulation assets

Download the robot models, built-in objects, and background textures:

```bash
bash script/_download_assets.sh
```

Before enabling `use_custom_objects`, copy the example from
`examples/custom_assets/` into `our_assets/`:

```bash
mkdir -p our_assets/actor
cp -r examples/custom_assets/actor/apple our_assets/actor/
```

See the [custom asset example guide](examples/custom_assets/README.md) for the
full file layout.

## Quick start

`demo_complete` enables `use_custom_objects` by default. After copying the
example above, run:

```bash
python script/collect_data_complete.py pick_up demo_complete
```

To run without custom assets, disable `use_custom_objects` in
`task_config/demo_complete.yml` or use another task configuration.

Task behavior is controlled by `task_config/demo_complete.yml`. See the
[RoboTwin usage documentation](https://robotwin-platform.github.io/doc/usage/index.html)
for general configuration options.

## Custom asset preparation

RoboCousin expects each custom object instance under `our_assets/<bucket>/<name>/<id>/`
to contain at least a URDF, contact-point metadata, and mesh files:

```text
our_assets/
  actor/
    apple/
      0/
        sample.urdf
        model_data.json
        mesh/
          sample.obj                 # or sample.glb
          sample_collision.obj       # or sample_collision.glb
      1/
      2/
      3/
      4/
  non-actor/
    laptop/0/
      sample.urdf
      model_data.json
      mesh/
        sample.obj                 # or sample.glb
        sample_collision.obj       # or sample_collision.glb
  room/
    desk/1/
      sample.urdf
      model_data.json
      mesh/
        sample.glb
```

- `sample.urdf` provides the simulation-loading configuration and mesh scale.
- `model_data.json` stores contact points, scale, and bounding-box dimensions;
  it is required when grasping custom objects.
- Visual and collision meshes live under `mesh/`. OBJ pairs may also include
  `material.mtl` and texture files.

The repository ships ready-to-copy `actor` examples under
`examples/custom_assets/actor/`. The default `apple` example contains five
instances numbered `0` through `4`; additional examples cover batteries,
bottled drinks, boxes, canned food, markers, oranges, and toys. Copy the
categories you need into `our_assets/actor/` as shown in the setup section
above.

Ready-to-copy `non-actor` examples are available under
`examples/custom_assets/non-actor/`, covering cameras, computer monitors,
keyboards, and laptops.

Before using a newly imported graspable object, generate its contact-point
metadata if `model_data.json` is missing.

Process all custom assets:

```bash
python script/auto_generate_contact_points.py --all
```

Process one asset branch:

```bash
python script/auto_generate_contact_points.py \
  --assets_branch_dir our_assets/actor
```

Process one instance:

```bash
python script/auto_generate_contact_points.py \
  --object_dir our_assets/actor/coke/0 \
  --height_ratio 0.6 \
  --vertical_ratio 0.3
```

The command writes `model_data.json` into each processed instance directory.
Add `--no_skip_existing_model_data` to overwrite an existing file.

Custom-object support is task-specific. `pick_up`, `place_bread_basket`, and `place_object_stand` are the best starting points for new asset categories. Other tasks may require task-specific grasp, placement, and success-condition tuning.

## Room-based data collection

The primary entry point is `script/collect_data_complete.py`.

```bash
python script/collect_data_complete.py TASK_NAME CONFIG_NAME [options]
```

Useful options include:

- `--room-type`: select a room family such as `livingroom`, `bedroom`, or `diningroom`;
- `--room-layout-index`: select a layout within the room family;
- `--room-layout-json`: load an explicit layout JSON file;
- `--generate-room-layout --room-prompt`: generate a layout before collection;
- `--use-cousin-coordinate`: use a layout generated by the Desk Cousin workflow;
- `--render-freq`: control rendering frequency during collection.

For objects prepared under `threeD-generation/obj/`, the compatibility script
`threeD-generation/run_obj_data_collection.py` runs contact-point generation,
configuration creation, and data collection in sequence:

```bash
python threeD-generation/run_obj_data_collection.py --room-type livingroom
```

## Desk Cousin: image-to-layout

This optional workflow estimates a relative tabletop layout from one RGB image
and matches the detected objects against local `our_assets`. It then determines
the orientation of each instance, builds support relationships for stacked
objects, and exports a layout that RoboTwin can preview or execute.

It builds on the real-world extraction stage from [Digital Cousins](https://github.com/cremebrule/digital-cousins) and adds RoboTwin asset matching, camera-aware orientation selection, and explicit `ontop` support graphs.

The retained and adapted Digital Cousins components remain licensed under the
Apache License 2.0. Files modified for RoboCousin are identified in their source
headers. See the [Third-Party Notices](./THIRD_PARTY_NOTICES.md) for the affected
scope and license text.

<p align="center">
  <img src="docs/images/cousinsss.jpg" alt="Real tabletop scenes and generated digital cousins" width="100%">
</p>

### Additional prerequisites

1. Initialize all Git submodules as described above.
2. Download the required model checkpoints to a local directory.
3. Set the checkpoint directory:

```bash
export COUSIN_LAYOUT_CHECKPOINT_DIR=/path/to/checkpoints
```

4. To use captioning or recaptioning, set the API key in
   `cousin_layout/configs/local_keys.yaml`:

```yaml
pipeline:
  RealWorldExtractor:
    call:
      gpt_api_key: YOUR_API_KEY
```


### Generate a layout

```bash
python cousin_layout/scripts/image_to_relative_layout_with_matching.py \
  --input-image-path /path/to/desk.png \
  --save-dir cousin_layout/desk_layout/my_run \
  --snapshot-overwrite \
  --verbose
```

Important outputs include:

- `step_1_output_info.json`;
- `relative_layout/relative_layout_ontop_desktable.json`.

Useful options:

- `--semantic-model bert|qwen`: choose the class-name matching backend;
- `--snapshot-step-degrees`: control the orientation search interval;
- `--snapshot-distance-factor`: control snapshot camera distance;
- `--support-footprint-slice-frac`: tune support detection for tall or irregular objects.

### Preview the layout

```bash
python script/preview_cousin_layout.py \
  --task-config demo_complete \
  --input_layout cousin_layout/desk_layout/my_run/relative_layout/relative_layout_ontop_desktable.json
```

To call the preview from a UI, use the compatibility script:

```bash
python threeD-generation/preview_cousin_layout_ui.py \
  --input_layout cousin_layout/desk_layout/my_run/relative_layout/relative_layout_ontop_desktable.json
```

Optional viewer settings:

```bash
export ROBOTWIN_VIEWER_RES=960,540
export ROBOTWIN_VIEWER_PLACEMENT=bottom_left
export ROBOTWIN_VIEWER_MINIMAL_UI=1
```

### Collect data with the generated layout

Set the following fields in `task_config/demo_complete.yml`:

```yaml
use_cousin_coordinate: true
cousin_target_labels: [OBJECT_LABEL]
cousin_relative_layout_json: cousin_layout/desk_layout/my_run/relative_layout/relative_layout_ontop_desktable.json
```

Then run:

```bash
python script/collect_data_complete.py pick_up demo_complete
```

## Optional 3DGENERATION integration

RoboCousin provides compatibility scripts for
[3DGENERATION](https://github.com/Tele-DLife/3DGENERATION) under
`threeD-generation/`, but does not include the 3DGENERATION WebUI. The English
and Chinese WebUI entry points are `apps/app_demo_Eng.py` and
`apps/app_demo_CHN.py` in the
[3DGENERATION repository](https://github.com/Tele-DLife/3DGENERATION).

If 3DGENERATION is installed locally, configure it with the path to this
RoboCousin checkout. It can then call the asset extraction and synchronization,
Desk Cousin generation, scene preview, and data collection workflows. The two
repositories can be stored anywhere; no fixed `/home/...` layout is required.

The projects exchange layout results through
`relative_layout_ontop_desktable.json`. RoboCousin's command-line workflows do
not require the 3DGENERATION WebUI.

## Scene rendering and export

Render a generated scene:

```bash
python threeD-generation/ui_generated_scene_render_ui.py --steps 200
```

Export it as GLB:

```bash
python threeD-generation/ui_generated_scene_render_ui.py \
  --export \
  --export-path outputs/scene_export.glb
```

## Repository structure

```text
RoboCousin/
  assets/                 # downloaded RoboTwin runtime assets
  cousin_layout/          # image extraction, matching, and layout generation
  deps/                   # external model repositories as submodules
  envs/                   # simulation tasks and environment utilities
  our_assets/             # custom object assets
  script/                 # installation, processing, preview, and collection tools
  task_config/            # public task configurations
  threeD-generation/      # RoboCousin-side compatibility and rendering wrappers
```

## Troubleshooting

- **A submodule import fails:** run `git submodule update --init --recursive` and verify the expected repository exists under `deps/`, `PerspectiveFields/`, or `envs/pytorch3d/`.
- **A robot, object, or texture cannot be found:** run `bash script/_download_assets.sh` and verify the downloaded directories under `assets/`.
- **The Desk Cousin workflow cannot find a model:** verify `COUSIN_LAYOUT_CHECKPOINT_DIR` and the required checkpoint filenames.
- **Video export fails:** verify that `ffmpeg` is available in `PATH`.
- **Viewer placement does not work:** install `xdotool`; it is only applicable to supported X11 sessions.
- **The first run is slow:** model initialization and local caches can take significant time.

When reporting a problem, include the command, task configuration name, Python/CUDA/PyTorch versions, and the complete error traceback. Do not include API keys or private dataset paths.

## Acknowledgements

RoboCousin builds on or integrates ideas and components from:

- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
- [Digital Cousins (ACDC)](https://github.com/cremebrule/digital-cousins) and its [paper](https://arxiv.org/abs/2410.07408)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields)
- [DINOv2](https://github.com/facebookresearch/dinov2)

Please cite the upstream projects relevant to your use and cite RoboCousin for its custom-asset, Desk Cousin, and RoboTwin integration work.

## License

RoboCousin's original contributions are released under the [MIT License](./LICENSE). Portions derived from third-party projects remain subject to their original licenses. See [Third-Party Notices](./THIRD_PARTY_NOTICES.md) for details.

The MIT License for RoboCousin's original contributions does not replace the licenses of bundled, linked, or separately downloaded dependencies, model weights, or assets. In particular, PerspectiveFields is limited to noncommercial use and prohibits redistribution of its Research Materials.
