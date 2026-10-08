# Custom asset examples

This folder contains complete, ready-to-copy custom object examples that
match the layout expected by RoboCousin under `our_assets/`.

The bundled actor categories are:

| Category | Instances |
| --- | ---: |
| `apple` | 5 |
| `battery` | 7 |
| `bottle drink` | 14 |
| `box` | 3 |
| `canned food` | 8 |
| `marker` | 4 |
| `orange` | 3 |
| `toy` | 13 |

The bundled non-actor categories are:

| Category | Instances |
| --- | ---: |
| `camera` | 5 |
| `computer monitor` | 3 |
| `keyboard` | 6 |
| `laptop` | 7 |

Each instance directory contains:

- `sample.urdf` — simulation metadata and mesh references
- `model_data.json` — contact points and scale/extents (required for grasping)
- `mesh/sample.obj` or `mesh/sample.glb` — visual mesh
- `mesh/sample_collision.obj` or `mesh/sample_collision.glb` — collision mesh

All bundled meshes, textures, URDF files, and metadata were generated for this
project through the
[3DGENERATION](https://github.com/Tele-DLife/3DGENERATION) workflow. They are
project example assets and were not copied from a third-party asset repository.

Copy the example into your local runtime assets directory:

```bash
mkdir -p our_assets/actor
cp -r examples/custom_assets/actor/apple our_assets/actor/
```

To copy every bundled actor category instead:

```bash
cp -r examples/custom_assets/actor/. our_assets/actor/
```

To copy every bundled non-actor category:

```bash
mkdir -p our_assets/non-actor
cp -r examples/custom_assets/non-actor/. our_assets/non-actor/
```

Then run the public quick-start task:

```bash
python script/collect_data_complete.py pick_up demo_complete
```

If you add a new object from scratch, generate `model_data.json` with:

```bash
python script/auto_generate_contact_points.py \
  --object_dir our_assets/actor/<object_name>/0
```
