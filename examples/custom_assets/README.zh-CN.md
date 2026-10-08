# 自定义资产示例

本目录包含一组完整、可直接复制的自定义物体示例，结构与 RoboCousin 在
`our_assets/` 下期望的布局一致。

自带的 actor 类别如下：

| 类别 | 实例数 |
| --- | ---: |
| `apple` | 5 |
| `battery` | 7 |
| `bottle drink` | 14 |
| `box` | 3 |
| `canned food` | 8 |
| `marker` | 4 |
| `orange` | 3 |
| `toy` | 13 |

自带的 non-actor 类别如下：

| 类别 | 实例数 |
| --- | ---: |
| `camera` | 5 |
| `computer monitor` | 3 |
| `keyboard` | 6 |
| `laptop` | 7 |

每个实例目录包含：

- `sample.urdf` — 仿真元数据与 mesh 引用
- `model_data.json` — 接触点与 scale/extents（抓取必需）
- `mesh/sample.obj` 或 `mesh/sample.glb` — 视觉 mesh
- `mesh/sample_collision.obj` 或 `mesh/sample_collision.glb` — 碰撞 mesh

示例中的 apple mesh、纹理、URDF 和元数据均通过
[3DGENERATION](https://github.com/Tele-DLife/3DGENERATION) 工作流生成，作为
RoboCousin 的项目示例资产提供，并非复制自第三方资产仓库。

复制示例到本地运行时资产目录：

```bash
mkdir -p our_assets/actor
cp -r examples/custom_assets/actor/apple our_assets/actor/
```

如需复制全部自带 actor 类别：

```bash
cp -r examples/custom_assets/actor/. our_assets/actor/
```

如需复制全部自带 non-actor 类别：

```bash
mkdir -p our_assets/non-actor
cp -r examples/custom_assets/non-actor/. our_assets/non-actor/
```

然后运行公开 quick-start 任务：

```bash
python script/collect_data_complete.py pick_up demo_complete
```

如果你从零添加新物体，可用以下命令生成 `model_data.json`：

```bash
python script/auto_generate_contact_points.py \
  --object_dir our_assets/actor/<object_name>/0
```
