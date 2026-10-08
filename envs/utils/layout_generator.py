import os
import logging
from datetime import datetime
from envs.utils.qwen_clients import QWEN_CLIENT, QWENclient
import json
from pathlib import Path

try:
    from script.path_config import REPO_ROOT, ROOM_CONFIG_50ROOMS_DIR, UI_GENERATED_LAYOUT_JSON
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    ROOM_CONFIG_50ROOMS_DIR = (REPO_ROOT / "envs" / "room_config" / "50rooms").resolve()
    UI_GENERATED_LAYOUT_JSON = (REPO_ROOT / "envs" / "room_config" / "ui_generated_layout.json").resolve()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoomLayoutGenerator:
    """
    根据房间描述生成基于锚点（Anchor）逻辑的 JSON 布局配置文件。
    """

    def __init__(self, qwen_client: QWENclient):
        self.qwen_client = qwen_client
        # 定义核心 Prompt，强制模型理解相对位置关系
        # self.prompt_template = """
        # You are an expert interior design AI specialized in semantic layout generation. 
        # Your task is to design a furniture layout for a {room_description} using an anchor-based approach.

        # ### CRITICAL CONSTRAINTS (MUST FOLLOW):
        # 1. **STRICT WHITELIST**: You must ONLY select furniture categories from the provided list: [{furniture_categories}]. 
        # - DO NOT invent, infer, or add any furniture type NOT explicitly present in this list (e.g., if "bookshelf" is not in the list, DO NOT include it, even if the room seems to need one).
        # - If the provided list has fewer types than needed, simply use fewer items or duplicate existing types logically (e.g., two chairs), but NEVER introduce new categories.
        # 2. **Quantity**: Aim for 5 to 10 furniture items total. However, if the provided category list is too short to reach 5 items without repetition, prioritize strict adherence to the whitelist over the count.
        # 3. **Coordinate Logic**:
        # - The room center is (0.0, 0.0).
        # - Select 1 or more primary items as "anchors" (set `anchor`: None).
        # - All other items must be placed relative to an existing anchor using `rel_side` ("left", "right", "front", "back", "top").
        # - **IMPORTANT**: If `anchor` is None, set `rel_side` to null and `gap` to 0.0. Do not provide a side description for absolute placements.
        # - For "top" placement, `gap` must always be 0.0.

        # ### Layout Rules:
        # - **Rotation**: Use `rotation_deg` to ensure logical facing (e.g., chairs face desks, beds face away from walls or towards the room center).
        # - **Gap**: Use reasonable meter values (e.g., 0.5m for walking space).
        # - **Path Construction**: Construct the path strictly as: `{room_assets_root}/{{category}}/1/mesh/sample.glb` where `{{category}}` is the exact string from the whitelist.

        # ### Step-by-Step Reasoning (Internal Monologue):
        # Before generating the JSON, verify:
        # 1. Is every item's category strictly inside [{furniture_categories}]?
        # 2. Are all relative placements referencing a valid, previously defined anchor name?
        # 3. Did I avoid adding any unlisted items like "bookshelf" or "rug" unless they are in the input list?

        # ### Output Format:
        # Output ONLY a valid Python list assigned to the variable `CONFIG`. Do not include markdown code blocks (```), explanations, or any text outside the list.

        # Structure for each item:
        # {{
        #     "name": "UniqueName_Category_Index",  (e.g., "Bed_01", "Chair_Left_01")
        #     "category": "exact_category_from_list",
        #     "anchor": "NameOfAnchorItem" or None,
        #     "pos_abs": (x, y) if anchor is None else None,
        #     "rel_side": "left"|"right"|"front"|"back"|"top" or None (if anchor is None),
        #     "gap": float (0.0 if anchor is None or rel_side is "top"),
        #     "path": "{room_assets_root}/{{category}}/1/mesh/sample.glb",
        #     "rotation_deg": int,
        #     "fix_axis": "y_to_z",
        #     "is_static": True
        # }}

        # CONFIG = [
        #     // First item (Anchor)
        #     {{
        #         "name": "...",
        #         "category": "...",
        #         "anchor": None,
        #         "pos_abs": (0.0, 0.0),
        #         "rel_side": None,
        #         "gap": 0.0,
        #         "path": "...",
        #         "rotation_deg": 0,
        #         "fix_axis": "y_to_z",
        #         "is_static": True
        #     }},
        #     // Subsequent items
        #     ...
        # ]
        # """

        # self.prompt_template = """
        #     # 角色设定
        #     你是一位精通空间布局和功能性的资深室内设计师。

        #     # 任务描述
        #     请为类型为 {room_description} 的房间设计家具布局。
        #     你需要从提供的家具类别列表：["armchair","bed", "desk","desk lamp", "floor lamp", "flower vase", "nightstand", "potted plant", "rocking chair", "table", "wardrobe", "storage bin"] 中选择 8 到 12 件家具。
        #     布局逻辑采用“锚点逻辑（Anchor-based）”：先放置一个核心家具作为锚点（Anchor），其他家具根据与锚点家具的相对位置（左、右、前、后、上）和间距（gap）进行摆放。

        #     # 布局约束
        #     1. 空间中心：房间中心点 (x, y) 为 (0.0, 0.0)。假设房间是一个标准大小的空间。
        #     2. 锚点规则：
        #     - 选择主要的家具（至少有一个，可以根据总数量设置多个）的 "anchor" 为 None，并给定 "pos_abs" 绝对坐标。
        #     - 其他家具必须指定一个已存在的家具名称作为 "anchor"。
        #     - 相对位置（rel_side）必须是 ["left", "right", "front", "back", "top"] 之一。
        #     - 间距（gap）：以米为单位。若 rel_side 为 "top"，则 gap 必须为 0.0。前后距离最远1米，左右距离最远2米。挨着的关系比如床头柜和床，gap=0.02。
        #     3. 真实性：家具摆放需符合生活逻辑（例如：床头靠墙，床头柜在床侧，椅子面向书桌等）。
        #     4. 旋转角度：使用 "rotation_deg"（0-360度）描述家具朝向（90度是面朝右，-90度是面朝左）。anchor物体前的("font")物体可能需要朝着anchor物体。

        #     # 输出要求
        #     仅输出 Python 列表代码，确保格式可以直接被解析。
        #     路径格式："{room_assets_root}/{category}/1/mesh/sample.glb"

        #     CONFIG = [
        #         {
        #             "name": "Bed_1",
        #             "category": "bed",
        #             "anchor": None,
        #             "pos_abs": (0.0, -1.0),
        #             "path": "{room_assets_root}/bed/1/mesh/sample.glb",
        #             "rotation_deg": 0,
        #             "fix_axis": "y_to_z",
        #             "is_static": True
        #         },
        #         {
        #             "name": "Nightstand_L",
        #             "category": "nightstand",
        #             "anchor": "Bed_1",
        #             "rel_side": "left",
        #             "gap": 0.02,
        #             "path": "{room_assets_root}/nightstand/1/mesh/sample.glb",
        #             "rotation_deg": 0,
        #             "fix_axis": "y_to_z",
        #             "is_static": True
        #         },
        #         {
        #             "name": "Desk_Lamp",
        #             "category": "desk lamp",
        #             "anchor": "Desk_1",
        #             "rel_side": "top",
        #             "gap": 0.0,
        #             "path": "{room_assets_root}/desk lamp/1/mesh/sample.glb",
        #             "rotation_deg": 0,
        #             "fix_axis": "y_to_z",
        #             "is_static": True
        #         }
        #         # ... 继续生成其他 3-8 件家具
        #     ]
        # """

        self.prompt_template = """
            # 角色设定
            你是一位精通空间布局和功能性的资深室内设计师。

            # 任务描述
            请为类型为 {room_description} 的房间设计家具布局。

            ## 约束：必须根据房间类型严格选择家具
            1. **如果房间为 卧室**：必须仅从 ["bed", "desk", "desk lamp", "floor lamp", "flower vase", "nightstand", "potted tree", "small potted plant", "table lamp", "rocking chair", "chair", "wardrobe", "storage bin"] 中选择5-10件。
            2. **如果房间为 客厅**：必须仅从 [ "sofa", "floor lamp", "potted tree", "small potted plant", "coffee table", "flower vase","table lamp","armchair", "storage bin"] 中选择 6-8 件。
            3. **如果房间为 餐厅**：必须仅从 ["dining table", "dining chair", "wall cabinet", "base cabinet", "microwave", "pot", "refrigerator", "storage cabinet"] 中选择 6-8 件。
            - 重点强调：所有 `category` 必须来自于上述对应房间类型的列表，不得出现跨房间家具（例如客厅严禁出现 bed）。

            # 布局约束
            1. 空间中心：中心点 (x,y) 为 (0.0, -1.5)。y为-3.0时靠在后墙。
            2. 锚点规则：
               - 核心大件家具（如 bed，sofa，wardrobe）使用 `anchor: None` 和 `pos_abs`。都面朝中心。
               - 其他物体根据摆放位置选择参照物来摆放。比如桌子以核心大件家具床为参照物了，椅子以桌子为参照物。
            3. 朝向与旋转逻辑 (关键)：
                - **面向中心规则 (针对 pos_abs 大件家具的硬性规定)**：
                    * **如果在右墙**：当 `pos_abs` 的 x 为负数（如 -1.5），`rotation_deg` 必须设为 **-90**（或 270），使其向左面朝中心。
                    * **如果在左墙**：当 `pos_abs` 的 x 为正数（如 1.5），`rotation_deg` 必须设为 **90**，使其向右面朝中心。
                    * **如果在前墙**：当 `pos_abs` 的 y 为负数（如 2.0），`rotation_deg` 必须设为 **0**，使其向北面朝中心。
                    * **如果在后墙**：当 `pos_abs` 的 y 为正数（如 -2.0），`rotation_deg` 必须设为 **180**，使其向南面朝中心。
                              - **桌椅动态方位规则 (确保人从房间中心坐向桌子)**：
               - **桌椅耦合规则 (重要)**：
                 * **原则**：椅子必须摆放在桌子靠近房间中心的一侧。
                        * **条件映射**：
                        1. 如果桌子 `rotation_deg` 是 **180** (在 Front)：椅子 `rel_side` 必须为 **back**。
                        2. 如果桌子 `rotation_deg` 是 **0** (在 Back)：椅子 `rel_side` 必须为 **front**。
                        3. 如果桌子 `rotation_deg` 是 **90** (在 Right)：椅子 `rel_side` 必须为 **left**。
                        4. 如果桌子 `rotation_deg` 是 **-90** (在 Left)：椅子 `rel_side` 必须为 **right**。
                        * **间距**：椅子与桌子 `gap: 0.0`。
                 * **椅子旋转**：椅子的正面必须朝向桌子。如果桌子旋转了 $R$ 度，椅子在桌子 `back` 时，其旋转应为 $R+180$。

            4. 真实间距：
               - 挨着的物品（如床和床头柜,椅子和桌子）：gap = 0.02。
               - 普通物体之间间隔（如沙发和桌子，沙发和盆栽,落地灯和盆栽）： gap应在0.1到0.2米之间
               - 需要人通行的物品之间（如衣柜和对面的桌子）：gap 应在 0.5到 3 米之间。
            5. 轴向：
               - (-1.5, 0.0)在中心右侧（right），(0.0, 2.0)在中心前侧（front）。

            6. # 物理承载与空间分布逻辑 (防止拥挤与违背常理)
            - **高度禁止项**：严禁以 [wardrobe, refrigerator, wall cabinet] 为 anchor 时使用 `rel_side: top`。
            - **表面物件 (Surface Objects)**：[desk lamp, table lamp, flower vase, apple, pot, microwave] 必须放置在表面。
            - **分布式摆放原则 (重要)**：
                1. **一对一优先**：每一个小物体应优先寻找一个独立的承载物（desk, nightstand, coffee table, cabinet）。
                2. **禁止过度堆叠**：除非房间内没有多余的承载物表面，否则严禁在同一个 `top` 上摆放超过 2 个物体。
                3. **特定绑定**：`desk lamp` 必须优先绑定在 `desk` 上；`table lamp` 或 `flower vase` 必须优先分配给 `nightstand`。
            - **地面物件 (Ground Objects)**：[potted tree, floor lamp, rocking chair, chair, storage bin] 必须通过 `left/right/front/back` 锚定在大件家具旁或使用 `pos_abs`，**严禁**使用 `rel_side: top`。
            - **链式平铺**：只有当必须在同一表面摆放第二个物体时，才允许使用链式锚点（如 Vase 锚定在 table lamp 的 right）。当同一个表面摆放超过一个物体时（如 Vase 锚定在 table lamp 的 right），必须使用链式锚点避免物体重叠，设置gap=0.02。

            # 输出要求
            仅输出一个纯净的 JSON 数组（List of objects），不要输出括号 ()，坐标请用数组 []。category只能用空格不能用下划线。
            "path": "{room_assets_root}/{{category}}/RANDOM_INDEX/mesh/sample.glb"

            # 输出格式示例（仅参考结构，严禁照抄家具种类）
            示例1:
            [
                {{
                    "name": "Bed",
                    "category": "bed",
                    "anchor": null,
                    "pos_abs": [0.0,-3.0],
                    "path": "{room_assets_root}/bed/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 0,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }},
                {{
                    "name": "Nightstand_L",
                    "category": "nightstand",
                    "anchor": "Bed",
                    "rel_side": "left",
                    "gap":0.02,
                    "path": "{room_assets_root}/nightstand/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 0,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }},
                {{
                    "name": "Nightstand_R",
                    "category": "nightstand",
                    "anchor": "Bed",
                    "rel_side": "right",
                    "gap": 0.02,
                    "path": "{room_assets_root}/nightstand/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 0,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }},
                {{
                    "name": "Wardrobe_Cabinet",
                    "category": "wardrobe cabinet",
                    "anchor": null,
                    "pos_abs": [1.9, -1.5],# 靠房间左下角
                    "path": "{room_assets_root}/wardrobe cabinet/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 90,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }},
                {{
                    "name": "Desk",
                    "category": "desk",
                    "anchor": "Wardrobe",   # 以衣柜为锚点
                    "rel_side": "right",    # 在衣柜右边排开
                    "gap": 2.9,             # 留出2.7m间距
                    "path": "{room_assets_root}/desk/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": -90,    # 旋转使桌面对着床的方向
                    "fix_axis": "y_to_z",
                    "is_static": true
                }},
                {{
                    "name": "Arm_Chair",
                    "category": "arm chair",
                    "anchor": "Desk",
                    "rel_side": "left",
                    "gap": 0.02,
                    "path": "{room_assets_root}/armchair/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 90,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }}
            ]

            示例2：
            # 输出格式示例 (关键逻辑展示)：
            [
                {{
                    "name": "Table_01",
                    "category": "coffee table",
                    "anchor": null,
                    "pos_abs": [1.0, 0.5],
                    "path": "{room_assets_root}/coffee table/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 90,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }},
                {{
                    "name": "Lamp_On_Table",
                    "category": "desk lamp",
                    "anchor": "Table_01",
                    "rel_side": "top",
                    "gap": 0.0,
                    "path": "{room_assets_root}/desk lamp/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 0,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }},
                {{
                    "name": "Vase_On_Desk",
                    "category": "flower vase",
                    "anchor": "Lamp_On_Table",  # 链式锚点：绑定在灯上而非桌上
                    "rel_side": "right",       # 在灯的右侧平铺
                    "gap": 0.02,                # 保持间距防止重叠
                            "path": "{room_assets_root}/flower vase/RANDOM_INDEX/mesh/sample.glb",
                    "rotation_deg": 0,
                    "fix_axis": "y_to_z",
                    "is_static": true
                }}
            ]
            按照上面的格式输出。
        
       
        """

#  示例3:
#             {{
#                 "name": "Cabinet",
#                 "category": "cabinet",
#                 "anchor": None,  # 基座
#                 "pos_abs": (1.0, 0.0),
#                 "path": "{room_assets_root}/cabinet/0/mesh/sample.glb",
#                 "rotation_deg": 0,
#                 "fix_axis": 'y_to_z',
#                 "is_static": True
#             }},
#             {{
#                 "name": "Microwave",
#                 "category": "microwave",
#                 "anchor": "Cabinet", 
#                 "rel_side": "top",
#                 "gap": 0.0,
#                 "path": "{room_assets_root}/microwave/0/mesh/sample.glb",
#                 "rotation_deg": 0,
#                 "fix_axis": 'y_to_z',
#                 "is_static": True
#             }},
#             {{
#                 "name": "Wall_Cabinet",
#                 "category": "wall cabinet",
#                 "anchor": "Microwave",   # 三个物品竖直依次从上到下时，用链式参考坐标
#                 "rel_side": "top",
#                 "gap": 0.4,     
#                 "path": "{room_assets_root}/wall cabinet/0/mesh/sample.glb",
#                 "rotation_deg": 0,
#                 "fix_axis": 'y_to_z',
#                 "is_static": True
#             }}


    # def __call__(self, room_description: str, furniture_categories: str, output_path: str):
    def __call__(self, room_description: str, furniture_categories: str, output_path: str):
        """执行生成并存入 .json 文件"""
        room_assets_root = "our_assets/room"
        prompt = self.prompt_template.format(
            room_description=room_description,
            furniture_categories=furniture_categories,
            room_assets_root=room_assets_root,
        )
        # prompt = self.prompt_template
        print(prompt)

        # 调用 Qwen
        response = self.qwen_client.query(
            text_prompt=prompt,
            system_role="You are an expert in JSON format and interior design.",
            params={"temperature": 0.7, "max_tokens": 2500}
        )

        if not response:
            logger.error("Qwen 响应为空")
            return None
        
        import random
        import re
 
        # 清洗代码块标签
        # clean_code = response.replace("```python", "").replace("```py", "").replace("```", "").strip()
        clean_json = response.replace("```json", "").replace("```", "").strip()

        def replace_random_index(match):
            category = match.group(1)
            base_dir = str((REPO_ROOT / "our_assets" / "room" / category).resolve())
            
            if os.path.exists(base_dir):
                # 获取所有数字命名的子目录
                indices = [d for d in os.listdir(base_dir) if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))]
                if indices:
                    selected_index = random.choice(indices)
                    return str((Path("our_assets") / "room" / category / selected_index / "mesh" / "sample.glb").as_posix())
            
            # 如果没找到，默认返回 0
            return str((Path("our_assets") / "room" / category / "0" / "mesh" / "sample.glb").as_posix())

        # 使用正则匹配路径并动态替换
        # 匹配类似 .../our_assets/.../xxx/RANDOM_INDEX/... 的字符串

        clean_json = re.sub(
            r"(?:.*/)?our_assets/room/([^/]+)/RANDOM_INDEX/mesh/sample\.glb",
            replace_random_index,
            clean_json
        )

        print("Cleaned Code:\n", clean_json)



        # 3. 验证并保存 JSON
        try:
            # 尝试解析一次以确保模型输出的是合法 JSON
            layout_data = json.loads(clean_json)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                # 写入 JSON 文件
                json.dump(layout_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"布局配置已成功写入 JSON: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"解析或解析 JSON 失败: {e}\n原始输出内容:\n{clean_json}")
            return None


 
import argparse
import sys

if __name__ == "__main__":
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Room Layout Generator CLI")
    parser.add_argument("--prompt", type=str, default="", help="房间场景描述文本")
    parser.add_argument("--template", type=str, default="", help="预设模板路径")    
    args = parser.parse_args()

    save_path = str(UI_GENERATED_LAYOUT_JSON)
    # 2. 逻辑判断
    # 情况 A: 提供了模板路径
    if args.template and os.path.exists(args.template):
        print(f"ℹ️ 检测到模板路径: {args.template}")
        print("✅ 使用现有模板。")
        sys.exit(0) # 正常退出，由后续渲染脚本处理该模板

    # 情况 B: 提供了 Prompt
    elif args.prompt.strip():
        print(f"🚀 正在根据描述生成场景: {args.prompt}")
        
        # 初始化客户端和生成器
        generator = RoomLayoutGenerator(QWEN_CLIENT)
        
        # 获取资产目录下的所有类别作为参考
        asset_root = str((REPO_ROOT / "our_assets" / "room").resolve())
        if os.path.exists(asset_root):
            furniture_categories = ", ".join([f for f in os.listdir(asset_root) if os.path.isdir(os.path.join(asset_root, f))])
        else:
            furniture_categories = "bed, desk, chair, sofa, table" # 备选兜底

        # 执行生成
        result = generator(
            room_description=args.prompt, 
            furniture_categories=furniture_categories, 
            output_path=save_path #args.output
        )
        
        if result:
            print(f"✅ 布局已生成至: {result}")
            sys.exit(0)
        else:
            print("❌ 布局生成失败。")
            sys.exit(1)

    # 情况 C: 均未提供
    else:
        print("⚠️ 提示：请给出需要搭建的场景描述 (Prompt) 或 场景模板路径 (Template Path)。")
        sys.exit(1)





