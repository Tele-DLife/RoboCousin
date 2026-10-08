import os
import random
import logging
import shutil
import sys
from pathlib import Path

try:
    from script.path_config import ROOM_CONFIG_50ROOMS_DIR, UI_GENERATED_LAYOUT_JSON
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parents[2]
    ROOM_CONFIG_50ROOMS_DIR = (_repo_root / "envs" / "room_config" / "50rooms").resolve()
    UI_GENERATED_LAYOUT_JSON = (_repo_root / "envs" / "room_config" / "ui_generated_layout.json").resolve()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoomLayoutMatcher:
    def __init__(self, base_path: str, target_path: str):
        # 基础库路径
        self.base_path = base_path
        # 最终生效的路径 (现在应该是 .json 后缀)
        self.target_path = target_path
        # 严格限制的房间类型映射
        self.type_map = {
            "卧室": "bedroom",
            "客厅": "livingroom",
            "餐厅": "diningroom",
            "卫生间": "bathroom",
            "儿童房": "kidsroom",
            "书房": "study",
        }

    def apply_random_layout(self, room_type: str) -> bool:
        """从对应目录随机选一个 .json 文件并覆盖到 target_path"""
        if room_type not in self.type_map:
            logger.error(f"❌ 不支持的房间类型: {room_type}")
            return False

        subdir = self.type_map[room_type]
        source_dir = os.path.join(self.base_path, subdir)
        
        if not os.path.exists(source_dir):
            logger.error(f"❌ 目录不存在: {source_dir}")
            return False

        # --- 修改点：获取目录下所有 .json 文件 ---
        config_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

        if not config_files:
            logger.error(f"❌ 在 {source_dir} 目录下没找到任何 .json 文件")
            return False

        # 1. 随机选择
        selected_filename = random.choice(config_files)
        source_file = os.path.join(source_dir, selected_filename)

        # 2. 执行复制
        try:
            # 确保目标目录存在
            target_dir = os.path.dirname(self.target_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            
            # 物理覆盖目标文件
            shutil.copy(source_file, self.target_path)
            
            # logger.info(f"✅ 已随机选中 JSON: {source_file}")
            # logger.info(f"🚀 已应用到: {self.target_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 复制文件失败: {e}")
            return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Room Layout JSON Shuffler")
    parser.add_argument("--type", type=str, required=True, 
                        choices=["卧室", "客厅", "餐厅", "卫生间", "儿童房", "书房"], 
                        help="指定房间类型")
    args = parser.parse_args()

    # --- 修改点：配置路径后缀改为 .json ---
    BASE_LIB = str(ROOM_CONFIG_50ROOMS_DIR.parent)
    # 这里的名字要和你在 SAPIEN 脚本里读取的名字一致
    TARGET_FILE = str(UI_GENERATED_LAYOUT_JSON)

    # for demostration
    import time
    # time.sleep(20)

    matcher = RoomLayoutMatcher(BASE_LIB, TARGET_FILE)
    success = matcher.apply_random_layout(args.type)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
