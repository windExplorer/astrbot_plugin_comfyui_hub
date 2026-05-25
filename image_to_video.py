import json
import random
import time
from io import BytesIO
from typing import Optional

from PIL import Image as PILImage
from astrbot.api import logger

from .comfyui_api import ComfyUIAPI


class ImageToVideo:
    def __init__(self, api: ComfyUIAPI, workflow_path: str,
                 positive_node: str = "3", negative_node: str = "4",
                 input_node: str = "2"):
        self.api = api
        self.workflow = self._load_workflow(workflow_path)
        self.positive_node = positive_node
        self.negative_node = negative_node
        self.input_node = input_node

    @staticmethod
    def _load_workflow(path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _set_prompt(node: dict, prompt: str) -> bool:
        if not node or "inputs" not in node:
            return False
        inputs = node["inputs"]
        if not inputs:
            return False
        first_key = next(iter(inputs))
        inputs[first_key] = prompt
        return True

    @staticmethod
    def _extract_first_frame_if_gif(image_data: bytes) -> Optional[bytes]:
        """如果输入是动图（GIF/WebP），提取第一帧"""
        try:
            with PILImage.open(BytesIO(image_data)) as img:
                is_animated = getattr(img, 'is_animated', False)
                if is_animated:
                    logger.info("[ComfyUI] 检测到动图，将使用首帧")
                    img.seek(0)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    output = BytesIO()
                    img.save(output, format='PNG')
                    return output.getvalue()
                return image_data
        except Exception as e:
            logger.error(f"[ComfyUI] 动图检测/处理失败: {e}")
            return None

    async def generate(self, image_data: bytes, prompt: str, negative: str = "",
                       max_wait: float = 300.0, on_wait_callback=None,
                       on_submitted_callback=None) -> Optional[bytes]:
        """生成视频

        Args:
            image_data: 输入图片数据
            prompt: 正面提示词
            negative: 负面提示词
        """
        workflow = json.loads(json.dumps(self.workflow))

        # 处理输入图片（动图首帧）
        processed = self._extract_first_frame_if_gif(image_data)
        if processed is None:
            logger.error("[ComfyUI] 输入图片处理失败")
            return None

        filename = f"i2v_input_{int(time.time() * 1000)}_{random.randint(1000, 9999)}.png"
        try:
            await self.api.upload_image(filename, processed)
        except Exception as e:
            logger.error(f"[ComfyUI] 上传图片失败: {e}")
            return None

        # 写入 LoadImage 节点
        load_image_set = False
        if self.input_node and self.input_node in workflow:
            node_data = workflow[self.input_node]
            if isinstance(node_data, dict) and node_data.get("class_type") == "LoadImage":
                node_data["inputs"]["image"] = filename
                load_image_set = True

        if not load_image_set:
            for node_id, node_data in workflow.items():
                if isinstance(node_data, dict) and node_data.get("class_type") == "LoadImage":
                    node_data["inputs"]["image"] = filename
                    load_image_set = True
                    break

        if not load_image_set:
            logger.error("[ComfyUI] 工作流中未找到 LoadImage 节点")
            return None

        # 设置正面提示词
        pos_node = workflow.get(self.positive_node)
        if not pos_node:
            logger.error(f"[ComfyUI] 找不到正面提示词节点 {self.positive_node}")
            return None
        if not self._set_prompt(pos_node, prompt):
            logger.error(f"[ComfyUI] 节点 {self.positive_node} 没有输入字段")
            return None

        # 设置负面提示词
        if self.negative_node and negative:
            neg_node = workflow.get(self.negative_node)
            if neg_node:
                self._set_prompt(neg_node, negative)

        # 随机化种子
        base_seed = random.randint(1, 999999999999999)
        offset = 0
        for node_data in workflow.values():
            if isinstance(node_data, dict):
                inputs = node_data.get("inputs", {})
                if "seed" in inputs:
                    inputs["seed"] = base_seed + offset
                    offset += 1
                if "noise_seed" in inputs:
                    inputs["noise_seed"] = base_seed + offset
                    offset += 1

        result = await self.api.queue_and_wait_video(
            workflow,
            max_wait=max_wait,
            on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback
        )

        if not result:
            logger.error("[ComfyUI] 图生视频生成失败或等待结果超时")

        return result