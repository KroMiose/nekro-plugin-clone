"""用户头像和工具函数"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from nekro_agent.api import core
from nekro_agent.tools.common_util import download_file
from nekro_agent.tools.image_utils import process_image_data_url


async def get_user_avatar_base64(user_qq: str, chat_key: str) -> Optional[str]:
    """获取用户头像并转换为base64

    Args:
        user_qq: 用户QQ号
        chat_key: 聊天频道标识

    Returns:
        Optional[str]: base64编码的头像数据URL,失败返回None
    """
    try:
        # 下载头像
        file_path, _ = await download_file(
            f"https://q1.qlogo.cn/g?b=qq&nk={user_qq}&s=640",
            from_chat_key=chat_key,
            use_suffix=".png",
        )

        # 读取并转换为base64
        file_path_obj = Path(file_path)
        image_data = file_path_obj.read_bytes()

        # 压缩图片以减少存储
        img = Image.open(BytesIO(image_data))
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)

        # 保存为优化后的JPEG
        output = BytesIO()
        img.convert("RGB").save(output, format="JPEG", quality=85, optimize=True)
        compressed_data = output.getvalue()

        # 转换为base64
        base64_encoded = base64.b64encode(compressed_data).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_encoded}"

        # 使用系统的图片处理工具进行进一步压缩和优化（500KB限制）
        return await process_image_data_url(data_url)
    except Exception as e:
        core.logger.error(f"获取用户头像失败: {e}")
        return None

