"""AI生成相关函数"""

import datetime
import json
import re
from typing import Dict, List, Optional

from nekro_agent.api import core
from nekro_agent.models.db_chat_message import DBChatMessage
from nekro_agent.services.agent.openai import OpenAIResponse, gen_openai_chat_response

from .collectors import collect_user_dialogs_with_context
from .formatters import (
    format_dialogs_for_analysis,
    format_messages_for_analysis,
    sample_messages,
)
from .models import PRESET_GENERATION_PROMPT
from .plugin import config


async def generate_preset_with_ai(
    user_id: str,
    nickname: str,
    messages: List[DBChatMessage],
    chat_key: str,
    model_group_name: str,
) -> Dict[str, str]:
    """使用AI生成人设描述

    Args:
        user_id: 用户ID
        nickname: 用户昵称
        messages: 历史消息列表（包含上下文）
        chat_key: 聊天频道标识
        model_group_name: 模型组名称

    Returns:
        Dict[str, str]: 包含人设信息的字典
    """
    # 尝试分段对话收集（按时间窗）
    dialogs: Optional[List[List[DBChatMessage]]] = None
    try:
        dialogs = await collect_user_dialogs_with_context(
            user_id=user_id,
            chat_key=chat_key,
            max_dialogs=config.MAX_DIALOG_COUNT,
            context_before=config.CONTEXT_BEFORE,
            context_after=config.CONTEXT_AFTER,
            time_window_minutes=config.DIALOG_TIME_WINDOW_MINUTES,
        )
        if not dialogs:
            core.logger.info("未生成对话分段，改用平铺消息格式化")
            dialogs = None
    except Exception as e:
        core.logger.warning(f"对话分段收集失败，改用平铺消息: {e}")
        dialogs = None

    # 统计消息（支持对话分段）
    flat_messages = list(messages)
    if dialogs is not None:
        flat_messages = [m for d in dialogs for m in d]

    target_message_count = sum(1 for msg in flat_messages if msg.platform_userid == user_id)
    total_message_count = len(flat_messages)

    core.logger.info(
        f"准备分析: 目标用户消息 {target_message_count} 条, 总消息(含上下文) {total_message_count} 条",
    )

    # 采样：仅在未分段时使用消息采样；分段场景保持对话完整性
    if dialogs is None and len(flat_messages) > config.MAX_MESSAGE_ANALYZE:
        core.logger.info(f"消息数量({len(flat_messages)})超过分析上限({config.MAX_MESSAGE_ANALYZE}),进行随机采样")
        messages = await sample_messages(flat_messages, config.MAX_MESSAGE_ANALYZE)
        flat_messages = messages
        target_message_count = sum(1 for msg in flat_messages if msg.platform_userid == user_id)
        total_message_count = len(flat_messages)
        core.logger.info(
            f"采样后: 目标用户消息 {target_message_count} 条, 总消息 {total_message_count} 条",
        )

    if not flat_messages:
        raise ValueError("没有可用于分析的消息")

    # 格式化历史记录（支持分段与块标记）
    if dialogs is not None:
        chat_history = await format_dialogs_for_analysis(
            dialogs,
            target_user_id=user_id,
            max_length=config.MAX_MESSAGE_LENGTH,
            add_block_markers=config.ADD_DIALOG_BLOCK_MARKERS,
        )
    else:
        chat_history = await format_messages_for_analysis(
            flat_messages,
            target_user_id=user_id,
            max_length=config.MAX_MESSAGE_LENGTH,
        )

    # 计算时间跨度
    if len(flat_messages) > 1:
        first_msg_time = datetime.datetime.fromtimestamp(flat_messages[0].send_timestamp)
        last_msg_time = datetime.datetime.fromtimestamp(flat_messages[-1].send_timestamp)
        time_delta = last_msg_time - first_msg_time
        days = time_delta.days
        time_span = f"{days}天" if days > 0 else "1天内"
    else:
        time_span = "未知"

    # 构造提示词
    prompt = PRESET_GENERATION_PROMPT.format(
        nickname=nickname,
        user_id=user_id,
        target_message_count=target_message_count,
        total_message_count=total_message_count,
        time_span=time_span,
        chat_history=chat_history,
    )

    # Debug日志
    if config.ENABLE_PROMPT_DEBUG:
        core.logger.info(f"完整请求 (长度: {len(prompt)}):\n{prompt}\n")

    # 获取模型组配置
    model_group = core.config.MODEL_GROUPS.get(model_group_name)
    if not model_group:
        raise ValueError(f"模型组 {model_group_name} 不存在")

    # 调用AI生成
    core.logger.info(f"开始使用 {model_group.CHAT_MODEL} 分析用户人设...")

    llm_response: OpenAIResponse = await gen_openai_chat_response(
        model=model_group.CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        base_url=model_group.BASE_URL,
        api_key=model_group.API_KEY,
        stream_mode=False,
        proxy_url=model_group.CHAT_PROXY,
        max_wait_time=config.TIMEOUT,
        temperature=0.7,
    )

    core.logger.info(f"AI分析完成, 消耗Token: {llm_response.token_consumption}")

    # 解析AI返回的JSON
    response_text = llm_response.response_content.strip()
    json_str = response_text

    # 1. 尝试提取markdown代码块中的JSON (```json 或 ```)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 2. 如果没有代码块，尝试查找第一个 { 到最后一个 }
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and start < end:
            json_str = response_text[start : end + 1]

    # 3. 清理可能的多余空白和换行
    json_str = json_str.strip()

    # 4. 尝试解析
    try:
        preset_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        core.logger.error(
            f"解析AI返回的JSON失败: {e}\n提取的JSON字符串:\n{json_str}\n原始响应内容:\n{response_text}",
        )
        raise ValueError("AI返回的格式不正确,无法解析人设数据") from e

    # 5. 验证必需字段
    required_fields = ["name", "title", "description", "content", "tags"]
    missing_fields = [field for field in required_fields if field not in preset_data]
    if missing_fields:
        raise ValueError(f"AI返回的数据缺少必需字段: {', '.join(missing_fields)}")

    return preset_data
