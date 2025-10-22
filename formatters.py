"""消息格式化和采样相关函数"""

import datetime
import random
from typing import Dict, List

from nekro_agent.models.db_chat_message import DBChatMessage

from .plugin import config


async def format_messages_for_analysis(
    messages: List[DBChatMessage],
    target_user_id: str,
    max_length: int = 200,
) -> str:
    """格式化消息用于AI分析，标注目标用户

    Args:
        messages: 消息列表（包含上下文）
        target_user_id: 目标用户ID
        max_length: 单条消息最大长度

    Returns:
        str: 格式化后的消息文本
    """
    formatted_lines = []
    total_length = 0
    last_timestamp = 0
    # 为非目标用户分配顺序化的显示昵称：用户1、用户2、...
    other_user_label_map: Dict[str, str] = {}
    next_index = 1

    for msg in messages:
        content = msg.content_text.strip()

        if content:
            # 限制单条消息长度
            if len(content) > max_length:
                content = content[:max_length] + "..."

            # 格式化时间 - 只在时间间隔较大时显示
            time_str = ""
            if last_timestamp == 0 or (msg.send_timestamp - last_timestamp) > 3600:
                time_str = f"[{datetime.datetime.fromtimestamp(msg.send_timestamp).strftime('%Y-%m-%d %H:%M')}] "
            last_timestamp = msg.send_timestamp

            # 标注目标用户 - 使用特殊标记，并为非目标用户分配"用户N"显示名
            is_target = msg.platform_userid == target_user_id
            if is_target:
                nickname_display = msg.sender_nickname
            else:
                label = other_user_label_map.get(msg.platform_userid)
                if not label:
                    label = f"用户{next_index}"
                    other_user_label_map[msg.platform_userid] = label
                    next_index += 1
                nickname_display = label

            formatted_line = f"{time_str}{('[我]' if is_target else '')}{nickname_display}: {content}"

            # 检查总长度
            if total_length + len(formatted_line) > config.MAX_TOTAL_CONTENT_LENGTH:
                formatted_lines.append("[...更多消息已省略以控制长度...]")
                break

            formatted_lines.append(formatted_line)
            total_length += len(formatted_line) + 1

    return "\n".join(formatted_lines)


async def format_dialogs_for_analysis(
    dialogs: List[List[DBChatMessage]],
    target_user_id: str,
    max_length: int = 200,
    add_block_markers: bool = True,
) -> str:
    """分段格式化对话输出"""
    formatted_lines: List[str] = []
    total_length = 0
    other_user_label_map: Dict[str, str] = {}
    next_index = 1

    for idx, dialog in enumerate(dialogs, start=1):
        if add_block_markers:
            start_time = datetime.datetime.fromtimestamp(dialog[0].send_timestamp).strftime("%Y-%m-%d %H:%M")
            end_time = datetime.datetime.fromtimestamp(dialog[-1].send_timestamp).strftime("%Y-%m-%d %H:%M")
            header = f"【对话{idx} 开始】({start_time} ~ {end_time})"
            footer = f"【对话{idx} 结束】"
            if total_length + len(header) > config.MAX_TOTAL_CONTENT_LENGTH:
                break
            formatted_lines.append(header)
            total_length += len(header) + 1

        last_ts = 0
        for msg in dialog:
            content = msg.content_text.strip()
            if not content:
                continue
            if len(content) > max_length:
                content = content[:max_length] + "..."

            time_str = ""
            if last_ts == 0 or (msg.send_timestamp - last_ts) > 3600:
                time_str = f"[{datetime.datetime.fromtimestamp(msg.send_timestamp).strftime('%Y-%m-%d %H:%M')}] "
            last_ts = msg.send_timestamp

            is_target = msg.platform_userid == target_user_id
            if is_target:
                nickname_display = msg.sender_nickname
            else:
                label = other_user_label_map.get(msg.platform_userid)
                if not label:
                    label = f"用户{next_index}"
                    other_user_label_map[msg.platform_userid] = label
                    next_index += 1
                nickname_display = label

            line = f"{time_str}{('[我]' if is_target else '')}{nickname_display}: {content}"
            if total_length + len(line) > config.MAX_TOTAL_CONTENT_LENGTH:
                formatted_lines.append("[...更多消息已省略以控制长度...]")
                break
            formatted_lines.append(line)
            total_length += len(line) + 1

        if add_block_markers:
            if total_length + len(footer) > config.MAX_TOTAL_CONTENT_LENGTH:
                break
            formatted_lines.append(footer)
            total_length += len(footer) + 1

    return "\n".join(formatted_lines)


async def sample_messages(messages: List[DBChatMessage], target_count: int) -> List[DBChatMessage]:
    """从消息列表中随机采样,保持时间顺序

    Args:
        messages: 原始消息列表(已按时间排序)
        target_count: 目标采样数量

    Returns:
        List[DBChatMessage]: 采样后的消息列表(保持时间顺序)
    """
    if len(messages) <= target_count:
        return messages

    # 随机选择要保留的索引,但保持顺序
    indices = list(range(len(messages)))
    selected_indices = sorted(random.sample(indices, target_count))
    return [messages[i] for i in selected_indices]

