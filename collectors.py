"""消息收集相关函数"""

import datetime
from typing import List

from nekro_agent.api import core
from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.models.db_chat_message import DBChatMessage

from .plugin import config


async def collect_user_messages_with_context(
    user_id: str,
    chat_key: str,
    max_count: int = 200,
    context_before: int = 1,
    context_after: int = 1,
) -> List[DBChatMessage]:
    """收集用户历史消息及其上下文（分页加载策略）

    为了更好地理解用户的语言风格和互动模式,不仅收集目标用户的消息,
    还收集每条消息前后的上下文(其他用户的消息),以便AI能理解完整的对话场景。

    采用分页加载策略:
    1. 先加载最近的一批消息
    2. 如果目标用户消息数量不足,再往前加载更多批次
    3. 避免一次性加载所有历史数据造成数据库压力

    Args:
        user_id: 用户平台ID
        chat_key: 聊天频道标识
        max_count: 最大目标用户消息数量
        context_before: 每条消息前的上下文消息数
        context_after: 每条消息后的上下文消息数

    Returns:
        List[DBChatMessage]: 包含上下文的消息列表（按时间排序，已去重）
    """
    # 获取聊天频道
    db_chat_channel = await DBChatChannel.get_or_none(chat_key=chat_key)
    if not db_chat_channel:
        raise ValueError("未找到聊天频道")

    # 确定查询起始时间
    if config.IGNORE_SESSION_RESET:
        conversation_start_timestamp = 0
        core.logger.info("已启用忽略会话重置时间,将收集所有历史消息")
    else:
        conversation_start_time: datetime.datetime = db_chat_channel.conversation_start_time
        conversation_start_timestamp = int(conversation_start_time.timestamp())

    # 分页加载策略
    target_messages: List[DBChatMessage] = []
    batch_size = config.BATCH_LOAD_SIZE
    offset = 0
    max_iterations = 10  # 最多迭代10次,避免无限循环

    for iteration in range(max_iterations):
        # 查询一批消息
        batch_messages = (
            await DBChatMessage.filter(
                chat_key=chat_key,
                platform_userid=user_id,
                send_timestamp__gte=conversation_start_timestamp,
            )
            .order_by("-send_timestamp")
            .offset(offset)
            .limit(batch_size)
        )

        if not batch_messages:
            core.logger.info(f"已到达消息历史末尾,共收集 {len(target_messages)} 条目标用户消息")
            break

        target_messages.extend(batch_messages)
        core.logger.info(
            f"第 {iteration + 1} 批: 加载 {len(batch_messages)} 条消息, 累计 {len(target_messages)} 条目标用户消息",
        )

        if len(target_messages) >= max_count:
            target_messages = target_messages[:max_count]
            core.logger.info(f"已达到目标消息数量 {max_count},停止加载")
            break

        offset += batch_size

    if not target_messages:
        return []

    # 反转为时间正序
    target_messages = target_messages[::-1]

    # 收集每条目标消息的上下文
    all_messages_dict: dict[int, DBChatMessage] = {}

    for target_msg in target_messages:
        all_messages_dict[target_msg.id] = target_msg

        # 获取前面的上下文消息
        if context_before > 0:
            prev_messages = (
                await DBChatMessage.filter(
                    chat_key=chat_key,
                    send_timestamp__lt=target_msg.send_timestamp,
                    send_timestamp__gte=conversation_start_timestamp,
                )
                .order_by("-send_timestamp")
                .limit(context_before)
            )
            for msg in prev_messages:
                all_messages_dict[msg.id] = msg

        # 获取后面的上下文消息
        if context_after > 0:
            next_messages = (
                await DBChatMessage.filter(
                    chat_key=chat_key,
                    send_timestamp__gt=target_msg.send_timestamp,
                    send_timestamp__gte=conversation_start_timestamp,
                )
                .order_by("send_timestamp")
                .limit(context_after)
            )
            for msg in next_messages:
                all_messages_dict[msg.id] = msg

    # 按时间排序并返回
    sorted_messages = sorted(all_messages_dict.values(), key=lambda m: m.send_timestamp)

    core.logger.info(
        f"收集到目标用户 {len(target_messages)} 条消息, 包含上下文共 {len(sorted_messages)} 条消息",
    )

    return sorted_messages


async def collect_user_dialogs_with_context(
    user_id: str,
    chat_key: str,
    max_dialogs: int = 5,
    context_before: int = 3,
    context_after: int = 3,
    time_window_minutes: int = 15,
) -> List[List[DBChatMessage]]:
    """按时间窗与上下文分段收集对话"""
    db_chat_channel = await DBChatChannel.get_or_none(chat_key=chat_key)
    if not db_chat_channel:
        raise ValueError("未找到聊天频道")

    if config.IGNORE_SESSION_RESET:
        conversation_start_timestamp = 0
    else:
        conversation_start_time: datetime.datetime = db_chat_channel.conversation_start_time
        conversation_start_timestamp = int(conversation_start_time.timestamp())

    target_msgs: List[DBChatMessage] = (
        await DBChatMessage.filter(
            chat_key=chat_key,
            platform_userid=user_id,
            send_timestamp__gte=conversation_start_timestamp,
        )
        .order_by("send_timestamp")
        .limit(config.MAX_MESSAGE_SAMPLE)
    )
    if not target_msgs:
        return []

    dialogs: List[List[DBChatMessage]] = []
    used_ids: set[int] = set()
    window_seconds = max(1, time_window_minutes) * 60

    for anchor in target_msgs:
        if len(dialogs) >= max_dialogs:
            break
        if anchor.id in used_ids:
            continue

        window_start = anchor.send_timestamp - window_seconds
        window_end = anchor.send_timestamp + window_seconds
        window_messages: List[DBChatMessage] = await DBChatMessage.filter(
            chat_key=chat_key,
            send_timestamp__gte=window_start,
            send_timestamp__lte=window_end,
        ).order_by("send_timestamp")
        if not window_messages:
            continue

        try:
            anchor_idx = next(i for i, m in enumerate(window_messages) if m.id == anchor.id)
        except StopIteration:
            continue

        prefix_candidates = window_messages[:anchor_idx]
        if len(prefix_candidates) < context_before:
            continue
        prefix = prefix_candidates[-context_before:]

        last_target_idx = anchor_idx
        i = anchor_idx
        while i < len(window_messages):
            msg = window_messages[i]
            if msg.platform_userid == user_id:
                last_target_idx = i
            after_count = max(0, i - last_target_idx)
            if last_target_idx >= anchor_idx and after_count >= context_after:
                break
            i += 1

        if last_target_idx < anchor_idx or (i >= len(window_messages) and max(0, i - last_target_idx) < context_after):
            continue

        postfix = window_messages[last_target_idx + 1 : last_target_idx + 1 + context_after]
        body = window_messages[anchor_idx : last_target_idx + 1]
        dialog_messages = prefix + body + postfix
        if not dialog_messages:
            continue

        span_seconds = dialog_messages[-1].send_timestamp - dialog_messages[0].send_timestamp
        if span_seconds > window_seconds:
            continue

        if any(m.id in used_ids for m in dialog_messages):
            continue
        for m in dialog_messages:
            used_ids.add(m.id)
        dialogs.append(dialog_messages)

    return dialogs

