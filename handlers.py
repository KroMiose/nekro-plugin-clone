"""命令处理器"""

import re
import time
from typing import Dict

from nonebot import on_command
from nonebot.adapters import Bot, Message
from nonebot.adapters.onebot.v11 import MessageEvent
from nonebot.matcher import Matcher
from nonebot.params import CommandArg

from nekro_agent.adapters.onebot_v11.matchers.command import command_guard, finish_with
from nekro_agent.api import core
from nekro_agent.models.db_preset import DBPreset
from nekro_agent.schemas.chat_message import ChatType

from .collectors import collect_user_messages_with_context
from .generator import generate_preset_with_ai
from .plugin import config, plugin
from .utils import get_user_avatar_base64

# 消息去重
_processed_message_ids: Dict[int, float] = {}
_PROCESSED_MSG_TTL_SECONDS: int = 60


@on_command("preset-clone", aliases={"preset_clone", "pclone"}, priority=5, block=True).handle()
async def handle_preset_clone(matcher: Matcher, event: MessageEvent, bot: Bot, arg: Message = CommandArg()):
    """处理人设克隆命令"""
    # 去重：同一条消息只处理一次
    now = time.time()
    for mid, ts in list(_processed_message_ids.items()):
        if now - ts > _PROCESSED_MSG_TTL_SECONDS:
            _processed_message_ids.pop(mid, None)
    if event.message_id in _processed_message_ids:
        return
    _processed_message_ids[event.message_id] = now

    username, cmd_content, chat_key, chat_type = await command_guard(event, bot, arg, matcher)

    # 参数验证
    if chat_type != ChatType.GROUP:
        await finish_with(matcher, message="人设克隆功能仅支持群聊使用")

    if not cmd_content:
        await finish_with(
            matcher,
            message="请指定要克隆的用户ID或@用户\n用法: /preset-clone <用户QQ号>\n或: /preset-clone @用户",
        )

    # 解析用户标识
    target_user_id = cmd_content.strip()
    at_match = re.search(r"@(\d+)", target_user_id)
    if at_match:
        target_user_id = at_match.group(1)
    elif not target_user_id.isdigit():
        await finish_with(matcher, message="用户ID格式不正确,请使用QQ号或@用户")

    # 发送处理提示
    await matcher.send(f"开始克隆用户 {target_user_id} 的人设...")

    # 业务逻辑处理
    try:
        # 1. 收集用户历史消息及上下文
        core.logger.info(
            f"开始收集用户 {target_user_id} 的历史消息 "
            f"(前置上下文:{config.CONTEXT_BEFORE}, 后置上下文:{config.CONTEXT_AFTER}, "
            f"IGNORE_SESSION_RESET:{config.IGNORE_SESSION_RESET})",
        )
        messages = await collect_user_messages_with_context(
            user_id=target_user_id,
            chat_key=chat_key,
            max_count=config.MAX_MESSAGE_SAMPLE,
            context_before=config.CONTEXT_BEFORE,
            context_after=config.CONTEXT_AFTER,
        )

        if not messages:
            error_msg = f"用户 {target_user_id} 没有历史消息,无法克隆"
            core.logger.warning(error_msg)
            await finish_with(matcher, message=error_msg)

        # 统计目标用户的实际消息数
        target_message_count = sum(1 for msg in messages if msg.platform_userid == target_user_id)

        if target_message_count < config.MIN_MESSAGE_COUNT:
            error_msg = (
                f"用户 {target_user_id} 的历史消息数量不足({target_message_count}/{config.MIN_MESSAGE_COUNT})\n"
                f"至少需要 {config.MIN_MESSAGE_COUNT} 条消息才能进行克隆 "
                f"(IGNORE_SESSION_RESET={config.IGNORE_SESSION_RESET})"
            )
            core.logger.warning(error_msg)
            await finish_with(matcher, message=error_msg)

        core.logger.info(
            f"收集完成: 目标用户 {target_message_count} 条, 总消息(含上下文) {len(messages)} 条",
        )

        # 获取用户昵称
        try:
            user_nickname = next(
                m.sender_nickname
                for m in reversed(messages)
                if m.platform_userid == target_user_id and m.sender_nickname
            )
        except StopIteration:
            user_nickname = str(target_user_id)

        await matcher.send(
            f"已收集到用户 {target_user_id} 的 {target_message_count} 条历史消息"
            f"(含上下文共 {len(messages)} 条),正在分析中...",
        )

        # 2. 使用AI生成人设
        preset_data = await generate_preset_with_ai(
            user_id=target_user_id,
            nickname=user_nickname,
            messages=messages,
            chat_key=chat_key,
            model_group_name=config.USE_CLONE_MODEL_GROUP,
        )

        # 3. 获取用户头像
        avatar_base64 = await get_user_avatar_base64(target_user_id, chat_key)
        if not avatar_base64:
            core.logger.warning("获取用户头像失败,使用默认头像")
            avatar_base64 = (
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )

        # 4. 添加标签
        tags = preset_data.get("tags", "")
        if config.AUTO_ADD_TAG and config.AUTO_ADD_TAG not in tags:
            tags = f"{config.AUTO_ADD_TAG},{tags}" if tags else config.AUTO_ADD_TAG

        # 5. 保存到数据库
        preset = await DBPreset.create(
            name=preset_data.get("name", user_nickname),
            title=preset_data.get("title", f"{user_nickname}的克隆体"),
            avatar=avatar_base64,
            content=preset_data.get("content", ""),
            description=preset_data.get("description", ""),
            tags=tags,
            author=username,
            on_shared=False,
        )

        core.logger.info(f"人设克隆完成,已保存: ID={preset.id}, Name={preset.name}")

        # 成功完成
        success_msg = (
            f"✅ 人设克隆完成!\n\n"
            f"人设名称: {preset.name}\n"
            f"人设标题: {preset.title}\n"
            f"人设描述: {preset.description}\n"
            f"标签: {preset.tags}\n\n"
            f"人设ID: {preset.id}\n"
            f"已保存到本地人设库,可在Web管理页面查看和编辑"
        )
        await finish_with(matcher, message=success_msg)

    except ValueError as e:
        # 业务逻辑错误
        error_msg = f"❌ 人设克隆失败: {e}"
        core.logger.error(error_msg)
        await finish_with(matcher, message=error_msg)
    except Exception as e:
        # 检查是否是 nonebot 的异常
        if e.__class__.__name__ in ("FinishedException", "StopPropagation", "SkippedException"):
            raise

        # 未知错误
        error_msg = f"❌ 人设克隆发生错误: {e}"
        core.logger.exception(f"人设克隆发生未知错误: {e}")
        await finish_with(matcher, message=error_msg)


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件资源"""
    core.logger.info("人设克隆插件资源已清理")

