"""
# 人设克隆插件 (Preset Clone)

通过分析用户的历史发言记录,自动生成该用户的人设配置。

## 主要功能

- **克隆用户人设**: 通过命令 `/preset-clone <用户QQ号>` 克隆指定用户的语言风格和性格特征
- **智能分析**: AI 分析用户的历史消息,提取语言习惯、性格特点、常用词汇等
- **自动保存**: 生成的人设自动保存到本地人设数据库,方便后续使用

## 使用方法

1. 使用命令 `/preset-clone <用户QQ号>` 指定要克隆的目标用户
2. 插件会自动收集该用户的历史发言、头像、昵称等信息
3. AI 分析这些数据并生成人设描述
4. 人设自动添加到本地人设库

## 注意事项

- 目标用户至少需要有 50 条历史消息才能进行克隆
- 建议在克隆前确保该用户有足够多样的聊天内容
## 提示

- 本插件不保证克隆的人设完全准确,仅作为参考
- 如造成经济损失 如：过多的Token消耗 请自行解决
- 本插件仍有一些问题待解决 如:提示词构建还是有点小问题
- 日志等级开到Debug可以看到具体请求内容

"""

import base64
import datetime
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from nonebot import on_command
from nonebot.adapters import Bot, Message
from nonebot.adapters.onebot.v11 import MessageEvent
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from PIL import Image
from pydantic import Field

from nekro_agent.adapters.onebot_v11.matchers.command import command_guard, finish_with
from nekro_agent.api import core
from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.models.db_chat_message import DBChatMessage
from nekro_agent.models.db_preset import DBPreset
from nekro_agent.schemas.chat_message import ChatType
from nekro_agent.services.agent.openai import OpenAIResponse, gen_openai_chat_response
from nekro_agent.tools.common_util import download_file

# 插件信息
plugin = NekroPlugin(
    name="克隆群友Bata",
    module_name="nekro_plugin_clone",
    description="通过分析用户历史发言自动克隆群友并生成 AI 人设",
    version="1.0.1.Bata",
    author="KroMiose",
    url="https://github.com/KroMiose/nekro-plugin-clone",
    support_adapter=["onebot_v11"],
)

# 插件配置
@plugin.mount_config()
class PresetCloneConfig(ConfigBase):
    """人设克隆配置"""

    MIN_MESSAGE_COUNT: int = Field(
        default=60,
        title="最小消息数量",
        description="克隆用户至少需要的历史消息数量",
        ge=10,
    )
    MAX_MESSAGE_SAMPLE: int = Field(
        default=240,
        title="最大采样消息数",
        description="从历史消息中采样的最大数量",
        ge=50,
    )
    MAX_MESSAGE_ANALYZE: int = Field(
        default=120,
        title="最大分析消息数",
        description="实际用于分析的消息数量,从采样中随机抽取以获得更长时间跨度",
        ge=30,
    )
    MAX_MESSAGE_LENGTH: int = Field(
        default=200,
        title="单条消息最大长度",
        description="分析时单条消息的最大字符数,超出部分会被截断",
        ge=50,
    )
    MAX_TOTAL_CONTENT_LENGTH: int = Field(
        default=10000,
        title="总内容最大长度",
        description="所有消息拼接后的最大总长度,避免超出模型上下文限制",
        ge=5000,
    )
    USE_CLONE_MODEL_GROUP: str = Field(
        default="default",
        title="克隆分析模型组",
        json_schema_extra=ExtraField(ref_model_groups=True, required=True, model_type="chat").model_dump(),
        description="用于分析和生成人设的模型组,建议使用智能程度较高的模型",
    )
    CONTEXT_BEFORE: int = Field(
        default=5,
        title="前置上下文消息数",
        description="收集目标用户每条消息前的上下文消息数量",
        ge=0,
        le=10,
    )
    CONTEXT_AFTER: int = Field(
        default=1,
        title="后置上下文消息数",
        description="收集目标用户每条消息后的上下文消息数量",
        ge=0,
        le=10,
    )
    # 新增：对话时间窗与分段设置
    DIALOG_TIME_WINDOW_MINUTES: int = Field(
        default=15,
        title="对话时间窗口（分钟）",
        description="单个对话的最大持续时间（从前置消息到后置消息），用于筛选符合时间条件的消息",
        ge=1,
        le=1440,
    )
    MAX_DIALOG_COUNT: int = Field(
        default=8,
        title="最大对话段数",
        description="按时间窗口与上下文规则分段的最大对话数量，用于格式化输出",
        ge=1,
        le=50,
    )
    ADD_DIALOG_BLOCK_MARKERS: bool = Field(
        default=True,
        title="添加对话分隔块",
        description="在不同的对话前后插入分隔标记，便于区分",
    )
    IGNORE_SESSION_RESET: bool = Field(
        default=False,
        title="忽略会话重置时间",
        description="是否忽略会话重置时间限制,启用后将收集所有历史消息(默认不启用,仅收集当前会话消息)",
    )
    BATCH_LOAD_SIZE: int = Field(
        default=500,
        title="分批加载大小",
        description="每次从数据库读取的消息批次大小,避免一次性加载过多数据",
    )
    TIMEOUT: int = Field(
        default=120,
        title="分析超时时间",
        description="人设分析的超时时间(秒)",
        ge=30,
    )
    AUTO_ADD_TAG: str = Field(
        default="克隆",
        title="自动添加标签",
        description="克隆生成的人设自动添加的标签",
    )


# 获取配置
config: PresetCloneConfig = plugin.get_config(PresetCloneConfig)
_processed_message_ids: Dict[int, float] = {}
_PROCESSED_MSG_TTL_SECONDS: int = 60


# 生成人设的提示词模板
PRESET_GENERATION_PROMPT = """You are a top-tier psychological profiler and AI Character Architect (Prompt Engineer). Your core capability is to deeply analyze fragmented chat logs to deconstruct a real person's inner psychology, personality traits, linguistic fingerprint, and behavioral patterns. You will then reconstruct this analysis into a highly realistic digital persona, suitable for AI role-playing.

# ⚠️ ATTENTION: Target Profile!!!
# ⚠️ CRITICAL WARNING: Outputting an incorrect User ID or nickname will result in a completely erroneous analysis!!!

**Nickname**: {nickname}
**User ID**: {user_id}

**Remember this critical rule:** Your **sole and exclusive target** for analysis is the person with the nickname "{nickname}" and User ID "{user_id}".

In the chat history below, all messages marked with **`[我]`** were sent by this person.
**You must ONLY analyze messages marked with `[我]`. Messages from Other User(s) are provided strictly for contextual understanding!**

## Data Statistics
- **The Target's ({nickname}) Message Count**: {target_message_count}
- **Total Message Count (including Other User(s))**: {total_message_count}
- **Time Span**: {time_span}

## Chat History

**⚠️ Attention:**

1.  Messages marked with **`[我]`** = Sent by the person you are analyzing (`{nickname}`, ID: `{user_id}`).
2.  Messages without the **`[我]`** mark = Any information sent by Other User(s), for contextual reference only.
3.  **You MUST generate the persona based exclusively on messages marked with `[我]`. Do not incorporate traits from Other User(s).**
4.  **System Messages**: This type of information is an automatically generated system message and should not be analyzed.
    *   Messages like `戳一戳 (揉了揉 12345678 的脑袋)` are "poke" actions and must be ignored.
    *   Messages like `[@id:12345678;nickname:用户名@]` are system messages for mentioning (@) a user and must be ignored. The nickname in this format is unreliable.
    *   Messages from the user "system" are automatically generated and must be ignored.

Below is the chat history of The Target (`{nickname}`):

{chat_history}

## Task Requirements
Final Confirmation: Your analysis is focused on the chat characteristics of the user with nickname "{nickname}" (User ID: "{user_id}").

**数据量预判：** 如果目标用户的有效消息（`[我]` 标记的消息）总数少于20条，信息可能不足以进行深入分析。在这种情况下，你的分析应更加谨慎，并在 `description` 字段的末尾标注“(信息量较少，分析基于有限数据)”。同时，集中分析最明显的语言习惯和性格特征，避免过度推断

Now, direct your focus only to the messages marked with [我] (nickname: "{nickname}", User ID: "{user_id}") and perform an in-depth analysis of this real person's traits:

1.  **Personality Traits**: The authentic personality reflected in the messages (e.g., cheerful, introverted, humorous, rigorous, direct, gentle). **Do not just use these single words; provide a more detailed description.**
2.  **Linguistic Style**: The authentic way of speaking, vocabulary habits, and tone (e.g., colloquial, formal, uses internet slang, regional dialect, emoji usage). **Do not just use these single words; provide a more detailed description.**
3.  **Interests and Hobbies**: Frequently discussed topics, areas of focus, and subjects they are knowledgeable about.
4.  **Behavioral Patterns**: Interaction style, response habits, thought processes, and methods of expressing emotion.
5.  **Unique Characteristics**: Distinctive catchphrases, signature expressions, or special linguistic habits.
6.  **Core Identity / Worldview**: Based on the dialogue, summarize the person's core self-perception, social role (e.g., an optimistic student, a cynical office worker, a current affairs commentator), and their fundamental perspectives and values.
7.  **Detail Requirement**: You must describe every characteristic you analyze in extensive detail. Do not just use adjectives; provide specific, descriptive explanations. For example: "You prefer to reply with short sentences, often sent as multiple messages in a row," or "You like to send a complete, logically structured paragraph all at once." **The final persona content must be at least 2000-5000 words.**

## Required Return Format

You must strictly return the result in the following JSON format without any additional text or explanations (use Chinese labels and content unless otherwise specified):

```json
{{
  "name": "A concise persona name. Use the user's nickname directly or a suitable variation (only if the nickname is too complex or unsuitable as a persona name). Example: {nickname}",
  "title": "The display title for the persona. Example: {nickname}",
  "description": "A one-sentence summary of the character's core features (50-100 characters).",
  "content": "A detailed character profile (300-800 words). Use the second person ('You are...') to describe this real person's characteristics.",
  "tags": "【优化】Based on the analysis, generate 3-5 tags that best represent their core characteristics, covering dimensions like personality, interests, and linguistic style. Example: 吐槽役, 游戏爱好者, 细节控, 互联网黑话"
}}
```

---

## ⚠️ ATTENTION: `content` (内容) Writing Requirements (Important!)

### Incorrect Examples (AVOID):
- ❌ "You are an AI that needs to role-play as {nickname}"
- ❌ "As an AI character, you should imitate {nickname}"
- ❌ "Your mission is to learn {nickname}'s speaking style"
- ...or any other text that reminds the AI of its identity as an AI. This is incorrect as it may lead the AI to incorporate its own AI-like traits.

### Key Principles:
- ✅ **Describe the person directly**: "You are {nickname}, you like..."
- ✅ **Use natural, human-like descriptions**: As if introducing a real person.
- ✅ **Emphasize authenticity**: Extract features based *only* on the actual messages, without fabrication.
- ✅ **Be specific and clear**: Provide concrete guidance on their manner of speaking, e.g., "You tend to reply with short sentences, often sending them as multiple consecutive messages," or "You prefer to send a single, complete paragraph with clear logic."
- ✅ **Include typical expressions**: Quote the person's characteristic words or catchphrases as examples.
- ✅ **Stylistic Consistency**: When writing the `content`, try to adopt a tone that is close to the target user's linguistic style. For instance, if the user is witty and humorous, your description can be more light-hearted; if the user is formal and rigorous, the description should be more objective and structured.

## ⚠️ Final Checklist (Must Read)

Before you begin generating, confirm the following one last time:
1. ✅ The person I am analyzing is: **{nickname}** (User ID: {user_id})
2. ✅ I will only look at messages marked with `[我]`
3. ✅ I will not mix characteristics of Other User(s) into the persona (their messages are for context only!).
4. ✅ The generated `name` field must be "{nickname}".

If you have understood all the points above, begin your analysis now and generate the persona for **{nickname}**.

Remember:
- This is for creating a role-playing persona of a real person, not an instruction manual for an AI.
- The subject of your analysis is exclusively the person with the nickname "{nickname}".
- If you are uncertain whether a specific trait belongs to {nickname}, do not include it.
- Use Chinese labels and content unless otherwise specified.
"""

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
        # 忽略会话重置,查询所有历史消息
        conversation_start_timestamp = 0
        core.logger.info("已启用忽略会话重置时间,将收集所有历史消息")
    else:
        # 仅查询当前会话的消息
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
            .order_by("-send_timestamp")  # 按时间倒序
            .offset(offset)
            .limit(batch_size)
        )

        if not batch_messages:
            # 没有更多消息了
            core.logger.info(f"已到达消息历史末尾,共收集 {len(target_messages)} 条目标用户消息")
            break

        # 添加到结果集
        target_messages.extend(batch_messages)

        core.logger.info(
            f"第 {iteration + 1} 批: 加载 {len(batch_messages)} 条消息, "
            f"累计 {len(target_messages)} 条目标用户消息",
        )

        # 检查是否已达到目标数量
        if len(target_messages) >= max_count:
            # 截断到目标数量
            target_messages = target_messages[:max_count]
            core.logger.info(f"已达到目标消息数量 {max_count},停止加载")
            break

        # 准备下一批
        offset += batch_size

    if not target_messages:
        return []

    # 反转为时间正序
    target_messages = target_messages[::-1]

    # 2. 收集每条目标消息的上下文
    all_messages_dict: dict[int, DBChatMessage] = {}  # 使用 message_id 去重

    for target_msg in target_messages:
        # 添加目标消息本身
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

    # 3. 按时间排序并返回
    sorted_messages = sorted(all_messages_dict.values(), key=lambda m: m.send_timestamp)

    core.logger.info(
        f"收集到目标用户 {len(target_messages)} 条消息, 包含上下文共 {len(sorted_messages)} 条消息",
    )

    return sorted_messages


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
        from io import BytesIO

        img = Image.open(BytesIO(image_data))
        # 调整大小到合适的尺寸
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)

        # 保存为优化后的JPEG
        output = BytesIO()
        img.convert("RGB").save(output, format="JPEG", quality=85, optimize=True)
        compressed_data = output.getvalue()

        # 转换为base64
        base64_encoded = base64.b64encode(compressed_data).decode("utf-8")
    except Exception as e:
        core.logger.error(f"获取用户头像失败: {e}")
        return None
    else:
        return f"data:image/jpeg;base64,{base64_encoded}"


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
        # 提取纯文本内容
        content = msg.content_text.strip()

        if content:  # 只包含有文本内容的消息
            # 限制单条消息长度
            if len(content) > max_length:
                content = content[:max_length] + "..."

            # 格式化时间 - 只在时间间隔较大时显示
            time_str = ""
            if last_timestamp == 0 or (msg.send_timestamp - last_timestamp) > 3600:  # 超过1小时
                time_str = f"[{datetime.datetime.fromtimestamp(msg.send_timestamp).strftime('%Y-%m-%d %H:%M')}] "
            last_timestamp = msg.send_timestamp

            # 标注目标用户 - 使用特殊标记，并为非目标用户分配“用户N”显示名
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
                # 如果超出总长度限制,停止添加
                formatted_lines.append("[...更多消息已省略以控制长度...]")
                break

            formatted_lines.append(formatted_line)
            total_length += len(formatted_line) + 1  # +1 for newline

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
    import random

    # 生成所有索引
    indices = list(range(len(messages)))
    # 随机选择target_count个索引
    selected_indices = sorted(random.sample(indices, target_count))
    # 按索引提取消息
    return [messages[i] for i in selected_indices]


async def generate_preset_with_ai(
    user_id: str,
    nickname: str,
    messages: List[DBChatMessage],
    model_group_name: str,
    dialogs: Optional[List[List[DBChatMessage]]] = None,
) -> Dict[str, str]:
    """使用AI生成人设描述

    Args:
        user_id: 用户ID
        nickname: 用户昵称
        messages: 历史消息列表（包含上下文）
        model_group_name: 模型组名称

    Returns:
        Dict[str, str]: 包含人设信息的字典
    """
    # 统计消息（支持对话分段）
    flat_messages = [m for m in messages]
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

    core.logger.info(f"参考上下文内容： (长度: {len(chat_history)}) \n{chat_history}")

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
    # 新增：打印完整请求内容到日志
    core.logger.debug(f"完整请求： (长度: {len(prompt)}) \n{prompt}\n")
    
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
        temperature=0.7,  # 适当的创造性
    )

    core.logger.info(f"AI分析完成, 消耗Token: {llm_response.token_consumption}")

    # 解析AI返回的JSON
    response_text = llm_response.response_content.strip()

    # 清理和提取JSON内容
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
            f"解析AI返回的JSON失败: {e}\n"
            f"提取的JSON字符串:\n{json_str}\n"
            f"原始响应内容:\n{response_text}",
        )
        raise ValueError("AI返回的格式不正确,无法解析人设数据") from e

    # 5. 验证必需字段
    required_fields = ["name", "title", "description", "content", "tags"]
    missing_fields = [field for field in required_fields if field not in preset_data]
    if missing_fields:
        raise ValueError(f"AI返回的数据缺少必需字段: {', '.join(missing_fields)}")

    return preset_data


@on_command("preset-clone", aliases={"preset_clone"}, priority=5, block=True).handle()
async def _(matcher: Matcher, event: MessageEvent, bot: Bot, arg: Message = CommandArg()):
    """处理人设克隆命令"""
    # 去重：同一条消息只处理一次
    import time
    now = time.time()
    # 清理过期记录
    for mid, ts in list(_processed_message_ids.items()):
        if now - ts > _PROCESSED_MSG_TTL_SECONDS:
            _processed_message_ids.pop(mid, None)
    if event.message_id in _processed_message_ids:
        return
    _processed_message_ids[event.message_id] = now

    username, cmd_content, chat_key, chat_type = await command_guard(event, bot, arg, matcher)
    
    # 参数验证 - 这些会直接finish_with，不在try块中
    if chat_type != ChatType.GROUP:
        await finish_with(matcher, message="人设克隆功能仅支持群聊使用")
    
    if not cmd_content:
        await finish_with(
            matcher,
            message=("请指定要克隆的用户ID或@用户\n用法: /preset-clone <用户QQ号>\n或: /preset-clone @用户"),
        )
    
    # 解析用户标识
    target_user_id = cmd_content.strip()
    
    # 检查是否是@用户格式
    at_match = re.search(r"@(\d+)", target_user_id)
    if at_match:
        target_user_id = at_match.group(1)
    elif not target_user_id.isdigit():
        await finish_with(matcher, message="用户ID格式不正确,请使用QQ号或@用户")
    
    # 发送处理提示（通过去重确保只发送一次）
    await matcher.send(f"开始克隆用户 {target_user_id} 的人设,请稍候...")
    
    # 业务逻辑处理
    try:
        # 1. 收集用户历史消息及上下文
        core.logger.info(
            f"开始收集用户 {target_user_id} 的历史消息 "
            f"(前置上下文:{config.CONTEXT_BEFORE}, 后置上下文:{config.CONTEXT_AFTER}, IGNORE_SESSION_RESET:{config.IGNORE_SESSION_RESET})",
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
                f"至少需要 {config.MIN_MESSAGE_COUNT} 条消息才能进行克隆 (IGNORE_SESSION_RESET={config.IGNORE_SESSION_RESET})"
            )
            core.logger.warning(error_msg)
            await finish_with(matcher, message=error_msg)
    
        core.logger.info(
            f"收集完成: 目标用户 {target_message_count} 条, 总消息(含上下文) {len(messages)} 条",
        )
    
        # 获取用户昵称(使用最新的昵称)
        user_nickname = messages[-1].sender_nickname
    
        await matcher.send(
            f"已收集到目标用户 {target_message_count} 条消息(含上下文共 {len(messages)} 条),正在分析中...",
        )
        
        # 分段对话收集（按时间窗）
        dialogs = await collect_user_dialogs_with_context(
            user_id=target_user_id,
            chat_key=chat_key,
            max_dialogs=config.MAX_DIALOG_COUNT,
            context_before=config.CONTEXT_BEFORE,
            context_after=config.CONTEXT_AFTER,
            time_window_minutes=config.DIALOG_TIME_WINDOW_MINUTES,
        )
        if not dialogs:
            core.logger.info("未生成对话分段，改用平铺消息格式化")
            dialogs = None
        
        # 2. 使用AI生成人设
        preset_data = await generate_preset_with_ai(
            user_id=target_user_id,
            nickname=user_nickname,
            messages=messages,
            model_group_name=config.USE_CLONE_MODEL_GROUP,
            dialogs=dialogs,
        )
    
        # 3. 获取用户头像
        avatar_base64 = await get_user_avatar_base64(target_user_id, chat_key)
        if not avatar_base64:
            # 使用默认头像
            core.logger.warning("获取用户头像失败,使用默认头像")
            avatar_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
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
        # 检查是否是 nonebot 的异常（如 FinishedException）
        # 这些异常应该向上传播，不应该被捕获
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


# 新增：按时间窗与上下文分段的消息收集
async def collect_user_dialogs_with_context(
    user_id: str,
    chat_key: str,
    max_dialogs: int = 5,
    context_before: int = 3,
    context_after: int = 3,
    time_window_minutes: int = 15,
) -> List[List[DBChatMessage]]:
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
        window_messages: List[DBChatMessage] = (
            await DBChatMessage.filter(
                chat_key=chat_key,
                send_timestamp__gte=window_start,
                send_timestamp__lte=window_end,
            )
            .order_by("send_timestamp")
        )
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


# 新增：分段格式化输出
async def format_dialogs_for_analysis(
    dialogs: List[List[DBChatMessage]],
    target_user_id: str,
    max_length: int = 200,
    add_block_markers: bool = True,
) -> str:
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
    import random

    # 生成所有索引
    indices = list(range(len(messages)))
    # 随机选择target_count个索引
    selected_indices = sorted(random.sample(indices, target_count))
    # 按索引提取消息
    return [messages[i] for i in selected_indices]


async def generate_preset_with_ai(
    user_id: str,
    nickname: str,
    messages: List[DBChatMessage],
    model_group_name: str,
    dialogs: Optional[List[List[DBChatMessage]]] = None,
) -> Dict[str, str]:
    """使用AI生成人设描述

    Args:
        user_id: 用户ID
        nickname: 用户昵称
        messages: 历史消息列表（包含上下文）
        model_group_name: 模型组名称

    Returns:
        Dict[str, str]: 包含人设信息的字典
    """
    # 统计消息（支持对话分段）
    flat_messages = [m for m in messages]
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

    core.logger.info(f"克隆群友插件：分析上下文 (长度: {len(chat_history)}) \n{chat_history}\n")

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
    # 新增：打印完整请求内容到日志
    core.logger.debug(f"克隆群友插件：完整请求 (长度: {len(prompt)}) \n{prompt}\n")
    
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
        temperature=0.7,  # 适当的创造性
    )

    core.logger.info(f"AI分析完成, 消耗Token: {llm_response.token_consumption}")

    # 解析AI返回的JSON
    response_text = llm_response.response_content.strip()

    # 清理和提取JSON内容
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
            f"解析AI返回的JSON失败: {e}\n"
            f"提取的JSON字符串:\n{json_str}\n"
            f"原始响应内容:\n{response_text}",
        )
        raise ValueError("AI返回的格式不正确,无法解析人设数据") from e

    # 5. 验证必需字段
    required_fields = ["name", "title", "description", "content", "tags"]
    missing_fields = [field for field in required_fields if field not in preset_data]
    if missing_fields:
        raise ValueError(f"AI返回的数据缺少必需字段: {', '.join(missing_fields)}")

    return preset_data


@on_command("preset-clone", aliases={"preset_clone"}, priority=5, block=True).handle()
async def _(matcher: Matcher, event: MessageEvent, bot: Bot, arg: Message = CommandArg()):
    """处理人设克隆命令"""
    # 去重：同一条消息只处理一次
    import time
    now = time.time()
    # 清理过期记录
    for mid, ts in list(_processed_message_ids.items()):
        if now - ts > _PROCESSED_MSG_TTL_SECONDS:
            _processed_message_ids.pop(mid, None)
    if event.message_id in _processed_message_ids:
        return
    _processed_message_ids[event.message_id] = now

    username, cmd_content, chat_key, chat_type = await command_guard(event, bot, arg, matcher)
    
    # 参数验证 - 这些会直接finish_with，不在try块中
    if chat_type != ChatType.GROUP:
        await finish_with(matcher, message="人设克隆功能仅支持群聊使用")
    
    if not cmd_content:
        await finish_with(
            matcher,
            message=("请指定要克隆的用户ID或@用户\n用法: /preset-clone <用户QQ号>\n或: /preset-clone @用户"),
        )
    
    # 解析用户标识
    target_user_id = cmd_content.strip()
    
    # 检查是否是@用户格式
    at_match = re.search(r"@(\d+)", target_user_id)
    if at_match:
        target_user_id = at_match.group(1)
    elif not target_user_id.isdigit():
        await finish_with(matcher, message="用户ID格式不正确,请使用QQ号或@用户")
    
    # 发送处理提示（通过去重确保只发送一次）
    await matcher.send(f"开始克隆用户 {target_user_id} 的人设,请稍候...")
    
    # 业务逻辑处理
    try:
        # 1. 收集用户历史消息及上下文
        core.logger.info(
            f"开始收集用户 {target_user_id} 的历史消息 "
            f"(前置上下文:{config.CONTEXT_BEFORE}, 后置上下文:{config.CONTEXT_AFTER}, IGNORE_SESSION_RESET:{config.IGNORE_SESSION_RESET})",
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
                f"至少需要 {config.MIN_MESSAGE_COUNT} 条消息才能进行克隆 (IGNORE_SESSION_RESET={config.IGNORE_SESSION_RESET})"
            )
            core.logger.warning(error_msg)
            await finish_with(matcher, message=error_msg)
    
        core.logger.info(
            f"收集完成: 目标用户 {target_message_count} 条, 总消息(含上下文) {len(messages)} 条",
        )
    
        # 获取用户昵称(使用最新的昵称)
        user_nickname = messages[-1].sender_nickname
    
        await matcher.send(
            f"已收集到目标用户 {target_message_count} 条消息(含上下文共 {len(messages)} 条),正在分析中...",
        )
        
        # 分段对话收集（按时间窗）
        dialogs = await collect_user_dialogs_with_context(
            user_id=target_user_id,
            chat_key=chat_key,
            max_dialogs=config.MAX_DIALOG_COUNT,
            context_before=config.CONTEXT_BEFORE,
            context_after=config.CONTEXT_AFTER,
            time_window_minutes=config.DIALOG_TIME_WINDOW_MINUTES,
        )
        if not dialogs:
            core.logger.info("未生成对话分段，改用平铺消息格式化")
            dialogs = None
        
        # 2. 使用AI生成人设
        preset_data = await generate_preset_with_ai(
            user_id=target_user_id,
            nickname=user_nickname,
            messages=messages,
            model_group_name=config.USE_CLONE_MODEL_GROUP,
            dialogs=dialogs,
        )
    
        # 3. 获取用户头像
        avatar_base64 = await get_user_avatar_base64(target_user_id, chat_key)
        if not avatar_base64:
            # 使用默认头像
            core.logger.warning("获取用户头像失败,使用默认头像")
            avatar_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
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
        # 检查是否是 nonebot 的异常（如 FinishedException）
        # 这些异常应该向上传播，不应该被捕获
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
