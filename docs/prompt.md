# 插件初始提示词存档

现在我们需要开发一个新的插件，具体是通过命令指定克隆某个用户，模仿其语言和互动风格，生成一份该用户的人设信息，并且添加到本地人设中方便用户后续使用，具体运行流程如下：

1. 注册 /preset-clone 命令，让用户通过该命令指定要克隆的目标用户
2. 通过数据库查询收集该用户的历史发言消息、头像（只支持onebot适配器）、昵称等信息
3. 构造合适的提示词，包括任务需求、返回约定、生成标准、内容引导等，调用LLM生成所需的人设信息
4. 将生成结果人设添加到本地人设数据中

你可能需要的支持有：

1. 插件的基本架构和开发规则说明： @__init__.py  @plugin-rules.mdc 
2. 消息历史记录数据库查询方式： @history_travel.py 
3. LLM配置与调用： @generate_chat_response_with_image_support  @plugin.py （这里的 model_type 应该用 chat）
4. 人设相关编辑工具与方法用法：（来自其他插件片段）

from nekro_agent.models.db_chat_channel import DefaultPreset
from nekro_agent.models.db_preset import DBPreset

...

@plugin.mount_on_user_message()
async def on_user_message(_ctx: AgentCtx, message: ChatMessage) -> MsgSignal:
    """用户消息处理器 - 提供AI生成回复前的友好提示"""
    bot_type: ChatBotType = await get_chat_bot_type(_ctx.chat_key)

    use_preset: Optional[Union[DBPreset, DefaultPreset]] = None
    if bot_type == ChatBotType.EQUIPMENT:
        use_preset = await _ctx.get_preset_by_id(config.EQUIPMENT_SERVE_PRESET_ID)
    elif bot_type == ChatBotType.SUPPLIER:
        use_preset = await _ctx.get_preset_by_id(config.SUPPLIER_SERVE_PRESET_ID)
    if not use_preset:
        return MsgSignal.CONTINUE

    current_preset = await _ctx.current_preset()

    if use_preset.id != (getattr(current_preset, "id", -1)):
        logger.info(
            f"会话 {_ctx.chat_key} 应用人设不一致，切换到人设: {use_preset.name}",
        )
        await _ctx.set_preset(use_preset.id)

...

5. 人设编辑相关接口： @presets.py 
6. 指令注册与使用： @ai_voice.py 
7. 用户头像文件获取： @get_user_avatar （不要直接引用这个方法，而是仿造其实现，因为这是另一个插件的功能）
现在综合上述所有信息，结合对我们应用基础的理解，深度合理分析与细化这个插件需求，在我提供的插件模板基础上改造来实现这个插件
