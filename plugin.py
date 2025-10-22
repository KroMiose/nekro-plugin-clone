"""插件配置和元数据"""

from pydantic import Field

from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin

# 创建插件实例
plugin = NekroPlugin(
    name="克隆群友Bata",
    module_name="nekro_plugin_clone",
    description="通过分析用户历史发言自动克隆群友并生成 AI 人设",
    version="1.0.3",
    author="KroMiose",
    url="https://github.com/KroMiose/nekro-plugin-clone",
    support_adapter=["onebot_v11"],
)


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
        le=100,
    )
    CONTEXT_AFTER: int = Field(
        default=1,
        title="后置上下文消息数",
        description="收集目标用户每条消息后的上下文消息数量",
        ge=0,
        le=100,
    )
    DIALOG_TIME_WINDOW_MINUTES: int = Field(
        default=15,
        title="对话时间窗口（分钟）",
        description="单个对话的最大持续时间（从前置消息到后置消息），用于筛选符合时间条件的消息",
        ge=1,
        le=14400,
    )
    MAX_DIALOG_COUNT: int = Field(
        default=8,
        title="最大对话段数",
        description="按时间窗口与上下文规则分段的最大对话数量，用于格式化输出",
        ge=1,
        le=10240,
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
        ge=100,
        le=2000,
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
    ENABLE_PROMPT_DEBUG: bool = Field(
        default=False,
        title="启用提示词Debug日志",
        description="开启后将把发送给AI的完整提示词打印到日志（可能很长）",
    )


# 获取配置实例
config: PresetCloneConfig = plugin.get_config(PresetCloneConfig)

