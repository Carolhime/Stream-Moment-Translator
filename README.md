#  SMT Beta — Stream Moment Translator

**实时 AI 同声传译桌面应用** — 输入直播/视频链接，自动捕获音频 → Whisper 语音识别 → LLM 翻译，逐句实时输出双语字幕。

Real-time AI simultaneous interpretation desktop app. Feed it a stream/video URL, and it captures audio → transcribes via Whisper → translates via LLM, displaying bilingual subtitles in real time.

---

## ⚠️ 重要声明

本软件为开源软件，完全免费。使用前请自行申请所需 API Key。

---

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| 🎯 **实时同声传译** | 按语音停顿逐句识别+翻译，不是事后整段处理 |
| 🌐 **多源支持** | YouTube 直播/录播、m3u8、本地视频/音频文件 |
| 🧠 **Whisper 语音识别** | 基于 faster-whisper，支持 CUDA GPU 加速，可切换模型大小 |
| 🤖 **LLM 翻译** | 通过 OpenAI 兼容 API 调用任意大模型（千问、GPT、DeepSeek 等） |
| 📖 **术语字典** | 自定义强制翻译对照表，确保专有名词精准翻译 |
| 🎭 **主题设定** | 为直播/视频指定背景设定，提升翻译贴合度 |
| 🧠 **记忆引擎** | 每 3 分钟自动压缩上下文，让 LLM 理解话题演变 |
| 💾 **历史去重** | 重播时命中相同时间戳直接复用缓存，省 API 费用 |
| ⏱️ **时间区间控制** | 支持跳过广告段、仅翻译指定时间段 |
| 🔌 **代理支持** | 配置 HTTP 代理访问境外流媒体和 API |
| 🖥️ **原生桌面 UI** | 基于 tkinter，零 Web 依赖，轻量直接 |

---

## 🚀 快速开始

### 1. 环境要求

- **Python 3.10+**
- **ffmpeg**（需在系统 PATH 中可用）
- **NVIDIA 显卡 + CUDA**（可选，CPU 也能跑但较慢）

### 2. 安装依赖

```bash
pip install numpy openai faster-whisper yt-dlp
```

> `faster-whisper` 首次运行会自动下载 Whisper 模型（large-v3 约 3GB），请确保网络通畅。

### 3. 获取 API Key

SMT Beta 默认使用**阿里百炼 DashScope** 做翻译（兼容 OpenAI 协议），也支持任意兼容接口。

- 阿里百炼：https://dashscope.aliyuncs.com
- 硅基流动：https://siliconflow.cn
- DeepSeek：https://platform.deepseek.com
- OpenAI：https://platform.openai.com

### 4. 配置文件

首次运行会自动在 `src/` 目录生成 `config.json`：

```json
{
    "api_key": "你的API_KEY",
    "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "llm_model_name": "qwen3.5-plus",
    "proxy_url": "http://127.0.0.1:7897",
    "whisper_model_size": "large-v3",
    "whisper_device": "cuda",
    "whisper_compute_type": "float16",
    "stream_context": "",
    "target_lang": "中文 (Chinese)",
    "cookies_path": ""
}
```

| 字段 | 说明 |
|---|---|
| `api_key` | LLM API 密钥（必填） |
| `api_base_url` | API 地址，换用其他模型时改这里 |
| `llm_model_name` | 模型名，如 `qwen3.5-plus` / `gpt-4o` / `deepseek-chat` |
| `proxy_url` | HTTP 代理地址，不需要则留空 |
| `whisper_model_size` | Whisper 模型：`tiny` / `base` / `small` / `medium` / `large-v3` |
| `whisper_device` | `cuda`（NVIDIA GPU）或 `cpu` |
| `whisper_compute_type` | GPU 用 `float16`，CPU 用 `int8` |
| `stream_context` | 全局背景设定，如 "这是一场科技发布会" |
| `target_lang` | 目标翻译语言 |
| `cookies_path` | YouTube cookies.txt 路径（遇到需登录时使用） |

### 5. 运行

```bash
cd src
python stream_beta.py
```

弹出桌面窗口，粘贴 URL 点击「▶ 开始」即可。

---

## 📖 使用说明

### 主界面

```
┌──────────────────────────────────────────────────┐
│ URL: [https://youtube.com/watch?v=...]  [📁 浏览]│
│ 起播: [00:00:00]  跳过: [__:__-__:__]  仅译: [ ] │
│ [▶ 开始]  [■ 停止]  [⏸ 暂停]  [▶ 继续]  [📋 迷你]│
├──────────────────────────────────────────────────┤
│ 术语字典 / 主题设定                      [🔄 同步]│
│ ┌──────────────────────────────────────────────┐ │
│ │ 主题：科技发布会                              │ │
│ │ GPU = 图形处理器                              │ │
│ │ LLM = 大语言模型                              │ │
│ └──────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────┤
│ 翻译输出                                         │
│ ⏱️ [00:05:23]                                    │
│ 🗣️ Hello everyone, welcome to the stream         │
│ 💬 大家好，欢迎来到直播间                          │
└──────────────────────────────────────────────────┘
```

### 支持的输入

- **YouTube**: 完整链接或短链 `youtu.be/xxx`
- **直播流**: m3u8、flv 等直链
- **本地文件**: 绝对路径，如 `D:\videos\stream.mp4`，或点击「📁 浏览」选择

### 时间区间

| 字段 | 格式 | 说明 |
|---|---|---|
| 起播 | `时:分:秒` | 从视频指定时间开始翻译 |
| 跳过 | `开始-结束, 开始-结束` | 跳过不需要翻译的段落 |
| 仅译 | `开始-结束` | 白名单模式，只翻译指定区间 |

示例：`跳过: 00:30-02:00, 15:00-16:30`（跳过两段广告）

### 术语字典

在文本框编辑后点「🔄 同步」生效，支持以下格式：

```
# 强制术语
Hololive = ホロライブ
NFT = 数字藏品

# 主题设定（以 主题:/背景:/设定: 开头）
主题：虚拟主播杂谈直播，主播在聊日常生活

# 也支持
VTuber 是 虚拟主播
GPU -> 图形处理器
```

## 🏗️ 项目结构

```
Stream Moment Translator/
├── src/
│   ├── stream_beta.py       # 主程序
│   ├── config.json          # 配置文件（自动生成）
│   └── exports_beta/        # 翻译记录存档
├── requirements.txt         # Python 依赖
└── README.md
```

---

## 🔧 常见问题

**Q: 提示找不到 ffmpeg？**
A: 下载 [ffmpeg](https://ffmpeg.org) 并将 `bin` 目录加入系统 PATH。Windows 可用 `winget install ffmpeg`。

**Q: Whisper 加载失败或显存不足？**
A: 修改 `config.json`：`whisper_device` 改 `cpu`，`whisper_model_size` 改 `base` 或 `small`。

**Q: YouTube 解析失败？**
A: 常见原因：
- 直播需要登录 → 用浏览器插件导出 cookies.txt，在 `cookies_path` 中指定路径
- 代理节点被拒 → 更换代理地址
- yt-dlp 版本过旧 → `pip install --upgrade yt-dlp`

**Q: 翻译不准？**
A: 在术语字典中添加专有名词对照，并写一句「主题」描述直播背景。

**Q: API 调用贵不贵？**
A: 历史去重已经帮你跳过了重复片段。翻译用小模型（如 qwen-turbo）成本极低。

---

## 📄 许可证

MIT License
