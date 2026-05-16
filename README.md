# 🎙️ SMT Beta — Stream Moment Translator

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
| 🍪 **Cookies 防风控** | 挂载 cookies.txt 绕过 YouTube 登录墙 |
| 💬 **字幕悬浮窗** | 独立置顶字幕窗，像视频字幕一样叠加在画面上，可拖拽 |
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

首次运行会自动在 `src/` 目录生成 `config.json`。用任意文本编辑器（记事本、VS Code 等）打开编辑：

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

#### 🔑 API 与 LLM 设置（必填）

| 字段 | 说明 |
|---|---|
| `api_key` | LLM API 密钥，**留空则无法翻译** |
| `api_base_url` | API 地址（兼容 OpenAI 协议的均可） |
| `llm_model_name` | 模型名称 |
| `proxy_url` | HTTP 代理地址，不需要则设为空字符串 `""` |

**不同 LLM 服务商的配置示例：**

| 服务商 | `api_base_url` | `llm_model_name` 示例 |
|---|---|---|
| 阿里百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.6-plus` / `qwen-turbo` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-flash` |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` / `deepseek-ai/DeepSeek-V3` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` / `gpt-4o-mini` |
| 其他 Ollama 本地模型 | `http://localhost:11434/v1` | `llama3` / `qwen3.5` |

> **省钱建议**：日常翻译用 `qwen3.5-plus`（百炼）或 `deepseek-v4-flash`（DeepSeek），速度快且成本极低；对翻译质量要求高时换 `qwen3.6-plus` 或 `gpt-5.5`。

#### 🧠 Whisper 听写模型设置

| 字段 | 说明 | 可选值 |
|---|---|---|
| `whisper_model_size` | 模型大小 | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `whisper_device` | 运行设备 | `cuda`（NVIDIA 显卡）/ `cpu`（纯 CPU） |
| `whisper_compute_type` | 计算精度 | `float16`（GPU）/ `int8`（CPU 或低显存 GPU） |

**模型选择指南：**

| 模型 | 大小 | 显存占用 | 速度 | 精度 | 适用场景 |
|---|---|---|---|---|---|
| `tiny` | ~150MB | ~1GB | 极快 | 一般 | 低配电脑、实时性要求极高 |
| `base` | ~290MB | ~1GB | 很快 | 尚可 | 入门级 GPU / CPU 用户 |
| `small` | ~950MB | ~2GB | 快 | 较好 | 4GB 显存显卡 |
| `medium` | ~3GB | ~5GB | 中等 | 好 | 6GB 显存显卡 |
| `large-v3` | ~3GB | ~6GB+ | 较慢 | 最佳 | 8GB+ 显存显卡，追求最佳识别率 |

**CPU vs CUDA 怎么选：**

| 方式 | 配置 | 适用人群 |
|---|---|---|
| **CUDA（推荐）** | `"whisper_device": "cuda"` + `"whisper_compute_type": "float16"` | 有 NVIDIA 显卡（GTX 1060 以上），速度是 CPU 的 5-20 倍 |
| **CPU** | `"whisper_device": "cpu"` + `"whisper_compute_type": "int8"` | 无独显 / AMD 显卡 / Intel 核显，模型建议选 `base` 或 `small` |
| **低显存 CUDA** | `"whisper_device": "cuda"` + `"whisper_compute_type": "int8"` | 显存 4GB，模型选 `small` 或 `base` |

> **注意**：`faster-whisper` 首次运行会自动从 HuggingFace 下载模型文件，请确保网络通畅。若下载失败，可设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。

#### 🎯 其他设置

| 字段 | 说明 |
|---|---|
| `stream_context` | 全局背景设定，如 `"这是一场虚拟主播杂谈直播"`，帮助 LLM 理解语境 |
| `target_lang` | 翻译目标语言，如 `"中文 (Chinese)"` / `"English"` / `"日本語"` |
| `cookies_path` | YouTube cookies.txt 完整路径，浏览器登录 YouTube 后用插件导出 |

### 5. 运行

```bash
cd src
python stream_beta.py
```

弹出桌面窗口，粘贴 URL 点击「▶ 开始」即可。

---

## 📖 使用说明

### 两个窗口

**控制面板**（主窗口）— 所有操作在这里：

```
┌──────────────────────────────────────────────────┐
│ URL: [https://youtube.com/watch?v=...]  [📁 浏览]│
│ 起播: [00:00:00]  跳过: [__:__-__:__]  仅译: [ ] │
│ [▶ 开始] [■ 停止] [⏸ 暂停] [▶ 继续] [💬 字幕:开] │
├──────────────────────────────────────────────────┤
│ 术语字典 / 主题设定                      [🔄 同步]│
│ ┌──────────────────────────────────────────────┐ │
│ │ 主题：科技发布会                              │ │
│ │ GPU = 图形处理器                              │ │
│ └──────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────┤
│ 翻译记录（历史回溯）                              │
│ ⏱️ [00:05:23]                                    │
│ 🗣️ Hello everyone                                │
│ 💬 大家好                                         │
└──────────────────────────────────────────────────┘
```

**字幕悬浮窗**（独立窗口）— 叠加在视频上：

```
┌─────────────────────────────────────────────┐
│ [00:05:23]  Hello everyone, welcome!         │
│ 大家好，欢迎来到直播间                         │
└─────────────────────────────────────────────┘
```
- 始终置顶，可拖拽到屏幕任意位置
- 右键菜单可隐藏/显示
- 主窗口「💬 字幕」按钮控制开关

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

### 字幕悬浮窗

字幕窗默认开启，是一个独立置顶窗口，像视频内嵌字幕一样浮在屏幕底部。可拖拽到任意位置，右键菜单可隐藏/显示。主窗口的「💬 字幕」按钮控制开关。

---

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
A: 下载 [ffmpeg](https://ffmpeg.org) 并将 `bin` 目录加入系统 PATH。Windows 可用 `winget install ffmpeg`，或手动下载后把 `ffmpeg.exe` 所在目录添加到环境变量。

**Q: Whisper 模型下载失败 / 网络错误？**
A: HuggingFace 在国内可能被墙。设置镜像环境变量后重试：
```bash
set HF_ENDPOINT=https://hf-mirror.com
python stream_beta.py
```
模型只下载一次，之后离线可用。

**Q: 没显卡 / AMD 显卡 / 显存不够怎么办？**
A: 修改 `config.json` 中三项：
```json
"whisper_model_size": "base",
"whisper_device": "cpu",
"whisper_compute_type": "int8"
```
- `base` 模型只有 290MB，CPU 也能流畅跑
- 如果 CPU 性能较好（如 R7/i7 以上），可以尝试 `small` 模型
- AMD 显卡用户目前只能选 `cpu`（faster-whisper 不支持 ROCm/HIP）

**Q: 有 NVIDIA 显卡但提示 CUDA 错误？**
A: 
- 确认已安装 NVIDIA 驱动：命令行执行 `nvidia-smi` 看是否有输出
- 确认 CUDA 版本 ≥ 11.8：`nvidia-smi` 右上角有 CUDA Version
- 如果 CUDA 版本太老（10.x 等），升级驱动即可

**Q: 如何确认 Whisper 跑在 GPU 上？**
A: 启动后在控制面板状态栏看到 `🟢 Whisper 就绪 (CUDA)` 即表示 GPU 模式。如果显示 `CPU` 说明没挂上 GPU，检查 `whisper_device` 配置。

**Q: 如何切换 LLM（大模型）服务商？**
A: 修改 `config.json` 中三处：
```json
// 例1：切换为 DeepSeek
"api_key": "sk-xxxxxxxx",
"api_base_url": "https://api.deepseek.com",
"llm_model_name": "deepseek-chat"

// 例2：切换为本地 Ollama
"api_key": "ollama",
"api_base_url": "http://localhost:11434/v1",
"llm_model_name": "qwen2.5:7b"
```

**Q: YouTube 解析失败？**
A: 分情况处理：
- **提示 Sign in / pass-cookies** → YouTube 要求登录验证。用浏览器插件（Get cookies.txt LOCALLY）导出 cookies.txt，放在任意位置，在 `cookies_path` 中填完整路径
- **提示网络超时** → 代理问题，检查 `proxy_url` 是否正确，或尝试留空直连
- **yt-dlp 版本过旧** → `pip install --upgrade yt-dlp`

**Q: 翻译结果不准确？**
A: 三道防线提升质量：
1. 术语字典：添加该视频的专有名词对照（人名、地名、圈内黑话）
2. 主题设定：写一句背景描述，帮助 LLM 理解语境
3. 换更强模型：`llm_model_name` 换成 `qwen3.5-plus` 或 `gpt-4o`

**Q: API 调用费用高吗？**
A: 两道省钱机制：
- 历史去重：重播同一视频时命中缓存直接跳过 API
- 推荐用小模型（qwen-turbo / deepseek-chat），百万 token 仅几毛钱

---

## 📄 许可证

MIT License
