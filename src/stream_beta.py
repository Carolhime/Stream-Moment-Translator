"""
SMT Beta — 完整功能桌面版同声传译引擎
纯 tkinter 原生界面，零 Web 依赖，完整保留原版全部功能。
直接运行: python "stream moment translator_beta.py"
"""
import platform, re, os, sys, time, json, queue, threading, subprocess, hashlib
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

# ── 编码与 GPU DLL 路径 ──
if sys.stdout and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

try:
    import site
    for sp in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            p = os.path.join(sp, *sub.split("/"))
            if os.path.exists(p):
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, 'add_dll_directory'): os.add_dll_directory(p)
except Exception:
    pass

from openai import OpenAI
from faster_whisper import WhisperModel

CREATE_NO_WINDOW = 0x08000000

# ═══════════════════════════════════════════════════════════════
# 完整引擎（原版全部功能）
# ═══════════════════════════════════════════════════════════════
class SMTEngine:
    def __init__(self):
        self.is_listening = False
        self.is_paused = False
        self.model = None
        self.llm_client = None
        self.ffmpeg_process = None
        self.current_video_time = 0.0
        self.start_time_str = "00:00:00"
        self.skip_times_str = ""
        self.target_times_str = ""
        self.skip_zones = []
        self.target_zones = []

        # 队列系统
        self.task_queue = queue.Queue()
        self.broadcast_queue = queue.Queue()

        # 记录与去重系统
        self.existing_records = {}
        self.log_file_path = ""
        self.export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports_beta")
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        # 记忆引擎与上下文系统
        self.audio_buffer = bytearray()
        self.context_history = []
        self.unsummarized_source = []
        self.global_summary = ""
        self.active_dict_str = ""
        self.last_summary_video_time = 0.0
        self.api_key = ""

        self.load_config()
        threading.Thread(target=self._whisper_worker, daemon=True).start()

        if hasattr(self, 'stream_context') and self.stream_context:
            self.sys_log(f"✨ [系统] 加载初始主题成功！AI 已植入专属设定。")
        else:
            self.sys_log(f"💡 [系统] 未检测到特殊主题，使用标准同传模式。")

    # ── 时间处理 ──
    def parse_time_to_seconds(self, time_str):
        if not time_str.strip(): return 0
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2: return parts[0] * 60 + parts[1]
        return 0

    def parse_time_zones(self, input_str):
        zones = []
        if not input_str.strip(): return zones
        ranges = input_str.split(',')
        for r in ranges:
            try:
                start_str, end_str = r.split('-')
                start_sec = self.parse_time_to_seconds(start_str.strip())
                end_sec = self.parse_time_to_seconds(end_str.strip())
                zones.append((start_sec, end_sec))
            except Exception: pass
        return zones

    def check_time_zone_action(self, current_sec):
        if len(self.target_zones) > 0:
            in_target = any(s <= current_sec <= e for s, e in self.target_zones)
            if not in_target: return False, "不在白名单"
        if len(self.skip_zones) > 0:
            if any(s <= current_sec <= e for s, e in self.skip_zones): return False, "黑名单拦截"
        return True, ""

    def format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"[{h:02d}:{m:02d}:{s:02d}]"

    # ── 日志 ──
    def sys_log(self, text: str):
        timestamp = time.strftime("[%H:%M:%S]")
        log_line = f"{timestamp} {text}"
        print(log_line)
        self.broadcast_queue.put({"type": "log", "content": text})

    # ── 历史记录与去重 ──
    def load_history_records(self):
        self.existing_records.clear()
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            pattern = r"⏱️ \[(\d{2}:\d{2}:\d{2})\]\n🗣️ ([^\n]+)\n💬 ([^\n]+)"
            matches = re.findall(pattern, content)
            for match in matches:
                time_str, source_text, translated_text = match
                sec = self.parse_time_to_seconds(time_str)
                self.existing_records[sec] = (source_text.strip(), translated_text.strip())
            if len(self.existing_records) > 0:
                self.sys_log(f"💾 [本地记忆] 成功唤醒 {len(self.existing_records)} 条历史记录，遇重复片段将免 API 直出！")
        except Exception as e:
            self.sys_log(f"🟡 [本地记忆] 读取历史记录失败: {e}")

    def find_existing_record(self, current_sec, tolerance=2):
        for s in self.existing_records:
            if abs(s - current_sec) <= tolerance: return self.existing_records[s]
        return None

    def display_memory_record(self, source_text, translated_text, video_timestamp_sec):
        time_str = self.format_time(video_timestamp_sec)
        display_text = f"⏱️ {time_str}\n🗣️ {source_text}\n💬 {translated_text}\n\n"
        self.context_history.append({"text": f"原文: {source_text} | 译文: {translated_text}"})
        if len(self.context_history) > 4: self.context_history.pop(0)
        self.broadcast_queue.put({
            "type": "translation",
            "display_text": display_text,
            "source": source_text,
            "target": translated_text,
            "timestamp": time_str
        })

    # ── 配置加载 ──
    def load_config(self):
        default_config = {
            "api_key": "",
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.config_path = os.path.join(parent_dir, "config.json")
        if not os.path.exists(self.config_path):
            self.config_path = os.path.join(current_dir, "config.json")
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.sys_log(f"[配置] 读取异常，使用默认配置: {e}")
        else:
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
            except: pass

        self.config = default_config
        self.stream_context = self.config.get("stream_context", "").strip()
        self.target_lang = self.config.get("target_lang", "中文 (Chinese)")
        self.api_key = self.config.get("api_key", "")

        proxy = self.config.get("proxy_url", "")
        if proxy:
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy

        threading.Thread(target=self.load_whisper_model).start()

    def load_whisper_model(self):
        try:
            device_str = self.config['whisper_device'].upper()
            self.sys_log(f"[系统] 开始基于外部配置加载 Whisper 引擎...")
            self.model = WhisperModel(
                self.config["whisper_model_size"],
                device=self.config["whisper_device"],
                compute_type=self.config["whisper_compute_type"]
            )
            self.sys_log(f"🟢 [系统] Whisper 引擎加载成功！运行在 {device_str} 上。")
        except Exception as e:
            self.sys_log(f"🔴 [致命错误] Whisper 加载失败: {e}")
            self.sys_log("💡 提示：如果是小白电脑，请在 config.json 将 device 改为 cpu，model 改为 base。")

    # ── 启动 / 停止 / 暂停 / 恢复 ──
    def start_listening(self, stream_url: str, start_time="00:00:00", skip_times="", target_times=""):
        if not self.model:
            self.sys_log("[警告] 语音模型尚未准备完毕。")
            return False
        if self.is_listening:
            return False
        source = stream_url.strip().strip('"').strip("'")
        if not source: return False

        # 生成日志文件路径
        if source.startswith("http"):
            if "youtube.com" in source or "youtu.be" in source:
                match = re.search(r'(?:v=|youtu\.be/)([^&?]+)', source)
                if match:
                    video_id = match.group(1)
                    self.log_file_path = os.path.join(self.export_dir, f"YT_{video_id}_同传记录.txt")
                else:
                    self.log_file_path = os.path.join(self.export_dir, f"Web_{hashlib.md5(source.encode()).hexdigest()[:6]}_同传记录.txt")
            else:
                self.log_file_path = os.path.join(self.export_dir, f"Web_{hashlib.md5(source.encode()).hexdigest()[:6]}_同传记录.txt")
        else:
            basename = os.path.basename(source)
            name_part = os.path.splitext(basename)[0][:7]
            self.log_file_path = os.path.join(self.export_dir, f"Local_{name_part}_同传记录.txt")

        self.sys_log(f"📄 [文档归档] 本次记录将保存在: {self.log_file_path}")
        self.load_history_records()

        self.is_listening = True
        self.is_paused = False
        self.start_time_str = start_time
        self.skip_zones = self.parse_time_zones(skip_times)
        self.target_zones = self.parse_time_zones(target_times)
        start_sec = self.parse_time_to_seconds(start_time)
        self.current_video_time = start_sec

        self.audio_buffer.clear()
        self.task_queue.queue.clear()
        self.unsummarized_source.clear()
        self.context_history.clear()
        self.last_summary_video_time = start_sec
        self.global_summary = ""
        self.broadcast_queue.put({"type": "theme_update", "content": ""})

        self.sys_log(f"🔵 准备执行... 起播: {start_sec}秒")
        threading.Thread(target=self._ffmpeg_listen_thread, args=(source,), daemon=True).start()
        return True

    def stop_listening(self):
        self.is_listening = False
        if self.ffmpeg_process:
            try: self.ffmpeg_process.kill()
            except: pass
        self.sys_log("🟡 [手动断开] 已切断媒体流。")
        return True

    # ── FFmpeg 音频捕获线程（完整版） ──
    def _ffmpeg_listen_thread(self, input_source):
        start_sec = self.parse_time_to_seconds(self.start_time_str)
        PROXY_URL = self.config.get("proxy_url", "").strip()
        COOKIES_PATH = self.config.get("cookies_path", "").strip()

        try:
            ffmpeg_cmd = ['ffmpeg', '-y']

            if input_source.startswith("http") and not input_source.endswith((".m3u8", ".flv", ".mp3", ".wav", ".aac", ".mp4")):
                self.sys_log(f"🌐 [嗅探雷达] 正在请求 JSON 密码本{' (使用代理)' if PROXY_URL else ' (直连模式)'}...")
                ytdlp_cmd = [
                    'yt-dlp', '-f', 'bestaudio/best', '--dump-json',
                    '--quiet', '--no-warnings', '--no-check-certificate'
                ]
                if COOKIES_PATH and os.path.exists(COOKIES_PATH):
                    self.sys_log("🍪 [高级防风控] 已挂载本地 cookies.txt 通行证！")
                    ytdlp_cmd.extend(['--cookies', COOKIES_PATH])
                if PROXY_URL:
                    ytdlp_cmd.extend(['--proxy', PROXY_URL])
                ytdlp_cmd.append(input_source)

                process = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           creationflags=CREATE_NO_WINDOW, text=True)
                out, err = process.communicate()

                if process.returncode != 0:
                    err_msg = err.strip()
                    if "pass-cookies-to-yt-dlp" in err_msg or "Sign in" in err_msg:
                        self.sys_log("🔴 [风控拦截]: YouTube 拒绝了代理节点，或该直播需要登录。")
                        self.sys_log("💡 [解决方案]: 请导入 cookies.txt 后重试！")
                    else:
                        self.sys_log(f"🔴 [yt-dlp 解析失败]: {err_msg[-200:]}")
                    self.stop_listening()
                    return

                try:
                    info = json.loads(out)
                    real_url = info.get('url')
                    headers = info.get('http_headers', {})
                    is_live = info.get('is_live', False)
                    self.sys_log(f"✅ [解析成功] 识别为: {'🔴 直播流' if is_live else '🎬 录播'}")

                    if start_sec > 0 and not is_live:
                        ffmpeg_cmd.extend(['-ss', str(start_sec)])
                        self.sys_log(f"⏩ [寻址] 快进至 {start_sec} 秒处")

                    if headers:
                        user_agent = headers.pop('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36')
                        ffmpeg_cmd.extend(['-user_agent', user_agent])
                        header_str = "".join([f"{k}: {v}\r\n" for k, v in headers.items()])
                        if header_str:
                            ffmpeg_cmd.extend(['-headers', header_str])

                    if is_live:
                        ffmpeg_cmd.extend([
                            '-reconnect', '1', '-reconnect_streamed', '1',
                            '-reconnect_delay_max', '5', '-reconnect_on_network_error', '1',
                            '-http_persistent', '0'
                        ])
                    else:
                        ffmpeg_cmd.extend(['-reconnect', '1', '-reconnect_delay_max', '5'])

                    if PROXY_URL:
                        ffmpeg_cmd.extend(['-http_proxy', PROXY_URL])

                    ffmpeg_cmd.extend(['-i', real_url])

                except Exception as e:
                    self.sys_log(f"🔴 [密码本崩溃]: {str(e)}")
                    self.stop_listening()
                    return
            else:
                if not input_source.startswith("http"):
                    if not os.path.exists(input_source):
                        self.sys_log(f"🔴 [文件缺失] 找不到本地文件: {input_source}")
                        self.sys_log("💡 [提示] 受浏览器安全限制，【浏览】按钮只能获取文件名。若是本地视频，请手动复制【完整绝对路径】粘贴到输入框！")
                        self.stop_listening()
                        return

                if start_sec > 0:
                    ffmpeg_cmd.extend(['-ss', str(start_sec)])

                if not os.path.isfile(input_source):
                    ffmpeg_cmd.extend([
                        '-reconnect', '1', '-reconnect_streamed', '1',
                        '-reconnect_delay_max', '5', '-reconnect_on_network_error', '1',
                        '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36',
                        '-http_persistent', '0'
                    ])
                    if PROXY_URL:
                        ffmpeg_cmd.extend(['-http_proxy', PROXY_URL])

                ffmpeg_cmd.extend(['-i', input_source])

            ffmpeg_cmd.extend([
                '-f', 's16le', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-'
            ])

            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            CHUNK_SEC = 0.5
            CHUNK_BYTES = int(16000 * 2 * CHUNK_SEC)
            VOLUME_THRESHOLD = 0.015
            SILENCE_LIMIT_CHUNKS = 1
            MAX_SENTENCE_CHUNKS = 20

            audio_buffer = []
            silence_counter = 0
            is_speaking = False
            sentence_start_time = self.current_video_time
            processed_audio_sec = 0.0
            loop_start_time = None

            while self.is_listening:
                if self.ffmpeg_process is None: break
                try:
                    in_bytes = self.ffmpeg_process.stdout.read(CHUNK_BYTES)
                except Exception:
                    in_bytes = b''

                if not in_bytes:
                    if self.ffmpeg_process is not None:
                        self.sys_log("🟡 [流结束] 视频流自然结束或遇到了无法恢复的网络中断。")
                    self.stop_listening()
                    break

                if loop_start_time is None:
                    loop_start_time = time.time()
                    self.sys_log("🟢 [同步锁] 引擎启动，极速同传中！")

                audio_chunk = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0

                processed_audio_sec += CHUNK_SEC
                elapsed_real_time = time.time() - loop_start_time
                if processed_audio_sec > elapsed_real_time:
                    time.sleep(processed_audio_sec - elapsed_real_time)

                self.current_video_time += CHUNK_SEC

                if self.is_paused:
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0
                    continue

                volume = np.max(np.abs(audio_chunk)) if len(audio_chunk) > 0 else 0

                if volume > VOLUME_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        sentence_start_time = self.current_video_time - CHUNK_SEC
                    silence_counter = 0
                    audio_buffer.append(audio_chunk)
                else:
                    if is_speaking:
                        silence_counter += 1
                        audio_buffer.append(audio_chunk)

                if is_speaking and (silence_counter >= SILENCE_LIMIT_CHUNKS or len(audio_buffer) >= MAX_SENTENCE_CHUNKS):
                    final_audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0

                    existing = self.find_existing_record(sentence_start_time)
                    if existing:
                        self.display_memory_record(existing[0], existing[1], sentence_start_time)
                    else:
                        start_time_perf = time.time()
                        self.task_queue.put((final_audio, start_time_perf, sentence_start_time))

        except Exception as e:
            self.sys_log(f"🔴 [致命错误] 引擎崩溃: {str(e)}")
            self.stop_listening()

    # ── Whisper 工作线程 ──
    def _whisper_worker(self):
        while True:
            audio_data, start_perf, timestamp = self.task_queue.get()
            try:
                self.process_sentence(audio_data, start_perf, timestamp)
            except Exception as e:
                self.sys_log(f"⚠️ [工人线程异常]: {e}")
            finally:
                self.task_queue.task_done()

    def process_sentence(self, audio_data, start_time, sentence_timestamp):
        if not self.model: return
        if self.is_paused: return

        should_process, reason = self.check_time_zone_action(sentence_timestamp)
        if not should_process: return

        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            transcribed_text = "".join([segment.text for segment in segments]).strip()

            hallucinations = [
                "ご視聴ありがとうございました", "자막 제공", "배달의민족",
                "MBC 뉴스", "Thank you.", "Thank you", "Beep",
                "다음 영상에서 만나요", "으음", "Oh", "아무튼"
            ]

            if transcribed_text:
                is_hallucination = any(h in transcribed_text for h in hallucinations)
                if not is_hallucination and len(transcribed_text) > 1:
                    self.sys_log(f"🟠 [Whisper] 完整听写: {transcribed_text}")
                    self.call_qwen_text_api(transcribed_text, start_time, sentence_timestamp)
                else:
                    self.sys_log(f"🔕 [防伪拦截] 已过滤无效杂音或幻觉: {transcribed_text}")

        except Exception as e:
            self.sys_log(f"🔴 [Whisper 报错]: {str(e)}")

    # ── LLM 翻译（完整上下文 + 字典 + 主题 + 记忆引擎） ──
    def call_qwen_text_api(self, source_text, start_time, video_timestamp_sec):
        if self.is_paused: return
        try:
            current_api_key = self.api_key.strip()
            if not current_api_key: return

            if self.llm_client is None:
                api_base = self.config.get("api_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                self.llm_client = OpenAI(
                    api_key=current_api_key,
                    base_url=api_base,
                    timeout=15.0
                )

            raw_dict = getattr(self, 'active_dict_str', '').strip()
            dynamic_theme = ""
            valid_dict_lines = []
            theme_prefixes = ("主题:", "主题：", "背景:", "背景：", "设定:", "设定：", "当前:", "当前：")

            for line in raw_dict.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'): continue
                is_theme = False
                for prefix in theme_prefixes:
                    if line.startswith(prefix):
                        dynamic_theme = line[len(prefix):].strip()
                        is_theme = True
                        break
                if is_theme: continue
                if '=' in line:
                    valid_dict_lines.append(line)
                elif '是' in line:
                    parts = line.split('是', 1)
                    valid_dict_lines.append(f"{parts[0].strip()} = {parts[1].strip()}")
                elif '->' in line:
                    parts = line.split('->', 1)
                    valid_dict_lines.append(f"{parts[0].strip()} = {parts[1].strip()}")

            dict_str = "\n".join(valid_dict_lines)

            system_prompt = f"你是一个专业的同声传译专家。请将以下识别出的语音文本翻译为{self.target_lang}，要求口语化、符合人类表达习惯。\n"

            current_context = dynamic_theme if dynamic_theme else self.stream_context
            if current_context:
                system_prompt += f"\n【本次直播/视频特殊背景设定】（非常重要，请务必贴合此语境翻译）：\n{current_context}\n"

            if dict_str:
                system_prompt += f"\n【🚨最高优先级指令：强制术语字典】\n当原文读音或拼写疑似以下词汇时，必须强制纠正并翻译为等号右侧的内容：\n{dict_str}\n"

            if self.global_summary:
                system_prompt += f"\n【全局背景与名词对照】（极其重要）：\n{self.global_summary}\n\n"
                system_prompt += "⚠️注意：结合背景还原缩略语，绝不要生硬音译！\n"

            if len(self.context_history) > 0:
                system_prompt += f"\n【最近几句的前情提要】：\n" + "\n".join([item["text"] for item in self.context_history])

            system_prompt += "\n\n请直接输出纯翻译结果，不要带任何解释。若原文全是语气词毫无意义，回复'NONE'。"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"【原文】：{source_text}"}
            ]

            response = self.llm_client.chat.completions.create(
                model=self.config.get("llm_model_name", "qwen3.5-plus").strip(),
                messages=messages,
                extra_body={"enable_thinking": False}
            )

            if response.choices and len(response.choices) > 0:
                result_text = response.choices[0].message.content
                if self.is_listening:
                    self.update_result_and_save(source_text, result_text, video_timestamp_sec)
                self.broadcast_queue.put({
                    "type": "theme_update",
                    "content": self.global_summary
                })
        except Exception as e:
            self.sys_log(f"🔴 [大模型网络崩溃]: {str(e)}")

    # ── 结果保存 + 上下文更新 + 记忆压缩触发 ──
    def update_result_and_save(self, source_text, translated_text, video_timestamp_sec):
        clean_trans = translated_text.strip()
        if clean_trans and clean_trans != "NONE":
            time_str = self.format_time(video_timestamp_sec)

            self.context_history.append({"text": f"原文: {source_text} | 译文: {clean_trans}"})
            if len(self.context_history) > 4: self.context_history.pop(0)

            self.unsummarized_source.append(f"听写原文: {source_text}")

            if video_timestamp_sec - self.last_summary_video_time >= 180:
                if len(self.unsummarized_source) > 0:
                    history_chunk = self.unsummarized_source.copy()
                    self.unsummarized_source.clear()
                    self.last_summary_video_time = video_timestamp_sec
                    threading.Thread(target=self.compress_memory_background, args=(history_chunk,), daemon=True).start()

            display_text = f"⏱️ {time_str}\n🗣️ {source_text}\n💬 {clean_trans}\n\n"

            if self.log_file_path:
                try:
                    with open(self.log_file_path, "a", encoding="utf-8") as f:
                        f.write(display_text)
                except Exception: pass

            self.broadcast_queue.put({
                "type": "translation",
                "display_text": display_text,
                "source": source_text,
                "target": clean_trans,
                "timestamp": time_str
            })

    # ── 记忆压缩（每3分钟） ──
    def compress_memory_background(self, source_chunk):
        try:
            current_api_key = self.api_key.strip()
            if not current_api_key: return

            if self.llm_client is None:
                api_base = self.config.get("api_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                self.llm_client = OpenAI(
                    api_key=current_api_key,
                    base_url=api_base,
                    timeout=15.0
                )

            raw_dict = getattr(self, 'active_dict_str', '').strip()
            dynamic_theme = ""
            valid_dict_lines = []
            theme_prefixes = ("主题:", "主题：", "背景:", "背景：", "设定:", "设定：", "当前:", "当前：")

            for line in raw_dict.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'): continue
                is_theme = False
                for prefix in theme_prefixes:
                    if line.startswith(prefix):
                        dynamic_theme = line[len(prefix):].strip()
                        is_theme = True
                        break
                if is_theme: continue
                if '=' in line:
                    valid_dict_lines.append(line)
                elif '是' in line:
                    parts = line.split('是', 1)
                    valid_dict_lines.append(f"{parts[0].strip()} = {parts[1].strip()}")
                elif '->' in line:
                    parts = line.split('->', 1)
                    valid_dict_lines.append(f"{parts[0].strip()} = {parts[1].strip()}")

            dict_str = "\n".join(valid_dict_lines)

            compress_prompt = "你是一个高级同传导播。你的任务是阅读过去3分钟的语音识别原文，并更新全局背景。\n"

            current_context = dynamic_theme if dynamic_theme else getattr(self, 'stream_context', '')
            if current_context:
                compress_prompt += f"【当前直播核心大背景】：{current_context}\n\n"
            elif hasattr(self, 'stream_context') and self.stream_context:
                compress_prompt += f"【本次直播的初始大背景】：{self.stream_context}\n\n"

            if dict_str:
                compress_prompt += f"【强制术语对照表】(遇到以下发音请自动纠正):\n{dict_str}\n\n"

            compress_prompt += (
                f"【当前已有背景】：{self.global_summary if self.global_summary else '暂无'}\n\n"
                "【更新要求】：\n"
                "1. 敏锐察觉话题转变。提取人名、地名等。\n"
                "2. 丢弃已结束的老话题，确立新话题。\n"
                "【输出格式】（严格控制在150字以内）：\n"
                "当前场景与话题：[一句话精准概括]。\n核心名词：[A], [B]。"
            )

            logs_text = "\n".join(source_chunk)

            response = self.llm_client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": compress_prompt},
                    {"role": "user", "content": f"过去3分钟的原声记录：\n{logs_text}"}
                ]
            )

            if response.choices and len(response.choices) > 0:
                new_summary = response.choices[0].message.content.strip()
                self.global_summary = new_summary
                self.sys_log(f"🧠 [记忆引擎 3min 轮询] 背景提纯完毕: {self.global_summary}")

        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# tkinter 原生桌面 UI
# ═══════════════════════════════════════════════════════════════
class SMTBetaUI:
    def __init__(self, engine: SMTEngine):
        self.engine = engine
        self._subtitle_on = True
        self._subtitle_win = None

        # ── 样式 ──
        self.colors = {
            "bg_dark": "#0f172a", "bg_panel": "#1e293b", "bg_input": "#0f172a",
            "border": "#334155", "accent": "#38bdf8", "accent_deep": "#0284c7",
            "text_main": "#e2e8f0", "text_muted": "#94a3b8", "text_dim": "#64748b",
            "green": "#22c55e", "red": "#ef4444", "yellow": "#f59e0b",
        }
        c = self.colors

        # ═══════════════════════════════════════════════════════
        # 主窗口（控制面板）
        # ═══════════════════════════════════════════════════════
        self.root = tk.Tk()
        self.root.title("SMT Beta — 控制面板")
        self.root.geometry("900x680")
        self.root.minsize(700, 500)
        self.root.configure(bg=c["bg_dark"])

        # ── 控制面板 ──
        ctrl_frame = tk.Frame(self.root, bg=c["bg_panel"])
        ctrl_frame.pack(fill="x", padx=8, pady=(8, 0))

        # Row 0: URL + 浏览
        url_frame = tk.Frame(ctrl_frame, bg=c["bg_panel"])
        url_frame.pack(fill="x", pady=(0, 6))
        tk.Label(url_frame, text="URL:", fg=c["text_muted"], bg=c["bg_panel"],
                 font=("Microsoft YaHei", 10)).pack(side="left", padx=(0, 8))
        self.url_var = tk.StringVar()
        self.url_entry = tk.Entry(url_frame, textvariable=self.url_var, font=("Consolas", 10),
                                  bg=c["bg_input"], fg=c["text_main"], insertbackground=c["text_main"],
                                  relief="flat", bd=1, highlightthickness=1,
                                  highlightbackground=c["border"], highlightcolor=c["accent"])
        self.url_entry.pack(side="left", fill="x", expand=True)
        self.url_entry.bind("<Return>", lambda e: self._start())

        browse_btn = tk.Button(url_frame, text="📁 浏览", command=self._browse_file,
                               bg="#334155", fg=c["text_main"], font=("Microsoft YaHei", 9),
                               relief="flat", padx=10, pady=3, cursor="hand2",
                               activebackground="#475569", activeforeground=c["text_main"])
        browse_btn.pack(side="left", padx=(8, 0))

        # Row 1: 起播 / 跳过 / 仅译
        time_frame = tk.Frame(ctrl_frame, bg=c["bg_panel"])
        time_frame.pack(fill="x", pady=(0, 6))
        for label, var_name in [("起播:", "start_time"), ("跳过:", "skip_times"), ("仅译:", "target_times")]:
            tk.Label(time_frame, text=label, fg=c["text_muted"], bg=c["bg_panel"],
                     font=("Microsoft YaHei", 9)).pack(side="left", padx=(0, 4))
            sv = tk.StringVar(value="00:00:00" if var_name == "start_time" else "")
            setattr(self, f"{var_name}_var", sv)
            ent = tk.Entry(time_frame, textvariable=sv, font=("Consolas", 9),
                           bg=c["bg_input"], fg=c["text_main"], insertbackground=c["text_main"],
                           relief="flat", bd=1, highlightthickness=1,
                           highlightbackground=c["border"], highlightcolor=c["accent"],
                           width=12 if var_name == "start_time" else 16)
            ent.pack(side="left", padx=(0, 12))

        # Row 2: 按钮行
        btn_frame = tk.Frame(ctrl_frame, bg=c["bg_panel"])
        btn_frame.pack(fill="x")

        self.btn_start = tk.Button(btn_frame, text="▶ 开始", command=self._start,
                                   bg=c["accent_deep"], fg="white", font=("Microsoft YaHei", 10, "bold"),
                                   relief="flat", padx=16, pady=4, cursor="hand2",
                                   activebackground="#0369a1", activeforeground="white")
        self.btn_start.pack(side="left", padx=(0, 6))

        self.btn_stop = tk.Button(btn_frame, text="■ 停止", command=self._stop,
                                  bg=c["red"], fg="white", font=("Microsoft YaHei", 10, "bold"),
                                  relief="flat", padx=16, pady=4, cursor="hand2",
                                  activebackground="#b91c1c", activeforeground="white")

        self.btn_pause = tk.Button(btn_frame, text="⏸ 暂停", command=self._pause,
                                   bg=c["yellow"], fg="white", font=("Microsoft YaHei", 10, "bold"),
                                   relief="flat", padx=16, pady=4, cursor="hand2",
                                   activebackground="#b45309", activeforeground="white")

        self.btn_resume = tk.Button(btn_frame, text="▶ 继续", command=self._resume,
                                    bg=c["green"], fg="white", font=("Microsoft YaHei", 10, "bold"),
                                    relief="flat", padx=16, pady=4, cursor="hand2",
                                    activebackground="#15803d", activeforeground="white")

        self.btn_subtitle = tk.Button(btn_frame, text="💬 字幕: 开", command=self._toggle_subtitle,
                                      bg=c["accent_deep"], fg="white", font=("Microsoft YaHei", 9),
                                      relief="flat", padx=10, pady=3, cursor="hand2",
                                      activebackground="#0369a1", activeforeground="white")
        self.btn_subtitle.pack(side="right", padx=(6, 0))

        clear_btn = tk.Button(btn_frame, text="清屏", command=self._clear_output,
                              bg="#334155", fg=c["text_main"], font=("Microsoft YaHei", 9),
                              relief="flat", padx=10, pady=3, cursor="hand2",
                              activebackground="#475569", activeforeground=c["text_main"])
        clear_btn.pack(side="right", padx=(6, 0))

        # ── 字典面板 ──
        dict_frame = tk.Frame(self.root, bg=c["bg_panel"])
        dict_frame.pack(fill="x", padx=8, pady=(6, 0))

        dict_header = tk.Frame(dict_frame, bg=c["bg_panel"])
        dict_header.pack(fill="x", pady=(0, 4))
        tk.Label(dict_header, text="术语字典 / 主题设定", fg=c["accent"], bg=c["bg_panel"],
                 font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        tk.Button(dict_header, text="🔄 同步", command=self._sync_dict,
                  bg=c["accent_deep"], fg="white", font=("Microsoft YaHei", 9),
                  relief="flat", padx=12, pady=2, cursor="hand2",
                  activebackground="#0369a1", activeforeground="white").pack(side="right")

        dict_container = tk.Frame(dict_frame, bg=c["border"])
        dict_container.pack(fill="x")
        self.dict_text = tk.Text(dict_container, height=5, font=("Consolas", 10),
                                 bg=c["bg_input"], fg=c["text_main"], insertbackground=c["text_main"],
                                 relief="flat", wrap="word",
                                 padx=8, pady=6, undo=True,
                                 highlightthickness=1, highlightbackground=c["border"])
        self.dict_text.pack(side="left", fill="x", expand=True)

        if engine.stream_context:
            self.dict_text.insert("1.0", f"主题：{engine.stream_context}\n")
        self.dict_text.insert("end",
            "# 格式说明：\n"
            "#   术语: Hololive = ホロライブ\n"
            "#   主题：这是一场关于AI的科技发布会\n"
            "#   可直接写 原文 = 译文, 也可以用 原文->译文 或 原文是译文\n"
        )

        # ── 翻译记录面板（紧凑） ──
        out_header = tk.Frame(self.root, bg=c["bg_dark"])
        out_header.pack(fill="x", padx=8)
        tk.Label(out_header, text="翻译记录", fg=c["accent"], bg=c["bg_dark"],
                 font=("Microsoft YaHei", 10, "bold")).pack(side="left")

        out_container = tk.Frame(self.root, bg=c["border"])
        out_container.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self.out_text = tk.Text(out_container, font=("Microsoft YaHei", 10),
                                bg=c["bg_input"], fg=c["text_main"], insertbackground=c["text_main"],
                                relief="flat", wrap="word",
                                padx=10, pady=8, state="disabled",
                                highlightthickness=1, highlightbackground=c["border"])
        self.out_scroll = tk.Scrollbar(out_container, orient="vertical", command=self.out_text.yview)
        self.out_text.configure(yscrollcommand=self.out_scroll.set)
        self.out_scroll.pack(side="right", fill="y")
        self.out_text.pack(side="left", fill="both", expand=True)

        self.out_text.tag_configure("ts", foreground=c["text_dim"], font=("Consolas", 9))
        self.out_text.tag_configure("src", foreground=c["text_muted"], font=("Microsoft YaHei", 10))
        self.out_text.tag_configure("tgt", foreground=c["text_main"], font=("Microsoft YaHei", 11, "bold"))
        self.out_text.tag_configure("log", foreground="#475569", font=("Microsoft YaHei", 9))
        self.out_text.tag_configure("separator", foreground=c["border"], font=("Consolas", 4))

        # ── 状态栏 ──
        self.status_var = tk.StringVar(value="🟡 等待 Whisper 模型加载...")
        status_bar = tk.Frame(self.root, bg=c["bg_panel"])
        status_bar.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(status_bar, textvariable=self.status_var, fg=c["text_muted"], bg=c["bg_panel"],
                 font=("Microsoft YaHei", 9)).pack(side="left")

        # ── 字幕悬浮窗 ──
        self._create_subtitle_overlay()

        # ── 事件绑定 ──
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_messages()
        self._update_buttons("idle")

    # ═══════════════════════════════════════════════════════
    # 字幕悬浮窗
    # ═══════════════════════════════════════════════════════
    def _create_subtitle_overlay(self):
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.88)
        win.configure(bg="#000000")
        # 窗口透明色（Windows 下配合 bg 实现伪透明）
        try:
            win.wm_attributes("-transparentcolor", "#000000")
        except Exception:
            pass

        c = self.colors

        # 内容容器
        container = tk.Frame(win, bg=c["bg_panel"])
        container.pack(fill="both", expand=True, padx=2, pady=2)

        # 顶部拖拽条（细条，用于移动窗口）
        drag_bar = tk.Frame(container, bg=c["border"], height=4, cursor="fleur")
        drag_bar.pack(fill="x")
        drag_bar.bind("<ButtonPress-1>", self._subtitle_drag_start)
        drag_bar.bind("<B1-Motion>", self._subtitle_drag_move)
        # 整个窗口也可拖拽
        container.bind("<ButtonPress-1>", self._subtitle_drag_start)
        container.bind("<B1-Motion>", self._subtitle_drag_move)

        # 文本区域
        text_frame = tk.Frame(container, bg=c["bg_panel"])
        text_frame.pack(fill="both", expand=True, padx=16, pady=(8, 12))

        self.sub_source_label = tk.Label(
            text_frame, text="", fg=c["text_muted"], bg=c["bg_panel"],
            font=("Microsoft YaHei", 11), anchor="w", justify="left", wraplength=900)
        self.sub_source_label.pack(fill="x")

        self.sub_target_label = tk.Label(
            text_frame, text="等待翻译...", fg="white", bg=c["bg_panel"],
            font=("Microsoft YaHei", 18, "bold"), anchor="w", justify="left", wraplength=900)
        self.sub_target_label.pack(fill="x", pady=(4, 0))

        # 右键菜单
        sub_menu = tk.Menu(win, tearoff=0, bg=c["bg_panel"], fg=c["text_main"],
                           font=("Microsoft YaHei", 9))
        sub_menu.add_command(label="隐藏字幕窗", command=self._toggle_subtitle)
        sub_menu.add_command(label="退出程序", command=self._on_close)
        container.bind("<Button-3>", lambda e: sub_menu.post(e.x_root, e.y_root))
        text_frame.bind("<Button-3>", lambda e: sub_menu.post(e.x_root, e.y_root))
        self.sub_source_label.bind("<Button-3>", lambda e: sub_menu.post(e.x_root, e.y_root))
        self.sub_target_label.bind("<Button-3>", lambda e: sub_menu.post(e.x_root, e.y_root))

        # 默认位置：屏幕底部居中
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        ww, wh = 960, 100
        win.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{sh-wh-40}")

        self._subtitle_win = win
        self._subtitle_drag_x = 0
        self._subtitle_drag_y = 0

        # 如果主窗口关闭，一并关掉字幕窗
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _subtitle_drag_start(self, event):
        self._subtitle_drag_x = event.x
        self._subtitle_drag_y = event.y

    def _subtitle_drag_move(self, event):
        dx = event.x - self._subtitle_drag_x
        dy = event.y - self._subtitle_drag_y
        x = self._subtitle_win.winfo_x() + dx
        y = self._subtitle_win.winfo_y() + dy
        self._subtitle_win.geometry(f"+{x}+{y}")

    def _update_subtitle(self, source, target, timestamp):
        """更新字幕悬浮窗的内容"""
        if not self._subtitle_win or not self._subtitle_on:
            return
        try:
            self.sub_source_label.configure(text=f"{timestamp}  {source}")
            self.sub_target_label.configure(text=target)
        except tk.TclError:
            pass  # 窗口已被销毁

    def _toggle_subtitle(self):
        if self._subtitle_on:
            self._subtitle_win.withdraw()
            self._subtitle_on = False
            self.btn_subtitle.configure(text="💬 字幕: 关", bg="#334155")
        else:
            self._subtitle_win.deiconify()
            self._subtitle_on = True
            self.btn_subtitle.configure(text="💬 字幕: 开", bg=self.colors["accent_deep"])

    # ═══════════════════════════════════════════════════════
    # UI 操作
    # ═══════════════════════════════════════════════════════
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="请选择要同传的媒体文件",
            filetypes=[("视频/音频文件", "*.mp4 *.mp3 *.wav *.flv *.mkv *.aac *.m3u8"), ("所有文件", "*.*")]
        )
        if path:
            self.url_var.set(path)

    def _start(self):
        url = self.url_var.get().strip()
        if not url:
            self._append_log("⚠️ 请输入 URL 或文件路径")
            return
        ok = self.engine.start_listening(
            url,
            start_time=self.start_time_var.get(),
            skip_times=self.skip_times_var.get(),
            target_times=self.target_times_var.get()
        )
        if ok:
            self._update_buttons("listening")
            self.status_var.set("🟢 翻译中 | 等待音频流...")

    def _stop(self):
        self.engine.stop_listening()
        self._update_buttons("idle")
        self.status_var.set("🟡 已停止")

    def _pause(self):
        self.engine.is_paused = True
        self.engine.sys_log("⏸️ [暂歇截流] 已开启：时间轴继续推进，当前音频已被静默丢弃！")
        self._update_buttons("paused")
        self.status_var.set("⏸️ 已暂停")

    def _resume(self):
        self.engine.is_paused = False
        self.engine.sys_log("▶️ [恢复截流] 已关闭：重新接管音频流，开始翻译！")
        self._update_buttons("listening")
        self.status_var.set("🟢 翻译中")

    def _sync_dict(self):
        content = self.dict_text.get("1.0", "end-1c")
        self.engine.active_dict_str = content
        self.engine.sys_log("📖 [字典] 术语字典已同步至引擎！")

    def _clear_output(self):
        self.out_text.configure(state="normal")
        self.out_text.delete("1.0", "end")
        self.out_text.configure(state="disabled")

    def _update_buttons(self, state):
        if state == "idle":
            self.btn_start.pack(side="left", padx=(0, 6))
            self.btn_stop.pack_forget()
            self.btn_pause.pack_forget()
            self.btn_resume.pack_forget()
        elif state == "listening":
            self.btn_start.pack_forget()
            self.btn_stop.pack(side="left", padx=(0, 6))
            self.btn_pause.pack(side="left", padx=(0, 6))
            self.btn_resume.pack_forget()
        elif state == "paused":
            self.btn_start.pack_forget()
            self.btn_stop.pack(side="left", padx=(0, 6))
            self.btn_pause.pack_forget()
            self.btn_resume.pack(side="left", padx=(0, 6))

    def _on_close(self):
        self.engine.stop_listening()
        if self._subtitle_win:
            try: self._subtitle_win.destroy()
            except: pass
        self.root.destroy()

    # ═══════════════════════════════════════════════════════
    # 消息轮询
    # ═══════════════════════════════════════════════════════
    def _poll_messages(self):
        try:
            while True:
                msg = self.engine.broadcast_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _handle_message(self, msg):
        msg_type = msg.get("type", "")
        if msg_type == "translation":
            source = msg.get("source", "")
            target = msg.get("target", "")
            timestamp = msg.get("timestamp", "")
            self._append_translation(source, target, timestamp)
            self._update_subtitle(source, target, timestamp)
        elif msg_type == "log":
            self._append_log(msg.get("content", ""))
            content = msg.get("content", "")
            if "已切断" in content or "已停止" in content:
                self.root.after(0, lambda: self._update_buttons("idle"))
                self.root.after(0, lambda: self.status_var.set("🟡 已停止"))
            elif "引擎启动" in content or "同步锁" in content:
                self.root.after(0, lambda: self.status_var.set("🟢 翻译中 | Whisper 已就绪"))
            elif "Whisper 引擎加载成功" in content:
                self.root.after(0, lambda: self.status_var.set("🟢 就绪 | 等待输入 URL"))

    def _append_translation(self, source, target, timestamp):
        self.out_text.configure(state="normal")
        self.out_text.insert("end", f"⏱️ {timestamp}\n", "ts")
        self.out_text.insert("end", f"🗣️ {source}\n", "src")
        self.out_text.insert("end", f"💬 {target}\n", "tgt")
        self.out_text.insert("end", "\n", "separator")
        self.out_text.see("end")
        self.out_text.configure(state="disabled")

    def _append_log(self, text):
        self.out_text.configure(state="normal")
        self.out_text.insert("end", f"  {text}\n", "log")
        self.out_text.see("end")
        self.out_text.configure(state="disabled")

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  SMT Beta — 完整版同声传译引擎")
    print("  原生桌面应用 (tkinter)")
    print("=" * 50)
    engine = SMTEngine()
    ui = SMTBetaUI(engine)
    ui.run()
