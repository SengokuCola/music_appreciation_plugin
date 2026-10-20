"""音乐赏析插件：从网易云获取歌曲音频，交给百炼 Qwen-Omni 听歌赏析。"""

from typing import Any, Dict, List, Optional
import base64
import json
import re

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType
import httpx

NETEASE_BASE = "https://music.163.com"
NETEASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
    "Referer": "https://music.163.com/",
}

# 链接形如 music.163.com/song?id=123、/#/song?id=123、/song/123、y.music.163.com/m/song?id=123
SONG_ID_PATTERNS = (
    re.compile(r"[?&]id=(\d+)"),
    re.compile(r"/song/(\d+)"),
)
SHORT_LINK_PATTERN = re.compile(r"https?://(?:163cn\.tv|[\w.]*music\.163\.com)/[^\s]+")

LRC_TIMESTAMP_PATTERN = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")

SYSTEM_PROMPT = (
    "你是一位细腻的音乐赏析者。请认真听用户提供的音频，结合歌曲信息和歌词（如有），"
    "用简洁中文写一份赏析笔记，涵盖：整体风格与情绪、编曲与配器、人声表现、节奏与段落结构、"
    "歌词主题与意象（如有歌词）、最打动人的瞬间。只描述你确实听到的内容，不要编造；"
    "如果音频被截取过，只评价听到的部分。歌词受版权保护，笔记中最多引用一两句，不要大段复述。"
)


class PluginSettings(PluginConfigBase):
    config_version: str = Field(default="0.1.0", description="配置版本", json_schema_extra={"label": "配置版本"})


class MusicConfig(PluginConfigBase):
    plugin: PluginSettings = Field(default_factory=PluginSettings, description="插件设置", json_schema_extra={"label": "插件设置"})
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="支持音频输入的 OpenAI 兼容接口地址，默认为阿里云百炼",
        json_schema_extra={"label": "模型接口地址"},
    )
    api_key: str = Field(default="", description="模型服务的 API 密钥", json_schema_extra={"label": "模型 API 密钥", "x-widget": "password"})
    model: str = Field(default="qwen3-omni-flash", description="支持音频输入的模型名称", json_schema_extra={"label": "音频模型"})
    max_audio_mb: float = Field(default=8.0, description="发送给模型的音频大小上限，单位 MB，超出部分截断", json_schema_extra={"label": "音频大小上限"})
    max_lyrics_chars: int = Field(default=3000, description="随音频发送的歌词最大字数", json_schema_extra={"label": "歌词字数上限"})
    max_output_tokens: int = Field(default=1500, description="赏析笔记的最大输出 token 数", json_schema_extra={"label": "输出长度上限"})
    request_timeout: float = Field(default=180.0, description="模型请求超时时间，单位秒", json_schema_extra={"label": "请求超时"})
    netease_cookie: str = Field(
        default="", description="可选，填写 MUSIC_U 的值或完整 Cookie，用于会员歌曲",
        json_schema_extra={"label": "网易云登录凭证", "x-widget": "password"},
    )


def extract_song_id(text: str) -> Optional[str]:
    """从链接或纯数字中解析网易云歌曲 ID；无法直接解析时返回 None。"""
    text = text.strip()
    if text.isdigit():
        return text
    for pattern in SONG_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def clean_lyrics(raw: str, max_chars: int) -> str:
    """去掉 LRC 时间戳和空行，并限制长度。"""
    lines = [LRC_TIMESTAMP_PATTERN.sub("", line).strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line)[:max_chars]


class MusicAppreciationPlugin(MaiBotPlugin):
    config_model = MusicConfig

    async def on_load(self) -> None:
        return None

    async def on_unload(self) -> None:
        return None

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        return None

    def _parse_cookie(self) -> Dict[str, str]:
        """解析配置里的 Cookie：不含 "=" 时视为 MUSIC_U 的值，否则按 "k=v; k=v" 解析。"""
        raw = self.config.netease_cookie.strip()
        if not raw:
            return {}
        if "=" not in raw:
            return {"MUSIC_U": raw}
        pairs = (item.split("=", 1) for item in raw.split(";") if "=" in item)
        return {key.strip(): value.strip() for key, value in pairs}

    def _netease_client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        """创建网易云客户端；Cookie 仅作用于 music.163.com，不会随跳转发给 CDN。"""
        client = httpx.AsyncClient(timeout=timeout, headers=NETEASE_HEADERS)
        for name, value in self._parse_cookie().items():
            client.cookies.set(name, value, domain="music.163.com")
        return client

    async def _fetch_account(self) -> Optional[Dict[str, Any]]:
        """查询当前 Cookie 对应的账号；Cookie 无效或已过期时返回 None。"""
        async with self._netease_client(15.0) as client:
            response = await client.get(f"{NETEASE_BASE}/api/nuser/account/get")
            response.raise_for_status()
            data = response.json()
        if not data.get("profile"):
            return None
        return {
            "nickname": data["profile"]["nickname"],
            "is_vip": (data.get("account") or {}).get("vipType", 0) > 0,
        }

    @Command("netease_status", description="查看网易云登录状态", pattern=r"^/网易云状态$")
    async def handle_status(self, stream_id: str = "", **kwargs: Any):
        del kwargs
        if not self._parse_cookie():
            text = "未配置网易云 Cookie，只能获取免费歌曲；请在插件配置中填写 netease_cookie"
        else:
            account = await self._fetch_account()
            if account is None:
                text = "已配置的网易云 Cookie 无效或已过期，请重新获取并填写"
            else:
                text = f"已登录：{account['nickname']}（{'会员' if account['is_vip'] else '非会员'}）"
        await self.ctx.send.text(text, stream_id)
        return True, text, True

    async def _resolve_song_id(self, client: httpx.AsyncClient, query: str) -> Optional[str]:
        """歌名走搜索，链接直接解析；短链先跟随跳转再解析。"""
        song_id = extract_song_id(query)
        if song_id:
            return song_id
        short_link = SHORT_LINK_PATTERN.search(query)
        if short_link:
            response = await client.get(short_link.group(0), follow_redirects=True)
            song_id = extract_song_id(str(response.url))
            if song_id:
                return song_id
            raise ValueError("无法从该链接中解析出歌曲 ID（可能不是单曲链接）")

        response = await client.post(
            f"{NETEASE_BASE}/api/search/get/web",
            data={"s": query, "type": 1, "offset": 0, "limit": 10, "total": "true"},
        )
        response.raise_for_status()
        songs = (response.json().get("result") or {}).get("songs") or []
        if not songs:
            return None
        # 综合搜索常把翻唱排在原唱前面：优先选歌手名出现在查询中的结果。
        for song in songs:
            if any(artist["name"] in query for artist in song.get("artists") or []):
                return str(song["id"])
        return str(songs[0]["id"])

    async def _fetch_song_info(self, client: httpx.AsyncClient, song_id: str) -> Dict[str, Any]:
        response = await client.get(f"{NETEASE_BASE}/api/song/detail", params={"ids": f"[{song_id}]"})
        response.raise_for_status()
        songs = response.json().get("songs") or []
        if not songs:
            raise ValueError(f"网易云中不存在歌曲 {song_id}")
        song = songs[0]
        album = song.get("album") or {}
        return {
            "id": song_id,
            "title": song["name"],
            "artists": [artist["name"] for artist in song.get("artists") or []],
            "album": album.get("name", ""),
            "duration_seconds": round(song.get("duration", 0) / 1000),
            "url": f"{NETEASE_BASE}/song?id={song_id}",
        }

    async def _fetch_lyrics(self, client: httpx.AsyncClient, song_id: str) -> str:
        response = await client.get(f"{NETEASE_BASE}/api/song/lyric", params={"id": song_id, "lv": 1})
        response.raise_for_status()
        return ((response.json().get("lrc") or {}).get("lyric")) or ""

    async def _download_audio(self, client: httpx.AsyncClient, song_id: str) -> Optional[Dict[str, Any]]:
        """通过外链下载音频前 max_audio_mb；歌曲无版权或需会员时返回 None。"""
        limit = int(self.config.max_audio_mb * 1024 * 1024)
        async with client.stream(
            "GET",
            f"{NETEASE_BASE}/song/media/outer/url",
            params={"id": f"{song_id}.mp3"},
            headers={"Range": f"bytes=0-{limit - 1}"},
            follow_redirects=True,
        ) as response:
            # 不可用的歌曲会被重定向到 /404，或返回非音频内容。
            content_type = response.headers.get("content-type", "")
            if response.status_code not in (200, 206) or not content_type.startswith("audio"):
                return None
            total = response.headers.get("content-range", "").rsplit("/", 1)[-1]
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) >= limit:
                    break
        # 服务端不支持 Range 时可能整首返回，这里统一截断。
        truncated = len(data) >= limit or (total.isdigit() and int(total) > len(data))
        return {"data": bytes(data[:limit]), "truncated": truncated}

    async def _ask_omni(self, audio: bytes, song: Dict[str, Any], lyrics: str, focus: str) -> str:
        """以流式调用 Omni 模型（百炼 Omni 仅支持 stream=true）并拼接文本。"""
        config = self.config
        info_lines = [
            f"歌名：{song['title']}",
            f"歌手：{'、'.join(song['artists'])}",
            f"专辑：{song['album']}",
            f"时长：{song['duration_seconds']} 秒",
        ]
        text = "\n".join(info_lines)
        if lyrics:
            text += f"\n\n歌词（仅供理解，勿大段复述）：\n{lyrics}"
        text += "\n\n请赏析这首歌。" + (f"\n重点关注：{focus}" if focus else "")

        payload = {
            "model": config.model,
            "stream": True,
            "modalities": ["text"],
            "max_tokens": config.max_output_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:;base64,{base64.b64encode(audio).decode()}",
                                "format": "mp3",
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                },
            ],
        }
        pieces: List[str] = []
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            async with client.stream(
                "POST",
                f"{config.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.api_key.strip()}"},
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Omni 模型请求失败 HTTP {response.status_code}: {body[:500]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    for choice in json.loads(data).get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            pieces.append(content)
        return "".join(pieces).strip()

    @Tool(
        "appreciate_music",
        description="听一首网易云音乐上的歌并赏析。输入歌名（可带歌手）或网易云歌曲链接，"
        "会获取音频与歌词，由音频理解模型给出风格、编曲、人声、结构与歌词意象的赏析笔记。"
        "按歌名搜索可能命中翻唱版，结果里的 song 是实际赏析的版本，必要时向用户确认或请用户提供链接。",
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="歌名（可附歌手，如“晴天 周杰伦”）或网易云歌曲链接",
                required=True,
            ),
            ToolParameterInfo(
                name="focus",
                param_type=ToolParamType.STRING,
                description="想侧重赏析的角度，如“编曲”“歌词意象”“情绪变化”；可留空",
                required=False,
                default="",
            ),
        ],
    )
    async def appreciate_music(self, query: str, focus: str = "", **kwargs: Any) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            return {"success": False, "message": "歌名或链接不能为空"}
        if not self.config.api_key.strip():
            return {"success": False, "message": "请先在插件配置中填写百炼 api_key"}

        async with self._netease_client() as client:
            try:
                song_id = await self._resolve_song_id(client, query)
            except ValueError as exc:
                return {"success": False, "message": str(exc)}
            if not song_id:
                return {"success": False, "message": f"网易云没有搜到「{query}」"}
            song = await self._fetch_song_info(client, song_id)
            lyrics = clean_lyrics(await self._fetch_lyrics(client, song_id), self.config.max_lyrics_chars)
            audio = await self._download_audio(client, song_id)

        if audio is None:
            return {
                "success": False,
                "message": "这首歌暂时拿不到音频（可能无版权、需要会员或已下架）。如是会员歌曲，可让管理员在插件配置中填写会员账号的 netease_cookie",
                "song": song,
            }

        appreciation = await self._ask_omni(audio["data"], song, lyrics, focus.strip())
        if not appreciation:
            return {"success": False, "message": "Omni 模型返回了空内容", "song": song}
        return {
            "success": True,
            "song": song,
            "matched_by_search": not SHORT_LINK_PATTERN.search(query) and not extract_song_id(query),
            "audio_truncated": audio["truncated"],
            "appreciation": appreciation,
        }


def create_plugin() -> MusicAppreciationPlugin:
    return MusicAppreciationPlugin()
