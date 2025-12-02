"""
Chrome DevTools Protocol (CDP) Interceptor for Video URL Capture

This module connects to a Chrome browser instance running with remote debugging
enabled (--remote-debugging-port=9222) and intercepts network requests to
capture video and audio URLs from streaming sites.

Usage:
    # Start Chrome with debugging enabled:
    # chrome.exe --remote-debugging-port=9222 --user-data-dir="C:/ChromeDebug"
    
    from src.cdp import CDPInterceptor
    
    def on_media_found(media_info):
        print(f"Found: {media_info.url}")
    
    interceptor = CDPInterceptor(on_media=on_media_found)
    await interceptor.start()
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Type of media resource."""
    VIDEO = "video"
    AUDIO = "audio"
    MANIFEST = "manifest"
    UNKNOWN = "unknown"


@dataclass
class MediaInfo:
    """Information about a captured media resource."""
    url: str
    media_type: MediaType
    content_type: str = ""
    title: str = ""
    duration: Optional[float] = None
    source_url: str = ""  # Page URL where media was found
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def __str__(self):
        return f"MediaInfo({self.media_type.value}: {self.url[:80]}...)"


# URL patterns for video/audio content
MEDIA_PATTERNS = [
    # YouTube
    (r'googlevideo\.com/videoplayback', MediaType.VIDEO),
    (r'youtube\.com/.+\.m3u8', MediaType.MANIFEST),
    (r'youtubei\.googleapis\.com/.+/player', MediaType.VIDEO),
    
    # Generic video formats
    (r'\.mp4(\?|$)', MediaType.VIDEO),
    (r'\.webm(\?|$)', MediaType.VIDEO),
    (r'\.m4a(\?|$)', MediaType.AUDIO),
    (r'\.mp3(\?|$)', MediaType.AUDIO),
    (r'\.aac(\?|$)', MediaType.AUDIO),
    (r'\.opus(\?|$)', MediaType.AUDIO),
    (r'\.ogg(\?|$)', MediaType.AUDIO),
    
    # HLS/DASH streaming
    (r'\.m3u8(\?|$)', MediaType.MANIFEST),
    (r'\.mpd(\?|$)', MediaType.MANIFEST),
    (r'/manifest\.', MediaType.MANIFEST),
    (r'/playlist\.', MediaType.MANIFEST),
    
    # Common CDN patterns
    (r'cloudfront\.net/.+\.(mp4|webm|m3u8)', MediaType.VIDEO),
    (r'akamaihd\.net/.+\.(mp4|webm|m3u8)', MediaType.VIDEO),
    (r'fastly\.net/.+\.(mp4|webm)', MediaType.VIDEO),
    
    # Twitch
    (r'usher\.ttvnw\.net', MediaType.MANIFEST),
    (r'ttvnw\.net/.+\.ts', MediaType.VIDEO),
    
    # Vimeo
    (r'vimeocdn\.com/.+\.mp4', MediaType.VIDEO),
    (r'player\.vimeo\.com/.+/video', MediaType.VIDEO),
    
    # Netflix (limited without cookies)
    (r'nflxvideo\.net', MediaType.VIDEO),
    
    # Amazon Prime
    (r'aiv-cdn\.net', MediaType.VIDEO),
    
    # Generic streaming patterns
    (r'/chunk[_-]?\d+', MediaType.VIDEO),
    (r'/segment\d+', MediaType.VIDEO),
    (r'mime_type=video', MediaType.VIDEO),
    (r'mime_type=audio', MediaType.AUDIO),
]

# Content-Type patterns
CONTENT_TYPE_MEDIA = [
    'video/',
    'audio/',
    'application/vnd.apple.mpegurl',  # HLS
    'application/dash+xml',  # DASH
    'application/x-mpegurl',  # HLS variant
]


def classify_url(url: str, content_type: str = "") -> Optional[MediaType]:
    """
    Classify a URL as video, audio, manifest, or None.
    
    Args:
        url: The URL to classify
        content_type: HTTP Content-Type header if available
        
    Returns:
        MediaType or None if not a media URL
    """
    # Check content type first
    if content_type:
        ct_lower = content_type.lower()
        if 'video/' in ct_lower:
            return MediaType.VIDEO
        if 'audio/' in ct_lower:
            return MediaType.AUDIO
        if 'mpegurl' in ct_lower or 'dash' in ct_lower:
            return MediaType.MANIFEST
    
    # Check URL patterns
    url_lower = url.lower()
    for pattern, media_type in MEDIA_PATTERNS:
        if re.search(pattern, url_lower):
            return media_type
    
    return None


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from common video platform URLs.
    
    Supports: YouTube, Vimeo, Twitch, etc.
    """
    parsed = urlparse(url)
    
    # YouTube
    if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
        if parsed.path == '/watch':
            qs = parse_qs(parsed.query)
            return qs.get('v', [None])[0]
        elif parsed.netloc == 'youtu.be':
            return parsed.path.lstrip('/')
    
    # Vimeo
    if 'vimeo.com' in parsed.netloc:
        match = re.search(r'/(\d+)', parsed.path)
        if match:
            return match.group(1)
    
    return None


class CDPInterceptor:
    """
    Chrome DevTools Protocol interceptor for capturing media URLs.
    
    Connects to Chrome via CDP and monitors network requests to detect
    video and audio streams.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9222,
        on_media: Optional[Callable[[MediaInfo], None]] = None,
        filter_duplicates: bool = True,
        min_video_size: int = 100000,  # 100KB minimum for video
    ):
        """
        Initialize CDP interceptor.
        
        Args:
            host: Chrome debugger host
            port: Chrome debugger port (--remote-debugging-port)
            on_media: Callback when media URL is captured
            filter_duplicates: Skip duplicate URLs
            min_video_size: Minimum content size to consider as video
        """
        self.host = host
        self.port = port
        self.on_media = on_media
        self.filter_duplicates = filter_duplicates
        self.min_video_size = min_video_size
        
        self._ws = None
        self._running = False
        self._seen_urls: set[str] = set()
        self._current_page_url = ""
        self._current_page_title = ""
        self._msg_id = 0
        self._pending_requests: dict[str, dict] = {}
        
    @property
    def endpoint(self) -> str:
        """Chrome debugger WebSocket endpoint."""
        return f"http://{self.host}:{self.port}/json"
    
    async def check_chrome(self) -> tuple[bool, str]:
        """
        Check if Chrome is running with debugging enabled.
        
        Returns:
            Tuple of (is_available, status_message)
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoint, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        tabs = await resp.json()
                        page_tabs = [t for t in tabs if t.get('type') == 'page']
                        return True, f"Connected ({len(page_tabs)} tabs)"
                    return False, f"HTTP {resp.status}"
        except aiohttp.ClientConnectorError:
            return False, "Chrome not running with --remote-debugging-port=9222"
        except asyncio.TimeoutError:
            return False, "Connection timeout"
        except Exception as e:
            return False, str(e)
    
    async def get_tabs(self) -> list[dict]:
        """Get list of open Chrome tabs."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoint) as resp:
                    tabs = await resp.json()
                    return [t for t in tabs if t.get('type') == 'page']
        except Exception as e:
            logger.error(f"Failed to get tabs: {e}")
            return []
    
    async def start(self, tab_index: int = 0):
        """
        Start intercepting media URLs.
        
        Args:
            tab_index: Index of tab to monitor (0 = first tab)
        """
        import websockets
        
        tabs = await self.get_tabs()
        if not tabs:
            logger.error("No Chrome tabs available")
            return
        
        # Prefer tabs with video content in URL
        video_tab_index = None
        for i, t in enumerate(tabs):
            url = t.get('url', '').lower()
            if '/watch' in url or 'video' in url or 'vimeo.com/' in url:
                video_tab_index = i
                break
        
        if video_tab_index is not None:
            tab_index = video_tab_index
            logger.info(f"Found video tab at index {tab_index}")
        elif tab_index >= len(tabs):
            tab_index = 0
        
        tab = tabs[tab_index]
        ws_url = tab.get('webSocketDebuggerUrl')
        
        if not ws_url:
            logger.error("Tab doesn't have WebSocket URL - is another debugger connected?")
            return
        
        self._current_page_url = tab.get('url', '')
        self._current_page_title = tab.get('title', '')
        
        logger.info(f"Connecting to tab: {self._current_page_title}")
        logger.info(f"URL: {self._current_page_url}")
        
        self._running = True
        
        try:
            async with websockets.connect(ws_url) as ws:
                self._ws = ws
                
                # Enable network monitoring
                await self._send_command('Network.enable')
                await self._send_command('Page.enable')
                
                logger.info("CDP interceptor started - monitoring network requests")
                
                # Listen for events
                async for message in ws:
                    if not self._running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Message handling error: {e}")
                        
        except websockets.exceptions.ConnectionClosed:
            logger.info("CDP connection closed")
        except Exception as e:
            logger.error(f"CDP error: {e}")
        finally:
            self._running = False
            self._ws = None
    
    def stop(self):
        """Stop intercepting."""
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())
    
    async def _send_command(self, method: str, params: dict = None) -> int:
        """Send CDP command."""
        self._msg_id += 1
        msg = {
            'id': self._msg_id,
            'method': method,
            'params': params or {}
        }
        await self._ws.send(json.dumps(msg))
        return self._msg_id
    
    async def _handle_message(self, data: dict):
        """Handle CDP message."""
        method = data.get('method', '')
        params = data.get('params', {})
        
        if method == 'Network.requestWillBeSent':
            await self._on_request(params)
        elif method == 'Network.responseReceived':
            await self._on_response(params)
        elif method == 'Page.frameNavigated':
            frame = params.get('frame', {})
            if not frame.get('parentId'):  # Main frame only
                self._current_page_url = frame.get('url', '')
                self._current_page_title = frame.get('name', '')
                logger.info(f"Navigated to: {self._current_page_url}")
    
    async def _on_request(self, params: dict):
        """Handle network request event."""
        request = params.get('request', {})
        url = request.get('url', '')
        request_id = params.get('requestId', '')
        
        # Store request for later matching with response
        self._pending_requests[request_id] = {
            'url': url,
            'method': request.get('method', 'GET'),
            'headers': request.get('headers', {}),
            'timestamp': params.get('timestamp', 0),
        }
        
        # Quick classification for logging
        media_type = classify_url(url)
        if media_type:
            logger.debug(f"Potential media request: {media_type.value} - {url[:100]}")
    
    async def _on_response(self, params: dict):
        """Handle network response event."""
        response = params.get('response', {})
        url = response.get('url', '')
        request_id = params.get('requestId', '')
        
        # Get content type from headers
        headers = response.get('headers', {})
        content_type = headers.get('content-type', headers.get('Content-Type', ''))
        content_length = headers.get('content-length', headers.get('Content-Length', '0'))
        
        try:
            content_length = int(content_length)
        except (ValueError, TypeError):
            content_length = 0
        
        # Debug: Log googlevideo URLs
        if 'googlevideo' in url.lower() or 'videoplayback' in url.lower():
            logger.info(f"Found googlevideo URL: {url[:150]}...")
        
        # Classify the URL
        media_type = classify_url(url, content_type)
        
        if media_type:
            # Skip small responses (likely not actual video content)
            if media_type == MediaType.VIDEO and content_length < self.min_video_size:
                if content_length > 0:  # Only log if we know the size
                    logger.debug(f"Skipping small video ({content_length} bytes): {url[:80]}")
                    return
            
            # Skip duplicates
            if self.filter_duplicates:
                # Normalize URL for dedup (remove some query params)
                norm_url = self._normalize_url(url)
                if norm_url in self._seen_urls:
                    return
                self._seen_urls.add(norm_url)
            
            # Create media info
            media_info = MediaInfo(
                url=url,
                media_type=media_type,
                content_type=content_type,
                title=self._current_page_title,
                source_url=self._current_page_url,
                metadata={
                    'content_length': content_length,
                    'headers': dict(headers),
                }
            )
            
            logger.info(f"Captured {media_type.value}: {url[:100]}...")
            
            # Notify callback
            if self.on_media:
                try:
                    self.on_media(media_info)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        # Clean up pending request
        self._pending_requests.pop(request_id, None)
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Keep host and path, remove most query params
        qs = parse_qs(parsed.query)
        
        # Keep only significant params
        keep_params = {'v', 'id', 'video_id', 'itag', 'range'}
        filtered_qs = {k: v for k, v in qs.items() if k in keep_params}
        
        # Rebuild
        from urllib.parse import urlencode
        query = urlencode(filtered_qs, doseq=True) if filtered_qs else ''
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query}" if query else f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    def clear_seen(self):
        """Clear seen URL cache."""
        self._seen_urls.clear()


# Convenience function for checking Chrome status
async def check_chrome_debugging(host: str = "localhost", port: int = 9222) -> tuple[bool, str]:
    """Check if Chrome is running with remote debugging enabled."""
    interceptor = CDPInterceptor(host=host, port=port)
    return await interceptor.check_chrome()


def get_chrome_launch_command(port: int = 9222) -> str:
    """Get command to launch Chrome with remote debugging enabled."""
    import platform
    
    if platform.system() == "Windows":
        return f'start chrome.exe --remote-debugging-port={port} --user-data-dir="%TEMP%\\ChromeDebug"'
    elif platform.system() == "Darwin":  # macOS
        return f'/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port={port} --user-data-dir="/tmp/ChromeDebug" &'
    else:  # Linux
        return f'google-chrome --remote-debugging-port={port} --user-data-dir="/tmp/ChromeDebug" &'
