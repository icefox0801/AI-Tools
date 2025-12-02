"""
Pre-transcription Pipeline

Orchestrates the full pre-transcription workflow:
1. Capture video URL from browser via CDP
2. Download and extract audio
3. Transcribe with timestamps
4. Provide timestamped segments for playback sync
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from .downloader import MediaDownloader, DownloadResult, DownloadStatus
from .transcriber import PreTranscriber, TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Status of the pre-transcription pipeline."""
    IDLE = "idle"
    WAITING_FOR_MEDIA = "waiting_for_media"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    READY = "ready"
    PLAYING = "playing"
    ERROR = "error"


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    status: PipelineStatus = PipelineStatus.IDLE
    progress: float = 0.0  # 0-1
    message: str = ""
    current_url: str = ""
    title: str = ""
    error: str = ""
    transcript: Optional[TranscriptResult] = None
    audio_path: str = ""
    duration: float = 0.0


class PreTranscribePipeline:
    """
    Complete pre-transcription pipeline that:
    - Monitors browser for video URLs via CDP
    - Downloads and transcribes content ahead of playback
    - Provides timestamp-synced captions
    """
    
    def __init__(
        self,
        asr_host: str = "localhost",
        asr_port: int = 8001,  # Whisper default
        cdp_host: str = "localhost",
        cdp_port: int = 9222,
        on_state_change: Optional[Callable[[PipelineState], None]] = None,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize pre-transcription pipeline.
        
        Args:
            asr_host: ASR service host
            asr_port: ASR service port
            cdp_host: Chrome DevTools Protocol host
            cdp_port: Chrome DevTools Protocol port
            on_state_change: Callback when pipeline state changes
            on_segment: Callback when a new segment should be displayed
            output_dir: Directory for temporary files
        """
        self.asr_host = asr_host
        self.asr_port = asr_port
        self.cdp_host = cdp_host
        self.cdp_port = cdp_port
        self.on_state_change = on_state_change
        self.on_segment = on_segment
        self.output_dir = output_dir or tempfile.mkdtemp(prefix='pretranscribe_')
        
        # Components
        self.downloader = MediaDownloader(
            output_dir=self.output_dir,
            on_progress=self._on_download_progress
        )
        self.transcriber = PreTranscriber(
            host=asr_host,
            port=asr_port,
            on_progress=self._on_transcribe_progress
        )
        
        # State
        self.state = PipelineState()
        self._running = False
        self._cdp_interceptor = None
        self._playback_task = None
        self._playback_start_time: Optional[float] = None
        self._pending_urls: asyncio.Queue = asyncio.Queue()
    
    def _update_state(self, **kwargs):
        """Update state and notify callback."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        if self.on_state_change:
            try:
                self.on_state_change(self.state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def _on_download_progress(self, progress: float, message: str):
        """Handle download progress."""
        self._update_state(progress=progress * 0.5, message=message)  # 0-50%
    
    def _on_transcribe_progress(self, progress: float, message: str):
        """Handle transcription progress."""
        self._update_state(progress=0.5 + progress * 0.5, message=message)  # 50-100%
    
    async def check_services(self) -> tuple[bool, dict[str, tuple[bool, str]]]:
        """
        Check if all required services are available.
        
        Returns:
            Tuple of (all_ok, service_status_dict)
        """
        results = {}
        
        # Check ASR service
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.asr_host}:{self.asr_port}/",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    results['asr'] = (True, f"Ready (port {self.asr_port})")
        except Exception as e:
            results['asr'] = (False, f"Not available: {e}")
        
        # Check CDP (Chrome)
        from ..cdp import CDPInterceptor
        interceptor = CDPInterceptor(host=self.cdp_host, port=self.cdp_port)
        cdp_ok, cdp_msg = await interceptor.check_chrome()
        results['chrome'] = (cdp_ok, cdp_msg)
        
        all_ok = all(ok for ok, _ in results.values())
        return all_ok, results
    
    async def start(self):
        """Start the pre-transcription pipeline."""
        self._running = True
        self._update_state(
            status=PipelineStatus.WAITING_FOR_MEDIA,
            message="Monitoring browser for media..."
        )
        
        # Start CDP interceptor
        await self._start_cdp()
        
        # Process media URLs as they come in
        while self._running:
            try:
                # Wait for media URL with timeout
                url_info = await asyncio.wait_for(
                    self._pending_urls.get(),
                    timeout=1.0
                )
                
                # Process the media (with direct URL as fallback)
                await self._process_media(
                    url_info['url'], 
                    url_info.get('title', ''),
                    direct_url=url_info.get('direct_url')
                )
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                self._update_state(status=PipelineStatus.ERROR, error=str(e))
    
    async def stop(self):
        """Stop the pipeline."""
        self._running = False
        
        if self._cdp_interceptor:
            self._cdp_interceptor.stop()
        
        if self._playback_task:
            self._playback_task.cancel()
        
        self._update_state(status=PipelineStatus.IDLE, message="Stopped")
    
    async def _start_cdp(self):
        """Start CDP interceptor."""
        from ..cdp import CDPInterceptor, MediaType
        
        # Track processed URLs to avoid duplicates
        processed_pages = set()
        
        def on_media(media_info):
            # Use the PAGE URL (source_url) for yt-dlp, not the stream URL
            page_url = media_info.source_url
            
            logger.debug(f"on_media callback: type={media_info.media_type}, page={page_url[:80] if page_url else 'None'}")
            
            # Skip if already processed this page
            if page_url in processed_pages:
                logger.debug(f"Skipping already processed page: {page_url[:60]}")
                return
            
            # Only process video pages from supported sites
            if not self._is_supported_video_page(page_url):
                logger.debug(f"Skipping unsupported page: {page_url[:60] if page_url else 'None'}")
                return
            
            # Only trigger on video/manifest requests (indicates video is playing)
            if media_info.media_type in (MediaType.VIDEO, MediaType.MANIFEST):
                logger.info(f"Detected video on page: {page_url}")
                processed_pages.add(page_url)
                
                try:
                    self._pending_urls.put_nowait({
                        'url': page_url,
                        'title': media_info.title,
                        'direct_url': media_info.url,
                    })
                except asyncio.QueueFull:
                    pass
        
        self._cdp_interceptor = CDPInterceptor(
            host=self.cdp_host,
            port=self.cdp_port,
            on_media=on_media
        )
        
        # Start in background
        asyncio.create_task(self._cdp_interceptor.start())
    
    def _is_supported_video_page(self, url: str) -> bool:
        """Check if URL is a supported video page for yt-dlp."""
        if not url:
            return False
        
        url_lower = url.lower()
        
        # YouTube
        if 'youtube.com/watch' in url_lower or 'youtu.be/' in url_lower:
            return True
        
        # Vimeo
        if 'vimeo.com/' in url_lower and '/video/' not in url_lower:
            return True
        
        # Twitch
        if 'twitch.tv/' in url_lower:
            return True
        
        # Dailymotion
        if 'dailymotion.com/video/' in url_lower:
            return True
        
        # Bilibili
        if 'bilibili.com/video/' in url_lower:
            return True
        
        # Twitter/X videos
        if ('twitter.com/' in url_lower or 'x.com/' in url_lower) and '/status/' in url_lower:
            return True
        
        return False
    
    async def _process_media(self, url: str, title: str, direct_url: str = None):
        """Process a media URL through the pipeline."""
        self._update_state(
            status=PipelineStatus.DOWNLOADING,
            current_url=url,
            title=title,
            progress=0,
            message=f"Downloading: {title or url[:50]}"
        )
        
        # Download and extract audio
        # First try with yt-dlp using page URL
        result = await self.downloader.download(url, title)
        
        # If yt-dlp fails and we have a direct stream URL, try that
        if not result.success and direct_url:
            logger.info(f"yt-dlp failed, trying direct stream URL...")
            self._update_state(message="Trying direct download...")
            result = await self.downloader.download(direct_url, title)
        
        if not result.success:
            self._update_state(
                status=PipelineStatus.ERROR,
                error=result.error
            )
            return
        
        # Transcribe
        self._update_state(
            status=PipelineStatus.TRANSCRIBING,
            audio_path=result.audio_path,
            duration=result.duration or 0,
            message="Transcribing audio..."
        )
        
        transcript = await self.transcriber.transcribe_file(result.audio_path)
        
        if not transcript or not transcript.segments:
            self._update_state(
                status=PipelineStatus.ERROR,
                error="Transcription failed or returned no content"
            )
            return
        
        # Ready for playback
        self._update_state(
            status=PipelineStatus.READY,
            transcript=transcript,
            progress=1.0,
            message=f"Ready: {len(transcript.segments)} segments ({transcript.duration:.1f}s)"
        )
        
        logger.info(f"Pre-transcription complete: {len(transcript.segments)} segments")
    
    async def process_url(self, url: str, title: str = ""):
        """
        Manually process a URL (without CDP).
        
        Args:
            url: Video URL to process
            title: Optional title
        """
        await self._process_media(url, title)
    
    def start_playback(self, start_time: float = 0):
        """
        Start synced playback of captions.
        
        Args:
            start_time: Starting position in seconds (for seeking)
        """
        if self.state.status != PipelineStatus.READY or not self.state.transcript:
            logger.warning("No transcript ready for playback")
            return
        
        self._playback_start_time = asyncio.get_event_loop().time() - start_time
        self._update_state(status=PipelineStatus.PLAYING)
        
        # Start playback task
        if self._playback_task:
            self._playback_task.cancel()
        
        self._playback_task = asyncio.create_task(self._run_playback())
    
    def stop_playback(self):
        """Stop playback."""
        if self._playback_task:
            self._playback_task.cancel()
            self._playback_task = None
        
        self._playback_start_time = None
        
        if self.state.status == PipelineStatus.PLAYING:
            self._update_state(status=PipelineStatus.READY)
    
    def seek(self, position: float):
        """
        Seek to position in the video.
        
        Args:
            position: Position in seconds
        """
        if self._playback_start_time is not None:
            self._playback_start_time = asyncio.get_event_loop().time() - position
    
    def get_current_time(self) -> float:
        """Get current playback position."""
        if self._playback_start_time is None:
            return 0
        return asyncio.get_event_loop().time() - self._playback_start_time
    
    async def _run_playback(self):
        """Run caption playback loop."""
        if not self.state.transcript:
            return
        
        segments = self.state.transcript.segments
        current_segment_idx = 0
        
        try:
            while self._running and current_segment_idx < len(segments):
                current_time = self.get_current_time()
                
                # Find segment for current time
                while current_segment_idx < len(segments):
                    segment = segments[current_segment_idx]
                    
                    if current_time < segment.start:
                        # Wait until segment starts
                        wait_time = segment.start - current_time
                        if wait_time > 0:
                            await asyncio.sleep(min(wait_time, 0.1))
                        break
                    
                    if segment.start <= current_time < segment.end:
                        # Display current segment
                        if self.on_segment:
                            self.on_segment(segment)
                        break
                    
                    # Move to next segment
                    current_segment_idx += 1
                
                await asyncio.sleep(0.05)  # 50ms update rate
        
        except asyncio.CancelledError:
            pass
        finally:
            if self.state.status == PipelineStatus.PLAYING:
                self._update_state(status=PipelineStatus.READY)
    
    def get_segment_at(self, time: float) -> Optional[TranscriptSegment]:
        """Get transcript segment at given time."""
        if self.state.transcript:
            return self.state.transcript.get_text_at(time)
        return None
    
    def get_all_segments(self) -> list[TranscriptSegment]:
        """Get all transcript segments."""
        if self.state.transcript:
            return self.state.transcript.segments
        return []
    
    def cleanup(self):
        """Clean up temporary files."""
        self.downloader.cleanup()


async def quick_pretranscribe(
    url: str,
    asr_host: str = "localhost",
    asr_port: int = 8001,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> Optional[TranscriptResult]:
    """
    Quick function to download and transcribe a video URL.
    
    Args:
        url: Video URL
        asr_host: ASR service host
        asr_port: ASR service port
        on_progress: Progress callback
        
    Returns:
        TranscriptResult or None on failure
    """
    downloader = MediaDownloader(on_progress=lambda p, m: on_progress(p * 0.5, m) if on_progress else None)
    transcriber = PreTranscriber(
        host=asr_host,
        port=asr_port,
        on_progress=lambda p, m: on_progress(0.5 + p * 0.5, m) if on_progress else None
    )
    
    # Download
    result = await downloader.download(url)
    if not result.success:
        logger.error(f"Download failed: {result.error}")
        return None
    
    # Transcribe
    transcript = await transcriber.transcribe_file(result.audio_path)
    
    # Cleanup
    if result.audio_path:
        try:
            os.remove(result.audio_path)
        except Exception:
            pass
    
    return transcript
