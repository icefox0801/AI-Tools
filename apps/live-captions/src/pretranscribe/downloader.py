"""
Media Downloader for Pre-transcription

Downloads video/audio content from URLs and extracts audio for transcription.
Uses yt-dlp for best quality audio extraction from supported sites.
Falls back to direct download for unsupported URLs.
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DownloadStatus(Enum):
    """Status of a download operation."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadResult:
    """Result of a download operation."""
    status: DownloadStatus
    audio_path: Optional[str] = None  # Path to extracted audio file
    duration: Optional[float] = None  # Duration in seconds
    title: str = ""
    error: str = ""
    source_url: str = ""
    metadata: dict = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status == DownloadStatus.COMPLETE and self.audio_path is not None


def check_ffmpeg() -> Optional[str]:
    """Check if ffmpeg is available and return path."""
    import shutil
    
    # Check PATH first
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Check common Windows locations
    common_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        os.path.expanduser(r'~\ffmpeg\bin\ffmpeg.exe'),
        os.path.expanduser(r'~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe'),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def check_ytdlp() -> bool:
    """Check if yt-dlp Python module is available."""
    try:
        import yt_dlp
        return True
    except ImportError:
        return False


def get_audio_duration(audio_path: str, ffprobe_path: Optional[str] = None) -> Optional[float]:
    """Get duration of audio file using ffprobe."""
    import shutil
    
    # Find ffprobe
    if ffprobe_path:
        probe = ffprobe_path.replace('ffmpeg', 'ffprobe')
    else:
        probe = shutil.which('ffprobe')
        if not probe:
            # Try same directory as ffmpeg
            ffmpeg = check_ffmpeg()
            if ffmpeg:
                probe = ffmpeg.replace('ffmpeg', 'ffprobe')
    
    if not probe or not os.path.exists(probe):
        return None
    
    try:
        result = subprocess.run([
            probe, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None


class MediaDownloader:
    """
    Downloads and extracts audio from video URLs for pre-transcription.
    
    Supports:
    - YouTube, Vimeo, Twitch, and 1000+ sites via yt-dlp
    - Direct video/audio URLs
    - HLS/DASH manifests (limited)
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        target_sample_rate: int = 16000,
        cookies_file: Optional[str] = None,
    ):
        """
        Initialize media downloader.
        
        Args:
            output_dir: Directory for downloaded files (uses temp dir if None)
            on_progress: Callback for progress updates (progress 0-1, message)
            target_sample_rate: Target audio sample rate for ASR
            cookies_file: Path to Netscape-format cookies.txt file for authentication
        """
        self.output_dir = output_dir or tempfile.mkdtemp(prefix='pretranscribe_')
        self.on_progress = on_progress
        self.target_sample_rate = target_sample_rate
        self.cookies_file = cookies_file
        
        # Auto-detect cookies file if not specified
        if not self.cookies_file:
            default_cookies = os.path.join(os.path.dirname(__file__), '..', '..', 'cookies.txt')
            if os.path.exists(default_cookies):
                self.cookies_file = default_cookies
                logger.info(f"Using cookies file: {self.cookies_file}")
        
        self._has_ytdlp = check_ytdlp()
        self._ffmpeg_path = check_ffmpeg()
        
        if not self._ffmpeg_path:
            logger.warning("ffmpeg not found - audio extraction may fail")
        else:
            logger.info(f"Using ffmpeg: {self._ffmpeg_path}")
        if not self._has_ytdlp:
            logger.warning("yt-dlp not found - will use direct download only")
    
    def _ensure_nodejs_in_path(self):
        """Ensure Node.js is in PATH for yt-dlp to use."""
        import shutil
        
        # Already available?
        if shutil.which('node'):
            return
        
        # Common Node.js install paths on Windows
        nodejs_paths = [
            r'C:\Program Files\nodejs',
            r'C:\Program Files (x86)\nodejs',
            os.path.expanduser(r'~\AppData\Roaming\nvm\current'),  # nvm-windows
            os.path.expanduser(r'~\scoop\apps\nodejs\current'),  # scoop
        ]
        
        for path in nodejs_paths:
            node_exe = os.path.join(path, 'node.exe')
            if os.path.exists(node_exe):
                # Add to PATH for this process
                current_path = os.environ.get('PATH', '')
                if path not in current_path:
                    os.environ['PATH'] = path + os.pathsep + current_path
                    logger.info(f"Added Node.js to PATH: {path}")
                return
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback."""
        if self.on_progress:
            try:
                self.on_progress(progress, message)
            except Exception:
                pass
    
    def _ytdlp_progress_hook(self, d):
        """Progress hook for yt-dlp."""
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                progress = 0.2 + (downloaded / total) * 0.5  # 20-70%
                self._report_progress(progress, f"Downloading: {int(progress * 100)}%")
        elif d['status'] == 'finished':
            self._report_progress(0.7, "Download complete, processing...")
    
    def _run_ytdlp(self, url: str, opts: dict) -> Optional[dict]:
        """Run yt-dlp synchronously (for thread pool)."""
        import yt_dlp
        
        # Ensure Node.js is in PATH for YouTube extraction
        self._ensure_nodejs_in_path()
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return info
        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp download error: {e}")
            return None
        except Exception as e:
            import traceback
            logger.error(f"yt-dlp error: {e}")
            logger.debug(f"yt-dlp traceback: {traceback.format_exc()}")
            return None
    
    async def download(self, url: str, title: str = "") -> DownloadResult:
        """
        Download and extract audio from URL.
        
        Args:
            url: Video or audio URL
            title: Optional title for the content
            
        Returns:
            DownloadResult with path to audio file
        """
        self._report_progress(0, f"Starting download: {title or url[:50]}")
        
        # Determine download method
        if self._should_use_ytdlp(url):
            return await self._download_ytdlp(url, title)
        else:
            return await self._download_direct(url, title)
    
    def _should_use_ytdlp(self, url: str) -> bool:
        """Check if yt-dlp should be used for this URL."""
        if not self._has_ytdlp:
            return False
        
        # Sites that yt-dlp handles well
        ytdlp_domains = [
            'youtube.com', 'youtu.be',
            'vimeo.com',
            'twitch.tv',
            'dailymotion.com',
            'bilibili.com',
            'twitter.com', 'x.com',
            'facebook.com',
            'instagram.com',
            'tiktok.com',
            'soundcloud.com',
        ]
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '')
        
        return any(d in domain for d in ytdlp_domains)
    
    async def _download_ytdlp(self, url: str, title: str) -> DownloadResult:
        """Download using yt-dlp Python module."""
        self._report_progress(0.1, "Extracting video info...")
        
        # Check for ffmpeg
        if not self._ffmpeg_path:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                error="ffmpeg not found - required for audio extraction. Install ffmpeg and add to PATH.",
                source_url=url
            )
        
        try:
            import yt_dlp
        except ImportError:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                error="yt-dlp module not installed",
                source_url=url
            )
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_title = "".join(c for c in title if c.isalnum() or c in ' -_')[:50] or "video"
        output_template = os.path.join(self.output_dir, f"{safe_title}_{timestamp}.%(ext)s")
        
        # yt-dlp options for audio extraction
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'outtmpl': output_template,
            'ffmpeg_location': os.path.dirname(self._ffmpeg_path),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'postprocessor_args': [
                '-ar', str(self.target_sample_rate),
                '-ac', '1',  # Mono
            ],
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [self._ytdlp_progress_hook],
            # Try to use android player client to avoid JS requirement
            'extractor_args': {'youtube': {'player_client': ['android']}},
        }
        
        # Add cookies if available
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts['cookiefile'] = self.cookies_file
            logger.info(f"Using cookies from: {self.cookies_file}")
        
        try:
            self._report_progress(0.2, "Downloading audio...")
            
            # Run yt-dlp in thread pool to not block async
            loop = asyncio.get_event_loop()
            result_info = await loop.run_in_executor(
                None,
                lambda: self._run_ytdlp(url, ydl_opts)
            )
            
            if result_info is None:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    error="yt-dlp download failed",
                    source_url=url
                )
            
            # Find the output WAV file
            output_file = output_template.replace('.%(ext)s', '.wav')
            
            if not os.path.exists(output_file):
                # Look for any audio file
                for ext in ['.wav', '.m4a', '.webm', '.mp3', '.opus']:
                    check_path = output_template.replace('.%(ext)s', ext)
                    if os.path.exists(check_path):
                        output_file = check_path
                        break
            
            if not os.path.exists(output_file):
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    error="Output file not found after download",
                    source_url=url
                )
            
            # Convert to WAV if needed
            self._report_progress(0.8, "Processing audio...")
            if not output_file.endswith('.wav'):
                wav_path = await self._convert_to_wav(output_file)
                if wav_path:
                    os.remove(output_file)  # Clean up original
                    output_file = wav_path
            
            # Get duration
            duration = get_audio_duration(output_file, self._ffmpeg_path)
            
            self._report_progress(1.0, "Download complete!")
            
            return DownloadResult(
                status=DownloadStatus.COMPLETE,
                audio_path=output_file,
                duration=duration,
                title=title,
                source_url=url
            )
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return DownloadResult(
                status=DownloadStatus.FAILED,
                error=str(e),
                source_url=url
            )
    
    async def _download_direct(self, url: str, title: str) -> DownloadResult:
        """Download directly from URL using ffmpeg."""
        if not self._ffmpeg_path:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                error="ffmpeg not available for direct download. Install ffmpeg and add to PATH.",
                source_url=url
            )
        
        self._report_progress(0.1, "Starting direct download...")
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_title = "".join(c for c in title if c.isalnum() or c in ' -_')[:50] or "media"
        output_path = os.path.join(self.output_dir, f"{safe_title}_{timestamp}.wav")
        
        # Use ffmpeg to download and convert
        cmd = [
            self._ffmpeg_path,
            '-i', url,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(self.target_sample_rate),
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            output_path
        ]
        
        try:
            self._report_progress(0.3, "Downloading and converting...")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode != 0 or not os.path.exists(output_path):
                error = stderr.decode('utf-8', errors='ignore')
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    error=f"ffmpeg failed: {error[:200]}",
                    source_url=url
                )
            
            duration = get_audio_duration(output_path)
            
            self._report_progress(1.0, "Download complete!")
            
            return DownloadResult(
                status=DownloadStatus.COMPLETE,
                audio_path=output_path,
                duration=duration,
                title=title,
                source_url=url
            )
            
        except Exception as e:
            logger.error(f"Direct download error: {e}")
            return DownloadResult(
                status=DownloadStatus.FAILED,
                error=str(e),
                source_url=url
            )
    
    async def _convert_to_wav(self, input_path: str) -> Optional[str]:
        """Convert audio file to WAV format."""
        if not self._ffmpeg_path:
            return None
        
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
        
        cmd = [
            self._ffmpeg_path,
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(self.target_sample_rate),
            '-ac', '1',
            '-y',
            output_path
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
        except Exception as e:
            logger.error(f"Conversion error: {e}")
        
        return None
    
    def cleanup(self, keep_files: list[str] = None):
        """Clean up downloaded files."""
        keep = set(keep_files or [])
        
        for file in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, file)
            if file_path not in keep:
                try:
                    os.remove(file_path)
                except Exception:
                    pass


async def download_and_extract_audio(
    url: str,
    title: str = "",
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> DownloadResult:
    """
    Convenience function to download and extract audio from a URL.
    
    Args:
        url: Video or audio URL
        title: Optional title
        output_dir: Output directory (uses temp if None)
        on_progress: Progress callback
        
    Returns:
        DownloadResult with path to extracted audio
    """
    downloader = MediaDownloader(output_dir=output_dir, on_progress=on_progress)
    return await downloader.download(url, title)
