"""FastAPI routes for audio upload and recordings API."""

import io
import wave
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from config import RECORDINGS_DIR, logger
from services.recordings import list_recordings, get_audio_duration

# Get app directory for static files
APP_DIR = Path(__file__).parent.parent


def setup_api_routes(app: FastAPI):
    """Setup custom API routes for audio upload."""
    
    @app.get("/favicon.ico")
    async def favicon():
        """Serve favicon."""
        favicon_path = APP_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/x-icon")
        raise HTTPException(status_code=404, detail="Favicon not found")
    
    @app.get("/icon-192.png")
    async def icon_192():
        """Serve 192x192 icon for PWA."""
        icon_path = APP_DIR / "icon-192.png"
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="Icon not found")
    
    @app.get("/icon-512.png")
    async def icon_512():
        """Serve 512x512 icon for PWA."""
        icon_path = APP_DIR / "icon-512.png"
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="Icon not found")
    
    @app.get("/manifest.json")
    async def manifest():
        """Serve PWA manifest."""
        manifest_path = APP_DIR / "manifest.json"
        if manifest_path.exists():
            return FileResponse(manifest_path, media_type="application/manifest+json")
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    @app.post("/api/upload-audio")
    async def upload_audio(
        audio: UploadFile = File(...),
        filename: str = Form(...),
        append: bool = Form(False)
    ):
        """Upload audio file from Live Captions."""
        try:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            
            safe_filename = Path(filename).name
            if not safe_filename.endswith('.wav'):
                safe_filename += '.wav'
            
            file_path = RECORDINGS_DIR / safe_filename
            audio_data = await audio.read()
            
            if append and file_path.exists():
                try:
                    with wave.open(str(file_path), 'rb') as existing:
                        params = existing.getparams()
                        existing_frames = existing.readframes(existing.getnframes())
                    
                    with io.BytesIO(audio_data) as new_audio_io:
                        with wave.open(new_audio_io, 'rb') as new_wav:
                            new_frames = new_wav.readframes(new_wav.getnframes())
                    
                    with wave.open(str(file_path), 'wb') as out:
                        out.setparams(params)
                        out.writeframes(existing_frames + new_frames)
                    
                    logger.info(f"Appended audio to {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to append, overwriting: {e}")
                    with open(file_path, 'wb') as f:
                        f.write(audio_data)
            else:
                with open(file_path, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"Saved audio to {file_path}")
            
            duration = get_audio_duration(str(file_path))
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            return JSONResponse({
                "status": "success",
                "path": str(file_path),
                "filename": safe_filename,
                "duration": duration,
                "size_mb": round(size_mb, 2)
            })
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/recordings")
    async def list_recordings_api():
        """List available recordings."""
        try:
            recordings = list_recordings()
            return JSONResponse({
                "status": "success",
                "recordings": recordings,
                "directory": str(RECORDINGS_DIR)
            })
        except Exception as e:
            logger.error(f"List recordings failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/health")
    async def api_health():
        """Health check endpoint."""
        return JSONResponse({
            "status": "ok",
            "recordings_dir": str(RECORDINGS_DIR),
            "recordings_count": len(list(RECORDINGS_DIR.glob("*.wav"))) if RECORDINGS_DIR.exists() else 0
        })
