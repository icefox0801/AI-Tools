"""FastAPI routes for audio upload and recordings API."""

import io
import wave
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from config import RECORDINGS_DIR, logger
from services.recordings import list_recordings, get_audio_duration


def setup_api_routes(app: FastAPI):
    """Setup custom API routes for audio upload."""
    
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
