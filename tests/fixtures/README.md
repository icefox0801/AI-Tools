# Test Fixtures

Shared test fixtures for E2E and integration tests.

## Audio Files

Test audio files in 16kHz mono WAV format for ASR testing:

| File | Content | Duration | Generated With |
|------|---------|----------|----------------|
| `hello_16k.wav` | "Hello" | ~1.5s | edge-tts (en-US-AriaNeural) |
| `numbers_16k.wav` | "One two three four five" | ~2.6s | edge-tts (en-US-AriaNeural) |
| `test_tone_16k.wav` | 440Hz sine wave | 1.0s | Python wave module |

## Audio Requirements

All audio files must be:
- **Sample rate**: 16000 Hz (16kHz)
- **Channels**: 1 (mono)
- **Bit depth**: 16-bit signed PCM
- **Format**: WAV (RIFF)

## Regenerating Audio

```bash
# Install edge-tts
pip install edge-tts

# Generate speech
python -c "
import asyncio
import edge_tts

async def main():
    # Hello
    comm = edge_tts.Communicate('Hello', 'en-US-AriaNeural')
    await comm.save('hello.mp3')
    
    # Numbers
    comm = edge_tts.Communicate('One two three four five', 'en-US-AriaNeural')
    await comm.save('numbers.mp3')

asyncio.run(main())
"

# Convert to 16kHz mono WAV
ffmpeg -i hello.mp3 -ar 16000 -ac 1 -acodec pcm_s16le hello_16k.wav
ffmpeg -i numbers.mp3 -ar 16000 -ac 1 -acodec pcm_s16le numbers_16k.wav
```

## Usage in Tests

```python
@pytest.fixture(scope="session")
def hello_audio(test_audio_dir: str) -> bytes:
    """Load 'hello' test audio."""
    with open(os.path.join(test_audio_dir, "hello_16k.wav"), "rb") as f:
        return f.read()
```
