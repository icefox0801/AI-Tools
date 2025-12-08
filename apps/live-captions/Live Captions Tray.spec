# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Live Captions Tray Application

This creates a single executable that:
- Runs as a system tray application
- Launches Live Captions with different backends
- Has no console window (windowed mode)

Build command:
    pyinstaller "Live Captions Tray.spec"
"""

import os
import sys
from pathlib import Path

# Get paths
spec_dir = os.path.dirname(os.path.abspath(SPEC))
project_root = os.path.dirname(os.path.dirname(spec_dir))

# Collect data files
datas = [
    # Include the main live_captions.py script
    ('live_captions.py', '.'),
    # Include the src directory
    ('src', 'src'),
    # Include shared modules
    (os.path.join(project_root, 'shared'), 'shared'),
    # Include build timestamp
    ('.build_time', '.'),
]

# Add icon if exists
if os.path.exists(os.path.join(spec_dir, 'icon.ico')):
    datas.append(('icon.ico', '.'))

a = Analysis(
    ['live_captions_tray.py'],
    pathex=[
        spec_dir,
        project_root,  # For shared module
    ],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'pystray',
        'pystray._win32',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        # Audio imports
        'pyaudio',
        'pyaudiowpatch',
        'numpy',
        'scipy',
        'scipy.signal',
        'websockets',
        'websockets.client',
        'asyncio',
        # Shared module imports
        'shared',
        'shared.config',
        'shared.client',
        'shared.config.backends',
        'shared.client.transcript',
        'shared.client.websocket_client',
        'shared.client.result',
        # Tkinter for caption window
        'tkinter',
        'tkinter.font',
        # Windows-specific
        'ctypes',
        'ctypes.wintypes',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary large packages
        'matplotlib',
        'pandas',
        'pytest',
        'IPython',
        'jupyter',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Live Captions',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window - tray app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
