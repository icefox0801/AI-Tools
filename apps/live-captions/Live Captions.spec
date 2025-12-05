# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Get the project root dynamically
spec_dir = os.path.dirname(os.path.abspath(SPEC))
project_root = os.path.dirname(os.path.dirname(spec_dir))

a = Analysis(
    ['live_captions_tray.py'],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=['shared', 'shared.config', 'shared.config.backends'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Live Captions',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Live Captions',
)
