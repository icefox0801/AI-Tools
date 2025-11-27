"""
Voice Transcribe - Desktop App Launcher
Opens the web GUI in a fixed 800x800 window using PyQt6
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl


class VoiceTranscribeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Transcribe")
        self.setFixedSize(800, 800)
        
        # Create web view
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("http://localhost:8080"))
        self.setCentralWidget(self.browser)


def main():
    app = QApplication(sys.argv)
    window = VoiceTranscribeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
