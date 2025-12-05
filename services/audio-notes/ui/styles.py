"""CSS styles for the Audio Notes UI."""

# Minimal CSS - the theme handles radio/checkbox hover colors
CUSTOM_CSS = """
/* Hide only progress bar elements with specific test ids */
[data-testid="progress-bar"],
[data-testid="progress-text"],
[data-testid="eta-bar"] { 
    display: none !important; 
}

/* Scrollable new recordings list - target the checkbox group container */
#new-recordings-list {
    max-height: 250px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

/* Scrollable transcribed recordings list */
#transcribed-recordings-list {
    max-height: 250px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

/* Disabled button style */
#chat-send-btn[disabled],
#chat-send-btn:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}
"""

# JavaScript to enable Ctrl+Enter to submit chat
CUSTOM_JS = """
function() {
    // Wait for DOM to be ready
    setTimeout(() => {
        document.addEventListener('keydown', function(e) {
            // Check for Ctrl+Enter
            if (e.ctrlKey && e.key === 'Enter') {
                // Find the chat input textarea
                const chatInput = document.querySelector('#chat-input textarea');
                if (chatInput && document.activeElement === chatInput) {
                    e.preventDefault();
                    // Find and click the Send button
                    const sendBtn = document.querySelector('#chat-send-btn');
                    if (sendBtn) {
                        sendBtn.click();
                    }
                }
            }
        });
    }, 1000);
    return [];
}
"""
