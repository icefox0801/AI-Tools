"""CSS styles for the Audio Notes UI."""

# CSS to hide all Gradio progress bars/spinners
CUSTOM_CSS = """
/* Hide ALL progress indicators globally - but not radio/checkbox inputs */
.progress-bar, .progress-text, .eta-bar { display: none !important; }
div.generating { display: none !important; }
div.pending { display: none !important; }
div[class*="progress"]:not([class*="radio"]):not([class*="checkbox"]) { display: none !important; }
span[class*="progress"]:not([class*="radio"]):not([class*="checkbox"]) { display: none !important; }
.progress-level { display: none !important; }
.progress-level-inner { display: none !important; }
.meta-text { display: none !important; }
.meta-text-center { display: none !important; }
.wrap.center.full.generating { display: none !important; }
.wrap.default.center.full.generating { display: none !important; }
/* Hide spinners */
.loader { display: none !important; }
div.loading { display: none !important; }
/* Hide processing overlay */
div.processing { display: none !important; }
[data-testid="progress"] { display: none !important; }

/* Scrollable new recordings list - target the checkbox group container */
#new-recordings-list {
    max-height: 250px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 8px;
    background: var(--background-fill-secondary, #f9fafb);
}

/* Disabled button style - make it visually obvious */
#chat-send-btn[disabled],
#chat-send-btn:disabled,
button#chat-send-btn[disabled] {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
    background-color: #9ca3af !important;
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
