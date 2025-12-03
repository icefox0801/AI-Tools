"""CSS styles for the Audio Notes UI."""

# CSS to hide all Gradio progress bars/spinners
CUSTOM_CSS = """
/* Hide ALL progress indicators globally */
.progress-bar, .progress-text, .eta-bar { display: none !important; }
.generating { display: none !important; }
.pending { display: none !important; }
div[class*="progress"] { display: none !important; }
span[class*="progress"] { display: none !important; }
.progress-level { display: none !important; }
.progress-level-inner { display: none !important; }
.meta-text { display: none !important; }
.meta-text-center { display: none !important; }
.wrap.center.full { display: none !important; }
.wrap.default.center.full { display: none !important; }
/* Hide spinners */
.loader { display: none !important; }
.loading { display: none !important; }
/* Hide processing overlay */
.processing { display: none !important; }
[data-testid="progress"] { display: none !important; }
/* Svelte specific */
.svelte-1txqlrd { display: none !important; }
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
