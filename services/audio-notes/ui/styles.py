"""CSS styles for the Audio Notes UI."""

# Minimal CSS - the theme handles radio/checkbox hover colors
CUSTOM_CSS = """
/* Hide Gradio's loading spinner in textboxes and other components */
.loading {
    display: none !important;
}

/* Hide the orange loading icon that appears in textboxes */
span[aria-label="Loading..."],
.loader {
    display: none !important;
}

/* Hide loading indicators in various Gradio components */
.wrap .loader,
.textarea-wrapper .loader,
.input-wrapper .loader {
    display: none !important;
}

/* Hide Gradio loading/pending borders that appear during requests */
.pending,
.generating,
[data-testid="block"][class*="pending"],
[data-testid="block"][class*="generating"] {
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Hide the orange pulsing border animation during loading */
.border-none,
.svelte-1kcgrqr,
[class*="pending"] {
    border: none !important;
    animation: none !important;
}

/* Target Gradio's loading state borders */
.wrap.svelte-1kcgrqr.generating,
.wrap.svelte-1kcgrqr.pending {
    border: none !important;
    box-shadow: none !important;
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

# JavaScript to enable Ctrl+Enter to submit chat and fix audio player initialization
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

        // Fix for audio player first-play issue
        // When audio source changes, ensure the audio element is properly loaded
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
                    const audio = mutation.target;
                    if (audio && audio.tagName === 'AUDIO' && audio.src) {
                        // Force reload the audio element
                        audio.load();
                    }
                }
                // Also watch for added audio elements
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === 1) {
                            const audios = node.querySelectorAll ? node.querySelectorAll('audio') : [];
                            audios.forEach((audio) => {
                                if (audio.src) {
                                    audio.load();
                                }
                            });
                        }
                    });
                }
            });
        });

        // Observe the entire document for audio changes
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['src']
        });
    }, 1000);
    return [];
}
"""
