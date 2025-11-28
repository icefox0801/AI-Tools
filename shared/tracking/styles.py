"""
CSS and JavaScript Styles for Chunk Tracking

Provides styling for HTML rendering of tracked transcripts.
"""

# CSS styles for HTML rendering
CHUNK_TRACKER_CSS = """
.transcript {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
}
.segment {
    display: inline;
    transition: background-color 0.3s, color 0.3s;
    border-radius: 2px;
    padding: 0 2px;
}
.segment.partial {
    color: #666;
    font-style: italic;
}
.segment.partial .partial-indicator {
    color: #999;
    animation: blink 1s infinite;
}
@keyframes blink {
    50% { opacity: 0.5; }
}
.segment.final {
    color: inherit;
}
.segment.regenerating {
    background-color: #fff3cd;
    color: #856404;
}
.segment.replaced {
    text-decoration: line-through;
    color: #999;
    display: none;  /* Hide replaced by default */
}
.segment:hover {
    background-color: #e3f2fd;
    cursor: pointer;
}
/* Show chunk info on hover */
.segment[data-chunks]:hover::after {
    content: " [" attr(data-chunks) "]";
    font-size: 0.7em;
    color: #888;
}
/* Highlight class for related segments */
.segment.highlight {
    background-color: #bbdefb !important;
}
"""

# JavaScript for interactive features
CHUNK_TRACKER_JS = """
<script>
// Highlight all segments from the same chunk on hover
document.querySelectorAll('.segment').forEach(span => {
    span.addEventListener('mouseenter', () => {
        const chunks = span.dataset.chunks.split(',');
        document.querySelectorAll('.segment').forEach(other => {
            const otherChunks = other.dataset.chunks.split(',');
            if (chunks.some(c => otherChunks.includes(c))) {
                other.classList.add('highlight');
            }
        });
    });
    span.addEventListener('mouseleave', () => {
        document.querySelectorAll('.segment').forEach(other => {
            other.classList.remove('highlight');
        });
    });
});

// Click to show segment details
document.querySelectorAll('.segment').forEach(span => {
    span.addEventListener('click', () => {
        const info = {
            segment: span.dataset.segment,
            chunks: span.dataset.chunks,
            status: span.dataset.status,
            text: span.textContent
        };
        console.log('Segment clicked:', info);
        // Dispatch custom event for external handlers
        span.dispatchEvent(new CustomEvent('segmentclick', { 
            detail: info, 
            bubbles: true 
        }));
    });
});
</script>
"""

# Dark theme variant
CHUNK_TRACKER_CSS_DARK = """
.transcript {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    color: #e0e0e0;
}
.segment {
    display: inline;
    transition: background-color 0.3s, color 0.3s;
    border-radius: 2px;
    padding: 0 2px;
}
.segment.partial {
    color: #888;
    font-style: italic;
}
.segment.partial .partial-indicator {
    color: #666;
    animation: blink 1s infinite;
}
@keyframes blink {
    50% { opacity: 0.5; }
}
.segment.final {
    color: inherit;
}
.segment.regenerating {
    background-color: #4a4000;
    color: #ffc107;
}
.segment.replaced {
    text-decoration: line-through;
    color: #666;
    display: none;
}
.segment:hover {
    background-color: #1e3a5f;
    cursor: pointer;
}
.segment[data-chunks]:hover::after {
    content: " [" attr(data-chunks) "]";
    font-size: 0.7em;
    color: #666;
}
.segment.highlight {
    background-color: #0d47a1 !important;
}
"""
