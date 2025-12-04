"""
Unit tests for langchain_chat module.

Tests the LangChain-based chat service helper functions.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add audio-notes services to path for testing
audio_notes_path = Path(__file__).parent.parent.parent / "services" / "audio-notes"
sys.path.insert(0, str(audio_notes_path))

# Mock config before importing langchain_chat
sys.modules["config"] = MagicMock()
sys.modules["config"].OLLAMA_MODEL = "llama3.2"
sys.modules["config"].OLLAMA_URL = "http://ollama:11434/api"

# Now import the module (after mocking config)
from services.langchain_chat import _build_system_prompt  # noqa: E402


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt function."""

    def test_basic_transcript(self):
        """System prompt includes transcript."""
        prompt = _build_system_prompt("This is a test transcript.")
        assert "This is a test transcript." in prompt
        assert "You are a helpful assistant" in prompt

    def test_empty_transcript(self):
        """Empty transcript produces valid prompt."""
        prompt = _build_system_prompt("")
        assert "You are a helpful assistant" in prompt
        assert "transcript for context" in prompt

    def test_transcript_truncation(self):
        """Long transcripts are truncated at 6000 chars."""
        long_transcript = "a" * 7000
        prompt = _build_system_prompt(long_transcript)
        assert "...[transcript truncated]..." in prompt
        # Should contain exactly 6000 'a' characters
        assert "a" * 6000 in prompt
        assert "a" * 6001 not in prompt

    def test_with_summary(self):
        """Summary is included when provided."""
        prompt = _build_system_prompt("transcript", "This is a summary.")
        assert "This is a summary." in prompt
        assert "summary of the transcript" in prompt

    def test_summary_excluded_when_loading(self):
        """Summary starting with ⏳ is excluded (loading state)."""
        prompt = _build_system_prompt("transcript", "⏳ Loading...")
        assert "⏳ Loading..." not in prompt
        assert "summary of the transcript" not in prompt

    def test_summary_excluded_when_error(self):
        """Summary starting with ⚠️ is excluded (error state)."""
        prompt = _build_system_prompt("transcript", "⚠️ Error occurred")
        assert "⚠️ Error occurred" not in prompt
        assert "summary of the transcript" not in prompt

    def test_summary_truncation(self):
        """Long summaries are truncated at 2000 chars."""
        long_summary = "b" * 3000
        prompt = _build_system_prompt("transcript", long_summary)
        # Should contain at most 2000 'b' characters in summary section
        assert "b" * 2000 in prompt


class TestConvertHistoryToMessages:
    """Tests for _convert_history_to_messages function."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for LangChain message classes before function is called."""
        import services.langchain_chat as lc

        # Pre-populate the global variables to skip _ensure_langchain_imports
        lc._ChatOllama = MagicMock()
        lc._ChatPromptTemplate = MagicMock()
        lc._MessagesPlaceholder = MagicMock()
        lc._HumanMessage = lambda content: {"type": "human", "content": content}
        lc._AIMessage = lambda content: {"type": "ai", "content": content}
        lc._SystemMessage = lambda content: {"type": "system", "content": content}
        yield
        # Reset after tests
        lc._ChatOllama = None
        lc._ChatPromptTemplate = None
        lc._MessagesPlaceholder = None
        lc._HumanMessage = None
        lc._AIMessage = None
        lc._SystemMessage = None

    def test_empty_history(self):
        """Empty history returns empty list."""
        from services.langchain_chat import _convert_history_to_messages

        result = _convert_history_to_messages([])
        assert result == []

    def test_dict_format_history(self):
        """Gradio dict format history is converted."""
        from services.langchain_chat import _convert_history_to_messages

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _convert_history_to_messages(history)

        assert len(result) == 2
        assert result[0]["type"] == "human"
        assert result[0]["content"] == "Hello"
        assert result[1]["type"] == "ai"
        assert result[1]["content"] == "Hi there!"

    def test_tuple_format_history(self):
        """Legacy tuple format history is converted."""
        from services.langchain_chat import _convert_history_to_messages

        history = [("Hello", "Hi there!"), ("How are you?", "I'm good!")]
        result = _convert_history_to_messages(history)

        assert len(result) == 4
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there!"
        assert result[2]["content"] == "How are you?"
        assert result[3]["content"] == "I'm good!"

    def test_complex_content_format(self):
        """Gradio complex content (list with text/images) is handled."""
        from services.langchain_chat import _convert_history_to_messages

        history = [
            {"role": "user", "content": ["Hello", "World"]},
            {"role": "user", "content": [{"text": "From dict"}]},
        ]
        result = _convert_history_to_messages(history)

        assert len(result) == 2
        assert result[0]["content"] == "Hello World"
        assert result[1]["content"] == "From dict"

    def test_skips_empty_messages(self):
        """Empty content messages are skipped."""
        from services.langchain_chat import _convert_history_to_messages

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "World"},
        ]
        result = _convert_history_to_messages(history)

        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "World"


class TestChatStreamingErrorHandling:
    """Tests for chat_streaming error handling."""

    def test_import_error_yields_message(self):
        """Import error yields error message."""
        import services.langchain_chat as lc
        from services.langchain_chat import chat_streaming

        # Reset globals to force reimport attempt
        lc._ChatOllama = None
        lc._ChatPromptTemplate = None
        lc._HumanMessage = None
        lc._AIMessage = None
        lc._SystemMessage = None

        # Use patch to properly mock the module function
        with patch(
            "services.langchain_chat._ensure_langchain_imports",
            side_effect=ImportError("No module named 'langchain_ollama'"),
        ):
            results = list(chat_streaming("test", [], "transcript"))
            assert len(results) == 1
            assert "Error" in results[0]
            assert "LangChain" in results[0]


class TestGenerateTitleErrorHandling:
    """Tests for generate_title error handling."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for LangChain components."""
        import services.langchain_chat as lc

        # Save original values
        original_chat = lc._ChatOllama
        original_human = lc._HumanMessage

        yield

        # Reset after tests
        lc._ChatOllama = original_chat
        lc._HumanMessage = original_human

    def test_error_returns_default_title(self):
        """Error during title generation returns default."""
        import services.langchain_chat as lc
        from services.langchain_chat import generate_title

        # Pre-populate globals to skip import
        lc._ChatOllama = MagicMock(side_effect=Exception("Connection error"))
        lc._ChatPromptTemplate = MagicMock()
        lc._MessagesPlaceholder = MagicMock()
        lc._HumanMessage = lambda content: {"content": content}
        lc._AIMessage = MagicMock()
        lc._SystemMessage = MagicMock()

        result = generate_title("What is this about?")
        assert result == "### 💬 Chat Session"

    def test_long_title_is_truncated(self):
        """Titles longer than 50 chars are truncated."""
        import services.langchain_chat as lc
        from services.langchain_chat import generate_title

        # Mock response with long title
        mock_response = MagicMock()
        mock_response.content = "A" * 60

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        # Pre-populate globals
        lc._ChatOllama = MagicMock(return_value=mock_llm)
        lc._ChatPromptTemplate = MagicMock()
        lc._MessagesPlaceholder = MagicMock()
        lc._HumanMessage = lambda content: {"content": content}
        lc._AIMessage = MagicMock()
        lc._SystemMessage = MagicMock()

        result = generate_title("test message")
        # Title should be truncated: "### 💬 " + 47 chars + "..."
        assert len(result) <= len("### 💬 ") + 50

    def test_quotes_removed_from_title(self):
        """Quotes are stripped from generated title."""
        import services.langchain_chat as lc
        from services.langchain_chat import generate_title

        mock_response = MagicMock()
        mock_response.content = '"Test Title"'

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        # Pre-populate globals
        lc._ChatOllama = MagicMock(return_value=mock_llm)
        lc._ChatPromptTemplate = MagicMock()
        lc._MessagesPlaceholder = MagicMock()
        lc._HumanMessage = lambda content: {"content": content}
        lc._AIMessage = MagicMock()
        lc._SystemMessage = MagicMock()

        result = generate_title("test message")
        assert '"' not in result
        assert result == "### 💬 Test Title"
