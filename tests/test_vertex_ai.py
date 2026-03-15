"""Tests for esg_kg.vertex_ai (internal helpers, no live API calls)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from esg_kg.vertex_ai import _build_prompt, _strip_markdown_fences, call_vertex_model


# ---------------------------------------------------------------------------
# _strip_markdown_fences
# ---------------------------------------------------------------------------


class TestStripMarkdownFences:
    def test_plain_json_unchanged(self):
        raw = '{"entities": [], "relations": []}'
        assert _strip_markdown_fences(raw) == raw

    def test_json_code_fence_stripped(self):
        raw = '```json\n{"entities": [], "relations": []}\n```'
        result = _strip_markdown_fences(raw)
        assert result == '{"entities": [], "relations": []}'

    def test_generic_code_fence_stripped(self):
        raw = '```\n{"entities": [], "relations": []}\n```'
        result = _strip_markdown_fences(raw)
        assert result == '{"entities": [], "relations": []}'

    def test_fence_uppercase_json_tag(self):
        raw = '```JSON\n{"entities": [], "relations": []}\n```'
        result = _strip_markdown_fences(raw)
        assert result == '{"entities": [], "relations": []}'

    def test_fence_with_leading_trailing_spaces(self):
        raw = '  ```json\n{"a": 1}\n```  '
        result = _strip_markdown_fences(raw)
        assert result == '{"a": 1}'

    def test_result_is_valid_json_after_strip(self):
        raw = '```json\n{"entities": [], "relations": []}\n```'
        result = _strip_markdown_fences(raw)
        parsed = json.loads(result)
        assert parsed == {"entities": [], "relations": []}


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_text(self):
        text = "Acme Corp reduced emissions by 30%."
        prompt = _build_prompt(text)
        assert text in prompt

    def test_prompt_contains_system_instructions(self):
        prompt = _build_prompt("some text")
        # Key phrases from the system prompt
        assert "ESG" in prompt
        assert "entities" in prompt.lower()
        assert "relations" in prompt.lower()

    def test_prompt_contains_text_section_header(self):
        prompt = _build_prompt("some text")
        assert "TEXT TO ANALYSE" in prompt


# ---------------------------------------------------------------------------
# call_vertex_model
# ---------------------------------------------------------------------------


_VALID_JSON = json.dumps({"entities": [], "relations": []})


class TestCallVertexModel:
    def _mock_model_response(self, text: str) -> MagicMock:
        response = MagicMock()
        response.text = text
        return response

    def test_returns_valid_json_string(self):
        mock_response = self._mock_model_response(_VALID_JSON)
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response

        with (
            patch("esg_kg.vertex_ai.vertexai.init"),
            patch(
                "esg_kg.vertex_ai.GenerativeModel",
                return_value=mock_model_instance,
            ),
        ):
            result = call_vertex_model("test text", project="proj", location="us")

        assert json.loads(result) == {"entities": [], "relations": []}

    def test_strips_markdown_fences_from_response(self):
        fenced = f"```json\n{_VALID_JSON}\n```"
        mock_response = self._mock_model_response(fenced)
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response

        with (
            patch("esg_kg.vertex_ai.vertexai.init"),
            patch(
                "esg_kg.vertex_ai.GenerativeModel",
                return_value=mock_model_instance,
            ),
        ):
            result = call_vertex_model("test text", project="proj", location="us")

        # Result should be parseable JSON (no fences)
        parsed = json.loads(result)
        assert parsed == {"entities": [], "relations": []}

    def test_raises_value_error_on_non_json_response(self):
        mock_response = self._mock_model_response("This is not JSON at all.")
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response

        with (
            patch("esg_kg.vertex_ai.vertexai.init"),
            patch(
                "esg_kg.vertex_ai.GenerativeModel",
                return_value=mock_model_instance,
            ),
        ):
            with pytest.raises(ValueError, match="non-JSON"):
                call_vertex_model("test text", project="proj", location="us")

    def test_vertexai_init_called_with_project_and_location(self):
        mock_response = self._mock_model_response(_VALID_JSON)
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response

        with (
            patch("esg_kg.vertex_ai.vertexai.init") as mock_init,
            patch(
                "esg_kg.vertex_ai.GenerativeModel",
                return_value=mock_model_instance,
            ),
        ):
            call_vertex_model(
                "text", project="my-project", location="europe-west1"
            )
            mock_init.assert_called_once_with(
                project="my-project", location="europe-west1"
            )

    def test_custom_model_name_used(self):
        mock_response = self._mock_model_response(_VALID_JSON)
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response

        with (
            patch("esg_kg.vertex_ai.vertexai.init"),
            patch("esg_kg.vertex_ai.GenerativeModel") as mock_gen_model,
        ):
            mock_gen_model.return_value = mock_model_instance
            call_vertex_model(
                "text",
                project="proj",
                location="us",
                model_name="gemini-2.0-flash",
            )
            args, _ = mock_gen_model.call_args
            assert args[0] == "gemini-2.0-flash"
