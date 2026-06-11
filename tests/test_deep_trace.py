"""Tests for deep_trace — bounded agentic loop."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(text: str, stop_reason: str = "end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = [block]
    return resp


def _tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tu_1"):
    tu = MagicMock()
    tu.type = "tool_use"
    tu.id = tool_id
    tu.name = tool_name
    tu.input = tool_input
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [tu]
    return resp


_TRACE_KWARGS = dict(
    company_name="Acme Holdings LP",
    entities=[{
        "canonical_name": "Acme Holdings LP",
        "entity_type": "company",
        "jurisdiction": "Cayman Islands",
    }],
    links=[],
    risk_score=35,
    risk_level="MEDIUM",
    findings=[{
        "rule": "secrecy_jurisdiction",
        "detail": "Acme Holdings LP — Cayman Islands",
        "weight": 30,
    }],
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunDeepTrace:

    def test_end_turn_immediately_returns_synthesis(self):
        db = MagicMock()
        with patch("deep_trace.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = _text_response(
                "No further concerns identified.", "end_turn"
            )
            from deep_trace import run_deep_trace
            result = run_deep_trace(**_TRACE_KWARGS, db=db)

        assert result["tool_calls_used"] == 0
        assert "No further concerns" in result["synthesis"]
        assert result["max_calls"] == 5  # DEEP_TRACE_MAX_TOOL_CALLS

    def test_tool_call_counted_and_logged(self):
        db = MagicMock()
        with patch("deep_trace.anthropic.Anthropic") as mock_cls:
            with patch("deep_trace._execute_tool") as mock_exec:
                mock_exec.return_value = {"status": "not_found", "entity_name": "SubCo"}
                mock_cls.return_value.messages.create.side_effect = [
                    _tool_use_response("investigate_entity", {"entity_name": "SubCo"}),
                    _text_response("SubCo not found in EDGAR.", "end_turn"),
                ]
                from deep_trace import run_deep_trace
                result = run_deep_trace(**_TRACE_KWARGS, db=db)

        assert result["tool_calls_used"] == 1
        assert len(result["tool_call_log"]) == 1
        assert result["tool_call_log"][0]["tool"] == "investigate_entity"

    def test_budget_enforced(self):
        """tool_calls_used must never exceed DEEP_TRACE_MAX_TOOL_CALLS."""
        db = MagicMock()
        with patch("deep_trace.anthropic.Anthropic") as mock_cls:
            with patch("deep_trace._execute_tool") as mock_exec:
                mock_exec.return_value = {"status": "not_found"}
                # Simulate Claude always wanting to use a tool; ensure budget caps it
                tool_resp = _tool_use_response("investigate_entity", {"entity_name": "X"})
                synth_resp = _text_response("Final synthesis.", "end_turn")
                # Enough responses for worst case
                mock_cls.return_value.messages.create.side_effect = (
                    [tool_resp] * 10 + [synth_resp]
                )
                from deep_trace import run_deep_trace
                from config import DEEP_TRACE_MAX_TOOL_CALLS
                result = run_deep_trace(**_TRACE_KWARGS, db=db)

        assert result["tool_calls_used"] <= DEEP_TRACE_MAX_TOOL_CALLS

    def test_api_error_captured_in_result(self):
        """An API error must not propagate — it is returned in the result dict."""
        db = MagicMock()
        import anthropic as ant
        with patch("deep_trace.anthropic.Anthropic") as mock_cls:
            # Raise a generic exception to simulate API failure
            mock_cls.return_value.messages.create.side_effect = ant.APIStatusError(
                "rate limit",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
            from deep_trace import run_deep_trace
            result = run_deep_trace(**_TRACE_KWARGS, db=db)

        assert "error" in result
        assert result["tool_calls_used"] == 0

    def test_synthesis_from_text_block_captured(self):
        """Text blocks after tool calls become the synthesis."""
        db = MagicMock()
        with patch("deep_trace.anthropic.Anthropic") as mock_cls:
            with patch("deep_trace._execute_tool") as mock_exec:
                mock_exec.return_value = {"status": "not_found"}
                mock_cls.return_value.messages.create.side_effect = [
                    _tool_use_response("search_cached_filings", {"query": "Cayman"}),
                    _text_response("Nothing additional found in filing cache.", "end_turn"),
                ]
                from deep_trace import run_deep_trace
                result = run_deep_trace(**_TRACE_KWARGS, db=db)

        assert "Nothing additional" in result["synthesis"]


class TestExecuteTools:

    def test_unknown_tool_returns_error(self):
        from deep_trace import _execute_tool
        db = MagicMock()
        result = _execute_tool("nonexistent_tool", {}, db)
        assert "error" in result

    def test_investigate_entity_empty_name_returns_error(self):
        from deep_trace import _investigate_entity
        db = MagicMock()
        result = _investigate_entity("", db)
        assert "error" in result

    def test_search_cached_filings_empty_query_returns_error(self):
        from deep_trace import _search_cached_filings
        db = MagicMock()
        result = _search_cached_filings("", db)
        assert "error" in result

    def test_search_cached_filings_store_error_captured(self):
        from deep_trace import _search_cached_filings
        db = MagicMock()
        # get_store is imported inside the function, so patch the source module
        with patch("vector_store.get_store") as mock_store:
            mock_store.side_effect = Exception("index offline")
            result = _search_cached_filings("Cayman Islands fund", db)
        assert "error" in result
