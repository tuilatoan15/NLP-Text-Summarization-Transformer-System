"""
tests/test_rag_agent.py — Unit tests for the Agentic RAG routing and query expansion.
"""
import unittest
from backend.services.rag.agent import classify_intent, expand_query

class TestRagAgent(unittest.TestCase):
    def test_classify_intent_general(self):
        # Without documents, hello message should be GENERAL
        intent = classify_intent("Chào bạn, bạn có thể giúp tôi không?", document_ids=[])
        self.assertEqual(intent, "GENERAL")

        # Basic greetings
        intent_greet = classify_intent("Hi assistant", document_ids=[])
        self.assertEqual(intent_greet, "GENERAL")

    def test_classify_intent_summarize(self):
        # With documents, summarize keywords should be SUMMARIZE
        intent = classify_intent("Hãy tóm tắt báo cáo này giúp tôi.", document_ids=["doc_1"])
        self.assertEqual(intent, "SUMMARIZE")

    def test_classify_intent_document_qa(self):
        # Queries with documents that are not greetings or summaries should be DOCUMENT_QA
        intent = classify_intent("Doanh thu quý 1 tăng trưởng bao nhiêu phần trăm?", document_ids=["doc_1"])
        self.assertEqual(intent, "DOCUMENT_QA")

    def test_expand_query_fallback(self):
        # Without LLM API active (using default env), query expansion returns an empty list
        expanded = expand_query("Báo cáo tài chính quý 1")
        self.assertIsInstance(expanded, list)
