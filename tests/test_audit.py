"""
tests/test_audit.py — Unit tests verifying the audit mode runner functions.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.audit_mode import (
    run_extraction_and_cleaning_audit,
    run_sentence_segmentation_audit,
    run_tokenization_audit,
    run_summarization_models_audit,
    run_evaluation_metrics_audit,
)


class TestAuditRunnerFunctions(unittest.TestCase):
    def test_run_extraction_and_cleaning_audit(self):
        passed = run_extraction_and_cleaning_audit()
        self.assertTrue(passed)

    def test_run_sentence_segmentation_audit(self):
        passed = run_sentence_segmentation_audit()
        self.assertTrue(passed)

    def test_run_tokenization_audit(self):
        passed = run_tokenization_audit()
        self.assertTrue(passed)

    def test_run_summarization_models_audit(self):
        passed = run_summarization_models_audit()
        self.assertTrue(passed)

    def test_run_evaluation_metrics_audit(self):
        passed = run_evaluation_metrics_audit()
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
