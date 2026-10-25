import logging
import warnings

import pytest

from tinyserve.quiet import quiet_logs


def test_quiet_logs_ignores_hf_token_warning():
    with warnings.catch_warnings(record=True) as caught:
        quiet_logs()
        warnings.warn("\nError while fetching `HF_TOKEN` secret value from your vault", UserWarning)
        warnings.warn("something else", UserWarning)
    assert [str(w.message) for w in caught] == ["something else"]


def test_quiet_logs_sets_transformers_to_error():
    transformers = pytest.importorskip("transformers")
    quiet_logs()
    assert transformers.logging.get_verbosity() == logging.ERROR
