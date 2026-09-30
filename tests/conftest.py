"""Shared pytest setup: isolate persistent data before application import."""

import os
import tempfile

import pytest

_TEMP_DATA = tempfile.mkdtemp(prefix="inspection-test-")
os.environ["INSPECTION_DATA_DIR"] = _TEMP_DATA

from inspection import synth  # noqa: E402


@pytest.fixture
def example_png():
    def _make(kind, variant=0):
        return (
            "%s_%d.png" % (kind, variant),
            synth.example_png(kind, variant),
            "image/png",
        )

    return _make
