import importlib

import qgemm

SUBMODULES = [
    "qgemm.formats",
    "qgemm.blocks",
    "qgemm.rounding",
    "qgemm.transforms",
    "qgemm.gemm",
    "qgemm.metrics",
    "qgemm.bounds",
    "qgemm.distributions",
    "qgemm.stats",
]


def test_import_qgemm():
    assert qgemm.__version__


def test_import_submodules():
    for name in SUBMODULES:
        importlib.import_module(name)
