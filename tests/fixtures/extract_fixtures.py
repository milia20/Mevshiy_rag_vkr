import os
import shutil
import tempfile

from pytest import fixture


@fixture
def temp_dir():
    """Создает временную директорию для тестов"""
    tmpdir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(tmpdir)
    yield tmpdir
    os.chdir(original_cwd)
    shutil.rmtree(tmpdir, ignore_errors=True)
