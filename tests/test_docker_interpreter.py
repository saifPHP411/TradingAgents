"""The Docker image must expose one interpreter to both the app and a shell.

A Debian login shell sources ``/etc/profile``, which *overwrites* ``PATH`` and
so discards the ``ENV PATH="/opt/venv/bin:$PATH"`` the Dockerfile sets. Any
``python -m ...`` run from such a shell then resolves to the base image's
interpreter, which has none of the project's dependencies, and dies with
``ModuleNotFoundError`` (the reported symptom was ``No module named 'dotenv'``).

These tests pin the facts that together prevent that regression, without needing
a Docker daemon: the dependency is declared, the package imports it at import
time, and the Dockerfile restores the venv path for login shells.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

VENV_BIN = "/opt/venv/bin"
PROFILE_SCRIPT = "/etc/profile.d/10-venv-path.sh"
EXPORT_LINE = f'export PATH="{VENV_BIN}:$PATH"'


@pytest.mark.unit
def test_python_dotenv_is_a_declared_runtime_dependency():
    """dotenv is imported by the package, so it must be a core dependency."""
    assert re.search(r'"python-dotenv[>=]', PYPROJECT), (
        "python-dotenv must be a core dependency: tradingagents/__init__.py "
        "imports it at import time, so a bare install cannot work without it"
    )


@pytest.mark.unit
def test_the_package_imports_dotenv_at_import_time():
    """This coupling is what turns a wrong interpreter into an import crash."""
    init = (REPO_ROOT / "tradingagents" / "__init__.py").read_text(encoding="utf-8")
    assert "from dotenv import" in init


@pytest.mark.unit
def test_the_dockerfile_builds_and_prepends_a_venv():
    assert "venv /opt/venv\n" in DOCKERFILE
    assert f'ENV PATH="{VENV_BIN}:$PATH"' in DOCKERFILE


@pytest.mark.unit
def test_the_dockerfile_restores_the_venv_path_for_login_shells():
    """A profile.d script must re-prepend the venv after /etc/profile resets PATH.

    /etc/profile runs first (overwriting PATH), then sources /etc/profile.d/*,
    so a script there is the only place a login shell can recover the venv.
    """
    assert PROFILE_SCRIPT in DOCKERFILE, (
        "the Dockerfile must write a /etc/profile.d script exporting PATH"
    )
    assert EXPORT_LINE in DOCKERFILE
    # The export must prepend, so the base image's PATH is still appended.
    assert EXPORT_LINE.endswith(f'{VENV_BIN}:$PATH"')


@pytest.mark.unit
def test_the_profile_script_is_written_before_the_user_switches_to_non_root():
    """Only root can write /etc/profile.d; the switch to appuser must come after."""
    assert DOCKERFILE.index(PROFILE_SCRIPT) < DOCKERFILE.index("USER appuser")


@pytest.mark.unit
def test_the_profile_script_export_is_valid_shell():
    """The emitted line must actually prepend the venv when a shell sources it."""
    result = subprocess.run(
        ["sh", "-c", f'PATH="/usr/bin:/bin"; {EXPORT_LINE}; printf %s "$PATH"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith(f"{VENV_BIN}:"), result.stdout
