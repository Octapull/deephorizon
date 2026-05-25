"""Compatibility shims for known third-party noise on Python 3.13.

Import and call `silence_ehtim_warnings()` BEFORE importing `ehtim`.

    from services.ml._compat import silence_ehtim_warnings
    silence_ehtim_warnings()
    import ehtim  # quiet from here

Or, equivalently for ad-hoc scripts:

    import services.ml._compat as _; _.silence_ehtim_warnings()
"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version


def silence_ehtim_warnings() -> None:
    """Suppress runtime ehtim warnings on Python 3.13.

    Targets:
    1. UserWarning about `pkg_resources` deprecation — fires on every
       `import ehtim`. This filter catches it. (`setuptools<82` in the
       `data` extras keeps the underlying API present.)
    2. SyntaxWarning from `\\m` / `\\c` escape sequences inside ehtim
       (will become SyntaxError on 3.14; the `requires-python<3.14` cap
       in pyproject.toml buys us time.) **Caveat:** SyntaxWarning fires
       at module *compilation*, before user code runs. The first import
       in a fresh venv will still print them once; subsequent imports
       are silent because Python caches the compiled `.pyc`. To kill
       even the first import noise, set
       `PYTHONWARNINGS=ignore::SyntaxWarning:ehtim` in your shell rc.
    """
    warnings.filterwarnings(
        "ignore",
        category=SyntaxWarning,
        module=r"ehtim(\..*)?",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated.*",
        category=UserWarning,
    )


def ehtim_version() -> str:
    """Return the installed ehtim version.

    ehtim does not expose `__version__`, so we read it from package metadata.
    Returns "unknown" if ehtim is not installed.
    """
    try:
        return version("ehtim")
    except PackageNotFoundError:
        return "unknown"
