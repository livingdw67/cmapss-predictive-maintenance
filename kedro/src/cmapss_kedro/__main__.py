"""Run the project as `python -m cmapss_kedro` (same as `kedro run`)."""

import sys
from pathlib import Path
from typing import Any

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs) -> Any:
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    kwargs["standalone_mode"] = not hasattr(sys, "ps1")
    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    main()
