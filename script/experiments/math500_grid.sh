#!/usr/bin/env bash
# math500 main grid; see run_grid.sh for the environment overrides.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_grid.sh" math500 "$@"
