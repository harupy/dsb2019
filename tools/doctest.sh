#!/usr/bin/env bash

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest --verbose --doctest-modules src