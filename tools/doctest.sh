#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest --verbose --doctest-modules src