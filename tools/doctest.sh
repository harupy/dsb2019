#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest --doctest-modules src