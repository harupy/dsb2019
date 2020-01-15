import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb


@pytest.fixture(autouse=True)
def add_packages(doctest_namespace):
    doctest_namespace['np'] = np
    doctest_namespace['pd'] = pd
    doctest_namespace['lgb'] = lgb