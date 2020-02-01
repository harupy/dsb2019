import os
import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb


@pytest.fixture(autouse=True)
def inject_items(doctest_namespace):
    doctest_namespace['os'] = os
    doctest_namespace['np'] = np
    doctest_namespace['pd'] = pd
    doctest_namespace['lgb'] = lgb
