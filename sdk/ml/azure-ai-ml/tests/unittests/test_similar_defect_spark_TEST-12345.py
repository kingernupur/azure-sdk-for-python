# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import contextlib
import pytest

from azure.ai.ml.entities import SparkJob
from azure.ai.ml.constants._common import LOCAL_COMPUTE_TARGET
from azure.ai.ml.exceptions import ValidationException


def does_not_raise():
    return contextlib.nullcontext()


@pytest.mark.unittest
@pytest.mark.defect_regression
class TestSimilarDefectSparkTEST12345:
    """Regression tests for SparkJob compute validation similar to IcM #TEST-12345.
    
    Root cause pattern: Missing compute validation for non-local runs can allow None to reach serialization
    and cause AttributeError if lowercasing or string ops are applied.
    """

    @pytest.mark.parametrize(
        "compute, expect",
        [
            (None, pytest.raises(ValidationException)),
            ("", pytest.raises(ValidationException)),
            (LOCAL_COMPUTE_TARGET, does_not_raise()),
            ("cpu-cluster", does_not_raise()),
        ],
        ids=["compute_none", "compute_empty", "local_compute_allows", "valid_compute"],
    )
    def test_spark_job_compute_validation(self, compute, expect):
        """Ensure SparkJob enforces non-empty compute for non-local runs and accepts local compute."""
        # SparkJob typically also requires an environment/code at submission time, but for validation
        # we focus on compute semantics consistent with CommandJob expectations.
        job = SparkJob(compute=compute)
        with expect:
            job._to_rest_object()
