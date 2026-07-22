# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import contextlib
import pytest

from azure.ai.ml.entities import CommandJob
from azure.ai.ml.constants._common import LOCAL_COMPUTE_TARGET
from azure.ai.ml.exceptions import ValidationException


def does_not_raise():
    return contextlib.nullcontext()


@pytest.mark.unittest
@pytest.mark.defect_regression
class TestDefectIcmTEST12345:
    """Tests for defect reported in IcM #TEST-12345.
    
    Root cause: CommandJob allows compute=None to pass local validation; _to_rest_object() in older versions
    invoked self.compute.lower() without a None check, causing AttributeError. Validation for compute is missing.
    """

    @pytest.mark.parametrize(
        "compute, expect",
        [
            (None, pytest.raises(ValidationException)),
            ("", pytest.raises(ValidationException)),
            (LOCAL_COMPUTE_TARGET, does_not_raise()),  # local runs may omit environment
            ("cpu-cluster", does_not_raise()),
        ],
        ids=["compute_none", "compute_empty", "local_compute_allows", "valid_compute"],
    )
    def test_command_job_compute_validation(self, compute, expect):
        """Regression test: ensure CommandJob with missing compute raises a clear ValidationException,
        and valid compute values succeed.
        
        - Before fix: compute=None crashes with AttributeError ('NoneType' has no attribute 'lower').
        - After fix: compute=None or '' raises ValidationException with a helpful message.
        """
        # Provide command and environment for non-local run cases; for local compute, environment can be None
        env = None if compute == LOCAL_COMPUTE_TARGET else "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1"
        job = CommandJob(command="echo hello", environment=env, compute=compute)
        with expect:
            # Entity-level conversion should trigger _validate() and surface a clear error for missing compute
            job._to_rest_object()
