# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
from contextlib import nullcontext as does_not_raise

from azure.ai.ml import command
from azure.ai.ml.exceptions import ValidationException


@pytest.mark.unittest
@pytest.mark.defect_regression
class TestDefectIcmTEST_12345:
    """Tests for defect reported in IcM #TEST-12345.
    
    Root cause: CommandJob serialization called .lower() on compute without validating None,
    and _validate() did not enforce compute for non-local runs, leading to AttributeError instead
    of a clear ValidationException.
    """

    @pytest.mark.parametrize(
        "compute_value, expected",
        [
            (None, pytest.raises(ValidationException)),
            ("", pytest.raises(ValidationException)),
            ("local", does_not_raise()),
            ("cpu-cluster", does_not_raise()),
        ],
        ids=["compute_is_none", "compute_is_empty", "compute_local", "compute_valid_cluster"],
    )
    def test_command_job_requires_compute_or_local(self, compute_value, expected):
        """Regression test: ensure CommandJob with missing compute fails validation with a clear error,
        and valid compute/local values succeed without raising.
        
        This would have caught the AttributeError('NoneType' has no attribute 'lower') thrown during
        _to_rest_object() when compute was None.
        """
        job = command(
            command="echo hello",
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
        )
        job.compute = compute_value

        with expected:
            # _to_rest_object triggers _validate() and serialization
            job._to_rest_object()

    def test_error_message_is_clear_when_compute_is_none(self):
        """Ensure the SDK raises a friendly ValidationException that mentions compute is required."""
        job = command(
            command="echo hello",
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
        )
        job.compute = None

        with pytest.raises(ValidationException) as ex:
            job._to_rest_object()

        # Message should clearly indicate compute is required (non-local)
        assert "compute" in str(ex.value).lower()
        assert "required" in str(ex.value).lower()
