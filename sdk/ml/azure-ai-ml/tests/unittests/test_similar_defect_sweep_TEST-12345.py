# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
from contextlib import nullcontext as does_not_raise

from azure.ai.ml import command
from azure.ai.ml.exceptions import ValidationException
from azure.ai.ml.sweep import SweepJob, Objective


@pytest.mark.unittest
@pytest.mark.defect_regression
class TestSimilarDefectSweepTEST_12345:
    """Similar defect test: SweepJob.compute=None should raise ValidationException instead of AttributeError.

    Root cause pattern: unsafe .lower() normalization on compute and missing validation for non-local runs.
    """

    @pytest.mark.parametrize(
        "compute_value, expected",
        [
            (None, pytest.raises(ValidationException)),
            ("", pytest.raises(ValidationException)),
            ("cpu-cluster", does_not_raise()),
        ],
        ids=["compute_none", "compute_empty", "compute_valid"],
    )
    def test_sweep_job_requires_compute(self, compute_value, expected):
        # Minimal trial command job used by sweep
        trial_cmd = command(
            command="echo trial",
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
        )

        sweep = SweepJob(
            trial=trial_cmd,
            sampling_algorithm="random",
            objective=Objective(goal="maximize", primary_metric="accuracy"),
            compute=compute_value,
        )

        with expected:
            sweep._to_rest_object()

    def test_sweep_job_message_is_clear_when_compute_missing(self):
        trial_cmd = command(
            command="echo trial",
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
        )
        sweep = SweepJob(
            trial=trial_cmd,
            sampling_algorithm="random",
            objective=Objective(goal="maximize", primary_metric="accuracy"),
            compute=None,
        )

        with pytest.raises(ValidationException) as ex:
            sweep._to_rest_object()

        assert "compute" in str(ex.value).lower()
        assert "required" in str(ex.value).lower()
