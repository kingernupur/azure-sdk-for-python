# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azure.ai.ml._schema.job import BaseJobSchema
from azure.ai.ml._schema.job.input_output_fields_provider import OutputsField
from azure.ai.ml._utils._experimental import experimental
from azure.ai.ml._schema.core.fields import (
    NestedField,
)
from ..queue_settings import QueueSettingsSchema
from ..job_resources import JobResourcesSchema

# This is meant to match the yaml definition NOT the models defined in _restclient


@experimental
class FineTuningJobSchema(BaseJobSchema):
    outputs = OutputsField()
    queue_settings = NestedField(QueueSettingsSchema)
    resources = NestedField(JobResourcesSchema)
