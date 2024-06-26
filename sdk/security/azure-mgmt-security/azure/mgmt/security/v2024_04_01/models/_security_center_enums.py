# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from enum import Enum
from azure.core import CaseInsensitiveEnumMeta


class ActionableRemediationState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """ActionableRemediation Setting.
    None - the setting was never set.
    Enabled - ActionableRemediation is enabled.
    Disabled - ActionableRemediation is disabled.
    """

    NONE = "None"
    DISABLED = "Disabled"
    ENABLED = "Enabled"


class AnnotateDefaultBranchState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Configuration of PR Annotations on default branch.

    Enabled - PR Annotations are enabled on the resource's default branch.
    Disabled - PR Annotations are disabled on the resource's default branch.
    """

    DISABLED = "Disabled"
    ENABLED = "Enabled"


class AutoDiscovery(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """AutoDiscovery states."""

    DISABLED = "Disabled"
    ENABLED = "Enabled"
    NOT_APPLICABLE = "NotApplicable"


class CreatedByType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of identity that created the resource."""

    USER = "User"
    APPLICATION = "Application"
    MANAGED_IDENTITY = "ManagedIdentity"
    KEY = "Key"


class DesiredOnboardingState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Onboarding states."""

    DISABLED = "Disabled"
    ENABLED = "Enabled"


class DevOpsProvisioningState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The provisioning state of the resource.

    Pending - Provisioning pending.
    Failed - Provisioning failed.
    Succeeded - Successful provisioning.
    Canceled - Provisioning canceled.
    PendingDeletion - Deletion pending.
    DeletionSuccess - Deletion successful.
    DeletionFailure - Deletion failure.
    """

    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELED = "Canceled"
    PENDING = "Pending"
    PENDING_DELETION = "PendingDeletion"
    DELETION_SUCCESS = "DeletionSuccess"
    DELETION_FAILURE = "DeletionFailure"


class InheritFromParentState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Update Settings.

    Enabled - Resource should inherit configurations from parent.
    Disabled - Resource should not inherit configurations from parent.
    """

    DISABLED = "Disabled"
    ENABLED = "Enabled"


class OnboardingState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Details about resource onboarding status across all connectors.

    OnboardedByOtherConnector - this resource has already been onboarded to another connector. This
    is only applicable to top-level resources.
    Onboarded - this resource has already been onboarded by the specified connector.
    NotOnboarded - this resource has not been onboarded to any connector.
    NotApplicable - the onboarding state is not applicable to the current endpoint.
    """

    NOT_APPLICABLE = "NotApplicable"
    ONBOARDED_BY_OTHER_CONNECTOR = "OnboardedByOtherConnector"
    ONBOARDED = "Onboarded"
    NOT_ONBOARDED = "NotOnboarded"


class RuleCategory(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Rule categories.
    Code - code scanning results.
    Artifact scanning results.
    Dependencies scanning results.
    IaC results.
    Secrets scanning results.
    Container scanning results.
    """

    CODE = "Code"
    ARTIFACTS = "Artifacts"
    DEPENDENCIES = "Dependencies"
    SECRETS = "Secrets"
    IA_C = "IaC"
    CONTAINERS = "Containers"
