# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential

from azure.mgmt.billing import BillingManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-billing
# USAGE
    python billing_property_patch_subscription_service_usage_address.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = BillingManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id="11111111-1111-1111-1111-111111111111",
    )

    response = client.billing_property.update(
        parameters={
            "properties": {
                "subscriptionServiceUsageAddress": {
                    "addressLine1": "Address line 1",
                    "addressLine2": "Address line 2",
                    "city": "City",
                    "country": "US",
                    "firstName": "Jenny",
                    "lastName": "Doe",
                    "middleName": "Ann",
                    "postalCode": "12345",
                    "region": "State",
                }
            }
        },
    )
    print(response)


# x-ms-original-file: specification/billing/resource-manager/Microsoft.Billing/stable/2024-04-01/examples/billingPropertyPatchSubscriptionServiceUsageAddress.json
if __name__ == "__main__":
    main()
