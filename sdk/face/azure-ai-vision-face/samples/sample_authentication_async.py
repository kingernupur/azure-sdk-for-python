# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
FILE: sample_authentication_async.py

DESCRIPTION:
    This sample demonstrates authenticating a client via:
        * api key
        * Microsoft Entra ID

USAGE:
    python sample_authentication_async.py

    Set the environment variables with your own values before running this sample:
    1) AZURE_FACE_API_ENDPOINT - the endpoint to your Face resource.
    The environment variable below is used for api key authentication.
    2) AZURE_FACE_API_ACCOUNT_KEY - your Face API key.
    The following environment variables are required for using azure-identity's DefaultAzureCredential.
    For more information, refer to https://aka.ms/azsdk/python/identity/docs#azure.identity.DefaultAzureCredential
    3) AZURE_TENANT_ID - the tenant ID in Microsoft Entra ID
    4) AZURE_CLIENT_ID - the application (client) ID registered in the AAD tenant
    5) AZURE_CLIENT_SECRET - the client secret for the registered application
"""
import asyncio
import os

from dotenv import find_dotenv, load_dotenv

from shared.constants import (
    CONFIGURATION_NAME_FACE_API_ACCOUNT_KEY,
    CONFIGURATION_NAME_FACE_API_ENDPOINT,
    DEFAULT_FACE_API_ACCOUNT_KEY,
    DEFAULT_FACE_API_ENDPOINT,
    TestImages,
)
from shared import helpers
from shared.helpers import beautify_json, get_logger


class FaceAuthentication:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.endpoint = os.getenv(CONFIGURATION_NAME_FACE_API_ENDPOINT, DEFAULT_FACE_API_ENDPOINT)
        self.key = os.getenv(CONFIGURATION_NAME_FACE_API_ACCOUNT_KEY, DEFAULT_FACE_API_ACCOUNT_KEY)
        self.logger = get_logger("sample_authentication_async")

    async def authentication_by_api_key(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.vision.face.aio import FaceClient
        from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel

        self.logger.info("Instantiate a FaceClient using an api key")
        async with FaceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key)) as face_client:
            sample_file_path = helpers.get_image_path(TestImages.DEFAULT_IMAGE_FILE)
            result = await face_client.detect(
                helpers.read_file_content(sample_file_path),
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=False,
            )

            self.logger.info(f"Detect faces from the file: {sample_file_path}")
            for idx, face in enumerate(result):
                self.logger.info(f"----- Detection result: #{idx+1} -----")
                self.logger.info(f"Face: {beautify_json(face.as_dict())}")

    async def authentication_by_aad_credential(self):
        from azure.identity.aio import DefaultAzureCredential
        from azure.ai.vision.face.aio import FaceClient
        from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel

        self.logger.info("Instantiate a FaceClient using a TokenCredential")
        async with DefaultAzureCredential() as credential, FaceClient(
            endpoint=self.endpoint, credential=credential
        ) as face_client:
            sample_file_path = helpers.get_image_path(TestImages.DEFAULT_IMAGE_FILE)
            result = await face_client.detect(
                helpers.read_file_content(sample_file_path),
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=False,
            )

            self.logger.info(f"Detect faces from the file: {sample_file_path}")
            for idx, face in enumerate(result):
                self.logger.info(f"----- Detection result: #{idx+1} -----")
                self.logger.info(f"Face: {beautify_json(face.as_dict())}")


async def main():
    sample = FaceAuthentication()
    await sample.authentication_by_api_key()
    await sample.authentication_by_aad_credential()


if __name__ == "__main__":
    asyncio.run(main())
