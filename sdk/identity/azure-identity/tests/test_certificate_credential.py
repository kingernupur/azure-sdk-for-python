# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
import json
import os
from unittest.mock import Mock, patch

from azure.core.pipeline.policies import ContentDecodePolicy, SansIOHTTPPolicy
from azure.identity import CertificateCredential, TokenCachePersistenceOptions
from azure.identity._enums import RegionalAuthority
from azure.identity._constants import EnvironmentVariables
from azure.identity._credentials.certificate import load_pkcs12_certificate
from azure.identity._internal.user_agent import USER_AGENT
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from msal import TokenCache
import msal
import pytest
from urllib.parse import urlparse

from helpers import (
    build_aad_response,
    build_id_token,
    id_token_claims,
    get_discovery_response,
    urlsafeb64_decode,
    mock_response,
    new_msal_validating_transport,
    Request,
    GET_TOKEN_METHODS,
)

PEM_CERT_PATH = os.path.join(os.path.dirname(__file__), "certificate.pem")
PEM_CERT_WITH_PASSWORD_PATH = os.path.join(os.path.dirname(__file__), "certificate-with-password.pem")
PFX_CERT_PATH = os.path.join(os.path.dirname(__file__), "certificate.pfx")
PFX_CERT_WITH_PASSWORD_PATH = os.path.join(os.path.dirname(__file__), "certificate-with-password.pfx")
CERT_PASSWORD = "password"
ALL_CERTS = (
    (PEM_CERT_PATH, None),
    (PEM_CERT_WITH_PASSWORD_PATH, CERT_PASSWORD),  # credential should accept passwords as str or bytes
    (PEM_CERT_WITH_PASSWORD_PATH, CERT_PASSWORD.encode("utf-8")),
    (PFX_CERT_PATH, None),
    (PFX_CERT_WITH_PASSWORD_PATH, CERT_PASSWORD),
    (PFX_CERT_WITH_PASSWORD_PATH, CERT_PASSWORD.encode("utf-8")),
)

EC_CERT_PATH = os.path.join(os.path.dirname(__file__), "ec-certificate.pem")


def test_non_rsa_key():
    """The credential should raise ValueError when given a cert without an RSA private key"""
    with pytest.raises(ValueError, match=".*RS256.*"):
        CertificateCredential("tenant-id", "client-id", EC_CERT_PATH)
    with pytest.raises(ValueError, match=".*RS256.*"):
        CertificateCredential("tenant-id", "client-id", certificate_data=open(EC_CERT_PATH, "rb").read())


def test_tenant_id_validation():
    """The credential should raise ValueError when given an invalid tenant_id"""

    valid_ids = {"c878a2ab-8ef4-413b-83a0-199afb84d7fb", "contoso.onmicrosoft.com", "organizations", "common"}
    for tenant in valid_ids:
        CertificateCredential(tenant, "client-id", PEM_CERT_PATH)

    invalid_ids = {"", "my tenant", "my_tenant", "/", "\\", '"my-tenant"', "'my-tenant'"}
    for tenant in invalid_ids:
        with pytest.raises(ValueError):
            CertificateCredential(tenant, "client-id", PEM_CERT_PATH)


@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_no_scopes(get_token_method):
    """The credential should raise ValueError when get_token is called with no scopes"""

    credential = CertificateCredential("tenant-id", "client-id", PEM_CERT_PATH)
    with pytest.raises(ValueError):
        getattr(credential, get_token_method)()


@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_policies_configurable(get_token_method):
    policy = Mock(spec_set=SansIOHTTPPolicy, on_request=Mock())

    transport = new_msal_validating_transport(
        requests=[Request()], responses=[mock_response(json_payload=build_aad_response(access_token="**"))]
    )

    credential = CertificateCredential(
        "tenant-id", "client-id", PEM_CERT_PATH, policies=[ContentDecodePolicy(), policy], transport=transport
    )

    getattr(credential, get_token_method)("scope")

    assert policy.on_request.called


@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_user_agent(get_token_method):
    transport = new_msal_validating_transport(
        requests=[Request(required_headers={"User-Agent": USER_AGENT})],
        responses=[mock_response(json_payload=build_aad_response(access_token="**"))],
    )

    credential = CertificateCredential("tenant-id", "client-id", PEM_CERT_PATH, transport=transport)

    getattr(credential, get_token_method)("scope")


@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_tenant_id(get_token_method):
    transport = new_msal_validating_transport(
        requests=[Request(required_headers={"User-Agent": USER_AGENT})],
        responses=[mock_response(json_payload=build_aad_response(access_token="**"))],
    )

    credential = CertificateCredential(
        "tenant-id", "client-id", PEM_CERT_PATH, transport=transport, additionally_allowed_tenants=["*"]
    )

    kwargs = {"tenant_id": "tenant_id"}
    if get_token_method == "get_token_info":
        kwargs = {"options": kwargs}
    getattr(credential, get_token_method)("scope", **kwargs)


@pytest.mark.parametrize("authority", ("localhost", "https://localhost"))
@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_authority(authority, get_token_method):
    """the credential should accept an authority, with or without scheme, as an argument or environment variable"""

    tenant_id = "expected-tenant"
    parsed_authority = urlparse(authority)
    expected_netloc = parsed_authority.netloc or authority
    expected_authority = "https://{}/{}".format(expected_netloc, tenant_id)

    mock_ctor = Mock(
        return_value=Mock(acquire_token_silent_with_error=lambda *_, **__: {"access_token": "**", "expires_in": 42})
    )

    credential = CertificateCredential(tenant_id, "client-id", PEM_CERT_PATH, authority=authority)
    with patch("msal.ConfidentialClientApplication", mock_ctor):
        # must call get_token because the credential constructs the MSAL application lazily
        getattr(credential, get_token_method)("scope")

    assert mock_ctor.call_count == 1
    _, kwargs = mock_ctor.call_args
    assert kwargs["authority"] == expected_authority
    mock_ctor.reset_mock()

    # authority can be configured via environment variable
    with patch.dict("os.environ", {EnvironmentVariables.AZURE_AUTHORITY_HOST: authority}, clear=True):
        credential = CertificateCredential(tenant_id, "client-id", PEM_CERT_PATH, authority=authority)
    with patch("msal.ConfidentialClientApplication", mock_ctor):
        getattr(credential, get_token_method)("scope")

    assert mock_ctor.call_count == 1
    _, kwargs = mock_ctor.call_args
    assert kwargs["authority"] == expected_authority


@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_regional_authority(get_token_method):
    """the credential should configure MSAL with a regional authority specified via kwarg or environment variable"""

    mock_confidential_client = Mock(
        return_value=Mock(acquire_token_silent_with_error=lambda *_, **__: {"access_token": "**", "expires_in": 3600}),
    )

    for region in RegionalAuthority:
        mock_confidential_client.reset_mock()

        # region can be configured via environment variable
        with patch.dict("os.environ", {EnvironmentVariables.AZURE_REGIONAL_AUTHORITY_NAME: region.value}, clear=True):
            credential = CertificateCredential("tenant", "client-id", PEM_CERT_PATH)
        with patch("msal.ConfidentialClientApplication", mock_confidential_client):
            getattr(credential, get_token_method)("scope")

        assert mock_confidential_client.call_count == 1
        _, kwargs = mock_confidential_client.call_args
        if region == RegionalAuthority.AUTO_DISCOVER_REGION:
            assert kwargs["azure_region"] == msal.ConfidentialClientApplication.ATTEMPT_REGION_DISCOVERY
        else:
            assert kwargs["azure_region"] == region.value


def test_requires_certificate():
    """the credential should raise ValueError when not given a certificate"""

    with pytest.raises(ValueError):
        CertificateCredential("tenant", "client-id")
    with pytest.raises(ValueError):
        CertificateCredential("tenant", "client-id", certificate_path=None)
    with pytest.raises(ValueError):
        CertificateCredential("tenant", "client-id", certificate_path="")
    with pytest.raises(ValueError):
        CertificateCredential("tenant", "client-id", certificate_data=None)
    with pytest.raises(ValueError):
        CertificateCredential("tenant", "client-id", certificate_path="", certificate_data=None)


@pytest.mark.parametrize("cert_path,cert_password", ALL_CERTS)
@pytest.mark.parametrize("send_certificate_chain", (True, False))
@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_request_body(cert_path, cert_password, send_certificate_chain, get_token_method):
    access_token = "***"
    authority = "authority.com"
    client_id = "client-id"
    expected_scope = "scope"
    tenant_id = "tenant"

    def mock_send(request, **kwargs):
        if not request.body:
            return get_discovery_response()

        assert request.body["grant_type"] == "client_credentials"
        assert request.body["scope"] == expected_scope

        with open(cert_path, "rb") as cert_file:
            validate_jwt(request, client_id, cert_file.read(), cert_password, expect_x5c=send_certificate_chain)

        return mock_response(json_payload=build_aad_response(access_token=access_token))

    cred = CertificateCredential(
        tenant_id,
        client_id,
        cert_path,
        password=cert_password,
        transport=Mock(send=mock_send),
        authority=authority,
        send_certificate_chain=send_certificate_chain,
    )
    token = getattr(cred, get_token_method)(expected_scope)
    assert token.token == access_token

    # credential should also accept the certificate as bytes
    with open(cert_path, "rb") as f:
        cert_bytes = f.read()

    cred = CertificateCredential(
        tenant_id,
        client_id,
        certificate_data=cert_bytes,
        password=cert_password,
        transport=Mock(send=mock_send),
        authority=authority,
        send_certificate_chain=send_certificate_chain,
    )
    token = getattr(cred, get_token_method)(expected_scope)
    assert token.token == access_token


def validate_jwt(request, client_id, cert_bytes, cert_password, expect_x5c=False):
    """Validate the request meets Microsoft Entra ID's expectations for a client credential grant using a certificate, as documented
    at https://learn.microsoft.com/entra/identity-platform/certificate-credentials
    """

    try:
        cert = x509.load_pem_x509_certificate(cert_bytes, default_backend())
    except ValueError:
        if cert_password:
            if isinstance(cert_password, str):
                cert_password = cert_password.encode("utf-8")
        cert_bytes = load_pkcs12_certificate(cert_bytes, cert_password).pem_bytes
        cert = x509.load_pem_x509_certificate(cert_bytes, default_backend())

    # jwt is of the form 'header.payload.signature'; 'signature' is 'header.payload' signed with cert's private key
    jwt = request.body["client_assertion"]
    if isinstance(jwt, bytes):
        jwt = jwt.decode("utf-8")
    header, payload, signature = (urlsafeb64_decode(s) for s in jwt.split("."))
    signed_part = jwt[: jwt.rfind(".")]

    claims = json.loads(payload.decode("utf-8"))
    assert claims["aud"] == request.url
    assert claims["iss"] == claims["sub"] == client_id

    deserialized_header = json.loads(header.decode("utf-8"))
    assert deserialized_header["alg"] == "RS256"
    assert deserialized_header["typ"] == "JWT"
    if expect_x5c:
        # x5c should have all the certs in the file, in order, in PEM format minus headers and footers
        pem_lines = cert_bytes.decode("utf-8").splitlines()
        header = "-----BEGIN CERTIFICATE-----"
        assert len(deserialized_header["x5c"]) == pem_lines.count(header)

        # concatenate the PEM file's certs, removing headers and footers
        chain_start = pem_lines.index(header)
        pem_chain_content = "".join(line for line in pem_lines[chain_start:] if not line.startswith("-" * 5))
        assert "".join(deserialized_header["x5c"]) == pem_chain_content, "JWT's x5c claim contains unexpected content"
    else:
        assert "x5c" not in deserialized_header
    assert urlsafeb64_decode(deserialized_header["x5t"]) == cert.fingerprint(hashes.SHA1())  # nosec

    cert.public_key().verify(signature, signed_part.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())


def validate_jwt_ps256(request, client_id, cert_bytes, cert_password, expect_x5c=False):
    """Validate the request meets Microsoft Entra ID's expectations for a client credential grant using a certificate, as documented
    at https://learn.microsoft.com/entra/identity-platform/certificate-credentials
    """

    try:
        cert = x509.load_pem_x509_certificate(cert_bytes, default_backend())
    except ValueError:
        if cert_password:
            if isinstance(cert_password, str):
                cert_password = cert_password.encode("utf-8")
        cert_bytes = load_pkcs12_certificate(cert_bytes, cert_password).pem_bytes
        cert = x509.load_pem_x509_certificate(cert_bytes, default_backend())

    # jwt is of the form 'header.payload.signature'; 'signature' is 'header.payload' signed with cert's private key
    jwt = request.body["client_assertion"]
    if isinstance(jwt, bytes):
        jwt = jwt.decode("utf-8")
    header, payload, signature = (urlsafeb64_decode(s) for s in jwt.split("."))
    signed_part = jwt[: jwt.rfind(".")]

    claims = json.loads(payload.decode("utf-8"))
    assert claims["aud"] == request.url
    assert claims["iss"] == claims["sub"] == client_id

    deserialized_header = json.loads(header.decode("utf-8"))
    assert deserialized_header["alg"] == "PS256"
    assert deserialized_header["typ"] == "JWT"
    if expect_x5c:
        # x5c should have all the certs in the file, in order, in PEM format minus headers and footers
        pem_lines = cert_bytes.decode("utf-8").splitlines()
        header = "-----BEGIN CERTIFICATE-----"
        assert len(deserialized_header["x5c"]) == pem_lines.count(header)

        # concatenate the PEM file's certs, removing headers and footers
        chain_start = pem_lines.index(header)
        pem_chain_content = "".join(line for line in pem_lines[chain_start:] if not line.startswith("-" * 5))
        assert "".join(deserialized_header["x5c"]) == pem_chain_content, "JWT's x5c claim contains unexpected content"
    else:
        assert "x5c" not in deserialized_header
    assert urlsafeb64_decode(deserialized_header["x5t#S256"]) == cert.fingerprint(hashes.SHA256())  # nosec

    cert.public_key().verify(
        signature,
        signed_part.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
        hashes.SHA256(),
    )


@pytest.mark.parametrize("cert_path,cert_password", ALL_CERTS)
@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_token_cache_persistent(cert_path, cert_password, get_token_method):
    """the credential should use a persistent cache if cache_persistence_options are configured"""

    access_token = "foo token"

    def send(request, **kwargs):
        # ensure the `claims` and `tenant_id` keywords from credential's `get_token` method don't make it to transport
        assert "claims" not in kwargs
        assert "tenant_id" not in kwargs
        parsed = urlparse(request.url)
        tenant = parsed.path.split("/")[1]
        if "/oauth2/v2.0/token" not in parsed.path:
            return get_discovery_response("https://{}/{}".format(parsed.netloc, tenant))
        return mock_response(json_payload=build_aad_response(access_token=access_token))

    with patch("azure.identity._internal.msal_credentials._load_persistent_cache") as load_persistent_cache:

        credential = CertificateCredential(
            "tenant",
            "client-id",
            cert_path,
            password=cert_password,
            cache_persistence_options=TokenCachePersistenceOptions(),
            transport=Mock(send=send),
        )

        assert load_persistent_cache.call_count == 0, "cache should not be loaded until a token is requested"
        assert credential._cache is None
        assert credential._cae_cache is None

        token = getattr(credential, get_token_method)("scope")
        assert token.token == access_token
        assert load_persistent_cache.call_count == 1
        assert credential._cache is not None
        assert credential._cae_cache is None

        kwargs = {"enable_cae": True}
        if get_token_method == "get_token_info":
            kwargs = {"options": kwargs}
        token = getattr(credential, get_token_method)("scope", **kwargs)
        assert load_persistent_cache.call_count == 2
        assert credential._cae_cache is not None


@pytest.mark.parametrize("cert_path,cert_password", ALL_CERTS)
@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_token_cache_memory(cert_path, cert_password, get_token_method):
    """The credential should default to in-memory cache if no persistence options are provided."""
    access_token = "foo token"

    def send(request, **kwargs):
        # ensure the `claims` and `tenant_id` keywords from credential's `get_token` method don't make it to transport
        assert "claims" not in kwargs
        assert "tenant_id" not in kwargs
        parsed = urlparse(request.url)
        tenant = parsed.path.split("/")[1]
        if "/oauth2/v2.0/token" not in parsed.path:
            return get_discovery_response("https://{}/{}".format(parsed.netloc, tenant))
        return mock_response(json_payload=build_aad_response(access_token=access_token))

    with patch("azure.identity._internal.msal_credentials._load_persistent_cache") as load_persistent_cache:
        credential = CertificateCredential(
            "tenant", "client-id", cert_path, password=cert_password, transport=Mock(send=send)
        )

        assert credential._cache is None
        token = getattr(credential, get_token_method)("scope")
        assert token.token == access_token
        assert isinstance(credential._cache, TokenCache)
        assert credential._cae_cache is None
        assert not load_persistent_cache.called

        kwargs = {"enable_cae": True}
        if get_token_method == "get_token_info":
            kwargs = {"options": kwargs}
        token = getattr(credential, get_token_method)("scope", **kwargs)
        assert isinstance(credential._cae_cache, TokenCache)
        assert not load_persistent_cache.called


@pytest.mark.parametrize("cert_path,cert_password", ALL_CERTS)
@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_persistent_cache_multiple_clients(cert_path, cert_password, get_token_method):
    """the credential shouldn't use tokens issued to other service principals"""

    access_token_a = "token a"
    access_token_b = "not " + access_token_a
    transport_a = new_msal_validating_transport(
        requests=[Request()], responses=[mock_response(json_payload=build_aad_response(access_token=access_token_a))]
    )
    transport_b = new_msal_validating_transport(
        requests=[Request()], responses=[mock_response(json_payload=build_aad_response(access_token=access_token_b))]
    )

    cache = TokenCache()
    with patch("azure.identity._internal.msal_credentials._load_persistent_cache") as mock_cache_loader:
        mock_cache_loader.return_value = Mock(wraps=cache)
        credential_a = CertificateCredential(
            "tenant",
            "client-a",
            cert_path,
            password=cert_password,
            transport=transport_a,
            cache_persistence_options=TokenCachePersistenceOptions(),
        )

        credential_b = CertificateCredential(
            "tenant",
            "client-b",
            cert_path,
            password=cert_password,
            transport=transport_b,
            cache_persistence_options=TokenCachePersistenceOptions(),
        )

        # A caches a token
        scope = "scope"
        token_a = getattr(credential_a, get_token_method)(scope)
        assert mock_cache_loader.call_count == 1, "credential should use the persistent cache"
        assert token_a.token == access_token_a
        assert transport_a.send.call_count == 2  # one MSAL discovery request, one token request

        # B should get a different token for the same scope
        token_b = getattr(credential_b, get_token_method)(scope)
        assert mock_cache_loader.call_count == 2, "credential should load the persistent cache"
        assert token_b.token == access_token_b
        assert transport_b.send.call_count == 2

        assert len(list(cache.search(TokenCache.CredentialType.ACCESS_TOKEN))) == 2


def test_certificate_arguments():
    """The credential should raise ValueError for mutually exclusive arguments"""

    with pytest.raises(ValueError) as ex:
        CertificateCredential("tenant-id", "client-id", certificate_path="...", certificate_data="...")
    message = str(ex.value)
    assert "certificate_data" in message and "certificate_path" in message


@pytest.mark.parametrize("cert_path,cert_password", ALL_CERTS)
@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_multitenant_authentication(cert_path, cert_password, get_token_method):
    first_tenant = "first-tenant"
    first_token = "***"
    second_tenant = "second-tenant"
    second_token = first_token * 2

    def send(request, **kwargs):
        # ensure the `claims` and `tenant_id` keywords from credential's `get_token` method don't make it to transport
        assert "claims" not in kwargs
        assert "tenant_id" not in kwargs
        parsed = urlparse(request.url)
        tenant = parsed.path.split("/")[1]
        assert tenant in (first_tenant, second_tenant, "common"), 'unexpected tenant "{}"'.format(tenant)
        if "/oauth2/v2.0/token" not in parsed.path:
            return get_discovery_response("https://{}/{}".format(parsed.netloc, tenant))

        token = first_token if tenant == first_tenant else second_token
        return mock_response(json_payload=build_aad_response(access_token=token))

    credential = CertificateCredential(
        first_tenant,
        "client-id",
        cert_path,
        password=cert_password,
        transport=Mock(send=send),
        additionally_allowed_tenants=["*"],
    )
    token = getattr(credential, get_token_method)("scope")
    assert token.token == first_token

    kwargs = {"tenant_id": first_tenant}
    if get_token_method == "get_token_info":
        kwargs = {"options": kwargs}
    token = getattr(credential, get_token_method)("scope", **kwargs)
    assert token.token == first_token

    kwargs = {"tenant_id": second_tenant}
    if get_token_method == "get_token_info":
        kwargs = {"options": kwargs}
    token = getattr(credential, get_token_method)("scope", **kwargs)
    assert token.token == second_token

    # should still default to the first tenant
    token = getattr(credential, get_token_method)("scope")
    assert token.token == first_token


@pytest.mark.parametrize("cert_path,cert_password", ALL_CERTS)
@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_multitenant_authentication_backcompat(cert_path, cert_password, get_token_method):
    expected_tenant = "expected-tenant"
    expected_token = "***"

    def send(request, **kwargs):
        # ensure the `claims` and `tenant_id` keywords from credential's `get_token` method don't make it to transport
        assert "claims" not in kwargs
        assert "tenant_id" not in kwargs
        parsed = urlparse(request.url)
        if "/oauth2/v2.0/token" not in parsed.path:
            return get_discovery_response("https://{}/{}".format(parsed.netloc, expected_tenant))

        tenant = parsed.path.split("/")[1]
        token = expected_token if tenant == expected_tenant else expected_token * 2
        return mock_response(json_payload=build_aad_response(access_token=token))

    credential = CertificateCredential(
        expected_tenant,
        "client-id",
        cert_path,
        password=cert_password,
        transport=Mock(send=send),
        additionally_allowed_tenants=["*"],
    )

    token = getattr(credential, get_token_method)("scope")
    assert token.token == expected_token

    kwargs = {"tenant_id": expected_tenant}
    if get_token_method == "get_token_info":
        kwargs = {"options": kwargs}
    # explicitly specifying the configured tenant is okay
    token = getattr(credential, get_token_method)("scope", **kwargs)
    assert token.token == expected_token

    kwargs = {"tenant_id": "un" + expected_tenant}
    if get_token_method == "get_token_info":
        kwargs = {"options": kwargs}
    token = getattr(credential, get_token_method)("scope", **kwargs)
    assert token.token == expected_token


def test_client_capabilities():
    """The credential should use the CAE-enabled client when enable_cae is True"""

    transport = Mock(send=Mock(side_effect=Exception("this test mocks MSAL, so no request should be sent")))

    credential = CertificateCredential("tenant-id", "client-id", PEM_CERT_PATH, transport=transport)
    with patch("msal.ConfidentialClientApplication") as ConfidentialClientApplication:
        credential._get_app()

        assert ConfidentialClientApplication.call_count == 1
        _, kwargs = ConfidentialClientApplication.call_args
        assert kwargs["client_capabilities"] == None

        credential._get_app(enable_cae=True)
        assert ConfidentialClientApplication.call_count == 2
        _, kwargs = ConfidentialClientApplication.call_args
        assert kwargs["client_capabilities"] == ["CP1"]


@pytest.mark.parametrize("get_token_method", GET_TOKEN_METHODS)
def test_claims_challenge(get_token_method):
    """get_token should pass any claims challenge to MSAL token acquisition APIs"""

    msal_acquire_token_result = dict(
        build_aad_response(access_token="**", id_token=build_id_token()),
        id_token_claims=id_token_claims("issuer", "subject", "audience", upn="upn"),
    )
    expected_claims = '{"access_token": {"essential": "true"}}'

    transport = Mock(send=Mock(side_effect=Exception("this test mocks MSAL, so no request should be sent")))
    credential = CertificateCredential("tenant-id", "client-id", PEM_CERT_PATH, transport=transport)
    with patch.object(CertificateCredential, "_get_app") as get_mock_app:
        msal_app = get_mock_app()
        msal_app.acquire_token_silent_with_error.return_value = None
        msal_app.acquire_token_for_client.return_value = msal_acquire_token_result

        kwargs = {"claims": expected_claims}
        if get_token_method == "get_token_info":
            kwargs = {"options": kwargs}
        getattr(credential, get_token_method)("scope", **kwargs)

        assert msal_app.acquire_token_silent_with_error.call_count == 1
        args, kwargs = msal_app.acquire_token_silent_with_error.call_args
        assert kwargs["claims_challenge"] == expected_claims

        assert msal_app.acquire_token_for_client.call_count == 1
        args, kwargs = msal_app.acquire_token_for_client.call_args
        assert kwargs["claims_challenge"] == expected_claims
