# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import httpx
import openai
import pytest
from google.genai.errors import ClientError
from litellm import exceptions as litellm_exceptions
from volcenginesdkarkruntime import _exceptions as ark_exceptions

from openviking.models.model_error import (
    ModelError,
    ModelErrorKind,
    model_error_boundary,
    model_error_from_exception,
    model_error_from_status,
)
from openviking.utils.model_retry import retry_async, retry_sync


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "https://api.example.test"))


def _http_status_error(status_code: int, body: dict) -> httpx.HTTPStatusError:
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.example.test"),
        json=body,
    )
    return httpx.HTTPStatusError(
        "provider request failed",
        request=response.request,
        response=response,
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            openai.RateLimitError(
                "quota",
                response=_response(429),
                body={"error": {"code": "insufficient_quota"}},
            ),
            ModelErrorKind.QUOTA_EXHAUSTED,
        ),
        (
            openai.RateLimitError(
                "rate limited",
                response=_response(429),
                body={"error": {"code": "rate_limit_exceeded"}},
            ),
            ModelErrorKind.RATE_LIMIT_ERROR,
        ),
        (
            openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
            ModelErrorKind.TRANSIENT_NETWORK_ERROR,
        ),
        (
            ark_exceptions.ArkRateLimitError(
                "quota",
                response=_response(429),
                body={"error": {"code": "AccountQuotaExceeded"}},
                request_id="request-id",
            ),
            ModelErrorKind.QUOTA_EXHAUSTED,
        ),
        (
            ark_exceptions.ArkContentFilterFinishReasonError(),
            ModelErrorKind.CONTENT_SAFETY,
        ),
        (
            litellm_exceptions.ContextWindowExceededError(
                "too large",
                model="model",
                llm_provider="openai",
            ),
            ModelErrorKind.INPUT_TOO_LARGE,
        ),
        (
            litellm_exceptions.ServiceUnavailableError(
                "unavailable",
                model="model",
                llm_provider="openai",
            ),
            ModelErrorKind.TRANSIENT_SERVICE_ERROR,
        ),
        (
            httpx.ReadTimeout("timed out"),
            ModelErrorKind.TRANSIENT_NETWORK_ERROR,
        ),
        (
            _http_status_error(429, {"code": "AccountQuotaExceeded"}),
            ModelErrorKind.QUOTA_EXHAUSTED,
        ),
        (
            openai.NotFoundError(
                "model not found",
                response=_response(404),
                body={"error": {"code": "model_not_found"}},
            ),
            ModelErrorKind.MODEL_NOT_FOUND,
        ),
        (
            _http_status_error(404, {}),
            ModelErrorKind.UNKNOWN,
        ),
    ],
)
def test_sdk_errors_map_to_model_error_kind(error, expected):
    converted = model_error_from_exception(error)

    assert converted is not None
    assert converted.kind == expected


def test_plain_http_404_requires_structured_model_code():
    converted = model_error_from_status(
        404,
        "not found",
        provider="http-provider",
    )

    assert converted.kind == ModelErrorKind.UNKNOWN


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("QUOTA_EXCEEDED", ModelErrorKind.QUOTA_EXHAUSTED),
        ("RATE_LIMIT_EXCEEDED", ModelErrorKind.RATE_LIMIT_ERROR),
        (None, ModelErrorKind.UNKNOWN),
    ],
)
def test_gemini_resource_exhausted_requires_structured_reason(reason, expected):
    details = [] if reason is None else [{"reason": reason}]
    error = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "resource exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": details,
            }
        },
    )

    converted = model_error_from_exception(error)

    assert converted is not None
    assert converted.kind == expected


def test_provider_boundary_preserves_original_sdk_error_as_cause():
    sdk_error = openai.BadRequestError(
        "bad request",
        response=_response(400),
        body={"error": {"code": "invalid_request_error"}},
    )

    @model_error_boundary
    def _call():
        raise sdk_error

    with pytest.raises(ModelError) as raised:
        _call()

    assert raised.value.kind == ModelErrorKind.INVALID_REQUEST
    assert raised.value.__cause__ is sdk_error


def test_provider_boundary_reports_actual_provider_name():
    class _Provider:
        provider = "jina"

        @model_error_boundary
        def call(self):
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://api.jina.ai")
            )

    with pytest.raises(ModelError) as raised:
        _Provider().call()

    assert raised.value.provider == "jina"


def test_provider_boundary_does_not_convert_local_program_error():
    error = ValueError("HTTP 503 is data, not an API failure")

    @model_error_boundary
    def _call():
        raise error

    with pytest.raises(ValueError) as raised:
        _call()

    assert raised.value is error


def test_unknown_model_error_uses_standard_retry_budget():
    attempts = 0

    def _call():
        nonlocal attempts
        attempts += 1
        raise ModelError(ModelErrorKind.UNKNOWN, "unclassified API error")

    with pytest.raises(ModelError):
        retry_sync(
            _call,
            max_retries=5,
            base_delay=0,
            jitter=False,
            model_errors_only=True,
        )

    assert attempts == 6


@pytest.mark.parametrize(
    "kind",
    [
        ModelErrorKind.AUTHENTICATION_ERROR,
        ModelErrorKind.PERMISSION_DENIED,
        ModelErrorKind.QUOTA_EXHAUSTED,
        ModelErrorKind.MODEL_NOT_FOUND,
        ModelErrorKind.UNSUPPORTED,
        ModelErrorKind.INVALID_REQUEST,
        ModelErrorKind.INPUT_TOO_LARGE,
        ModelErrorKind.CONTENT_SAFETY,
    ],
)
def test_strict_retry_propagates_unhandled_model_error_unchanged(kind):
    attempts = 0
    error = ModelError(kind, "handled by another layer")

    def _call():
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(ModelError) as raised:
        retry_sync(
            _call,
            max_retries=5,
            base_delay=0,
            jitter=False,
            model_errors_only=True,
        )

    assert raised.value is error
    assert attempts == 1


def test_strict_retry_does_not_parse_local_error_message():
    attempts = 0

    def _call():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("HTTP 503")

    with pytest.raises(RuntimeError):
        retry_sync(
            _call,
            max_retries=5,
            base_delay=0,
            jitter=False,
            model_errors_only=True,
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_provider_boundary_and_retry_preserve_cancellation():
    attempts = 0

    @model_error_boundary
    async def _call():
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_async(_call, max_retries=5, model_errors_only=True)

    assert attempts == 1
