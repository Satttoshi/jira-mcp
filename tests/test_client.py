"""Tests for read-only enforcement and base-URL detection (mocked HTTP)."""

from collections.abc import Callable

import httpx
import pytest

from jira_mcp.client import (
    AtlassianError,
    BaseUrlDetectionError,
    JiraClient,
    ReadOnlyViolationError,
)
from jira_mcp.config import Settings

SITE = "https://example.atlassian.net"
CLOUD_BASE = "https://api.atlassian.com/ex/jira/cid-123"
ACCOUNT = {"accountId": "abc", "displayName": "Ada", "emailAddress": "ada@x.io"}


def settings(cloud_id: str | None = None) -> Settings:
    return Settings(
        email="ada@x.io",
        api_token="secret",
        site_url=SITE,
        cloud_id=cloud_id,
        env_file_found=True,
    )


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], cloud_id: str | None = None
) -> JiraClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JiraClient(settings(cloud_id), http=http)


def unscoped_handler(request: httpx.Request) -> httpx.Response:
    """Site URL authenticates directly (unscoped token)."""
    url = str(request.url)
    if url == f"{SITE}/rest/api/3/myself":
        return httpx.Response(200, json=ACCOUNT)
    if url.startswith(SITE):
        return httpx.Response(200, json={"ok": True, "url": url})
    return httpx.Response(404)


class ScopedHandler:
    """Site URL returns anonymous-looking 200s; api.atlassian.com works."""

    def __init__(self) -> None:
        self.site_probes = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{SITE}/rest/api/3/myself":
            self.site_probes += 1
            return httpx.Response(200, json={})  # 200 but anonymous
        if url == f"{SITE}/_edge/tenant_info":
            return httpx.Response(200, json={"cloudId": "cid-123"})
        if url == f"{CLOUD_BASE}/rest/api/3/myself":
            return httpx.Response(200, json=ACCOUNT)
        if url.startswith(CLOUD_BASE):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


async def test_unscoped_token_uses_site_url() -> None:
    client = make_client(unscoped_handler)
    data = await client.get_json("/rest/api/3/issue/P-1")
    assert data["url"].startswith(SITE)
    assert client.base_url == SITE
    assert client.used_cloud_id_fallback is False
    assert client.account == ACCOUNT


async def test_scoped_token_falls_back_via_tenant_info() -> None:
    client = make_client(ScopedHandler())
    await client.get_json("/rest/api/3/issue/P-1")
    assert client.base_url == CLOUD_BASE
    assert client.used_cloud_id_fallback is True


async def test_401_on_site_also_triggers_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{SITE}/rest/api/3/myself":
            return httpx.Response(401)
        if url == f"{SITE}/_edge/tenant_info":
            return httpx.Response(200, json={"cloudId": "cid-123"})
        if url == f"{CLOUD_BASE}/rest/api/3/myself":
            return httpx.Response(200, json=ACCOUNT)
        return httpx.Response(200, json={})

    client = make_client(handler)
    await client.get_json("/rest/api/3/serverInfo")
    assert client.base_url == CLOUD_BASE


async def test_detection_result_is_cached() -> None:
    handler = ScopedHandler()
    client = make_client(handler)
    await client.get_json("/rest/api/3/issue/P-1")
    await client.get_json("/rest/api/3/issue/P-2")
    assert handler.site_probes == 1  # probed once, not per request


async def test_cloud_id_override_skips_detection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        assert "_edge/tenant_info" not in url
        assert not url.startswith(f"{SITE}/rest")  # site never probed
        if url == f"{CLOUD_BASE}/rest/api/3/myself":
            return httpx.Response(200, json=ACCOUNT)
        return httpx.Response(200, json={"ok": True})

    client = make_client(handler, cloud_id="cid-123")
    await client.get_json("/rest/api/3/issue/P-1")
    assert client.base_url == CLOUD_BASE


async def test_tenant_info_failure_names_the_escape_hatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = make_client(handler)
    with pytest.raises(BaseUrlDetectionError, match="ATLASSIAN_CLOUD_ID"):
        await client.get_json("/rest/api/3/issue/P-1")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/rest/api/3/issue"),
        ("PUT", "/rest/api/3/issue/P-1"),
        ("DELETE", "/rest/api/3/issue/P-1"),
        ("POST", "/rest/api/3/issue/P-1/comment"),
        ("PATCH", "/rest/api/3/issue/P-1"),
    ],
)
async def test_write_requests_are_blocked(method: str, path: str) -> None:
    client = make_client(unscoped_handler)
    with pytest.raises(ReadOnlyViolationError, match="read-only"):
        await client._request(method, path)


async def test_search_jql_post_is_the_only_allowed_post() -> None:
    client = make_client(unscoped_handler)
    data = await client.search_jql({"jql": "project = P"})
    assert data["ok"] is True


async def test_get_bytes_returns_content_and_mime() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{SITE}/rest/api/3/myself":
            return httpx.Response(200, json=ACCOUNT)
        # The JSON default must be widened, or Jira refuses raw files.
        assert request.headers["accept"] == "*/*"
        return httpx.Response(
            200,
            content=b"\x89PNG",
            headers={"Content-Type": "image/png; charset=binary"},
        )

    client = make_client(handler)
    data, mime = await client.get_bytes("/rest/api/3/attachment/content/10023")
    assert data == b"\x89PNG"
    assert mime == "image/png"  # charset parameter stripped


async def test_404_error_mentions_permissions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{SITE}/rest/api/3/myself":
            return httpx.Response(200, json=ACCOUNT)
        return httpx.Response(404)

    client = make_client(handler)
    with pytest.raises(AtlassianError, match="permission"):
        await client.get_json("/rest/api/3/issue/NOPE-1")
