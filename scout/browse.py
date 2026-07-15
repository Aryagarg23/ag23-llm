"""Keyless web tools for the scout: search + fetch.

No API key, on purpose — the scout must run with nothing but a network connection.
Search goes through DuckDuckGo's HTML endpoint; fetch is a plain GET with a couple of
conveniences (GitHub blob URLs are rewritten to raw). Everything degrades to an empty
result rather than raising, so a flaky network can't crash a scout run.
"""
from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import httpx

_UA = "Mozilla/5.0 (compatible; AG23-llm-scout/0.1; +https://aryagarg23.com)"
_TIMEOUT = 20

_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps outbound links as /l/?uddg=<encoded>. Unwrap to the real URL."""
    if "uddg=" in href:
        q = urllib.parse.urlparse(href).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


def search(query: str, *, max_results: int = 10) -> list[SearchHit]:
    """Web search via DuckDuckGo HTML. Returns [] on any failure."""
    try:
        r = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        r.raise_for_status()
    except Exception:
        return []

    hits: list[SearchHit] = []
    body = r.text
    snippets = _SNIPPET_RE.findall(body)
    for i, m in enumerate(_RESULT_RE.finditer(body)):
        url = _unwrap_ddg(m.group("href"))
        title = _clean(m.group("title"))
        snippet = _clean(snippets[i]) if i < len(snippets) else ""
        if url.startswith("http"):
            hits.append(SearchHit(title=title, url=url, snippet=snippet))
        if len(hits) >= max_results:
            break
    return hits


def _rawify_github(url: str) -> str:
    """github.com/owner/repo/blob/ref/path -> raw.githubusercontent.com/owner/repo/ref/path."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return url


def fetch(url: str, *, max_bytes: int = 400_000) -> Optional[str]:
    """GET a URL as text (truncated to max_bytes). None on failure."""
    url = _rawify_github(url)
    try:
        r = httpx.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT,
                      follow_redirects=True)
        r.raise_for_status()
    except Exception:
        return None
    return r.text[:max_bytes]


def fetch_json(url: str):
    """GET and parse JSON. None on failure."""
    try:
        r = httpx.get(url, headers={"User-Agent": _UA, "Accept": "application/json"},
                      timeout=_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
