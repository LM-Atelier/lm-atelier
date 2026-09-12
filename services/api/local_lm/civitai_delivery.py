"""Where a CivitAI download may send us, and what is allowed to travel there.

CivitAI does not serve a file from the endpoint you ask. The request to
`civitai.com` answers with a redirect to whichever delivery host actually holds
the bytes, so following exactly one hop is the normal path rather than an edge
case - a reader that refuses redirects never reaches any file at all.

The policy lives here rather than in either caller because both the model
transfer path and the workflow graph reader need the same answer, and a security
rule with two copies is a rule that will eventually disagree with itself.

TWO SEPARATE QUESTIONS, deliberately kept apart. Where a hop may point is
`permitted_delivery_url`. Whether our catalog credentials may travel on that hop
is `carries_catalog_credentials`, and the answer is no for every delivery host:
the redirect CivitAI issues is already signed, so the token buys nothing there
and sending it would hand our credentials to a host that never needed them.
"""

from __future__ import annotations

from urllib.parse import urlparse

# civitai.com issues the redirect, b2 serves smaller files, and anything of any
# size comes from their Cloudflare R2 delivery domain. Without the last of these
# every large model refused with "untrusted host" at the second hop.
ALLOWED_DOWNLOAD_HOSTS = (
    "civitai.com",
    "b2.civitai.com",
    ".r2.cloudflarestorage.com",
)

# One hop is the documented shape. The allowance is small rather than generous
# because a redirect chain that keeps going is a loop or a mistake, and either
# way the reader should say so instead of walking it.
MAX_DELIVERY_HOPS = 3

_CREDENTIALLED_HOST = "civitai.com"


def _host_permitted(host: str) -> bool:
    for allowed in ALLOWED_DOWNLOAD_HOSTS:
        if allowed.startswith("."):
            if host.endswith(allowed):
                return True
        elif host == allowed:
            return True
    return False


def permitted_delivery_url(url: str) -> str:
    """The url if a download may point at it, or a refusal saying it may not.

    Applied to EVERY hop, not only the first. A redirect is a value chosen by
    the far side, so checking only where we started would let the far side pick
    the destination.
    """

    parsed = urlparse(url)
    host = parsed.hostname or ""
    if (
        not url
        or len(url) > 2048
        or parsed.scheme != "https"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or not _host_permitted(host)
    ):
        raise ValueError(f"CivitAI download address is not an allowed delivery target: {url[:120]}")
    return url


def carries_catalog_credentials(url: str) -> bool:
    """Whether our catalog token may be sent to this url.

    Only the API host itself. A delivery host receives a signed address and
    needs nothing from us; sending the token anyway would disclose it to a third
    party for no gain, which is the whole reason this is a separate question
    from whether the hop is allowed.
    """

    return (urlparse(url).hostname or "") == _CREDENTIALLED_HOST
