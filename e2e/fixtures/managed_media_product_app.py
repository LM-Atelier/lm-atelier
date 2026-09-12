"""The product, unpatched, for a run where the media engine is really managed.

The browser-protocol fixture beside this one replaces the editor runtime
identity, because that run is about what a page may do and not about how the
identity is obtained. This run is the opposite: what it certifies is the trust
path itself - a workflow revision becoming reviewable because node information
came from a media worker the product started. Patching anything on that path
would leave the certification asserting its own stand-in.

So this adds exactly one thing to the shipped application, and that one thing is
about the RUN rather than the product: a way for the runner to recognise its own
process on a loopback port.
"""

from __future__ import annotations

from local_lm.main import create_app

from .runner_readiness import install_readiness

app = create_app()
install_readiness(app)
