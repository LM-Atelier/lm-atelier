"""Whether an extension a workflow needs may be trusted without asking.

The rule: a package from the ComfyUI Registry that passes the existing review
installs without asking, and anything from a git URL or an unreviewed source
pauses once with a plain-language explanation. That keeps most installs to a
single approval while code nobody has checked still gets a person's decision.

"The existing review" is the automated resolution and verification the product
already performs - an active package and version, no administrator security
tags, an identity that matches, a verified archive. It is not an upstream
"reviewed" flag, which the Registry client does not read, and it is not a manual
click.

This module DECIDES and never records. It reads no files and changes no rows, so
the decision can be shown to a person before anything is committed, and the
recorder beside `review_comfy_registry_install` applies it under verification.

TWO THINGS IT WILL NOT DO, both deliberate. It never turns a person's explicit
refusal back into trust: a package someone chose not to trust stays refused
until that person decides otherwise. And it treats a warning it does not
recognise as a reason to pause, because a trust boundary that fails open on a
signal nobody has seen yet is not a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .comfy_registry import ComfyNodeResolution
from .models import ComfyRegistryInstall

POLICY_ID = "registry-existing-review-v1"

Outcome = Literal["auto_trust", "already_trusted", "pause", "refuse"]

# Warnings the existing review attaches without refusing, that a person should
# still see. Deprecation is a notice, not a security clearance and not a pause.
_NOTICE_WARNINGS = frozenset({"deprecated_version"})

_PLAIN = {
    "source_review_required": (
        "This extension comes directly from a code repository rather than the "
        "ComfyUI Registry, so it has not been through the Registry's checks. "
        "Installing it runs its code on this computer. Continue only if you "
        "trust where it came from."
    ),
    "unreviewed_source": (
        "This extension does not come from a source the installer can check "
        "automatically. Installing it runs its code on this computer. Continue "
        "only if you trust where it came from."
    ),
    "unrecognised_warning": (
        "The ComfyUI Registry reported something about this extension that the "
        "installer does not recognise, so it will not trust it automatically. "
        "Continue only if you trust where it came from."
    ),
    "previously_refused": (
        "You chose not to trust this extension before, so it will not be trusted automatically now."
    ),
    "not_installed": ("This extension is not installed yet, so there is nothing to trust."),
    "identity_mismatch": (
        "The installed extension does not match the one the Registry described, "
        "so it will not be trusted."
    ),
}

_REFUSAL_PLAIN = (
    "The ComfyUI Registry check did not pass for this extension, so it cannot be installed."
)


@dataclass(frozen=True)
class RegistryTrustDecision:
    """One decision about one extension, carrying what a person should be told."""

    outcome: Outcome
    reason: str
    explanation: str
    notices: tuple[str, ...] = ()


def _person_refused(install: ComfyRegistryInstall) -> bool:
    """Whether a person, rather than this policy, decided not to trust the package.

    A review record written before the authority field existed carries no
    ``trust_authority`` at all. Until then the only writer of these records was
    the explicit review, so an unattributed refusal IS a person's refusal - and
    reading it as anything else would let the policy re-trust a package someone
    deliberately refused.
    """

    review = install.review_json if isinstance(install.review_json, dict) else {}
    return (
        isinstance(review.get("reviewed_at"), str)
        and review.get("trusted_by_local_user") is False
        and review.get("trust_authority") in (None, "local_user")
    )


def _identity_matches(resolution: ComfyNodeResolution, install: ComfyRegistryInstall) -> bool:
    """Every identity field the resolution and the stored install share, all equal.

    `install_kind` alone proves nothing about which package is installed, so a
    resolution for one package can never speak for another installed row.
    """

    return (
        resolution.package_id == install.package_id
        and resolution.declared_version == install.package_version
        and resolution.registry_record_id == install.registry_record_id
        and resolution.repository_url == install.repository_url
        and resolution.download_url == install.download_url
    )


def decide_registry_trust(
    resolution: ComfyNodeResolution,
    install: ComfyRegistryInstall | None,
) -> RegistryTrustDecision:
    """Decide for one resolved extension and the install that claims to be it."""

    if resolution.error_code is not None:
        return RegistryTrustDecision("refuse", resolution.error_code, _REFUSAL_PLAIN)
    if resolution.install_kind == "git_commit":
        reason = "source_review_required"
        return RegistryTrustDecision("pause", reason, _PLAIN[reason])
    if resolution.install_kind not in ("registry_archive", "already_installed"):
        reason = "unreviewed_source"
        return RegistryTrustDecision("pause", reason, _PLAIN[reason])
    if install is None:
        reason = "not_installed"
        return RegistryTrustDecision("pause", reason, _PLAIN[reason])
    if (
        install.package_id != resolution.package_id
        or install.package_version != resolution.declared_version
        or (
            resolution.install_kind == "registry_archive"
            and not _identity_matches(resolution, install)
        )
    ):
        reason = "identity_mismatch"
        return RegistryTrustDecision("refuse", reason, _PLAIN[reason])
    if _person_refused(install):
        reason = "previously_refused"
        return RegistryTrustDecision("pause", reason, _PLAIN[reason])

    notices = tuple(sorted(set(resolution.warnings) & _NOTICE_WARNINGS))
    unrecognised = set(resolution.warnings) - _NOTICE_WARNINGS
    if unrecognised:
        reason = "unrecognised_warning"
        return RegistryTrustDecision("pause", reason, _PLAIN[reason], notices)
    if install.trusted:
        # No new prompt - which is not permission to skip the exact verification
        # every later launch still performs.
        return RegistryTrustDecision("already_trusted", "already_trusted", "", notices)
    if resolution.install_kind == "already_installed":
        # Installed but never trusted, and not described by the Registry this
        # time: nothing here has been through the existing review.
        reason = "unreviewed_source"
        return RegistryTrustDecision("pause", reason, _PLAIN[reason], notices)
    return RegistryTrustDecision("auto_trust", POLICY_ID, "", notices)
