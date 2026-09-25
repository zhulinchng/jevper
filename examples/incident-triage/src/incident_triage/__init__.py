"""A support-incident triage service built on jevper.

The package is a consumer of the library, not a test of its internals: it only uses what
``jevper`` exports, and every scenario in :mod:`incident_triage.sweep` is something a real
service would do — classify a ticket, read a distribution, retry, explain, trace, cancel.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
