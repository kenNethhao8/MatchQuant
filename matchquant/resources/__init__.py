"""Packaged read-only resources for MatchQuant baseline-v0.1.

This subpackage carries the data files that must survive a real wheel install:

- ``m3_candidates.json``: the default candidate configuration for ``matchquant select``.
- ``examples/``: the same small synthetic examples as the repository ``examples/`` directory.
- ``demo/``: the synthetic demo payload exported by ``matchquant demo-export``
  (a frequency single-model runtime bundle, score-free fixtures, independent
  canonical results, and ``DEMO.md``).

Everything here is self-made synthetic teaching material. Nothing in this
package is real league data, and the demo bundle is a tutorial artifact, not a
production release.
"""
