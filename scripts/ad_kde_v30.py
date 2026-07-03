#!/usr/bin/env python3
"""Superseded: compatibility shim for the v2 correction release.

Figure scripts in this bundle import this module by its v30 name. None uses the
ISJ/Botev bandwidth, so their outputs are unchanged; this shim resolves the name
to ad_kde_v31, whose only difference is the corrected direct
Botev-Grotowski-Kroese fixed-point ISJ (the KDEpy improved_sheather_jones used
through v30/v1 undersmooths by about 2.6x and was dropped). See HISTORY.md.
"""
import sys as _sys
import ad_kde_v31 as _v31
_sys.modules[__name__] = _v31
