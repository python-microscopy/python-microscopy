"""
If this test begins failing for functional code we should quickly scrap it.
This is simply a claude-coded check for wx flags that the current wxWidgets (3.2.9)
and wxpython 4.2.5 will reject.
It does not run wx code, it just parses our source code and looks for faulty sizer calls.

wxWidgets 3.x added consistency checks to ``wxSizer::DoInsert`` which raise a
``wxAssertionError`` (i.e. a hard failure under wxPython) when an ``Add`` call
passes an alignment flag that the sizer would silently ignore. Because the
offending flag never had any effect, removing it is always a no-op for layout --
but leaving it in place turns the whole panel into a crash.

The rules below were established empirically against wxPython 4.2.5 /
wxWidgets 3.2.9:

1. ``EXPAND`` combined with any of ``ALIGN_RIGHT``, ``ALIGN_CENTER_HORIZONTAL``,
   ``ALIGN_BOTTOM`` or ``ALIGN_CENTER_VERTICAL`` ("wxEXPAND overrides alignment
   flags in box sizers").
2. A vertical box sizer with ``ALIGN_CENTER_VERTICAL`` or ``ALIGN_BOTTOM``
   ("only horizontal alignment flags can be used in vertical sizers").
3. A horizontal box sizer with ``ALIGN_CENTER_HORIZONTAL`` or ``ALIGN_RIGHT``
   ("only vertical alignment flags can be used in horizontal sizers").

``ALIGN_LEFT`` and ``ALIGN_TOP`` are both zero and so are always harmless, grid
sizers accept every alignment flag, and ``ALIGN_CENTER`` (both centre bits at
once) is specifically exempt from rules 2 and 3.

This is a static check: it parses the source rather than building any windows,
so it runs headless and does not need a display.
"""
import ast
import os
import warnings
from collections import namedtuple

import pytest

wx = pytest.importorskip('wx')

#: sizer classifications
BOX_V = 'vertical box sizer'
BOX_H = 'horizontal box sizer'
GRID = 'grid sizer'
UNKNOWN = 'unknown'

#: index of the ``flags`` argument, by sizer method name
_FLAG_ARG_INDEX = {'Add': 2, 'Prepend': 2, 'Insert': 3}

_GRID_CLASSES = ('GridSizer', 'FlexGridSizer', 'GridBagSizer')

_BAD_WITH_EXPAND = (wx.ALIGN_RIGHT | wx.ALIGN_CENTER_HORIZONTAL
                    | wx.ALIGN_BOTTOM | wx.ALIGN_CENTER_VERTICAL)
_BAD_IN_V = wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_BOTTOM
_BAD_IN_H = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_RIGHT

Violation = namedtuple('Violation', ['path', 'lineno', 'sizer', 'kind', 'flags', 'reason'])


def _dotted_name(node):
    """Return the dotted name of a Name/Attribute node, or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else '%s.%s' % (base, node.attr)
    return None


def _wx_attr(node):
    """Return the attribute name of a ``wx.XXX`` reference, or None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == 'wx':
        return node.attr
    return None


def _sizer_kind(call):
    """Classify a ``wx.*Sizer(...)`` constructor call.

    Parameters
    ----------
    call : ast.Call
        The candidate constructor call.

    Returns
    -------
    str or None
        One of `BOX_V`, `BOX_H`, `GRID`, `UNKNOWN`, or None if the call does not
        construct a sizer.
    """
    cls = _wx_attr(call.func)
    if cls is None or not cls.endswith('Sizer'):
        return None

    if cls in _GRID_CLASSES:
        return GRID
    if cls == 'StdDialogButtonSizer':
        return BOX_H

    # BoxSizer(orient), StaticBoxSizer(box, orient), WrapSizer(orient)
    orients = [_wx_attr(a) for a in call.args]
    orients += [_wx_attr(k.value) for k in call.keywords
                if k.arg in ('orient', 'orientation')]
    orients = [o for o in orients if o in ('VERTICAL', 'HORIZONTAL')]

    if len(orients) == 1:
        return BOX_V if orients[0] == 'VERTICAL' else BOX_H
    if cls == 'StaticBoxSizer':
        return BOX_V  # wx default when the orientation is omitted
    return UNKNOWN


def _eval_flags(node):
    """Evaluate an OR-ed expression of wx flag constants.

    Returns
    -------
    bits : int
        The combined flag value.
    resolved : bool
        False if any term could not be resolved to an integer, in which case
        `bits` is meaningless.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, left_ok = _eval_flags(node.left)
        right, right_ok = _eval_flags(node.right)
        return left | right, left_ok and right_ok
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value, True
    name = _wx_attr(node)
    if name is not None and isinstance(getattr(wx, name, None), int):
        return getattr(wx, name), True
    return 0, False


def _reasons(kind, bits):
    """Return the reasons wx would reject this flag combination (may be empty)."""
    if kind in (GRID, UNKNOWN):
        return []

    if (bits & wx.EXPAND) and (bits & _BAD_WITH_EXPAND):
        # this check fires first in wx and exempts nothing
        return ['EXPAND overrides alignment flags in box sizers']

    # wxALIGN_CENTRE (both centre bits) is accepted in either orientation
    if (bits & wx.ALIGN_CENTER_HORIZONTAL) and (bits & wx.ALIGN_CENTER_VERTICAL):
        bits &= ~(wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL)

    if kind == BOX_V and (bits & _BAD_IN_V):
        return ['only horizontal alignment flags can be used in vertical sizers']
    if kind == BOX_H and (bits & _BAD_IN_H):
        return ['only vertical alignment flags can be used in horizontal sizers']
    return []


def find_violations(source, path='<string>'):
    """Find sizer ``Add`` calls that pass flags wx will reject.

    Parameters
    ----------
    source : str
        Python source code.
    path : str, optional
        Name used in the returned records.

    Returns
    -------
    list of Violation
        Empty if the source is clean (or cannot be parsed).

    Notes
    -----
    A sizer's orientation is resolved by looking back for the most recent
    assignment of that name at or above the call's line. Calls whose sizer or
    flags cannot be resolved statically are ignored rather than reported, so
    this check yields false negatives but not false positives.
    """
    try:
        with warnings.catch_warnings():
            # parsing legacy sources re-raises their own escape/syntax warnings,
            # which have nothing to do with what we are checking here
            warnings.simplefilter('ignore', DeprecationWarning)
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return []

    # sizer name -> [(lineno, kind), ...] in source order
    assignments = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        kind = _sizer_kind(node.value)
        if kind is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = _dotted_name(target)
            if name:
                assignments.setdefault(name, []).append((node.lineno, kind))
    for records in assignments.values():
        records.sort()

    def kind_at(name, lineno):
        prior = [k for ln, k in assignments.get(name, []) if ln <= lineno]
        if prior:
            return prior[-1]
        # only assigned later (e.g. in another method) -- usable if unambiguous
        kinds = set(k for _, k in assignments.get(name, []))
        return kinds.pop() if len(kinds) == 1 else None

    violations = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        index = _FLAG_ARG_INDEX.get(node.func.attr)
        if index is None:
            continue
        sizer = _dotted_name(node.func.value)
        if sizer is None:
            continue
        kind = kind_at(sizer, node.lineno)
        if kind is None:
            continue

        flag_node = None
        if len(node.args) > index:
            flag_node = node.args[index]
        else:
            for keyword in node.keywords:
                if keyword.arg == 'flag':
                    flag_node = keyword.value
        if flag_node is None:
            continue

        bits, resolved = _eval_flags(flag_node)
        if not resolved:
            continue

        flags = ast.get_source_segment(source, flag_node) or ''
        for reason in _reasons(kind, bits):
            violations.append(Violation(path, node.lineno, sizer, kind,
                                        ' '.join(flags.split()), reason))
    return violations


def _pyme_source_files():
    """Yield the paths of all python files in the PYME package."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PYME'))
    if not os.path.isdir(root):
        pytest.skip('PYME source tree not found at %s' % root)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for filename in sorted(filenames):
            if filename.endswith('.py'):
                yield os.path.join(dirpath, filename)


def test_no_ignored_sizer_flags():
    """No sizer Add call in PYME may pass a flag that wx would reject."""
    violations = []
    for path in _pyme_source_files():
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        violations.extend(find_violations(source, os.path.relpath(path)))

    assert not violations, 'wx will raise on %d sizer Add call(s):\n%s' % (
        len(violations),
        '\n'.join('  %s:%d  %s.Add(..., %s) -- %s (%s)'
                  % (v.path, v.lineno, v.sizer, v.flags, v.reason, v.kind)
                  for v in violations))


# --- unit tests for the rules themselves ------------------------------------

def _check(body, sizer='wx.BoxSizer(wx.VERTICAL)'):
    source = 'import wx\ns = %s\ns.Add(w, 0, %s, 5)\n' % (sizer, body)
    return find_violations(source)


@pytest.mark.parametrize('flags, sizer', [
    # rule 1 -- EXPAND overrides alignment, in either orientation
    ('wx.EXPAND | wx.ALIGN_CENTER_VERTICAL', 'wx.BoxSizer(wx.VERTICAL)'),
    ('wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL', 'wx.BoxSizer(wx.HORIZONTAL)'),
    ('wx.EXPAND | wx.ALIGN_RIGHT', 'wx.BoxSizer(wx.VERTICAL)'),
    ('wx.EXPAND | wx.ALIGN_BOTTOM', 'wx.BoxSizer(wx.HORIZONTAL)'),
    ('wx.EXPAND | wx.ALIGN_CENTER', 'wx.BoxSizer(wx.VERTICAL)'),
    # rule 2 -- vertical alignment in a vertical sizer
    ('wx.ALIGN_CENTER_VERTICAL | wx.ALL', 'wx.BoxSizer(wx.VERTICAL)'),
    ('wx.ALIGN_BOTTOM', 'wx.BoxSizer(wx.VERTICAL)'),
    ('wx.ALIGN_CENTER_VERTICAL', 'wx.StaticBoxSizer(box, wx.VERTICAL)'),
    ('wx.ALIGN_CENTER_VERTICAL', 'wx.StaticBoxSizer(box)'),
    # rule 3 -- horizontal alignment in a horizontal sizer
    ('wx.ALIGN_CENTER_HORIZONTAL | wx.ALL', 'wx.BoxSizer(wx.HORIZONTAL)'),
    ('wx.ALIGN_RIGHT', 'wx.BoxSizer(wx.HORIZONTAL)'),
    ('wx.ALIGN_RIGHT', 'wx.StdDialogButtonSizer()'),
])
def test_rejected_combinations(flags, sizer):
    assert len(_check(flags, sizer)) == 1


@pytest.mark.parametrize('flags, sizer', [
    # perpendicular alignment is fine
    ('wx.ALIGN_CENTER_VERTICAL | wx.ALL', 'wx.BoxSizer(wx.HORIZONTAL)'),
    ('wx.ALIGN_CENTER_HORIZONTAL | wx.ALL', 'wx.BoxSizer(wx.VERTICAL)'),
    ('wx.ALIGN_RIGHT | wx.ALL', 'wx.BoxSizer(wx.VERTICAL)'),
    ('wx.ALIGN_BOTTOM', 'wx.BoxSizer(wx.HORIZONTAL)'),
    # ALIGN_CENTER (both centre bits) is exempt from rules 2 and 3
    ('wx.ALIGN_CENTER', 'wx.BoxSizer(wx.VERTICAL)'),
    ('wx.ALIGN_CENTER', 'wx.BoxSizer(wx.HORIZONTAL)'),
    ('wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL', 'wx.BoxSizer(wx.VERTICAL)'),
    # ALIGN_LEFT and ALIGN_TOP are zero
    ('wx.EXPAND | wx.ALIGN_LEFT | wx.ALIGN_TOP', 'wx.BoxSizer(wx.VERTICAL)'),
    # grid sizers accept everything
    ('wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT', 'wx.GridSizer(2, 2, 0, 0)'),
    ('wx.EXPAND | wx.ALIGN_CENTER_VERTICAL', 'wx.FlexGridSizer(2, 2, 0, 0)'),
    # plain border/expand flags
    ('wx.EXPAND | wx.ALL', 'wx.BoxSizer(wx.VERTICAL)'),
])
def test_accepted_combinations(flags, sizer):
    assert _check(flags, sizer) == []


def test_unresolvable_flags_are_not_reported():
    """A flag expression we cannot evaluate must not produce a false positive."""
    source = ('import wx\n'
              's = wx.BoxSizer(wx.VERTICAL)\n'
              's.Add(w, 0, some_flags | wx.ALIGN_CENTER_VERTICAL, 5)\n')
    assert find_violations(source) == []


def test_orientation_tracked_per_reassignment():
    """Reassigning a sizer name switches which flags are legal from there on."""
    source = ('import wx\n'
              's = wx.BoxSizer(wx.HORIZONTAL)\n'
              's.Add(w, 0, wx.ALIGN_CENTER_VERTICAL, 5)\n'   # legal
              's = wx.BoxSizer(wx.VERTICAL)\n'
              's.Add(w, 0, wx.ALIGN_CENTER_VERTICAL, 5)\n')   # illegal
    violations = find_violations(source)
    assert [v.lineno for v in violations] == [5]


def test_insert_and_prepend_flag_positions():
    """Insert takes the flags one argument later than Add and Prepend."""
    source = ('import wx\n'
              's = wx.BoxSizer(wx.VERTICAL)\n'
              's.Prepend(w, 0, wx.ALIGN_CENTER_VERTICAL, 5)\n'
              's.Insert(0, w, 0, wx.ALIGN_CENTER_VERTICAL, 5)\n')
    assert [v.lineno for v in find_violations(source)] == [3, 4]


def test_flag_keyword_argument():
    source = ('import wx\n'
              's = wx.BoxSizer(wx.VERTICAL)\n'
              's.Add(w, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)\n')
    assert len(find_violations(source)) == 1
