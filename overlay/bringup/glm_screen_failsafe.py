# SPDX-License-Identifier: Apache-2.0
"""Rank-consistent cold-start fallback for optional launch-time levers.

Only initialization and contract failures are recoverable. Runtime, transport,
and rollback failures remain fatal. Every rank votes before a failed lever is
self-disabled, so no rank can continue with a different effective stack.
"""
import os
import sys

REGISTRY = {}
_MISSING = object()


def enabled(env=None):
    return (os.environ if env is None else env).get('GLM_SCREEN_FAILSAFE', '0') == '1'


class Journal:
    """Undo only explicitly owned attributes; preserve later wrappers."""
    def __init__(self):
        self.rows = []

    def save(self, obj, name):
        self.rows.append([obj, name, getattr(obj, name, _MISSING), _MISSING])

    def mark(self, start=0):
        for row in self.rows[start:]:
            row[3] = getattr(row[0], row[1], _MISSING)

    def restore(self):
        for obj, name, value, installed in reversed(self.rows):
            current = getattr(obj, name, _MISSING)
            if installed is not _MISSING and current is not installed:
                if not callable(installed) or not callable(current):
                    raise RuntimeError('screen attribute changed after init: ' + name)
                seen = set()

                def unlink(fn):
                    if id(fn) in seen:
                        return False
                    seen.add(id(fn))
                    for cell in getattr(fn, '__closure__', ()) or ():
                        try:
                            child = cell.cell_contents
                        except ValueError:
                            continue
                        if child is installed:
                            cell.cell_contents = value
                            if getattr(fn, '__wrapped__', None) is installed:
                                fn.__wrapped__ = value
                            return True
                        if callable(child) and unlink(child):
                            return True
                    return False

                if not unlink(current):
                    raise RuntimeError('screen wrapper changed after init: ' + name)
                continue
            if value is _MISSING:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, value)
        self.rows.clear()


def disabled(name):
    item = REGISTRY.get(name)
    return item is not None and item.reason is not None


def register(name, restore=None, env=None):
    if name in REGISTRY:
        item = REGISTRY[name]
        if restore is not None and restore not in item.restores:
            item.restores.append(restore)
        return item
    item = Lever(name, restore, env)
    REGISTRY[name] = item
    return item


class Lever:
    def __init__(self, name, restore, env):
        self.name, self.env = name, env
        self.restores = [] if restore is None else [restore]
        self.error = None
        self.reason = None

    def stage(self, init):
        if not enabled(self.env):
            return init()
        if self.reason is not None or self.error is not None:
            return None
        try:
            return init()
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
            return None

    def run(self, init, vote=None):
        if not enabled(self.env):
            return init()
        if self.reason is not None:
            return None
        result = self.stage(init)
        if vote is None:
            from glm_draft_head import agree
            vote = agree
        from glm_draft_head import VoteRefused
        try:
            vote({'screen_lever': self.name}, self.error is None)
        except VoteRefused as refusal:
            if refusal.receipt.get('payload_mismatch'):
                raise
            reasons = [row['measured'].get('reason')
                       for row in refusal.receipt.get('votes', [])
                       if row.get('measured', {}).get('reason')]
            reason = '; '.join(sorted(set(reasons))) or 'init/contract vote refused on a TP rank'
            from glm_draft_head import gather_receipts
            rows = gather_receipts({'reason': self.error})
            detail = sorted(set(row['reason'] for row in rows if row.get('reason')))
            if detail:
                reason = '; '.join(detail)
            error = None
            try:
                for restore in reversed(self.restores):
                    restore()
            except Exception as exc:
                error = exc
            vote({'screen_lever_restored': self.name}, error is None)
            if error is not None:
                raise RuntimeError('screen rollback failed: ' + self.name) from error
            self.reason = ' '.join(reason.splitlines())
            sys.stderr.write(f'glm-lever {self.name} self-disabled: {self.reason}\n')
            sys.stderr.flush()
            return None
        return result


def finish(vote=None):
    """Vote all registered levers after TP init and before capture/forward."""
    if enabled():
        if vote is None:
            from glm_draft_head import agree
            vote = agree
        vote({'screen_registry': sorted(REGISTRY)}, True)
        for name in sorted(REGISTRY):
            REGISTRY[name].run(lambda: None, vote)


def options(name, check, fallback, env=None):
    """Defer an optional import-time contract to the first TP vote."""
    if not enabled(env):
        return check()
    item = register(name, env=env)
    if item.reason is not None:
        return fallback
    value = item.stage(check)
    return fallback if value is None else value


def install(name, init, attributes):
    """Stage a hook installation with an explicit list of owned attributes."""
    if not enabled():
        return init()
    journal = Journal()
    for obj, attr in attributes:
        journal.save(obj, attr)

    def checked():
        try:
            return init()
        finally:
            journal.mark()
    return register(name, journal.restore).stage(checked)
