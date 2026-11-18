################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Driver resource ownership (RF04).

The CUDA Driver allocates long-lived, process-global resources -- contexts
retained/created on a device and modules loaded into a context.  Leaking or
double-releasing them from Python is a correctness bug:

* a context created with ``cuCtxCreate_v2`` (or a primary context retained
  with ``cuDevicePrimaryCtxRetain``) must be released exactly once, with the
  matching release API, while a *borrowed* context (one owned by another
  component, e.g. an external framework) must never be released here;
* a loaded module must not be unloaded while live kernel launchers still
  reference its functions;
* ``close``/``unload`` must be idempotent and usable as a context manager.

This module owns those rules independent of ``ctypes`` binding details.  The
release/validate callables are injected by :class:`CUDARuntime
<little_kernel.backends.nv.common.runtime.cuda_runtime.CUDARuntime>`, which also performs the
checked Driver calls.  No background threads or implicit ``__del__`` releases
are used: resource release is an explicit, deterministic action.
"""

from __future__ import annotations

import enum
import threading
from typing import Any, Callable, Optional

__all__ = [
    "ContextOwnership",
    "ManagedContext",
    "ManagedModule",
    "ResourceClosedError",
]


class ResourceClosedError(RuntimeError):
    """Raised when a closed driver resource is used again."""


class ContextOwnership(enum.Enum):
    """How a :class:`ManagedContext` relates to its underlying handle.

    ``PRIMARY_RETAINED``: retained through ``cuDevicePrimaryCtxRetain`` and
    released once through the matching primary-context release API.
    ``CREATED``: created through the modern ``cuCtxCreate_v2`` API and
    destroyed once through the matching destroy API.
    ``BORROWED``: owned by another component; it is made current when needed
    but never released by this object.
    """

    PRIMARY_RETAINED = "primary-retained"
    CREATED = "created"
    BORROWED = "borrowed"


class ManagedContext:
    """Exactly-once ownership of one CUDA context handle.

    Parameters
    ----------
    handle:
        Raw ``CUcontext`` pointer value. A null handle is rejected.
    device:
        Device ordinal the context belongs to.
    ownership:
        One of :class:`ContextOwnership`.
    release:
        Called exactly once from :meth:`close` for retained/created contexts.
        It receives this :class:`ManagedContext` and must perform the checked
        Driver release call. It must not be supplied for borrowed contexts
        (they are never released here).
    validate:
        Optional setup validator invoked once during construction; it
        receives this object and must raise if the context is not actually
        usable (e.g. ``cuCtxSetCurrent`` fails or ``cuCtxGetCurrent`` reports
        a different context).
    label:
        Optional diagnostic label.
    """

    def __init__(
        self,
        handle: int,
        device: int,
        ownership: ContextOwnership,
        *,
        release: Optional[Callable[["ManagedContext"], None]] = None,
        validate: Optional[Callable[["ManagedContext"], None]] = None,
        label: Optional[str] = None,
    ) -> None:
        handle = int(handle or 0)
        if handle == 0:
            raise RuntimeError(f"Cannot manage a null CUDA context ({label or ownership})")
        if not isinstance(ownership, ContextOwnership):
            raise TypeError(f"ownership must be a ContextOwnership, got {ownership!r}")
        if ownership is ContextOwnership.BORROWED and release is not None:
            raise ValueError("A borrowed context must not carry a release callable: it is "
                             "owned by another component and must never be released here")
        self._handle = handle
        self._device = int(device)
        self._ownership = ownership
        self._release = release
        self._label = label
        self._closed = False
        self._lock = threading.RLock()
        # Validate the *actual* setup before ownership is advertised. If
        # validation fails on a context we own, release it immediately so a
        # failed construction never leaks the retained/created handle. A
        # failure during that cleanup is chained to -- never substituted
        # for -- the original setup error.
        if validate is not None:
            try:
                validate(self)
            except BaseException as setup_exc:
                if ownership is not ContextOwnership.BORROWED and release:
                    try:
                        release(self)
                    except BaseException as cleanup_exc:  # noqa: BLE001
                        raise setup_exc from cleanup_exc
                raise

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def device(self) -> int:
        return self._device

    @property
    def ownership(self) -> ContextOwnership:
        return self._ownership

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def borrowed(self) -> bool:
        return self._ownership is ContextOwnership.BORROWED

    def close(self) -> bool:
        """Release the context exactly once.

        Returns ``True`` when this call performed the release, ``False`` when
        it was already closed or the context is borrowed. A failed owned
        release leaves the object open and tracked so ``close`` can be
        retried; it is never marked released on error.
        """
        with self._lock:
            if self._closed:
                return False
            if self._ownership is ContextOwnership.BORROWED:
                self._closed = True
                return False
            if self._release is not None:
                self._release(self)
            # Mark closed only after a successful release.
            self._closed = True
            return True

    def __enter__(self) -> "ManagedContext":
        if self.closed:
            raise ResourceClosedError("cannot re-enter a closed context")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False  # never suppress exceptions

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"ManagedContext(handle=0x{self._handle:x}, device={self._device}, "
                f"ownership={self._ownership.value}, closed={self._closed})")


class ManagedModule:
    """Exactly-once ownership of one loaded ``CUmodule`` handle.

    A module may have live dependents (kernel launchers). Unloading a module
    while a launcher still references its functions would leave the caller
    holding a dangling ``CUfunction``; :meth:`close` therefore refuses to
    unload in that state rather than silently invalidating live launchers.

    A *borrowed* module (loaded by another component and registered only so
    launcher dependencies can be tracked) is never unloaded here.
    """

    def __init__(
        self,
        handle: int,
        *,
        unload: Optional[Callable[[int], None]] = None,
        borrowed: bool = False,
        label: Optional[str] = None,
    ) -> None:
        handle = int(handle or 0)
        if handle == 0:
            raise RuntimeError(f"Cannot manage a null CUmodule ({label or 'module'})")
        if borrowed and unload is not None:
            raise ValueError("A borrowed module must not carry an unload callable")
        self._handle = handle
        self._unload = unload
        self._borrowed = bool(borrowed)
        self._label = label
        self._live_users = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def borrowed(self) -> bool:
        return self._borrowed

    @property
    def live_users(self) -> int:
        """Number of live launchers depending on this module."""
        with self._lock:
            return self._live_users

    def attach_user(self) -> int:
        """Register a live dependent (a kernel launcher)."""
        with self._lock:
            if self._closed:
                raise ResourceClosedError(f"module 0x{self._handle:x} has already been unloaded")
            self._live_users += 1
            return self._live_users

    def detach_user(self) -> int:
        """Release a previously registered dependent."""
        with self._lock:
            if self._live_users > 0:
                self._live_users -= 1
            return self._live_users

    def close(self) -> bool:
        """Unload the module exactly once.

        Raises :class:`RuntimeError` if live launchers still depend on it.
        Returns ``True`` when this call performed the unload, ``False`` when
        it was already closed or the module is borrowed. A failed owned
        unload leaves the module open and tracked so ``close`` can be
        retried; it is never marked unloaded on error.
        """
        with self._lock:
            if self._closed:
                return False
            if self._live_users:
                raise RuntimeError(f"Refusing to unload module 0x{self._handle:x}: "
                                   f"{self._live_users} live launcher(s) still depend on it "
                                   f"({self._label or 'unlabeled'})")
            if self._borrowed:
                self._closed = True
                return False
            if self._unload is not None:
                self._unload(self._handle)
            # Mark closed only after a successful unload.
            self._closed = True
            return True

    # Driver terminology alias.
    unload = close

    def __enter__(self) -> "ManagedModule":
        if self.closed:
            raise ResourceClosedError("cannot re-enter an unloaded module")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"ManagedModule(handle=0x{self._handle:x}, "
                f"borrowed={self._borrowed}, live_users={self._live_users}, "
                f"closed={self._closed})")
