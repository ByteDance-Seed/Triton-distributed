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
"""Shared immutable build identities, source dependencies and atomic binary storage."""
from __future__ import annotations
import hashlib
import json
import os
import re
import shlex
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

_BUILD_SCHEMA = 2

_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*(?:(?P<quote>"(?P<q>[^"]+)")|(?P<angle><(?P<a>[^>]+)>))',
    re.MULTILINE,
)

_INCLUDE_NEXT_RE = re.compile(r'^\s*#\s*include_next\b', re.MULTILINE)

_NONLITERAL_INCLUDE_RE = re.compile(r'^\s*#\s*include(?:_next)?\b(?!\s*["<])\s*(?P<token>\S+)', re.MULTILINE)

_SEPARATE_INC_DIR_FLAGS = frozenset({
    "-I",
    "-iquote",
    "-isystem",
    "--include-path",
    "--system-include",
})

_FORCE_INCLUDE_FLAGS = frozenset({"-include", "--pre-include"})

_OPTIONS_FILE_PREFIXES = ("-optf", "--options-file")

_MAX_OPTIONS_FILE_DEPTH = 32

_PATH_LIKE = (str, os.PathLike)


class UncacheableBuildError(ValueError):
    """Raised when a build's dependency closure cannot be established.

    This covers cases such as an unreadable ``@response`` file or a forced
    include that cannot be located. Reusing a cache entry here would be
    unsafe, so caching is refused rather than silently reusing a wrong cubin.
    """


def normalize_options(options: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Return nvcc argv tokens exactly, preserving semantics.

    Order is significant (later ``-D``/``-U`` override earlier ones; paired
    flags such as ``-I`` take a following token), duplicates are significant
    and literal whitespace/token content is significant -- so tokens are
    **not** sorted, deduplicated or stripped. Two builds share an identity
    only when their argv is byte-for-byte identical.
    """
    if not options:
        return ()
    tokens = []
    for opt in options:
        if not isinstance(opt, str):
            raise TypeError(f"nvcc options must be strings, got {opt!r}")
        tokens.append(opt)
    return tuple(tokens)


def normalize_include_paths(include_paths: Optional[Sequence[Union[str, Path]]], ) -> Tuple[str, ...]:
    """Canonicalize include search paths while preserving search precedence.

    The order is significant (the first directory containing a quoted header
    wins), so paths are resolved but never sorted.
    """
    if not include_paths:
        return ()
    resolved = []
    for path in include_paths:
        resolved.append(os.path.realpath(os.path.abspath(os.fspath(path))))
    return tuple(resolved)


def _sha256(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _content_hash(content: bytes) -> str:
    return _sha256(content)[:16]


def _read_dependency_entry(entry: Any) -> Tuple[str, bytes]:
    """Normalize an explicit dependency into ``(name, bytes)``."""
    if isinstance(entry, tuple) and len(entry) == 2:
        name, value = entry
        if not isinstance(name, str) or not name:
            raise ValueError(f"dependency name must be a non-empty str, got {name!r}")
        if isinstance(value, str):
            return name, value.encode("utf-8")
        if isinstance(value, bytes):
            return name, value
        raise TypeError(f"dependency {name!r} content must be str or bytes, got {type(value)!r}")
    if isinstance(entry, _PATH_LIKE):
        path = Path(os.fspath(entry)).resolve()
        return os.fspath(path), path.read_bytes()
    raise TypeError(f"dependency entries must be paths or (name, content) tuples, got {entry!r}")


def _iter_includes(source: str):
    for match in _INCLUDE_RE.finditer(source):
        if match.group("quote") is not None:
            yield match.group("q"), False
        else:
            yield match.group("a"), True


def _split_attached_dir(token: str) -> Optional[str]:
    if token.startswith("-I") and len(token) > 2:
        return token[2:]
    for prefix in ("-iquote", "-isystem"):
        if token.startswith(prefix) and len(token) > len(prefix):
            rest = token[len(prefix):]
            if rest.startswith("="):
                rest = rest[1:]
            return rest
    for prefix in ("--include-path=", "--system-include="):
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _split_options_files(token: str) -> Tuple[Optional[str], ...]:
    """Split an options-file flag into its file names (comma-separated).

    Returns ``None`` when ``token`` is not an options-file flag; a tuple with
    an empty string when a file argument must come from the next argv token.
    Supports ``-optf``/``--options-file`` paired and attached ``=file[,file]``
    forms.
    """
    for prefix in _OPTIONS_FILE_PREFIXES:
        if token == prefix:
            return ("", )
        for sep in ("=", ):
            attached = prefix + sep
            if token.startswith(attached):
                return tuple(token[len(attached):].split(","))
    return None


def expand_option_dependencies(
        tokens: Sequence[str],
        *,
        _depth: int = 0,
        _seen: Optional[set] = None,
        _active: frozenset = frozenset(),
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Extract dependency-bearing information from raw nvcc argv tokens.

    Returns ``(system_dirs, quote_dirs, force_includes, response_hashes)``
    preserving command-line order:

    * ``-I``/``-isystem``/``--include-path`` directories apply to quoted and
      angle includes alike; ``-iquote`` directories apply to quoted includes;
    * ``-include``/``--pre-include`` names forced headers;
    * ``@file`` response files and ``-optf``/``--options-file`` options files
      (paired, attached and comma-separated forms) are read and expanded
      recursively; their contents are fingerprinted. Recursion is bounded and
      cycles do not overflow. An unreadable file raises
      :class:`UncacheableBuildError` rather than silently evading identity.
    """
    if _seen is None:
        _seen = set()
    if _depth > _MAX_OPTIONS_FILE_DEPTH:
        raise UncacheableBuildError(f"nvcc options/response file nesting exceeds "
                                    f"{_MAX_OPTIONS_FILE_DEPTH} levels; refusing to cache a build "
                                    f"whose dependency closure cannot be bounded")

    system_dirs: list = []
    quote_dirs: list = []
    force_includes: list = []
    response_hashes: list = []

    def add_dir(flag: str, directory: str):
        # NVCC include-path options are list options. Keep every search
        # directory in order rather than treating a comma list as one path.
        if "\\," in directory:
            raise UncacheableBuildError("escaped comma in include paths requires preprocessing; disable caching")
        directories = [directory] if flag == "-iquote" else directory.split(",")
        for item in directories:
            resolved = os.path.realpath(os.path.abspath(item))
            if flag == "-iquote":
                quote_dirs.append(resolved)
            else:
                system_dirs.append(resolved)

    def ingest_options_file(name: str, active: frozenset):
        path = Path(name)
        real = os.path.realpath(path)
        if real in active:
            raise UncacheableBuildError(f"recursive nvcc options/response file detected at {name!r}; "
                                        f"dependency closure is unbounded, refusing to cache")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise UncacheableBuildError(f"cannot read nvcc options/response file {name!r} to "
                                        f"establish cache identity: {exc}; remove it, pass an explicit "
                                        f"readable dependency, or disable caching") from exc
        response_hashes.append(f"{real}:{_content_hash(content)}")
        if real in _seen:  # already-expanded subtree: contents recorded
            return
        _seen.add(real)
        inner = shlex.split(content.decode("utf-8", errors="replace"))
        sys_inner, quote_inner, force_inner, resp_inner = expand_option_dependencies(
            inner, _depth=_depth + 1, _seen=_seen, _active=active | {real})
        system_dirs.extend(sys_inner)
        quote_dirs.extend(quote_inner)
        force_includes.extend(force_inner)
        response_hashes.extend(resp_inner)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("@") and len(token) > 1:
            ingest_options_file(token[1:], _active)
            i += 1
            continue

        options_files = _split_options_files(token)
        if options_files is not None:
            names = list(options_files)
            if names == [""]:
                if i + 1 >= len(tokens):
                    raise UncacheableBuildError(f"nvcc option {token!r} is missing its file argument")
                # The next token may itself list comma-separated files.
                names = tokens[i + 1].split(",")
                i += 1
            for name in names:
                if name:
                    ingest_options_file(name, _active)
            i += 1
            continue

        if token in _SEPARATE_INC_DIR_FLAGS:
            if i + 1 >= len(tokens):
                raise UncacheableBuildError(f"nvcc option {token!r} is missing its directory argument")
            add_dir(token, tokens[i + 1])
            i += 1
        elif token in _FORCE_INCLUDE_FLAGS:
            if i + 1 >= len(tokens):
                raise UncacheableBuildError(f"nvcc option {token!r} is missing its header argument")
            force_includes.extend(tokens[i + 1].split(","))
            i += 1
        elif any(token.startswith(flag + "=") for flag in _FORCE_INCLUDE_FLAGS):
            names = token.split("=", 1)[1].split(",")
            if not all(names):
                raise UncacheableBuildError(f"nvcc option {token!r} has an empty forced header")
            force_includes.extend(names)
        else:
            attached = _split_attached_dir(token)
            if attached is not None:
                add_dir(token.split("=", 1)[0], attached)
        i += 1

    return (
        tuple(system_dirs),
        tuple(quote_dirs),
        tuple(force_includes),
        tuple(response_hashes),
    )


def scan_user_dependencies(
        source: str,
        include_paths: Optional[Sequence[Union[str, Path]]] = None,
        extra_dependencies: Sequence[Any] = (),
        *,
        option_system_dirs: Sequence[str] = (),
        option_quote_dirs: Sequence[str] = (),
        forced_includes: Sequence[str] = (),
) -> Tuple[Tuple[str, str], ...]:
    """Return ``(identity-key, content-hash)`` pairs for every dependency.

    Both quoted and angle includes are followed when they resolve inside a
    user include directory. Resolution mirrors nvcc/gcc precedence:

    * a quoted include first searches the *including file's* directory, then
      ``-iquote`` directories, then ``-I``/include-path directories in order;
    * an angle include searches only the ``-I``/include-path directories.

    Each entry is keyed by its **resolved path**, so the same spelling
    resolving to different files (due to search-path order) produces a
    different identity. Angle headers not found in any user directory are
    assumed toolchain-provided: they are keyed by spelling only (toolchain
    identity covers their content). A quoted include that resolves nowhere is
    keyed by **(including file's directory, spelling)**: nvcc searches the
    includer's own directory first, so two includers in different directories
    that miss the same spelling must not collapse into one undifferentiated
    entry -- creating the header next to one includer later changes exactly
    that includer's resolution. The anonymous top-level source keys its
    missing includes with a ``null`` directory component.
    """
    if not isinstance(source, str):
        raise TypeError("source must be str")

    # Fail closed on macro-expanded/computed includes: their selected header
    # is only known after preprocessing, so a cache key cannot be complete.
    def reject_nonliteral(text: str) -> None:
        if _INCLUDE_NEXT_RE.search(text):
            raise UncacheableBuildError("#include_next requires preprocessor search-position tracking; disable caching")
        bad = _NONLITERAL_INCLUDE_RE.search(text)
        if bad is not None:
            raise UncacheableBuildError(f"non-literal #include directive {bad.group('token')!r} cannot be "
                                        f"resolved without preprocessing; preprocess the source first "
                                        f"or disable caching")

    base_system_dirs = [Path(p) for p in normalize_include_paths(include_paths)]
    base_system_dirs.extend(Path(p) for p in option_system_dirs)
    base_quote_dirs = [Path(p) for p in option_quote_dirs]

    entries: dict = {}
    visited: set = set()

    def resolve(name: str, includer_dir: Optional[Path], angle: bool) -> Optional[Path]:
        absolute = Path(name)
        if absolute.is_absolute():
            return absolute.resolve() if absolute.is_file() else None
        candidates = []
        if not angle:
            if includer_dir is not None:
                candidates.append(includer_dir / name)
            candidates.extend(d / name for d in base_quote_dirs)
        candidates.extend(d / name for d in base_system_dirs)
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_file():
                return resolved
        return None

    def walk(text: str, includer_dir: Optional[Path]) -> None:
        reject_nonliteral(text)
        for name, angle in _iter_includes(text):
            path = resolve(name, includer_dir, angle)
            if path is None:
                if angle:
                    # Toolchain-provided angle header: keyed by spelling only;
                    # the toolchain identity covers its content.
                    entries[f"system:{name}"] = "system"
                    continue
                # Missing quoted include: resolution prefers the including
                # file's own directory, so key by (includer directory,
                # spelling). JSON keeps the pair unambiguous even when a
                # directory or spelling contains a separator character.
                dir_key = os.fspath(includer_dir) if includer_dir is not None else None
                entries[f"missing:{json.dumps([dir_key, name])}"] = "missing"
                continue
            key = os.fspath(path)
            if key in visited:
                continue
            visited.add(key)
            content = path.read_bytes()
            entries[key] = _content_hash(content)
            walk(content.decode("utf-8", errors="replace"), path.parent)

    walk(source, None)

    for name in forced_includes:
        path = resolve(name, None, False)
        if path is None:
            raise UncacheableBuildError(f"forced include {name!r} cannot be resolved from the user "
                                        f"include paths; refusing to cache without its contents")
        key = os.fspath(path)
        if key not in visited:
            visited.add(key)
            content = path.read_bytes()
            entries[key] = _content_hash(content)
            walk(content.decode("utf-8", errors="replace"), path.parent)

    explicit_names = set()
    for entry in extra_dependencies:
        name, content = _read_dependency_entry(entry)
        if name in explicit_names:
            raise UncacheableBuildError(f"duplicate explicit dependency name {name!r}")
        explicit_names.add(name)
        entries[f"extra:{name}"] = _content_hash(content)

    return tuple(sorted(entries.items()))


@dataclass(frozen=True)
class BuildIdentity:
    """One complete, immutable cache identity.

    Two builds share a cached cubin if and only if their canonical
    representations are equal.
    """

    kernel_name: str
    source_hash: str
    target_arch: str
    options: Tuple[str, ...] = field(default_factory=tuple)
    include_paths: Tuple[str, ...] = field(default_factory=tuple)
    dependencies: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    toolchain_id: str = "unavailable"
    build_schema: int = _BUILD_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_name, str) or not self.kernel_name:
            raise ValueError("kernel_name must be a non-empty str")
        if not isinstance(self.target_arch, str) or not self.target_arch:
            raise ValueError("target_arch must be a resolved, non-empty arch string")
        if self.target_arch.lower() == "auto":
            raise ValueError("cache identity requires a resolved target architecture; "
                             "'auto' is unresolved and must never be used as a cache key")
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "include_paths", tuple(self.include_paths))
        object.__setattr__(self, "dependencies", tuple((str(n), str(h)) for n, h in self.dependencies))

    def canonical(self) -> str:
        """Deterministic JSON serialization used for the fingerprint."""
        return json.dumps(
            {
                "build_schema": self.build_schema,
                "kernel_name": self.kernel_name,
                "source_sha256": self.source_hash,
                "target_arch": self.target_arch,
                # Lists, not sets: order is semantically meaningful.
                "options": list(self.options),
                "include_paths": list(self.include_paths),
                "dependencies": [list(item) for item in self.dependencies],
                "toolchain_id": self.toolchain_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def fingerprint(self) -> str:
        return _sha256(self.canonical())

    @property
    def safe_name(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", self.kernel_name)

    @property
    def cache_key(self) -> str:
        return f"{self.safe_name}_{self.fingerprint[:24]}"

    def manifest(self) -> dict:
        return json.loads(self.canonical())


def atomic_publish(
    directory: Union[str, Path],
    filename: str,
    data: Union[str, bytes],
    mode: str = "wb",
) -> Path:
    """Atomically place ``data`` at ``directory/filename``.

    Writes go to a hidden temporary file inside ``directory``, are flushed and
    ``fsync``-ed, and then moved into place with :func:`os.replace` (atomic on
    POSIX and Windows for same-directory replacement).  The destination never
    exists in a partially written state.
    """
    if os.sep in filename or (os.altsep and os.altsep in filename) or filename in ("", ".", ".."):
        raise ValueError(f"unsafe cache filename {filename!r}")
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8" if "b" not in mode else None
    fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=os.fspath(out_dir))
    try:
        with os.fdopen(fd, mode, encoding=encoding) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, os.fspath(out_dir / filename))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return out_dir / filename


class AtomicBinaryCache:
    """Filesystem binary cache with atomic publication and in-process locking.

    Multiple threads building the same identity compile exactly once; other
    processes may also build concurrently -- each publishes via
    :func:`os.replace`, so readers always observe a complete file.
    """

    def __init__(self, cache_dir: Union[str, Path], *, binary_suffix=".bin", source_suffix=".cc"):
        self.cache_dir = Path(cache_dir)
        self.binary_suffix = binary_suffix
        self.source_suffix = source_suffix
        self._key_locks: dict = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def binary_path(self, identity: BuildIdentity) -> Path:
        return self.cache_dir / f"{identity.cache_key}{self.binary_suffix}"

    def source_path(self, identity: BuildIdentity) -> Path:
        return self.cache_dir / f"{identity.cache_key}{self.source_suffix}"

    def manifest_path(self, identity: BuildIdentity) -> Path:
        return self.cache_dir / f"{identity.cache_key}.json"

    def load(self, identity: BuildIdentity) -> Optional[bytes]:
        """Return a complete cached binary, or ``None`` on a cache miss.

        A zero-byte file (e.g. a legacy non-atomic write interrupted mid-way)
        is treated as a miss and replaced atomically on the next store.
        """
        path = self.binary_path(identity)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        if not data:
            return None
        return data

    def store(
        self,
        identity: BuildIdentity,
        binary: bytes,
        source: Optional[str] = None,
    ) -> Path:
        if not isinstance(binary, bytes) or not binary:
            raise ValueError("binary must be non-empty bytes")
        atomic_publish(self.cache_dir, self.binary_path(identity).name, binary)
        atomic_publish(
            self.cache_dir,
            self.manifest_path(identity).name,
            json.dumps(identity.manifest(), indent=2, sort_keys=True),
            mode="w",
        )
        if source is not None:
            atomic_publish(self.cache_dir, self.source_path(identity).name, source, mode="w")
        return self.binary_path(identity)

    def get_or_build(
        self,
        identity: BuildIdentity,
        builder: Callable[[], bytes],
        source: Optional[str] = None,
    ) -> bytes:
        """Return the cached binary or build, publish and return it.

        Within one process, concurrent callers for the same identity are
        serialized so the builder runs exactly once.
        """
        cached = self.load(identity)
        if cached is not None:
            return cached

        lock = self._lock_for(identity.cache_key)
        with lock:
            # Re-check after acquiring the lock: another thread may have won.
            cached = self.load(identity)
            if cached is not None:
                return cached
            binary = builder()
            if not isinstance(binary, bytes) or not binary:
                raise RuntimeError("binary builder produced empty output")
            self.store(identity, binary, source)
            return binary
