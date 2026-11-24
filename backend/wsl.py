import asyncio
import ipaddress
import posixpath
import time
from urllib.parse import urlparse, urlunparse

_IP_CACHE_TTL_SECONDS = 30.0
_ip_cache: dict[str, tuple[float, str]] = {}


def _decode_wsl_output(raw: bytes) -> str:
    """Decode WSL subprocess output handling UTF-16LE (Windows wsl.exe) and UTF-8."""
    if not raw:
        return ""
    if raw.startswith(b"\xff\xfe") or b"\x00" in raw:
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\ufeff", "").replace("\x00", "")


async def _run_wsl_command(args: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    """Run wsl.exe without blocking the event loop, returning (returncode, stdout, stderr)."""
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "wsl.exe",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, _decode_wsl_output(stdout_b), _decode_wsl_output(stderr_b)
    except asyncio.TimeoutError:
        if proc is not None and proc.returncode is None:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
        return -1, "", "WSL command timed out"
    except Exception as exc:
        return -1, "", str(exc)


async def list_wsl_distros() -> list[str]:
    """Return installed WSL distribution names, or [] if unavailable."""
    rc, stdout, _ = await _run_wsl_command(["--list", "--quiet"], timeout=3.0)
    if rc != 0:
        return []
    distros: list[str] = []
    for line in stdout.splitlines():
        name = line.strip()
        if name:
            distros.append(name)
    return distros


async def get_wsl_ip(distro: str = "", force_refresh: bool = False) -> str | None:
    """Resolve the WSL2 VM IPv4 address using hostname -I, with a short TTL cache."""
    cache_key = distro.strip()
    now = time.monotonic()
    if not force_refresh and cache_key in _ip_cache:
        cached_at, cached_ip = _ip_cache[cache_key]
        if now - cached_at < _IP_CACHE_TTL_SECONDS:
            return cached_ip

    args: list[str] = []
    if cache_key:
        args.extend(["-d", cache_key])
    args.extend(["-e", "hostname", "-I"])

    rc, stdout, _ = await _run_wsl_command(args, timeout=3.0)
    if rc != 0 or not stdout.strip():
        return None

    for token in stdout.split():
        try:
            addr = ipaddress.IPv4Address(token.strip())
            if not addr.is_loopback and not addr.is_link_local:
                ip_str = str(addr)
                _ip_cache[cache_key] = (now, ip_str)
                return ip_str
        except ValueError:
            continue
    return None


def is_loopback_url(url: str) -> bool:
    """Return True if the URL points to localhost / 127.0.0.1 / ::1."""
    try:
        host = (urlparse(url).hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"}
    except Exception:
        return False


def rewrite_url_with_host(url: str, new_host: str) -> str:
    """Replace the hostname in a URL while preserving scheme, port, and path."""
    parsed = urlparse(url)
    host_part = f"[{new_host}]" if ":" in new_host and not new_host.startswith("[") else new_host
    netloc = f"{host_part}:{parsed.port}" if parsed.port else host_part
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)).rstrip("/")


async def list_wsl_gguf_files(gguf_dir: str, distro: str = "") -> list[dict]:
    """Discover .gguf model files inside a WSL folder without shell injection."""
    target_dir = gguf_dir.strip()
    if not target_dir:
        return []

    args: list[str] = []
    if distro.strip():
        args.extend(["-d", distro.strip()])

    script = (
        'dir="$1"; '
        'case "$dir" in '
        '"~") dir="$HOME";; '
        '"~/"*) dir="$HOME/${dir#~/}";; '
        'esac; '
        '[ -d "$dir" ] && find "$dir" -maxdepth 3 -type f -name "*.gguf" -printf "%s\\t%p\\n" 2>/dev/null'
    )
    args.extend(["-e", "sh", "-lc", script, "_", target_dir])

    rc, stdout, _ = await _run_wsl_command(args, timeout=4.0)
    if rc != 0 and not stdout.strip():
        return []

    files: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        size_str, full_path = line.split("\t", 1)
        try:
            size_bytes = int(size_str)
        except ValueError:
            size_bytes = 0
        files.append(
            {
                "name": posixpath.basename(full_path),
                "path": full_path,
                "size_bytes": size_bytes,
            }
        )
    files.sort(key=lambda item: item["name"].lower())
    return files
