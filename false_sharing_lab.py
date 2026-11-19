import os
import sys
import json
import time
import ctypes
import hashlib
import platform
import threading
import statistics
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

try:  # optional: context switches + physical core count on Windows/macOS
    import psutil
except ImportError:
    psutil = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "ItachiTore"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000

STRIDES = (1, 4, 8, 16)
REPEATS = 5  # each layout is timed REPEATS times; median is reported
BUFFER_ALIGN = 128  # 128-B alignment => 64-B aligned AND each 128-B prefetch pair starts at line 0
GIL_DEMO_ITERS = 2_000_000  # per-thread updates for the (slow) pure-Python GIL demo

LCG_A = 6364136223846793005
LCG_C = 1442695040888963407


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    if sys.platform == "win32":
        size = _windows_l1d_line_size()
        if size:
            return size
    return 64


def _windows_l1d_line_size():
    """GetLogicalProcessorInformation -> CACHE_DESCRIPTOR(Level 1, Data).LineSize."""
    try:
        from ctypes import wintypes

        class CACHE_DESCRIPTOR(ctypes.Structure):
            _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                        ("LineSize", wintypes.WORD), ("Size", wintypes.DWORD),
                        ("Type", ctypes.c_int)]

        class _U(ctypes.Union):
            _fields_ = [("Cache", CACHE_DESCRIPTOR), ("Reserved", ctypes.c_ulonglong * 2)]

        class SLPI(ctypes.Structure):
            _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int), ("u", _U)]

        k32 = ctypes.windll.kernel32
        length = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformation(None, ctypes.byref(length))
        n = length.value // ctypes.sizeof(SLPI)
        arr = (SLPI * n)()
        if not k32.GetLogicalProcessorInformation(arr, ctypes.byref(length)):
            return None
        for e in arr:
            # RelationCache == 2, CacheData == 2 (CacheUnified=0, Instruction=1, Data=2, Trace=3)
            if e.Relationship == 2 and e.u.Cache.Level == 1 and e.u.Cache.Type == 2:
                return int(e.u.Cache.LineSize)
    except Exception:
        return None
    return None


def cpu_model_string() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           encoding="utf-8").strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ------------------------------------------------------------------------------
# Core affinity helpers (Linux: sched_setaffinity on the calling thread;
# Windows: SetThreadAffinityMask on the calling thread; macOS: unsupported)
# ------------------------------------------------------------------------------
def _distinct_physical_cores(logical: int):
    """Pick one logical CPU per physical core so that pinned threads never share
    an L1/L2 (SMT siblings share L1D and would hide coherence traffic)."""
    if sys.platform.startswith("linux"):
        seen, picks = set(), []
        for cpu in sorted(os.sched_getaffinity(0)):
            base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
            try:
                with open(f"{base}/core_id") as f:
                    core = f.read().strip()
                with open(f"{base}/physical_package_id") as f:
                    pkg = f.read().strip()
            except OSError:
                return list(range(logical)), "linux-no-topology"
            if (pkg, core) not in seen:
                seen.add((pkg, core))
                picks.append(cpu)
        return picks, "linux-sysfs-core_id"
    physical = psutil.cpu_count(logical=False) if psutil else None
    if sys.platform == "win32" and physical and logical == 2 * physical:
        # Windows enumerates SMT siblings adjacently: (0,1), (2,3), ...
        return list(range(0, logical, 2)), "windows-smt-adjacent"
    return list(range(logical)), "identity"


def pin_current_thread(cpu: int) -> bool:
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {cpu})  # pid 0 == calling thread on Linux
            return True
        if sys.platform == "win32":
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            return k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << cpu) != 0
    except Exception:
        return False
    return False


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    candidates, topo_method = _distinct_physical_cores(logical_cores)
    affinity_supported = hasattr(os, "sched_setaffinity") or sys.platform == "win32"
    pinned_cpus = candidates[:NUM_THREADS] if affinity_supported and len(candidates) >= NUM_THREADS else []

    return {
        "git_user": git_user.strip(),
        "cpu_sig": cpu_sig,
        "cpu_model": cpu_model_string(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": __import__("numba").__version__,
        "logical_cores": logical_cores,
        "physical_cores": psutil.cpu_count(logical=False) if psutil else None,
        "cache_line_size": line_size,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "topology_method": topo_method,
        "pinned_cpus": pinned_cpus,
    }


def thread_seeds(instance_seed: int):
    """SplitMix64 expansion of INSTANCE_SEED into one seed per thread."""
    mask = (1 << 64) - 1
    out, z = [], instance_seed
    for _ in range(NUM_THREADS):
        z = (z + 0x9E3779B97F4A7C15) & mask
        x = z
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        out.append(x ^ (x >> 31))
    return out


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
@njit(nogil=True, cache=False)
def worker_strided_accumulate(buffer, slot, iters, seed):
    """Read-modify-write buffer[slot] in memory on every iteration.

    The increment is the top byte of a per-thread LCG (+1), so every thread's
    final counter is a seed-dependent value that all layouts must reproduce.
    Register promotion is blocked by a data dependency: the index used by the
    next iteration is computed from the value just stored
    (idx = slot + (new >> 63)). The counter never reaches 2^63, so idx is
    always `slot`, but LLVM cannot prove that, so it must emit a real
    load + store of buffer[idx] each iteration.
    """
    a = uint64(LCG_A)
    c = uint64(LCG_C)
    s56 = uint64(56)
    s63 = uint64(63)
    one = uint64(1)
    x = uint64(seed)
    idx = slot
    for _ in range(iters):
        x = x * a + c
        new = buffer[idx] + (x >> s56) + one
        buffer[idx] = new
        idx = slot + np.int64(new >> s63)
    return x


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True, cache=False)
def worker_private_accumulate(buffer, slot, iters, seed):
    """Same arithmetic, but the accumulator lives in a register; one store at the end."""
    a = uint64(LCG_A)
    c = uint64(LCG_C)
    s56 = uint64(56)
    one = uint64(1)
    x = uint64(seed)
    acc = uint64(0)
    for _ in range(iters):
        x = x * a + c
        acc += (x >> s56) + one
    buffer[slot] = buffer[slot] + acc
    return x


# GIL-holding twin of the strided kernel (nogil=False) for Question 5
@njit(nogil=False, cache=False)
def worker_strided_accumulate_gil(buffer, slot, iters, seed):
    a = uint64(LCG_A)
    c = uint64(LCG_C)
    s56 = uint64(56)
    s63 = uint64(63)
    one = uint64(1)
    x = uint64(seed)
    idx = slot
    for _ in range(iters):
        x = x * a + c
        new = buffer[idx] + (x >> s56) + one
        buffer[idx] = new
        idx = slot + np.int64(new >> s63)
    return x


@njit(nogil=True, cache=False)
def worker_neighbor_read_accumulate(buffer, slot, iters, seed):
    """Supplementary: same RMW on buffer[slot] plus a read of the next thread's slot
    (`stride` elements away) every iteration. The extra load must be served from the
    coherent L1 line, not the core's own store buffer, so true line ping-pong
    becomes visible. `seed` carries the stride here."""
    one = uint64(1)
    s63 = uint64(63)
    stride = np.int64(seed)
    nb = slot + stride
    idx = slot
    acc = uint64(0)
    for _ in range(iters):
        acc += buffer[nb]
        new = buffer[idx] + one
        buffer[idx] = new
        idx = slot + np.int64((new | acc) >> s63)
    return acc


def python_bytecode_worker(buffer, slot, iters, seed):
    """Pure CPython version (holds the GIL between bytecodes)."""
    mask = (1 << 64) - 1
    x = seed
    for _ in range(iters):
        x = (x * LCG_A + LCG_C) & mask
        buffer[slot] = int(buffer[slot]) + (x >> 56) + 1
    return x


def expected_counter(seed: int, iters: int) -> int:
    """Closed reference computed with the private kernel on a scratch buffer."""
    scratch = np.zeros(1, dtype=np.uint64)
    worker_private_accumulate(scratch, 0, iters, np.uint64(seed))
    return int(scratch[0])


def aligned_uint64(n_elems: int, align: int = BUFFER_ALIGN) -> np.ndarray:
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    arr = raw[off:off + n_elems * 8].view(np.uint64)
    assert arr.ctypes.data % align == 0  # `arr.base` keeps `raw` alive
    return arr


def _ctx_switches():
    """(voluntary, involuntary) process context switches, or (None, None)."""
    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_nvcsw), int(ru.ru_nivcsw)
    if psutil is not None:
        cs = psutil.Process().num_ctx_switches()
        return int(cs.voluntary), int(cs.involuntary)
    return None, None


def _run_threads(ctx, kernel, buffer, slots, seeds, iters, pin=True):
    """Start NUM_THREADS workers behind a barrier; return elapsed ns + ctx switches."""
    n = len(slots)
    barrier = threading.Barrier(n + 1)
    done = threading.Barrier(n + 1)
    release = threading.Event()
    pinned = [False] * n
    starts = [0] * n
    ends = [0] * n
    cpus = ctx["pinned_cpus"]

    # Timestamps are taken inside the workers: a GIL-holding kernel would block the
    # main thread, so a main-thread t0 could be taken only after the kernel ended.
    def body(t):
        if pin and cpus:
            pinned[t] = pin_current_thread(cpus[t])
        barrier.wait()
        starts[t] = time.perf_counter_ns()
        kernel(buffer, slots[t], iters, seeds[t])
        ends[t] = time.perf_counter_ns()
        # Stay alive until main has sampled context switches (psutil on Windows only
        # sums the threads that still exist).
        done.wait()
        release.wait()

    threads = [threading.Thread(target=body, args=(t,)) for t in range(n)]
    for th in threads:
        th.start()
    v0, i0 = _ctx_switches()
    barrier.wait()
    done.wait()
    v1, i1 = _ctx_switches()
    release.set()
    for th in threads:
        th.join()
    return {
        "elapsed_ns": max(ends) - min(starts),
        "per_thread_ns": [e - s for s, e in zip(starts, ends)],
        "voluntary_ctx_switches": None if v0 is None else v1 - v0,
        "involuntary_ctx_switches": None if i0 is None else i1 - i0,
        "threads_pinned": all(pinned) if pin and cpus else False,
    }


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(ctx: dict) -> dict:
    seeds = [np.uint64(s) for s in thread_seeds(ctx["instance_seed"])]
    results = {}

    # JIT warm-up (compile both kernels once, untimed)
    warm = aligned_uint64(16)
    worker_strided_accumulate(warm, 0, 10, seeds[0])
    worker_private_accumulate(warm, 0, 10, seeds[0])

    layouts = [(f"stride_{s}", worker_strided_accumulate, s) for s in STRIDES]
    layouts.append(("private", worker_private_accumulate, 8))

    for name, kernel, stride in layouts:
        runs, finals = [], None
        slots = [t * stride for t in range(NUM_THREADS)]
        for _ in range(REPEATS):
            buf = aligned_uint64(NUM_THREADS * stride + 16)
            r = _run_threads(ctx, kernel, buf, slots, seeds, ITERATIONS_PER_THREAD)
            got = [int(buf[s]) for s in slots]
            if finals is None:
                finals = got
            elif got != finals:
                raise AssertionError(f"{name}: nondeterministic counters {got} vs {finals}")
            runs.append(r)
        times = [r["elapsed_ns"] for r in runs]
        med = int(statistics.median(times))
        base = buf.ctypes.data
        line = ctx["cache_line_size"]
        results[name] = {
            "kernel": kernel.__name__,
            "stride_elems": stride,
            "stride_bytes": stride * 8,
            "slot_byte_offsets": [s * 8 for s in slots],
            "cache_lines_touched": sorted({(base + 8 * s) // line - base // line for s in slots}),
            "elapsed_ns_runs": times,
            "elapsed_ns_median": med,
            "elapsed_ns_min": min(times),
            "ns_per_update_per_thread": med / ITERATIONS_PER_THREAD,
            "ns_per_update_aggregate": med / (ITERATIONS_PER_THREAD * NUM_THREADS),
            "voluntary_ctx_switches": [r["voluntary_ctx_switches"] for r in runs],
            "involuntary_ctx_switches": [r["involuntary_ctx_switches"] for r in runs],
            "threads_pinned": all(r["threads_pinned"] for r in runs),
            "final_counters": [str(v) for v in finals],
        }
        print(f"  {name:<10} stride={stride * 8:>4} B  median={med / 1e6:9.2f} ms  "
              f"min={min(times) / 1e6:9.2f} ms  {med / ITERATIONS_PER_THREAD:7.3f} ns/update/thread  "
              f"lines={results[name]['cache_lines_touched']}", flush=True)
    return results


def run_gil_demo(ctx: dict) -> dict:
    """Question 5: stride = 1 with GIL-holding threads, 1 thread vs NUM_THREADS threads."""
    seeds = [np.uint64(s) for s in thread_seeds(ctx["instance_seed"])]
    py_seeds = [int(s) for s in seeds]
    demo = {}

    # (a) pure CPython bytecode threads
    for n in (1, NUM_THREADS):
        buf = aligned_uint64(16)
        r = _run_threads(ctx, lambda b, s, it, sd: python_bytecode_worker(b, s, it, int(sd)),
                         buf, list(range(n)), py_seeds[:n], GIL_DEMO_ITERS, pin=True)
        demo[f"python_bytecode_{n}t"] = {
            "threads": n, "iters_per_thread": GIL_DEMO_ITERS,
            "elapsed_ns": r["elapsed_ns"],
            "ns_per_update_aggregate": r["elapsed_ns"] / (n * GIL_DEMO_ITERS),
            "voluntary_ctx_switches": r["voluntary_ctx_switches"],
            "involuntary_ctx_switches": r["involuntary_ctx_switches"],
        }

    # (b) numba-compiled but nogil=False -> the GIL is held for the whole call
    warm = aligned_uint64(16)
    worker_strided_accumulate_gil(warm, 0, 10, seeds[0])
    for n in (1, NUM_THREADS):
        buf = aligned_uint64(16)
        r = _run_threads(ctx, worker_strided_accumulate_gil, buf, list(range(n)), seeds[:n],
                         ITERATIONS_PER_THREAD)
        demo[f"numba_gil_held_{n}t"] = {
            "threads": n, "iters_per_thread": ITERATIONS_PER_THREAD,
            "elapsed_ns": r["elapsed_ns"],
            "ns_per_update_aggregate": r["elapsed_ns"] / (n * ITERATIONS_PER_THREAD),
            "voluntary_ctx_switches": r["voluntary_ctx_switches"],
            "involuntary_ctx_switches": r["involuntary_ctx_switches"],
        }

    # (c) nogil=True, single thread on stride 1 (reference for the 4-thread sweep)
    buf = aligned_uint64(16)
    r = _run_threads(ctx, worker_strided_accumulate, buf, [0], seeds[:1], ITERATIONS_PER_THREAD)
    demo["numba_nogil_1t"] = {
        "threads": 1, "iters_per_thread": ITERATIONS_PER_THREAD,
        "elapsed_ns": r["elapsed_ns"],
        "ns_per_update_aggregate": r["elapsed_ns"] / ITERATIONS_PER_THREAD,
    }
    for k, v in demo.items():
        print(f"  {k:<22} threads={v['threads']}  elapsed={v['elapsed_ns'] / 1e6:9.2f} ms  "
              f"{v['ns_per_update_aggregate']:8.3f} ns/update (aggregate)", flush=True)
    return demo


def run_supplementary(ctx: dict) -> dict:
    """Extra evidence (not part of parity): coherence cost once loads cannot be
    satisfied from the store buffer, and the effect of pinning / SMT siblings."""
    seeds = [np.uint64(s) for s in thread_seeds(ctx["instance_seed"])]
    sup = {}

    def med(kernel, stride, cpus, n=NUM_THREADS, seed_override=None):
        c = dict(ctx)
        c["pinned_cpus"] = cpus
        ts = []
        for _ in range(3):
            buf = aligned_uint64(NUM_THREADS * stride + 2 * stride + 16)
            sd = [np.uint64(seed_override)] * n if seed_override is not None else seeds[:n]
            ts.append(_run_threads(c, kernel, buf, [t * stride for t in range(n)], sd,
                                   ITERATIONS_PER_THREAD, pin=bool(cpus))["elapsed_ns"])
        return int(statistics.median(ts))

    warm = aligned_uint64(64)
    worker_neighbor_read_accumulate(warm, 0, 10, np.uint64(1))
    pinned = ctx["pinned_cpus"]
    for s in (1, 8, 16):
        sup[f"neighbor_read_stride_{s}_ns"] = med(worker_neighbor_read_accumulate, s, pinned,
                                                  seed_override=s)
    sup["strided_unpinned_stride_1_ns"] = med(worker_strided_accumulate, 1, [])
    sup["strided_unpinned_stride_8_ns"] = med(worker_strided_accumulate, 8, [])
    if sys.platform == "win32" or hasattr(os, "sched_setaffinity"):
        sib = [0, 1] if ctx["topology_method"] == "windows-smt-adjacent" else []
        if sib:
            sup["strided_smt_siblings_2t_stride_1_ns"] = med(worker_strided_accumulate, 1, sib, n=2)
            sup["strided_distinct_cores_2t_stride_1_ns"] = med(worker_strided_accumulate, 1, pinned[:2], n=2)
            sup["strided_distinct_cores_2t_stride_8_ns"] = med(worker_strided_accumulate, 8, pinned[:2], n=2)
    for k, v in sup.items():
        print(f"  {k:<40}: {v / 1e6:9.2f} ms", flush=True)
    return sup


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def _core_function_ir(llvm_text: str, py_name: str) -> str:
    """Extract the body of the native (non-wrapper) function from the module IR."""
    lines = llvm_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("define") and py_name in line and "cpython" not in line \
                and "cfunc" not in line:
            body = []
            for l2 in lines[i:]:
                body.append(l2)
                if l2.startswith("}"):
                    break
            return "\n".join(body)
    return ""


def audit_llvm_and_parity(ctx: dict, bench: dict) -> dict:
    seeds = thread_seeds(ctx["instance_seed"])
    expected = [str(expected_counter(s, ITERATIONS_PER_THREAD)) for s in seeds]
    parity = {name: (r["final_counters"] == expected) for name, r in bench.items()}
    parity_ok = all(parity.values())

    def ir_of(fn):
        d = fn.inspect_llvm()
        return "\n".join(d.values()) if isinstance(d, dict) else str(d)

    strided_ir = ir_of(worker_strided_accumulate)
    private_ir = ir_of(worker_private_accumulate)
    core = _core_function_ir(strided_ir, "worker_strided_accumulate")
    core_priv = _core_function_ir(private_ir, "worker_private_accumulate")
    with open("worker_strided_accumulate.ll", "w", encoding="utf-8") as f:
        f.write(core)
    with open("worker_private_accumulate.ll", "w", encoding="utf-8") as f:
        f.write(core_priv)

    # Find the loop: a basic block that branches back to itself / its header and stores i64
    def loop_blocks(fn_ir):
        blocks, cur, name = {}, [], "entry"
        for line in fn_ir.splitlines():
            s = line.strip()
            if s.endswith(":") or (": " in s and s.split(":")[0].replace(".", "").replace("_", "").isalnum()
                                   and not s.startswith(("%", "store", "br", "ret", "call", ";"))):
                if cur:
                    blocks[name] = cur
                name, cur = s.split(":")[0], []
            else:
                cur.append(s)
        if cur:
            blocks[name] = cur
        loops = {}
        for bname, instrs in blocks.items():
            if any(i.startswith("br ") and f"%{bname}" in i for i in instrs):
                loops[bname] = instrs
        return loops

    loops = loop_blocks(core)
    loop_store = any(any(i.startswith("store i64") for i in ins) for ins in loops.values())
    loop_load = any(any(" = load i64" in i for i in ins) for ins in loops.values())
    loops_p = loop_blocks(core_priv)
    priv_loop_store = any(any(i.startswith("store i64") for i in ins) for ins in loops_p.values())

    pyobj_markers = ("PyObject", "@PyNumber", "@PyLong", "NRT_decref", "@numba_unpickle", "PyErr")
    llvm = {
        "function_found": bool(core),
        "store_i64_count": core.count("store i64"),
        "load_i64_count": core.count("load i64"),
        "loop_blocks": sorted(loops.keys()),
        "store_i64_in_loop": loop_store,
        "load_i64_in_loop": loop_load,
        "python_object_calls": sum(core.count(m) for m in pyobj_markers),
        "nopython_signatures": [str(s) for s in worker_strided_accumulate.signatures],
        "private_store_i64_in_loop": priv_loop_store,
        "private_store_i64_count": core_priv.count("store i64"),
        "ir_sha256": hashlib.sha256(core.encode()).hexdigest(),
    }
    llvm_ok = (llvm["function_found"] and llvm["store_i64_in_loop"] and llvm["load_i64_in_loop"]
               and llvm["python_object_calls"] == 0 and not llvm["private_store_i64_in_loop"])

    print(f"  parity per layout       : {parity}", flush=True)
    print(f"  expected final counters : {expected}", flush=True)
    print(f"  LLVM strided kernel     : store i64={llvm['store_i64_count']} load i64={llvm['load_i64_count']} "
          f"loop blocks={llvm['loop_blocks']} store-in-loop={loop_store} "
          f"PyObject calls={llvm['python_object_calls']}", flush=True)
    print(f"  LLVM private kernel     : store-in-loop={priv_loop_store} "
          f"(store i64 total={llvm['private_store_i64_count']})", flush=True)
    return {"expected_counters": expected, "parity": parity, "parity_passed": parity_ok,
            "llvm": llvm, "llvm_passed": llvm_ok}


def main():
    git_user = sys.argv[1] if len(sys.argv) > 1 else GIT_USERNAME
    ctx = init_system_context(git_user)
    print("=" * 78)
    print("Cache Line & False Sharing Mechanics | Numba @njit(nogil=True) + threading")
    print("=" * 78)
    for k in ("git_user", "cpu_model", "cpu_sig", "platform", "logical_cores", "physical_cores",
              "cache_line_size", "topology_method", "pinned_cpus", "python", "numpy", "numba"):
        print(f"{k:<16}: {ctx[k]}")
    print(f"{'instance_seed':<16}: {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    print(f"{'threads x iters':<16}: {NUM_THREADS} x {ITERATIONS_PER_THREAD:,} updates, "
          f"{REPEATS} repeats per layout (median reported)")
    print("-" * 78)
    print("TASK 3/4: stride sweep + private-register baseline", flush=True)
    bench = run_stride_benchmark(ctx)
    print("-" * 78)
    print("QUESTION 5: GIL-held threads vs nogil threads (stride = 1)", flush=True)
    gil = run_gil_demo(ctx)
    print("-" * 78)
    print("SUPPLEMENTARY: neighbour-read kernel, pinning and SMT siblings", flush=True)
    sup = run_supplementary(ctx)
    print("-" * 78)
    print("TASK 5: parity + LLVM audit", flush=True)
    audit = audit_llvm_and_parity(ctx, bench)

    s1 = bench["stride_1"]["elapsed_ns_median"]
    s4 = bench["stride_4"]["elapsed_ns_median"]
    s8 = bench["stride_8"]["elapsed_ns_median"]
    s16 = bench["stride_16"]["elapsed_ns_median"]
    pv = bench["private"]["elapsed_ns_median"]
    ratios = {
        "stride1_over_stride8": s1 / s8,
        "stride4_over_stride8": s4 / s8,
        "stride8_over_stride16": s8 / s16,
        "stride8_over_private": s8 / pv,
        "stride1_over_private": s1 / pv,
        "gil_python_4t_over_1t": gil["python_bytecode_4t"]["elapsed_ns"] / gil["python_bytecode_1t"]["elapsed_ns"],
        "gil_numba_4t_over_1t": gil["numba_gil_held_4t"]["elapsed_ns"] / gil["numba_gil_held_1t"]["elapsed_ns"],
        "nogil_stride1_4t_over_1t": s1 / gil["numba_nogil_1t"]["elapsed_ns"],
        "neighbor_read_stride1_over_stride8": sup["neighbor_read_stride_1_ns"] / sup["neighbor_read_stride_8_ns"],
        "neighbor_read_stride8_over_stride16": sup["neighbor_read_stride_8_ns"] / sup["neighbor_read_stride_16_ns"],
        "unpinned_stride1_over_stride8": sup["strided_unpinned_stride_1_ns"] / sup["strided_unpinned_stride_8_ns"],
    }
    verification = audit["parity_passed"] and audit["llvm_passed"]

    print("-" * 78)
    for k, v in ratios.items():
        print(f"  {k:<26}: {v:7.3f}x")
    print(f"verification_passed: {verification}")

    manifest = {
        "schema": "cache-line-false-sharing-lab/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_username": ctx["git_user"],
        "instance_seed": str(ctx["instance_seed"]),
        "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
        "system": {k: ctx[k] for k in ("cpu_model", "cpu_sig", "platform", "logical_cores",
                                       "physical_cores", "cache_line_size", "affinity_supported",
                                       "topology_method", "pinned_cpus", "python", "numpy", "numba")},
        "config": {"num_threads": NUM_THREADS, "iterations_per_thread": ITERATIONS_PER_THREAD,
                   "strides": list(STRIDES), "repeats": REPEATS, "buffer_align_bytes": BUFFER_ALIGN,
                   "gil_demo_iters": GIL_DEMO_ITERS},
        "thread_seeds": [str(s) for s in thread_seeds(ctx["instance_seed"])],
        "benchmarks": bench,
        "gil_demo": gil,
        "supplementary": sup,
        "ratios": ratios,
        "parity": audit["parity"],
        "expected_counters": audit["expected_counters"],
        "parity_passed": audit["parity_passed"],
        "llvm_audit": audit["llvm"],
        "llvm_passed": audit["llvm_passed"],
        "verification_passed": verification,
    }
    with open(__file__, "rb") as f:
        manifest["script_sha256"] = hashlib.sha256(f.read()).hexdigest()
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote cache_lab_manifest.json")
    print("=" * 78)
    if not verification:
        sys.exit(2)


if __name__ == "__main__":
    main()
