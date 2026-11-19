# Cache Line & False Sharing Mechanics: Analysis (Q1–Q5)

**Student:** Kenzhekhanuly Torezhan (230103252), Git handle `ItachiTore`
**Host:** 11th Gen Intel Core i5-11400H (Tiger Lake-H, 6 cores / 12 threads), Windows 11, Python 3.13.7, NumPy 2.3.5, Numba 0.68.1
**Run:** `python false_sharing_lab.py` → `cache_lab_manifest.json` (`verification_passed: true`), full stdout in `run_log.txt`
**INSTANCE_SEED:** `0x043e3d9b30826867` = SHA-256(`ItachiTore::AMD64::Intel64 Family 6 Model 141 Stepping 1, GenuineIntel::12`)[:16]

**Setup**
- Cache line size: 64 B, read via `GetLogicalProcessorInformation` (L1D descriptor).
- 4 threads, each pinned with `SetThreadAffinityMask` to logical CPUs `[0, 2, 4, 6]`. These are 4 distinct physical cores, so no two threads are SMT siblings sharing an L1D.
- 60,000,000 updates per thread. Each layout runs 5 times and the median is reported.
- The buffer is 128-byte aligned. That makes it 64-byte aligned, and it also starts on an even line, so each 128-B adjacent-line pair is lines {0,1}, {2,3}, …

All four strided layouts and the register-privatized baseline end with **bit-identical counters** `[7710513407, 7709910813, 7710882197, 7709985136]`. These match the reference values computed independently from the per-thread seeds.

---

## Q1. MESI state transitions under `stride = 1`

With `stride = 1`, `buffer[0]` (thread 0, core 0) and `buffer[1]` (thread 1, core 2) are 8 B apart in **cache line 0**. The manifest records `cache_lines_touched = [0]` for this layout. Below, *L* is that line. Core 0 does `buffer[0] = buffer[0] + δ`, then core 1 immediately does the same on `buffer[1]`.

| step | event | core 0's copy of L | core 1's copy of L |
|---|---|---|---|
| 0 | initial: both cores have read L earlier | **S** | **S** |
| 1 | core 0 executes `store buffer[0]`. The store enters core 0's store buffer and can't commit to L1 while L is S, so core 0 issues an RFO (read-for-ownership / invalidate) on the ring. | S → (pending) | **S → I** (snoop invalidates it) |
| 2 | Core 0 receives the invalidate acks and owns L. The buffered store commits. | **M** | I |
| 3 | core 1 executes `load buffer[1]` (the read half of its RMW). It misses in L1 because L is I. | M | I (miss) |
| 4 | core 1 executes `store buffer[1]` and issues an RFO for L. Core 0's snoop hits a **Modified** line, so it writes L back (or forwards it core-to-core via the L3/CHA) and invalidates its own copy. | **M → I** | **I → M** |
| 5 | core 0's next iteration stores `buffer[0]` again → RFO → core 1 goes M → I and core 0 goes I → M … | ping-pong | ping-pong |

The **E** state appears only when a single core misses on L while no other core holds it, for example the very first access to a freshly allocated buffer. With 4 writers, every ownership change is effectively **M → I** on the old owner and **I → M** on the new one. Each change is a full 64-B line transfer across the ring plus invalidation acknowledgements, typically tens of ns, even though the threads never touch the same 8-byte word.

**Why the RFO stalls core 1's store buffer.** x86 TSO requires stores to become globally visible in program order, and a store can only leave the store buffer when its line is in **M** or **E** in the local L1. While core 1's RFO for L is outstanding, the oldest store in its store buffer cannot drain, and every younger store queues behind it. Once the store buffer fills (~56 entries on Sunny/Willow Cove cores), the allocation stage stalls and the core stops issuing new µops. The RFO therefore turns a remote coherence round-trip into a local pipeline stall, and every time another core steals L, the next store pays it again.

**What this run shows about that mechanism.** In this pure read-modify-write kernel the stall is mostly *hidden*. Each core's next `load buffer[slot]` is satisfied by **store-to-load forwarding from its own store buffer**, so it needs no line at all. Then, whenever the line arrives in M, the queue of buffered stores to that line commits as a burst. The penalty is therefore paid once per line transfer, not once per update. That is why `stride = 1` is only 1.17× slower than `stride = 8` (Q2). The supplementary kernel adds a read of the *neighbour's* slot every iteration, which forwarding cannot satisfy. Each iteration then has to obtain L in at least S, and the full ping-pong appears: **1255.86 ms vs 432.34 ms = 2.91×**, at **20.9 ns per update** vs 7.2 ns.

---

## Q2. Measured durations, slowdowns, ns per update

Wall-clock time from `time.perf_counter_ns()` for 4 threads × 60,000,000 updates, median of 5 runs:

| layout | stride | lines touched | median (ns) | min (ns) | ns / update / thread | ns / update (aggregate, ÷ 4) | vol. ctx switches (5 runs) |
|---|---|---|---|---|---|---|---|
| A, contiguous | 1 (8 B) | [0] | **172,419,800** | 166,752,200 | **2.874** | 0.718 | 52, 53, 58, 65, 51 |
| B, half-line | 4 (32 B) | [0, 1] | **168,111,600** | 165,143,700 | **2.802** | 0.700 | 70, 50, 85, 58, 55 |
| C, line-padded | 8 (64 B) | [0, 1, 2, 3] | **147,759,400** | 147,482,500 | **2.463** | 0.616 | 53, 58, 78, 53, 52 |
| D, prefetch-safe | 16 (128 B) | [0, 2, 4, 6] | **148,064,700** | 147,631,000 | **2.468** | 0.617 | 51, 56, 54, 68, 47 |
| private register | n/a (1 store) | [0, 1, 2, 3] | **58,533,700** | 57,819,300 | **0.976** | 0.244 | 29, 25, 29, 26, 31 |

The exact nanosecond values are in `cache_lab_manifest.json` → `benchmarks.*.elapsed_ns_runs`. Involuntary context switches were 0 in every run. On Windows, psutil reports all switches as voluntary.

**Slowdown factors**
- `stride 1 / stride 8` = **1.167×**. False sharing costs **+0.41 ns per update per thread**: 2.874 − 2.463 ns, or 24.7 ms over the run.
- `stride 4 / stride 8` = **1.138×**. Two-way sharing captures most of the four-way penalty, because both cases still move one line per ownership change.
- `stride 8 / stride 16` = **0.998×**: no difference (see Q3).
- `stride 8 / private` = **2.524×** and `stride 1 / private` = **2.946×**.
- A single nogil thread on `stride = 1` takes 140.68 ms (2.345 ns per update). So 4 threads in the false-sharing layout take **1.226×** as long as 1 thread doing the same per-thread work, while padded layouts run at ~1.05×, near-perfect scaling.

**What the regimes separate**
- **Private (0.976 ns per update):** the accumulator lives in a register. Per iteration there is only the LCG `mul` + `add` + shift + add chain, about 4 cycles at the ~4–4.5 GHz turbo clock. No memory traffic.
- **Stride 8 / 16 (≈2.46 ns):** each iteration does a real `load` → `add` → `store` to an L1 line the core owns exclusively. The ~1.5 ns over the private baseline (≈ 6 cycles) is the **L1 store-to-load forwarding latency** on the loop-carried memory dependency. It is purely core-local, with no coherence traffic.
- **Stride 1 / 4 (2.80–2.87 ns):** the extra **0.34–0.41 ns** per update is the amortised **cross-core coherence latency**: RFO round-trips that the store buffer could not fully hide.
- **Supplementary neighbour-read kernel:** 6.51 ns (stride 16), 7.21 ns (stride 8) and **20.93 ns** (stride 1) per update. Here the coherence cost is no longer amortised: +13.7 ns per update relative to stride 8.

**Pinning matters.** The same 4-thread stride sweep without affinity masks gives `stride 1 / stride 8` = 266.99 / 148.36 ms = **1.80×**. When the OS migrates threads or places two of them on SMT siblings, the line also bounces between hyper-threads competing for one store-buffer and L1 port. Two threads on SMT siblings {0,1} with `stride 1` take 242.38 ms. The same two threads on distinct cores {0,2} take 153.70 ms (stride 1) and 142.16 ms (stride 8).

---

## Q3. 64-byte vs 128-byte spacing and the L2 adjacent-line prefetcher

Measured: `stride 8` = 147.76 ms and `stride 16` = 148.06 ms, so **stride 8 / stride 16 = 0.998×** in the main kernel, statistically identical. In the supplementary neighbour-read kernel, where every iteration has to pull a remote line: `stride 8` = 432.34 ms and `stride 16` = 390.41 ms, so **stride 8 / stride 16 = 1.107×**. That is about 10 % extra cost from pairing at 64-byte spacing.

**Mechanism.** Intel cores, including this Tiger Lake, have an **L2 spatial (adjacent-line) prefetcher**. It treats memory as 128-byte-aligned *pairs* of 64-byte lines. When L2 misses on one line, it also requests the "buddy" line, i.e. the line whose address differs only in bit 6 (`addr ^ 64`). Lines 2k and 2k+1 therefore travel together. Because our buffer is 128-B aligned:
- **stride 8:** threads 0 and 1 own lines 0 and 1 (one pair), threads 2 and 3 own lines 2 and 3 (another pair). Their *data* is in different lines, so the coherence protocol never truly conflicts, and `cache_lines_touched = [0, 1, 2, 3]`.
- **stride 16:** each thread owns lines 0, 2, 4, 6, one per 128-B pair, so the buddy of every owned line is unused.

**When 64-byte padding still suffers secondary invalidations.** If core 0 misses on line 0 (for example after an eviction, or because it read a neighbour's data), the prefetcher also fetches line 1 in **S** state. Line 1 is held **M** by core 2 (thread 1). That prefetch snoops core 2, forces a write-back, and leaves the line shared. Core 2's next store then needs another RFO to invalidate core 0's useless prefetched copy. These are *secondary* invalidations caused by speculative prefetching, not by any real access. They only occur when a thread keeps **missing** in its L2. In the main kernel each core keeps its line in M and never misses, so the prefetcher never fires and stride 8 equals stride 16. In the neighbour-read kernel, constant misses trigger buddy-line prefetches, and 128-byte spacing (stride 16) becomes measurably faster. The general rule: pad to 128 B (Intel's `std::hardware_destructive_interference_size` on many toolchains) whenever per-thread data sits next to anything that is shared or frequently missed.

---

## Q4. LLVM IR: register promotion vs real memory stores

`worker_strided_accumulate.inspect_llvm()` (core function saved to `worker_strided_accumulate.ll`, SHA-256 `f9e6c89c…`):
- Nopython signature `(Array(uint64, 1, 'C', False, aligned=True), int64, int64, uint64)`. There is no object-mode fallback, and **0** `PyObject`/`PyNumber`/`NRT_decref` calls in the core function.
- **4 × `store i64`** and **3 × `load i64`** in the function. The hot loop block `B182` (unrolled ×2, plus a `B182.prol` remainder block) contains a `load i64` and a `store i64` per iteration:

```llvm
B182:                                  ; preds = %B182.preheader1, %B182
  %idx.2.096 = phi i64 [ %.241.1, %B182 ], ...
  %.166  = mul i64 %x.2.095, %const2           ; LCG  x = x*a
  %.168  = add i64 %.166, %const               ;      + c
  %3     = getelementptr i64, ptr %arg.buffer.4, i64 %idx.2.096
  %.200  = getelementptr i64, ptr %3, i64 %.186
  %.201  = load i64, ptr %.200, align 8        ; read  buffer[idx]
  %.203  = lshr i64 %.168, 56
  %.204  = add i64 %.201, 1
  %.206  = add i64 %.204, %.203
  store i64 %.206, ptr %.200, align 8          ; write buffer[idx]  (every iteration)
  %.239  = lshr i64 %.206, 63                  ; new >> 63
  %.241  = add nsw i64 %.239, %arg.slot        ; next idx = slot + (new >> 63)
  ...
```

In contrast, `worker_private_accumulate` has **no `store i64` inside its loop** (`private_store_i64_in_loop = false`). Its only array store is the single `buffer[slot] = buffer[slot] + acc` after the loop.

**Why a compiler wants to hoist `buffer[slot]`.** In `for _: buffer[slot] += f(x)` the address is loop-invariant. LLVM's LICM / *scalar promotion* pass sees that nothing else in the loop can alias `buffer[slot]` (there are no calls and no other stores). It is then allowed to load the value once before the loop, keep it in a register, and store it once after the loop. That is legal for a single thread because Python/Numba has no `volatile`/atomic semantics, and the result is identical, exactly the code of `worker_private_accumulate`. It would also erase the experiment: one store per thread means no false sharing.

**How the kernel prevents promotion.** The index for the next iteration is computed from the value just stored: `idx = slot + (new >> 63)`. The counter never reaches 2^63, so `idx == slot` always, but LLVM cannot prove that. The address of the next load therefore depends on data produced by the previous store (the `phi %idx.2.096 ← %.241`), so it is no longer loop-invariant and scalar promotion's precondition fails. LLVM must emit a real `load` from and `store` to `buffer[idx]` every iteration, and the IR above shows it did, even after unrolling. The timings confirm it: **2.463 ns per update** (stride 8) vs **0.976 ns** (private), a 2.52× difference that is exactly the per-iteration L1 load/store round trip.

---

## Q5. CPython GIL masking vs `nogil=True`

Same `stride = 1` layout (all threads write `buffer[0..3]` in one line):

| variant | 1 thread | 4 threads | 4t / 1t | ns / update (aggregate) 1t → 4t | vol. ctx switches 1t → 4t |
|---|---|---|---|---|---|
| pure CPython bytecode (2 M updates per thread) | 591.85 ms | 2405.72 ms | **4.07×** | 295.9 → 300.7 | 65 → 767 |
| Numba `@njit` **without** `nogil` (60 M per thread) | 144.17 ms | 572.93 ms | **3.97×** | 2.403 → 2.387 | 25 → 193 |
| Numba `@njit(nogil=True)` (60 M per thread) | 140.68 ms | 172.42 ms | **1.23×** | 2.345 → 0.718 | n/a → 52 |

**Why bytecode threads show no false sharing.** CPython lets only the thread holding the GIL execute bytecode. The four threads therefore never execute their `buffer[slot] = …` stores *simultaneously*. They run one after another in time slices: the switch interval is 5 ms on 3.13, or until a blocking call. Each slice performs thousands of updates on one core. When the GIL moves to another core, the line moves with it *once*, which is a single RFO amortised over ~17,000 updates (5 ms ÷ 296 ns). Cache-line contention is invisible. Total work is just serialised: 4 threads × 2 M updates take 4.07× as long as 1 × 2 M, with the same ~296–301 ns per update. That cost is interpreter overhead (bytecode dispatch, boxing Python ints, NumPy scalar `__setitem__`), about 120× slower than the native 2.4 ns.

**Why it gets slower with more threads.** Wall time grows linearly (4.07×) instead of staying flat, and per-update cost even rises slightly, from 295.9 to 300.7 ns. Voluntary context switches jump from 65 to **767**. Every GIL hand-off costs a mutex/condition-variable signal, an OS context switch and a cold-cache restart on the new core. Threads that time out waiting set `eval_breaker` and force the holder to drop the GIL. The bottleneck is **software lock contention on one global mutex**, and adding threads only adds hand-off overhead.

The Numba kernel compiled **without** `nogil` makes the same point even more clearly. It is native machine code (2.40 ns per update), but because the GIL is held for the whole call, 4 threads take **3.97×** as long as 1. The threads run strictly back-to-back, and again no two stores are ever concurrent.

**With `nogil=True`** the kernel releases the GIL on entry, and the four threads run truly simultaneously on cores 0/2/4/6. Total time is only 1.23× the single-thread time for 4× the work (aggregate 0.718 ns per update vs 2.345). The **only** thing that keeps it from perfect 1.0× scaling is the **hardware cache-line contention** measured in Q2: padded layouts reach 147.76 ms ≈ 1.05× of a single thread. GIL contention serialises the *threads*. Cache-line contention serialises only the *line*, and only when the cores really do execute concurrently, which `nogil=True` is required to expose.
