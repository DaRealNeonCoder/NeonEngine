# MCTS-VK v3: limitations resolved, timing vs MCTS-NC

All numbers: RTX 3050 Laptop (4 GB), driver 610.60, 2026-11-05. Connect Four, T = 64 trees,
m = 32 playouts per child, C = 1.4, 200 steps, 2 warm-ups, 10 timed reps (protocol of
`MCTS-NC/src/bench_mctsnc.py`; MCTS-VK side: `MctsBench ncbench`). NC and VK were run
alternately (NC, VK, NC, VK) in one session; wall ms is the mean of the two rounds.

## Empty board

| Variant | VK wall (ms) | NC wall (ms) | Speed-up (NC / VK) |
|---|---:|---:|---:|
| ACP-prodigal | 13.5 | 476 | 35x |
| ACP-thrifty  | 14.8 | 729 | 49x |
| OCP-prodigal | 6.1  | 284 | 47x |
| OCP-thrifty  | 7.7  | 564 | 73x |

Midgame: ACP-prodigal 10.3 vs 459 ms, ACP-thrifty 11.5 vs 711, OCP-prodigal 5.4 vs 298,
OCP-thrifty 7.3 vs 582. VK run-to-run sd is 0.1-0.35 ms; NC sd is 9-124 ms.

The old table (Table 4) had ACP-prodigal at 710 ms and ACP-thrifty at 772 ms for VK.

## Limitations

| Limitation (paper) | v3 status |
|---|---|
| Serial expansion | Parallel since v2 (one 8-thread group per tree, one thread per action). v3 fixes a bug in it, below. |
| Thrifty overhead | No host round trip (GPU-side compaction + `vkCmdDispatchIndirect`). Batch command buffers are recorded once and replayed; per-search recording fell from ~2.4 ms to ~0.07 ms. Wall - GPU time is now ~0.6 ms (4-9%). Thrifty remains ~10% slower than prodigal on the GPU at T = 64 (expand atomics). |
| Host-visible memory and atomics | Every buffer the kernels touch is device-local; root stats and counters are copied to a host buffer once per submit. Counter cost, interleaved paired reps (midgame, 200 steps): ACP-prodigal +1.1% (+-0.9), ACP-thrifty -0.5% (+-2.1), OCP-prodigal +1.8% (+-1.4), OCP-thrifty +2.2% (+-1.8). |
| Board representation | Bitboards since v2 (child record 20 B instead of 168 B). v3 picks each playout move straight from the "next free cell" mask: ~13% less playout time, identical results. |
| Warp divergence | Measured lane efficiency 0.46-0.54 (counter `lane_eff`). v3 can pack S children into one 32-lane group with lanes taking the next playout when they finish: lane efficiency 0.89 at S = 8. This only helps once the grid oversubscribes the GPU: at T = 1024, S = 4 cuts ACP-prodigal time by 24%; at T = 256 by 14%; at T = 64 it is slower (fewer warps). Default `slotsPerGroup = 0` picks S from the grid size. Results do not depend on S (checked bit for bit). |
| Statistical power | Harness plays 100 colour-swapped opening pairs and bootstraps over openings (`analyze_matches.py`). Not run yet for v3 (~1.5-2 h of GPU time). `MAX_ITERS` raised to 10000 so OCP is not capped at 200 ms/move. |
| Reproducibility | Every RNG stream depends only on (cfg seed, call seed, iteration index). Time-limited searches replay exactly with a fixed iteration count (10/10 per variant), and repeated searches are identical. |

## Bug fixed in v3

ACP expansion could create a child for a full column: in some expansions 7 threads wrote a
child while the group's legal-move count was 6. The SPIR-V from Slang is correct; the driver
appears to miscompile `uint64_t(1) << (th * 7 + 5)` (64-bit shift by a run-time amount). The
corrupted trees made ACP-prodigal non-deterministic for searches of more than ~100 iterations
and made it differ from ACP-thrifty. v3 has no 64-bit shift by a run-time amount; legal moves
are computed once per group with constant shifts. After the fix, prodigal and thrifty with the
same slot layout agree bit for bit for 300 iterations.

## Checks (`MctsBench sanity verify determinism replay bias counters ncbench`)

- sanity: 12/12 tactical positions correct, no anomalies, visits identical to v2
- verify: 252/252 searches identical across slots/group 1-8
- determinism: identical repeats for 10-300 iterations, all variants
- replay: 10/10 per variant
- bias: mirror test and empty-board symmetry ok
