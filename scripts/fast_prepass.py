"""Fast, bit-exact replacement for the sequential pre-pass of build_spectral_agg_par.py (--rng-mode stream).

The original pre-pass replays every node's random walks in pure Python with one numpy Generator
(PCG64) only to learn the generator state at the start of each chunk. Here:

  1. raw 64-bit PCG64 outputs are drawn in bulk with bit_generator.random_raw() (vectorised);
  2. a small C kernel consumes them exactly as numpy's Generator.integers(deg) would
     (Lemire bounded draw on 32-bit values, PCG64's 32-bit buffering, no draw when deg == 1)
     while following the same walks as walk_draws(), and records the 32-bit stream position at
     every chunk start;
  3. each chunk's state is rebuilt with PCG64.advance() plus the has_uint32/uinteger buffer.

The result is the same list of states the Python pre-pass produces; check_states() compares the two
on a prefix so this can be verified on every run.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_C_SRC = r"""
#include <stdint.h>

/* numpy: buffered_bounded_lemire_uint32 with rng = deg - 1 (deg >= 2) */
#define LEMIRE(OUTVAR, DEG)                                                     \
  {                                                                             \
    uint32_t rng_excl = (uint32_t)(DEG);                                        \
    if (pos >= len) { need_more = 1; break; }                                   \
    uint64_t m = (uint64_t)u32[pos++] * rng_excl;                               \
    uint32_t leftover = (uint32_t)m;                                            \
    if (leftover < rng_excl) {                                                  \
      uint32_t threshold = (uint32_t)(UINT32_MAX - (rng_excl - 1)) % rng_excl;  \
      while (leftover < threshold) {                                            \
        if (pos >= len) { need_more = 1; break; }                               \
        m = (uint64_t)u32[pos++] * rng_excl;                                    \
        leftover = (uint32_t)m;                                                 \
      }                                                                         \
      if (need_more) break;                                                     \
    }                                                                           \
    OUTVAR = (int64_t)(m >> 32);                                                \
  }

#define KERNEL(NAME, IPT, IXT)                                                                     \
int64_t NAME(const IPT *sip, const IXT *six, int64_t start_node, int64_t limit,                     \
             const uint32_t *u32, int64_t len, int64_t *pos_io,                                    \
             int64_t walks, int64_t m_max, int64_t *stamp,                                          \
             const int64_t *bounds, int64_t nb, int64_t *bidx_io,                                  \
             int64_t base, int64_t *rec_pos, uint32_t *rec_val, int64_t *epoch_io)                  \
{                                                                                                  \
  int64_t pos = *pos_io, bidx = *bidx_io, node, epoch = *epoch_io;                                 \
  for (node = start_node; node < limit; ++node) {                                                  \
    int64_t node_pos = pos;                                                                        \
    int need_more = 0;                                                                             \
    /* chunk boundary: record global 32-bit position and the value that would be buffered */     \
    int64_t save_bidx = bidx;                                                                      \
    while (bidx < nb && bounds[bidx] == node) {                                                    \
      rec_pos[bidx] = base + pos;                                                                  \
      rec_val[bidx] = (pos < len) ? u32[pos] : 0;                                                  \
      if (((base + pos) & 1) && pos >= len) { need_more = 1; break; }                              \
      ++bidx;                                                                                      \
    }                                                                                              \
    if (need_more) { bidx = save_bidx; break; }                                                    \
    int64_t chosen = 1, w, step, current;                                                          \
    ++epoch;                 /* fresh 'seen' set for every attempt (an undone attempt leaves marks) */ \
    stamp[node] = epoch;                                                                           \
    for (w = 0; w < walks; ++w) {                                                                  \
      current = node;                                                                              \
      for (step = 0; step < 3; ++step) {                                                           \
        int64_t lo = (int64_t)sip[current], hi = (int64_t)sip[current + 1];                        \
        if (hi <= lo) break;                                                                       \
        int64_t deg = hi - lo, r = 0;                                                              \
        if (deg > 1) LEMIRE(r, deg)                                                                \
        current = (int64_t)six[lo + r];                                                            \
        if (stamp[current] != epoch) {                                                             \
          stamp[current] = epoch;                                                                  \
          ++chosen;                                                                                \
          if (chosen >= m_max) break;                                                              \
        }                                                                                          \
      }                                                                                            \
      if (need_more || chosen >= m_max) break;                                                     \
    }                                                                                              \
    if (need_more) {         /* ran out of random numbers inside this node: undo and ask for more */ \
      pos = node_pos; bidx = save_bidx;                                                            \
      for (int64_t j = bidx; j < nb && bounds[j] == node; ++j) rec_pos[j] = -1;                    \
      break;                                                                                       \
    }                                                                                              \
  }                                                                                                \
  *pos_io = pos; *bidx_io = bidx; *epoch_io = epoch;                                               \
  return node;                                                                                     \
}

KERNEL(kern_64_64, int64_t, int64_t)
KERNEL(kern_64_32, int64_t, int32_t)
KERNEL(kern_32_64, int32_t, int64_t)
KERNEL(kern_32_32, int32_t, int32_t)
KERNEL(kern_64_u32, int64_t, uint32_t)
KERNEL(kern_u32_u32, uint32_t, uint32_t)
"""

_TYPES = {np.dtype(np.int64): "64", np.dtype(np.int32): "32", np.dtype(np.uint32): "u32"}
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    h = hashlib.sha1(_C_SRC.encode()).hexdigest()[:12]
    cache = Path(os.environ.get("FAST_PREPASS_CACHE", Path(__file__).resolve().parent / ".fast_prepass_build"))
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"fast_prepass_{h}.so"
    if not so.exists():
        src = cache / f"fast_prepass_{h}.c"
        src.write_text(_C_SRC)
        tmp = Path(tempfile.mkstemp(dir=cache, suffix=".so")[1])
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-O3", "-std=c99", "-shared", "-fPIC", str(src), "-o", str(tmp)], check=True)
        os.replace(tmp, so)
    _lib = ctypes.CDLL(str(so))
    return _lib


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def chunk_states(sip, six, bounds, walks, m_max, seed, block_u64=None):
    """States of np.random.default_rng(seed) at the start of each chunk [bounds[k], bounds[k+1]).

    Equivalent to the Python pre-pass in build_spectral_agg_par.py:
        rng = default_rng(seed); for each chunk: record rng state, then walk_draws(...) every node.
    """
    lib = _load()
    sip = np.asarray(sip)
    six = np.asarray(six)
    if sip.dtype not in _TYPES or six.dtype not in _TYPES:
        sip = sip.astype(np.int64, copy=False) if sip.dtype not in _TYPES else sip
        six = six.astype(np.int64, copy=False) if six.dtype not in _TYPES else six
    sip = np.ascontiguousarray(sip)
    six = np.ascontiguousarray(six)
    name = f"kern_{_TYPES[sip.dtype]}_{_TYPES[six.dtype]}"
    if not hasattr(lib, name):          # unusual combination: widen to int64
        sip = sip.astype(np.int64)
        six = six.astype(np.int64)
        name = "kern_64_64"
    kern = getattr(lib, name)
    kern.restype = ctypes.c_int64

    starts = np.ascontiguousarray(np.asarray(bounds[:-1], dtype=np.int64))
    limit = int(bounds[-1])
    nb = len(starts)
    rec_pos = np.full(nb, -1, dtype=np.int64)
    rec_val = np.zeros(nb, dtype=np.uint32)
    n_nodes = len(sip) - 1
    stamp = np.zeros(max(n_nodes, limit), dtype=np.int64)

    if block_u64 is None:                # each node draws at most 3*walks 32-bit values (+ rare rejections)
        block_u64 = int(min(1 << 24, max(1 << 14, limit * walks * 3 // 2 // 8)))
    gen = np.random.default_rng(seed)
    bg = gen.bit_generator
    buf = np.empty(0, dtype=np.uint32)   # unconsumed 32-bit values; buf[0] is global position `base`
    base = 0
    node = 0
    pos = ctypes.c_int64(0)
    bidx = ctypes.c_int64(0)
    epoch = ctypes.c_int64(0)
    while node < limit:
        new = np.asarray(bg.random_raw(block_u64), dtype=np.uint64)
        lo = (new & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        hi = (new >> np.uint64(32)).astype(np.uint32)
        fresh = np.empty(2 * len(new), dtype=np.uint32)
        fresh[0::2] = lo                 # PCG64 next_uint32: low half first, then the buffered high half
        fresh[1::2] = hi
        buf = np.ascontiguousarray(np.concatenate([buf, fresh]))
        pos.value = 0
        node = int(kern(_ptr(sip), _ptr(six), ctypes.c_int64(node), ctypes.c_int64(limit),
                        _ptr(buf), ctypes.c_int64(len(buf)), ctypes.byref(pos),
                        ctypes.c_int64(int(walks)), ctypes.c_int64(int(m_max)), _ptr(stamp),
                        _ptr(starts), ctypes.c_int64(nb), ctypes.byref(bidx),
                        ctypes.c_int64(base), _ptr(rec_pos), _ptr(rec_val), ctypes.byref(epoch)))
        base += pos.value
        buf = buf[pos.value:].copy()
    if (rec_pos < 0).any():
        raise RuntimeError("fast pre-pass: some chunk starts were not reached")

    states = []
    for p, v in zip(rec_pos.tolist(), rec_val.tolist()):
        g = np.random.default_rng(seed)
        g.bit_generator.advance((p + 1) // 2)        # number of 64-bit outputs drawn so far
        st = g.bit_generator.state
        if p % 2:                                    # high half of the last 64-bit output still buffered
            st["has_uint32"], st["uinteger"] = 1, int(v)
        else:
            st["has_uint32"], st["uinteger"] = 0, 0
        states.append(st)
    return states


def python_states(sip, six, bounds, walks, m_max, seed, walk_draws, n_chunks=None):
    """The original sequential pre-pass (optionally only the first n_chunks chunk starts)."""
    import copy
    rng = np.random.default_rng(seed)
    out = []
    nb = len(bounds) - 1 if n_chunks is None else min(n_chunks, len(bounds) - 1)
    for k in range(nb):
        out.append(copy.deepcopy(rng.bit_generator.state))
        if k == nb - 1 and n_chunks is not None:
            break
        for node in range(bounds[k], bounds[k + 1]):
            walk_draws(node, sip, six, walks, m_max, rng)
    return out


def same_state(a, b):
    return (a["state"]["state"] == b["state"]["state"] and a["state"]["inc"] == b["state"]["inc"]
            and int(a["has_uint32"]) == int(b["has_uint32"])
            and (not a["has_uint32"] or int(a["uinteger"]) == int(b["uinteger"])))