/* moe_scheduler.c
 * --------------------------------------------------------------------------
 * Full C port of Idea_Model/analytical_scheduler.py.
 *
 * This is a 1:1 functional translation — every search branch is preserved:
 *   - n==1  : Method-A (SOLO C2/C3 + SPLIT at `now`) + Method-B (early start
 *             on idle side, enumerate busy-side bw_change_pts)
 *   - n>=2 both_idle:
 *       * PAIR(top0, topK), K in {1,2,3}, both directions
 *       * PAIR(topK, topJ), K<J, K in {1..2}, J in {K+1..3}, both directions
 *         (S3 enumerated only over {B,C} to bound work, matches Python)
 *       * SPLIT(top0) with full cut set { ceil(n/2), floor(n/2),
 *                                          M_dim of every shape, n - M_dim }
 *       * WAIT-PAIR : send topK first SOLO, leave the other side waiting,
 *         next iter is both_idle at t_K and re-evaluates everything
 *       * Recursive _sim1 lookahead when only 1 expert remains
 *       * _greedy_heuristic LB for cost when >=2 experts remain
 *   - n>=2 one cluster busy:
 *       * SOLO top0 on idle side with early-start enumeration
 *
 * Snapshot mirrors Python FourStageSnap including prefetch fields.
 * cache_hit() uses pf_eid + pf_end semantics (NOT cur_eid) so that the
 * initial residency of `make_initial_snap` is honoured exactly like Python.
 *
 * Output: a moe_schedule_t whose tasks are dispatched to clusters in order.
 *         A post-pass fills advisory prefetch_eid hints based on the
 *         same-cluster next-task-with-different-expert lookup.  Current
 *         analytical timing does not assume those hints have already turned a
 *         later task into a cache hit.
 * --------------------------------------------------------------------------
 */
#include "moe_scheduler.h"

#include <stddef.h>
#include <stdint.h>

/* ============================================================
 *  Physical constants (must match four_stage_scheduler.py)
 * ============================================================ */
#define WBYTES_S1   2883584u   /* 2 * 2048 * 1408 * 0.5B (INT4 packed gate+up) */
#define WBYTES_S3   1441792u   /* 1 * 2048 * 1408 * 0.5B (INT4 packed down)    */
#define MAX_BW           128u  /* B/cc                              */
#define CACHE_S1_READY     1u  /* S1 gate/up already resident       */
#define CACHE_S3_READY     2u  /* S3 down already resident          */
#define INIT_CACHE_C2      1u
#define INIT_CACHE_C3      2u
#define EXACT_TAIL_MAX_TOKENS 4u

typedef struct {
    uint16_t m_dim;
    uint16_t bw_req;
    uint16_t alloc;
    uint16_t _pad;
    uint32_t T_s1;
    uint32_t T_s3;
    uint32_t t_dma_s1;
    uint32_t t_dma_s3;
} shape_def_t;

/*
 * Shape fields:
 *   M_dim : the first S1/S3 iteration consumes this many tokens.
 *   bw_req: ideal DMA bandwidth that makes DMA and compute finish together.
 *   alloc : actual reserved DMA bandwidth slot under the hardware allocator.
 */
static const shape_def_t SHAPE[3] = {
    /* A */ {8,  32, 64,  0, 90112u, 45056u, 45056u, 22528u},
    /* B */ {4,  64, 64,  0, 45056u, 22528u, 45056u, 22528u},
    /* C */ {2, 128,128,  0, 22528u, 11264u, 22528u, 11264u},
};
#define N_SHAPES 3
#define SHAPE_C_M_DIM 2u

/* ============================================================
 *  Helpers
 * ============================================================ */
static inline uint32_t udiv_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}
static inline uint32_t u32_min(uint32_t a, uint32_t b) { return a < b ? a : b; }
static inline uint32_t u32_max(uint32_t a, uint32_t b) { return a > b ? a : b; }

static inline uint32_t best_s2_compute(uint32_t remaining) {
    if (remaining == 0) return 0;
    return udiv_ceil(remaining, SHAPE_C_M_DIM) * SHAPE[MOE_SHAPE_C].T_s1;
}
static inline uint32_t best_s4_compute(uint32_t remaining) {
    if (remaining == 0) return 0;
    return udiv_ceil(remaining, SHAPE_C_M_DIM) * SHAPE[MOE_SHAPE_C].T_s3;
}

static uint32_t task_time_for_shapes(uint32_t ntok, int s1, int s3) {
    uint32_t rem_s2 = (ntok > SHAPE[s1].m_dim) ? (ntok - SHAPE[s1].m_dim) : 0u;
    uint32_t rem_s4 = (ntok > SHAPE[s3].m_dim) ? (ntok - SHAPE[s3].m_dim) : 0u;
    return SHAPE[s1].T_s1 + best_s2_compute(rem_s2)
         + SHAPE[s3].T_s3 + best_s4_compute(rem_s4);
}

/* Best per-cluster task time bounds (Python _best_task_time / _best_concurrent_task_time) */
static uint32_t best_task_time(uint32_t ntok) {
    /* Python _best_task_time: S1 and S3 shapes are optimized independently. */
    uint32_t best = UINT32_MAX;
    for (int s1 = 0; s1 < N_SHAPES; s1++)
    for (int s3 = 0; s3 < N_SHAPES; s3++) {
        uint32_t t = task_time_for_shapes(ntok, s1, s3);
        if (t < best) best = t;
    }
    return best;
}
static uint32_t best_concurrent_task_time(uint32_t ntok) {
    /* Python _best_concurrent_task_time: S1/S3 independently choose among
     * shapes that can coexist with another cluster under MAX_BW. */
    uint32_t best = UINT32_MAX;
    for (int s1 = 0; s1 < N_SHAPES; s1++) {
        if (SHAPE[s1].alloc * 2u > MAX_BW) continue;
        for (int s3 = 0; s3 < N_SHAPES; s3++) {
            if (SHAPE[s3].alloc * 2u > MAX_BW) continue;
            uint32_t t = task_time_for_shapes(ntok, s1, s3);
            if (t < best) best = t;
        }
    }
    return best;
}

static moe_shape_t best_solo_shape_s1(uint32_t ntok) {
    uint32_t best = UINT32_MAX;
    moe_shape_t best_shape = MOE_SHAPE_A;
    for (int s = 0; s < N_SHAPES; s++) {
        uint32_t rem = (ntok > SHAPE[s].m_dim) ? (ntok - SHAPE[s].m_dim) : 0u;
        uint32_t t = SHAPE[s].T_s1 + best_s2_compute(rem);
        if (t < best) { best = t; best_shape = (moe_shape_t)s; }
    }
    return best_shape;
}

static moe_shape_t best_solo_shape_s3(uint32_t ntok) {
    uint32_t best = UINT32_MAX;
    moe_shape_t best_shape = MOE_SHAPE_A;
    for (int s = 0; s < N_SHAPES; s++) {
        uint32_t rem = (ntok > SHAPE[s].m_dim) ? (ntok - SHAPE[s].m_dim) : 0u;
        uint32_t t = SHAPE[s].T_s3 + best_s4_compute(rem);
        if (t < best) { best = t; best_shape = (moe_shape_t)s; }
    }
    return best_shape;
}

static moe_shape_t best_conc_shape_s1(uint32_t ntok) {
    uint32_t best = UINT32_MAX;
    moe_shape_t best_shape = MOE_SHAPE_A;
    for (int s = 0; s < N_SHAPES; s++) {
        if (SHAPE[s].alloc > MAX_BW / 2u) continue;
        uint32_t rem = (ntok > SHAPE[s].m_dim) ? (ntok - SHAPE[s].m_dim) : 0u;
        uint32_t t = SHAPE[s].T_s1 + best_s2_compute(rem);
        if (t < best) { best = t; best_shape = (moe_shape_t)s; }
    }
    return best_shape;
}

static moe_shape_t best_conc_shape_s3(uint32_t ntok) {
    uint32_t best = UINT32_MAX;
    moe_shape_t best_shape = MOE_SHAPE_A;
    for (int s = 0; s < N_SHAPES; s++) {
        if (SHAPE[s].bw_req > MAX_BW / 2u) continue;
        uint32_t rem = (ntok > SHAPE[s].m_dim) ? (ntok - SHAPE[s].m_dim) : 0u;
        uint32_t t = SHAPE[s].T_s3 + best_s4_compute(rem);
        if (t < best) { best = t; best_shape = (moe_shape_t)s; }
    }
    return best_shape;
}

/* ============================================================
 *  Cluster state snapshot — mirrors Python FourStageSnap
 *  (including prefetch fields)
 * ============================================================ */
typedef struct {
    uint32_t task_start;
    uint32_t task_end;
    uint32_t dma1_end;
    uint32_t s1_end;
    uint32_t s2_end;
    uint32_t dma3_end;
    uint32_t s3_end;
    /* prefetch state (Python pf_start/pf_end/pf_eid/pf_bw) */
    int32_t  pf_start;     /* -1 = none */
    int32_t  pf_end;       /* -1 = none, 0 = ready since t=0 (initial cache) */
    int16_t  pf_eid;       /* -1 = none */
    uint16_t pf_bw;
    uint8_t  pf_full;      /* initial resident cache, not S1-only prefetch */
    uint16_t bw_s1;
    uint16_t bw_s3;
    int16_t  cur_eid;      /* -1 = idle */
    /* schedule output fields (for emit) */
    uint8_t  shape_s1;
    uint8_t  shape_s3;
    uint16_t ntok;
    uint8_t  skip_dma_s1;
    uint8_t  skip_dma_s3;
    int32_t  s2pf_start;    /* S2 down prefetch start, -1 = none */
    int32_t  s2pf_end;
    uint16_t s2pf_bw;
    uint8_t  is_wait;      /* 1 = synthetic waiting snap (no actual task) */
} snap_t;

static void snap_make_initial(snap_t *s, int16_t cached_eid) {
    /* Mirrors make_initial_snap(): if cached_eid >= 0, set pf_eid=cached_eid
     * with pf_end=0 (already loaded at t=0). cur_eid=-1 always. */
    s->task_start = 0;
    s->task_end = 0;
    s->dma1_end = 0;
    s->s1_end = 0;
    s->s2_end = 0;
    s->dma3_end = 0;
    s->s3_end = 0;
    s->bw_s1 = 0;
    s->bw_s3 = 0;
    s->cur_eid = -1;
    s->shape_s1 = 0;
    s->shape_s3 = 0;
    s->ntok = 0;
    s->skip_dma_s1 = 0;
    s->skip_dma_s3 = 0;
    s->s2pf_start = -1;
    s->s2pf_end = -1;
    s->s2pf_bw = 0;
    s->is_wait = 0;
    if (cached_eid >= 0) {
        s->pf_start = -1;
        s->pf_end   = 0;          /* already cached at t=0 */
        s->pf_eid   = cached_eid;
        s->pf_bw    = 0;
        s->pf_full  = 1;
    } else {
        s->pf_start = -1;
        s->pf_end   = -1;
        s->pf_eid   = -1;
        s->pf_bw    = 0;
        s->pf_full  = 0;
    }
}

/* Build a snap representing assigning task <eid, ntok> at time `start`,
 * with shapes (s1, s3). Mirrors FourStageSnap.from_assign — note that
 * pf_* is RESET to "none" (Python from_assign also does this). */
static void snap_assign(snap_t *out,
                        uint32_t start,
                        moe_shape_t s1, moe_shape_t s3,
                        uint16_t ntok, int16_t eid, uint8_t cache_flags) {
    const shape_def_t *S1 = &SHAPE[s1];
    const shape_def_t *S3 = &SHAPE[s3];
    uint8_t s1_cached = (cache_flags & CACHE_S1_READY) ? 1u : 0u;
    uint8_t s3_cached = (cache_flags & CACHE_S3_READY) ? 1u : 0u;
    uint32_t rem_s2  = (ntok > S1->m_dim) ? ((uint32_t)ntok - (uint32_t)S1->m_dim) : 0u;
    uint32_t rem_s4  = (ntok > S3->m_dim) ? ((uint32_t)ntok - (uint32_t)S3->m_dim) : 0u;
    uint32_t gate_end = start + S1->T_s1 + best_s2_compute(rem_s2);
    uint32_t down_end = gate_end + S3->T_s3 + best_s4_compute(rem_s4);
    out->task_start  = start;
    out->s1_end      = s1_cached ? gate_end : (start + S1->T_s1);
    out->s2_end      = gate_end;
    out->s3_end      = s3_cached ? down_end : (out->s2_end + S3->T_s3);
    out->task_end    = down_end;
    out->dma1_end    = s1_cached ? start : (start + S1->t_dma_s1);
    out->dma3_end    = s3_cached ? out->s2_end : (out->s2_end + S3->t_dma_s3);
    out->bw_s1       = s1_cached ? 0 : S1->alloc;
    out->bw_s3       = s3_cached ? 0 : S3->alloc;
    out->cur_eid     = eid;
    out->shape_s1    = (uint8_t)s1;
    out->shape_s3    = (uint8_t)s3;
    out->ntok        = ntok;
    out->skip_dma_s1 = s1_cached;
    out->skip_dma_s3 = s3_cached;
    out->s2pf_start  = -1;
    out->s2pf_end    = -1;
    out->s2pf_bw     = 0;
    out->is_wait     = 0;
    /* prefetch fields cleared (Python parity) */
    out->pf_start    = -1;
    out->pf_end      = -1;
    out->pf_eid      = -1;
    out->pf_bw       = 0;
    out->pf_full     = 0;
}

/* Build a synthetic "waiting" snap at absolute time t — used by WAIT-PAIR
 * to represent "this cluster will be idle and ready at time t". */
static void snap_wait(snap_t *out, uint32_t t) {
    out->task_start  = t;
    out->task_end    = t;
    out->dma1_end    = t;
    out->s1_end      = t;
    out->s2_end      = t;
    out->dma3_end    = t;
    out->s3_end      = t;
    out->bw_s1       = 0;
    out->bw_s3       = 0;
    out->cur_eid     = -1;
    out->shape_s1    = 0;
    out->shape_s3    = 0;
    out->ntok        = 0;
    out->skip_dma_s1 = 0;
    out->skip_dma_s3 = 0;
    out->s2pf_start  = -1;
    out->s2pf_end    = -1;
    out->s2pf_bw     = 0;
    out->is_wait     = 1;
    out->pf_start    = -1;
    out->pf_end      = -1;
    out->pf_eid      = -1;
    out->pf_bw       = 0;
    out->pf_full     = 0;
}

/* Active BW at time t for a snap (mirrors Python active_bw_at). */
static inline uint32_t snap_active_bw(const snap_t *s, uint32_t t) {
    uint32_t bw = 0;
    if (s->cur_eid >= 0 && t < s->task_end) {
        if (t >= s->task_start && t < s->dma1_end)      bw = s->bw_s1;
        else if (t >= s->s2_end && t < s->dma3_end)     bw = s->bw_s3;
    }
    if (s->pf_start >= 0 && (int32_t)t >= s->pf_start && (int32_t)t < s->pf_end)
        bw += s->pf_bw;
    if (s->s2pf_start >= 0 && (int32_t)t >= s->s2pf_start && (int32_t)t < s->s2pf_end)
        bw += s->s2pf_bw;
    return bw;
}

/* BW change points for a snap (max 11 with S2/S4 prefetch). */
static int snap_bw_change_pts(const snap_t *s, uint32_t *out) {
    int n = 0;
    out[n++] = s->task_start;
    out[n++] = s->dma1_end;
    out[n++] = s->s1_end;
    out[n++] = s->s2_end;
    out[n++] = s->dma3_end;
    out[n++] = s->s3_end;
    out[n++] = s->task_end;
    if (s->pf_start >= 0) {
        out[n++] = (uint32_t)s->pf_start;
        out[n++] = (uint32_t)s->pf_end;
    }
    if (s->s2pf_start >= 0) {
        out[n++] = (uint32_t)s->s2pf_start;
        out[n++] = (uint32_t)s->s2pf_end;
    }
    return n;
}

/* Two-cluster BW feasibility: at every change-point window the sum of
 * active BWs must not exceed MAX_BW. Mirrors Python bw_feasible(). */
static int bw_feasible(const snap_t *a, const snap_t *b) {
    uint32_t pts[22];
    int na = snap_bw_change_pts(a, pts);
    int nb = snap_bw_change_pts(b, pts + na);
    int n  = na + nb;
    /* insertion sort */
    for (int i = 1; i < n; i++) {
        uint32_t v = pts[i]; int j = i - 1;
        while (j >= 0 && pts[j] > v) { pts[j+1] = pts[j]; j--; }
        pts[j+1] = v;
    }
    for (int i = 0; i + 1 < n; i++) {
        if (pts[i] == pts[i+1]) continue;
        uint32_t t = pts[i];
        uint32_t bw = snap_active_bw(a, t) + snap_active_bw(b, t);
        if (bw > MAX_BW) return 0;
    }
    return 1;
}

#define S2PF_NO_START UINT32_MAX
#define S2PF_MAX_CANDS 64

static void add_unique_start(uint32_t *out, int *n, uint32_t value) {
    for (int i = 0; i < *n; i++) {
        if (out[i] == value) return;
    }
    if (*n < S2PF_MAX_CANDS) out[(*n)++] = value;
}

static void sort_starts(uint32_t *v, int n) {
    for (int i = 1; i < n; i++) {
        uint32_t x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
        v[j + 1] = x;
    }
}

static int collect_s2_down_prefetch_starts(const snap_t *sn, moe_shape_t s3,
                                           const snap_t *peer, uint32_t *out) {
    int n = 0;
    if (sn->cur_eid < 0 || sn->bw_s3 == 0) return 0;
    uint32_t dma = SHAPE[s3].t_dma_s3;
    if (sn->s2_end < sn->task_start + dma) return 0;
    uint32_t lo = sn->task_start;
    uint32_t hi = sn->s2_end - dma;
    add_unique_start(out, &n, lo);
    add_unique_start(out, &n, hi);
    if (sn->dma1_end >= lo && sn->dma1_end <= hi) add_unique_start(out, &n, sn->dma1_end);
    if (sn->s1_end >= lo && sn->s1_end <= hi) add_unique_start(out, &n, sn->s1_end);
    uint32_t self_pts[11];
    int ns = snap_bw_change_pts(sn, self_pts);
    for (int i = 0; i < ns; i++) {
        if (self_pts[i] >= lo && self_pts[i] <= hi) add_unique_start(out, &n, self_pts[i]);
        if (self_pts[i] >= dma) {
            uint32_t aligned = self_pts[i] - dma;
            if (aligned >= lo && aligned <= hi) add_unique_start(out, &n, aligned);
        }
    }
    if (peer != NULL) {
        uint32_t pts[11];
        int np = snap_bw_change_pts(peer, pts);
        for (int i = 0; i < np; i++) {
            if (pts[i] >= lo && pts[i] <= hi) add_unique_start(out, &n, pts[i]);
            if (pts[i] >= dma) {
                uint32_t aligned = pts[i] - dma;
                if (aligned >= lo && aligned <= hi) add_unique_start(out, &n, aligned);
            }
        }
    }
    sort_starts(out, n);
    return n;
}

static void snap_apply_s2_down_prefetch(snap_t *sn, moe_shape_t s3,
                                        uint32_t start) {
    sn->s2pf_start = (int32_t)start;
    sn->s2pf_end = (int32_t)(start + SHAPE[s3].t_dma_s3);
    sn->s2pf_bw = SHAPE[s3].alloc;
    sn->dma3_end = sn->s2_end;
    sn->s3_end = sn->task_end;
    sn->bw_s3 = 0;
    sn->skip_dma_s3 = 1;
}

static void apply_s2_down_prefetch_single(snap_t *sn, moe_shape_t s3,
                                          const snap_t *peer) {
    uint32_t starts[S2PF_MAX_CANDS];
    int ns = collect_s2_down_prefetch_starts(sn, s3, peer, starts);
    for (int i = 0; i < ns; i++) {
        snap_t cand = *sn;
        snap_apply_s2_down_prefetch(&cand, s3, starts[i]);
        snap_t idle;
        snap_wait(&idle, 0);
        if ((peer == NULL && bw_feasible(&cand, &idle)) ||
            (peer != NULL && bw_feasible(&cand, peer))) {
            *sn = cand;
            return;
        }
    }
}

static void apply_s2_down_prefetch_pair(snap_t *a, moe_shape_t s3a,
                                        snap_t *b, moe_shape_t s3b) {
    uint32_t starts_a[S2PF_MAX_CANDS];
    uint32_t starts_b[S2PF_MAX_CANDS];
    int na = collect_s2_down_prefetch_starts(a, s3a, b, starts_a);
    int nb = collect_s2_down_prefetch_starts(b, s3b, a, starts_b);
    snap_t best_a = *a;
    snap_t best_b = *b;
    int best_score = bw_feasible(a, b) ? 0 : -1;
    uint32_t best_start_sum = 0;
    for (int ia = 0; ia <= na; ia++) {
        snap_t cand_a = *a;
        uint32_t start_a = S2PF_NO_START;
        if (ia > 0) {
            start_a = starts_a[ia - 1];
            snap_apply_s2_down_prefetch(&cand_a, s3a, start_a);
        }
        for (int ib = 0; ib <= nb; ib++) {
            snap_t cand_b = *b;
            uint32_t start_b = S2PF_NO_START;
            if (ib > 0) {
                start_b = starts_b[ib - 1];
                snap_apply_s2_down_prefetch(&cand_b, s3b, start_b);
            }
            if (!bw_feasible(&cand_a, &cand_b)) continue;
            int score = (start_a != S2PF_NO_START) + (start_b != S2PF_NO_START);
            uint32_t start_sum = (start_a == S2PF_NO_START ? 0u : start_a)
                               + (start_b == S2PF_NO_START ? 0u : start_b);
            if (score > best_score ||
                (score == best_score && start_sum < best_start_sum)) {
                best_score = score;
                best_start_sum = start_sum;
                best_a = cand_a;
                best_b = cand_b;
            }
        }
    }
    *a = best_a;
    *b = best_b;
}

static int collect_next_s1_prefetch_starts(const snap_t *sn, moe_shape_t pf_shape,
                                           const snap_t *peer, uint32_t *out) {
    int n = 0;
    if (sn->cur_eid < 0 || sn->pf_eid >= 0) return 0;
    uint32_t dma = SHAPE[pf_shape].t_dma_s1;
    uint32_t lo = sn->s2_end;
    uint32_t hi = sn->task_end;
    if (hi < lo) return 0;
    add_unique_start(out, &n, lo);
    add_unique_start(out, &n, sn->dma3_end);
    add_unique_start(out, &n, sn->s3_end);
    add_unique_start(out, &n, hi);

    uint32_t self_pts[11];
    int ns = snap_bw_change_pts(sn, self_pts);
    for (int i = 0; i < ns; i++) {
        if (self_pts[i] >= lo && self_pts[i] <= hi) add_unique_start(out, &n, self_pts[i]);
        if (self_pts[i] >= dma) {
            uint32_t aligned = self_pts[i] - dma;
            if (aligned >= lo && aligned <= hi) add_unique_start(out, &n, aligned);
        }
    }
    if (peer != NULL) {
        uint32_t pts[11];
        int np = snap_bw_change_pts(peer, pts);
        for (int i = 0; i < np; i++) {
            if (pts[i] >= lo && pts[i] <= hi) add_unique_start(out, &n, pts[i]);
            if (pts[i] >= dma) {
                uint32_t aligned = pts[i] - dma;
                if (aligned >= lo && aligned <= hi) add_unique_start(out, &n, aligned);
            }
        }
    }
    sort_starts(out, n);
    return n;
}

static void snap_apply_next_s1_prefetch(snap_t *sn, int16_t next_eid,
                                        moe_shape_t pf_shape, uint32_t start) {
    sn->pf_start = (int32_t)start;
    sn->pf_end = (int32_t)(start + SHAPE[pf_shape].t_dma_s1);
    sn->pf_eid = next_eid;
    sn->pf_bw = SHAPE[pf_shape].alloc;
    sn->pf_full = 0;
}

static void collect_next_s1_prefetch_snaps(const snap_t *sn, const snap_t *peer,
                                           int16_t next_eid, snap_t *out, int *n) {
    if (*n < 8) out[(*n)++] = *sn;
    for (int s = 0; s < N_SHAPES; s++) {
        uint32_t starts[S2PF_MAX_CANDS];
        int ns = collect_next_s1_prefetch_starts(sn, (moe_shape_t)s, peer, starts);
        for (int i = 0; i < ns; i++) {
            snap_t cand = *sn;
            snap_apply_next_s1_prefetch(&cand, next_eid, (moe_shape_t)s, starts[i]);
            snap_t idle;
            snap_wait(&idle, 0);
            if ((peer == NULL && bw_feasible(&cand, &idle)) ||
                (peer != NULL && bw_feasible(&cand, peer))) {
                if (*n < 8) out[(*n)++] = cand;
                break;
            }
        }
    }
}

static void apply_next_s1_prefetch_pair(snap_t *a, snap_t *b, int16_t next_eid) {
    snap_t cand_a[8];
    snap_t cand_b[8];
    int na = 0;
    int nb = 0;
    collect_next_s1_prefetch_snaps(a, b, next_eid, cand_a, &na);
    collect_next_s1_prefetch_snaps(b, a, next_eid, cand_b, &nb);

    snap_t best_a = *a;
    snap_t best_b = *b;
    int best_score = -1;
    uint32_t best_end_sum = 0;
    for (int ia = 0; ia < na; ia++) {
        for (int ib = 0; ib < nb; ib++) {
            if (!bw_feasible(&cand_a[ia], &cand_b[ib])) continue;
            int score = (cand_a[ia].pf_eid == next_eid) + (cand_b[ib].pf_eid == next_eid);
            uint32_t end_sum = (cand_a[ia].pf_eid == next_eid ? (uint32_t)cand_a[ia].pf_end : 0u) +
                               (cand_b[ib].pf_eid == next_eid ? (uint32_t)cand_b[ib].pf_end : 0u);
            if (score > best_score || (score == best_score && end_sum < best_end_sum)) {
                best_score = score;
                best_end_sum = end_sum;
                best_a = cand_a[ia];
                best_b = cand_b[ib];
            }
        }
    }
    *a = best_a;
    *b = best_b;
}

/* Cache flags: initial resident cache is full (S1+S3), S4 prefetch is S1-only. */
static inline uint8_t cache_hit(const snap_t *cl, int16_t eid, uint32_t t) {
    if (cl->pf_eid != eid) return 0;
    if (cl->pf_end < 0) return 0;
    if ((uint32_t)cl->pf_end > t) return 0;
    return (uint8_t)(CACHE_S1_READY | (cl->pf_full ? CACHE_S3_READY : 0u));
}

/* ============================================================
 *  Output emission helpers
 * ============================================================ */
static int emit_task(moe_schedule_t *sch, moe_cluster_t cluster,
                     const snap_t *sn) {
    if (sch->n_tasks >= MOE_MAX_TASKS) return 0;
    moe_task_t *t = &sch->tasks[sch->n_tasks++];
    t->cluster      = cluster;
    t->expert_id    = (uint16_t)sn->cur_eid;
    t->ntokens      = sn->ntok;
    t->shape_s1     = (moe_shape_t)sn->shape_s1;
    t->shape_s3     = (moe_shape_t)sn->shape_s3;
    t->skip_dma_s1  = sn->skip_dma_s1;
    t->skip_dma_s3  = sn->skip_dma_s3;
    t->prefetch_eid = -1;          /* filled by post-pass                */
    t->_pad         = 0;
    t->est_start_cc = sn->task_start;
    t->est_end_cc   = sn->task_end;
    return 1;
}

/* ============================================================
 *  Working list of remaining experts
 * ============================================================ */
typedef struct { int16_t eid; uint16_t ntok; } entry_t;

/* ============================================================
 *  Greedy heuristic (Python _greedy_heuristic, multi-remaining)
 * ============================================================ */
static uint32_t greedy_heuristic(uint32_t c2_end, uint32_t c3_end,
                                 const uint16_t *rem_ntok, int n_rem) {
    if (n_rem == 0) return u32_max(c2_end, c3_end);

    uint32_t tasks_max = 0, tasks_sum = 0;
    for (int i = 0; i < n_rem; i++) {
        uint32_t t = best_concurrent_task_time(rem_ntok[i]);
        if (t > tasks_max) tasks_max = t;
        tasks_sum += t;
    }

    if (n_rem == 1) {
        uint32_t t_early = u32_min(c2_end, c3_end);
        uint32_t t_late  = u32_max(c2_end, c3_end);
        uint32_t solo_t  = best_task_time(rem_ntok[0]);
        uint32_t half    = (rem_ntok[0] + 1u) / 2u;
        uint32_t split_t = best_concurrent_task_time(half);
        uint32_t solo_cost  = u32_max(t_late, t_early + solo_t);
        uint32_t split_cost = u32_max(t_late, t_early + split_t);
        return u32_min(solo_cost, split_cost);
    }
    /* multi-remaining */
    uint32_t base = u32_max(c2_end, c3_end);
    uint32_t extra = u32_max(tasks_max, tasks_sum / 2u);
    return base + extra;
}

/* ============================================================
 *  _sim1 — exact 1-step lookahead for the LAST remaining expert
 *  Mirrors Python _sim1: enumerate Method-A (SOLO + SPLIT at now_s)
 *  and Method-B (early start on idle side over busy-side bw pts).
 * ============================================================ */
static uint32_t sim1(const snap_t *c2, const snap_t *c3,
                     int16_t eid, uint16_t ntok)
{
    snap_t c2_local = *c2;
    snap_t c3_local = *c3;
    apply_next_s1_prefetch_pair(&c2_local, &c3_local, eid);
    c2 = &c2_local;
    c3 = &c3_local;
    uint32_t t2 = c2->task_end, t3 = c3->task_end;
    uint32_t now_s = u32_max(t2, t3);
    uint8_t  c2c   = cache_hit(c2, eid, now_s);
    uint8_t  c3c   = cache_hit(c3, eid, now_s);
    uint32_t best  = UINT32_MAX;

    /* Method A: both idle at now_s — SOLO on either side, all 3x3 shapes. */
    for (int s1 = 0; s1 < N_SHAPES; s1++) {
        for (int s3 = 0; s3 < N_SHAPES; s3++) {
            snap_t sn;
            /* try SOLO on C2 (cache from c2) */
            snap_assign(&sn, now_s, (moe_shape_t)s1, (moe_shape_t)s3,
                        ntok, eid, c2c);
                apply_s2_down_prefetch_single(&sn, (moe_shape_t)s3, NULL);
            uint32_t cost = u32_max(sn.task_end, t3);
            if (cost < best) best = cost;
            /* try SOLO on C3 (cache from c3) */
            snap_assign(&sn, now_s, (moe_shape_t)s1, (moe_shape_t)s3,
                        ntok, eid, c3c);
                apply_s2_down_prefetch_single(&sn, (moe_shape_t)s3, NULL);
            cost = u32_max(sn.task_end, t2);
            if (cost < best) best = cost;
        }
    }

    /* Method A: SPLIT at now_s. Cuts = {ceil/floor halves} only (Python _sim1
     * uses the same minimal cut set for the last expert; the full main-loop
     * SPLIT enumerates more cuts for non-last experts). */
    if (ntok >= 2) {
        uint16_t cuts[2];
        int nc = 0;
        cuts[nc++] = (uint16_t)((ntok + 1) / 2);
        uint16_t lo = (uint16_t)(ntok / 2);
        if (lo != cuts[0]) cuts[nc++] = lo;
        for (int ci = 0; ci < nc; ci++) {
            uint16_t cA = cuts[ci];
            uint16_t cB = (uint16_t)(ntok - cA);
            for (int s1a = 0; s1a < N_SHAPES; s1a++) {
            for (int s1b = 0; s1b < N_SHAPES; s1b++) {
            for (int s3a = 0; s3a < N_SHAPES; s3a++) {
            for (int s3b = 0; s3b < N_SHAPES; s3b++) {
                snap_t a, b;
                snap_assign(&a, now_s, (moe_shape_t)s1a, (moe_shape_t)s3a,
                            cA, eid, c2c);
                snap_assign(&b, now_s, (moe_shape_t)s1b, (moe_shape_t)s3b,
                            cB, eid, c3c);
                apply_s2_down_prefetch_pair(&a, (moe_shape_t)s3a,
                                            &b, (moe_shape_t)s3b);
                if (!bw_feasible(&a, &b)) continue;
                uint32_t cost = u32_max(a.task_end, b.task_end);
                if (cost < best) best = cost;
            }}}}
        }
    }

    /* Method B: early start on idle side over busy-side bw_change_pts. */
    if (t2 != t3) {
        const snap_t *idle_cl = (t2 < t3) ? c2 : c3;
        const snap_t *busy_cl = (t2 < t3) ? c3 : c2;
        int idle_is_c2 = (t2 < t3);
        uint32_t idle_t = idle_cl->task_end;
        uint32_t starts[12]; int ns = 0;
        starts[ns++] = idle_t;
        uint32_t pts[11];
        int np = snap_bw_change_pts(busy_cl, pts);
        for (int i = 0; i < np; i++) {
            if (pts[i] >= idle_t && ns < 12) {
                int dup = 0;
                for (int j = 0; j < ns; j++) if (starts[j] == pts[i]) { dup=1; break; }
                if (!dup) starts[ns++] = pts[i];
            }
        }
        for (int k = 0; k < ns; k++) {
            uint32_t t_st = starts[k];
            uint8_t cc = cache_hit(idle_cl, eid, t_st);
            for (int s1 = 0; s1 < N_SHAPES; s1++) {
            for (int s3 = 0; s3 < N_SHAPES; s3++) {
                snap_t sn;
                snap_assign(&sn, t_st, (moe_shape_t)s1, (moe_shape_t)s3,
                            ntok, eid, cc);
                apply_s2_down_prefetch_single(&sn, (moe_shape_t)s3, busy_cl);
                int ok = idle_is_c2 ? bw_feasible(&sn, busy_cl)
                                    : bw_feasible(busy_cl, &sn);
                if (!ok) continue;
                uint32_t cost = u32_max(sn.task_end, busy_cl->task_end);
                if (cost < best) best = cost;
            }}
        }
    }
    return (best == UINT32_MAX) ? (u32_max(t2, t3) + best_task_time(ntok))
                                : best;
}

/* ============================================================
 *  Cost evaluation given (sna, snb) and remaining experts.
 *    new_rem may have 0, 1, or more entries.
 * ============================================================ */
static uint32_t eval_cost(const snap_t *sna, const snap_t *snb,
                          const uint16_t *rem_ntok, int n_rem,
                          int16_t last_eid)
{
    if (n_rem == 0)
        return u32_max(sna->task_end, snb->task_end);
    if (n_rem == 1) {
        snap_t a = *sna, b = *snb;
        return sim1(&a, &b, last_eid, rem_ntok[0]);
    }
    if (n_rem == 2 &&
        (uint32_t)rem_ntok[0] + (uint32_t)rem_ntok[1] <= EXACT_TAIL_MAX_TOKENS) {
        uint32_t t_early = u32_min(sna->task_end, snb->task_end);
        uint32_t t_late = u32_max(sna->task_end, snb->task_end);
        uint32_t solo_seq = t_early + best_task_time(rem_ntok[0])
                          + best_task_time(rem_ntok[1]);
        uint32_t pair_after_late = t_late +
            u32_max(best_concurrent_task_time(rem_ntok[0]),
                    best_concurrent_task_time(rem_ntok[1]));
        return u32_min(u32_max(t_late, solo_seq), pair_after_late);
    }
    return greedy_heuristic(sna->task_end, snb->task_end, rem_ntok, n_rem);
}

/* ============================================================
 *  Working list of remaining experts
 * ============================================================ */
static void sort_desc(entry_t *e, int n) {
    for (int i = 1; i < n; i++) {
        entry_t v = e[i]; int j = i - 1;
        while (j >= 0 && e[j].ntok < v.ntok) { e[j+1] = e[j]; j--; }
        e[j+1] = v;
    }
}

static int find_rem_index(const entry_t *rem, int n, int16_t eid) {
    if (eid < 0) return -1;
    for (int i = 0; i < n; i++) {
        if (rem[i].eid == eid) return i;
    }
    return -1;
}

static void remove_rem_index(entry_t *rem, int *n, int idx) {
    if (idx < 0 || idx >= *n) return;
    for (int i = idx; i + 1 < *n; i++) rem[i] = rem[i + 1];
    (*n)--;
}

static void snap_assign_best_cached(snap_t *out, uint32_t start,
                                    int16_t eid, uint16_t ntok) {
    uint32_t best_end = UINT32_MAX;
    snap_t best;
    uint8_t full_cache = (uint8_t)(CACHE_S1_READY | CACHE_S3_READY);
    for (int s1 = 0; s1 < N_SHAPES; s1++) {
        for (int s3 = 0; s3 < N_SHAPES; s3++) {
            snap_t sn;
            snap_assign(&sn, start, (moe_shape_t)s1, (moe_shape_t)s3,
                        ntok, eid, full_cache);
            if (sn.task_end < best_end) {
                best_end = sn.task_end;
                best = sn;
            }
        }
    }
    *out = best;
}

/* Remove entries by eid from a copy, return new count and ntok list. */
static int rem_excluding(const entry_t *src, int n, int16_t skip_a, int16_t skip_b,
                         uint16_t *out_ntok, int16_t *out_last_eid) {
    int k = 0;
    int16_t last = -1;
    for (int i = 0; i < n; i++) {
        if (src[i].eid == skip_a || src[i].eid == skip_b) continue;
        out_ntok[k] = src[i].ntok;
        last = src[i].eid;
        k++;
    }
    if (out_last_eid) *out_last_eid = last;
    return k;
}

static int rem_excluding_entries(const entry_t *src, int n,
                                 int16_t skip_a, int16_t skip_b,
                                 entry_t *out) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (src[i].eid == skip_a || src[i].eid == skip_b) continue;
        out[k++] = src[i];
    }
    return k;
}

static uint32_t eval_cost_entries(const snap_t *sna, const snap_t *snb,
                                  const entry_t *rem, int n_rem) {
    uint16_t rem_ntok[MOE_MAX_EXPERTS];
    int16_t last_eid = -1;
    for (int i = 0; i < n_rem; i++) {
        rem_ntok[i] = rem[i].ntok;
        last_eid = rem[i].eid;
    }
    return eval_cost(sna, snb, rem_ntok, n_rem, last_eid);
}

static void add_cut(uint16_t *cuts, int *n, uint16_t cut, uint16_t ntok) {
    if (cut < 1u || cut >= ntok) return;
    for (int i = 0; i < *n; i++) {
        if (cuts[i] == cut) return;
    }
    if (*n < 8) cuts[(*n)++] = cut;
}

static uint32_t split_hot_tail_cost(const snap_t *sna, const snap_t *snb,
                                    const entry_t *new_rem, int n_rem) {
    if (n_rem == 0) return u32_max(sna->task_end, snb->task_end);

    int16_t hot_eid = new_rem[0].eid;
    uint16_t hot_ntok = new_rem[0].ntok;
    int n_tail = n_rem - 1;
    if (hot_ntok < 2u || n_tail > 2) {
        return eval_cost_entries(sna, snb, new_rem, n_rem);
    }

    snap_t c2_hot = *sna;
    snap_t c3_hot = *snb;
    apply_next_s1_prefetch_pair(&c2_hot, &c3_hot, hot_eid);
    if (c2_hot.task_end != c3_hot.task_end) {
        return eval_cost_entries(&c2_hot, &c3_hot, new_rem, n_rem);
    }

    uint32_t start = c2_hot.task_end;
    uint8_t c2c_hot = cache_hit(&c2_hot, hot_eid, start);
    uint8_t c3c_hot = cache_hit(&c3_hot, hot_eid, start);
    uint16_t cuts[8];
    int nc = 0;
    add_cut(cuts, &nc, (uint16_t)((hot_ntok + 1u) / 2u), hot_ntok);
    add_cut(cuts, &nc, (uint16_t)(hot_ntok / 2u), hot_ntok);
    for (int s = 0; s < N_SHAPES; s++) {
        add_cut(cuts, &nc, SHAPE[s].m_dim, hot_ntok);
        if (hot_ntok > SHAPE[s].m_dim) {
            add_cut(cuts, &nc, (uint16_t)(hot_ntok - SHAPE[s].m_dim), hot_ntok);
        } else {
            add_cut(cuts, &nc, 1u, hot_ntok);
        }
    }

    uint32_t best = UINT32_MAX;
    for (int ci = 0; ci < nc; ci++) {
        uint16_t cA = cuts[ci];
        uint16_t cB = (uint16_t)(hot_ntok - cA);
        for (int s1a = 0; s1a < N_SHAPES; s1a++)
        for (int s1b = 0; s1b < N_SHAPES; s1b++)
        for (int s3a = 0; s3a < N_SHAPES; s3a++)
        for (int s3b = 0; s3b < N_SHAPES; s3b++) {
            snap_t a, b;
            snap_assign(&a, start, (moe_shape_t)s1a, (moe_shape_t)s3a,
                        cA, hot_eid, c2c_hot);
            snap_assign(&b, start, (moe_shape_t)s1b, (moe_shape_t)s3b,
                        cB, hot_eid, c3c_hot);
            apply_s2_down_prefetch_pair(&a, (moe_shape_t)s3a,
                                        &b, (moe_shape_t)s3b);
            if (!bw_feasible(&a, &b)) continue;
            uint32_t cost = eval_cost_entries(&a, &b, new_rem + 1, n_tail);
            if (cost < best) best = cost;
        }
    }
    return (best == UINT32_MAX) ? eval_cost_entries(&c2_hot, &c3_hot, new_rem, n_rem)
                                : best;
}

/* ============================================================
 *  Best-candidate tracker (used in main loop)
 * ============================================================ */
typedef struct {
    uint32_t cost;
    uint32_t snap_min;
    uint32_t snap_max;
    snap_t   c2_after;
    snap_t   c3_after;
    snap_t   pre_emit;
    int16_t  consumed_eid_a;
    int16_t  consumed_eid_b;
    int16_t  consumed_eid_c;
    moe_cluster_t pre_cluster;
    uint8_t  emit_pre;
    uint8_t  emit_c2;          /* 1 if c2_after is a real new task         */
    uint8_t  emit_c3;          /* 1 if c3_after is a real new task         */
} cand_t;

static void cand_init(cand_t *c) {
    c->cost = UINT32_MAX;
    c->snap_min = UINT32_MAX;
    c->snap_max = UINT32_MAX;
    snap_wait(&c->c2_after, 0);
    snap_wait(&c->c3_after, 0);
    snap_wait(&c->pre_emit, 0);
    c->consumed_eid_a = -1;
    c->consumed_eid_b = -1;
    c->consumed_eid_c = -1;
    c->pre_cluster = MOE_CLUSTER_C2;
    c->emit_pre = 0;
    c->emit_c2 = 0;
    c->emit_c3 = 0;
}

static int cand_better(const cand_t *best, uint32_t cost,
                       const snap_t *c2_after, const snap_t *c3_after) {
    uint32_t snap_min = u32_min(c2_after->task_end, c3_after->task_end);
    uint32_t snap_max = u32_max(c2_after->task_end, c3_after->task_end);
    if (cost != best->cost) return cost < best->cost;
    if (snap_max != best->snap_max) return snap_max < best->snap_max;
    return snap_min < best->snap_min;
}

static void cand_update(cand_t *best, uint32_t cost,
                        const snap_t *c2_after, const snap_t *c3_after,
                        int16_t consumed_a, int16_t consumed_b, int16_t consumed_c,
                        uint8_t emit_c2, uint8_t emit_c3) {
    best->cost = cost;
    best->snap_min = u32_min(c2_after->task_end, c3_after->task_end);
    best->snap_max = u32_max(c2_after->task_end, c3_after->task_end);
    best->c2_after = *c2_after;
    best->c3_after = *c3_after;
    best->consumed_eid_a = consumed_a;
    best->consumed_eid_b = consumed_b;
    best->consumed_eid_c = consumed_c;
    best->emit_pre = 0;
    best->emit_c2 = emit_c2;
    best->emit_c3 = emit_c3;
}

static void cand_set_pre_emit(cand_t *best, moe_cluster_t cluster,
                              const snap_t *pre_emit) {
    best->pre_cluster = cluster;
    best->pre_emit = *pre_emit;
    best->emit_pre = 1;
}

/* ============================================================
 *  Main scheduler
 * ============================================================ */
static moe_status_t moe_schedule_impl(const moe_request_t *req, moe_schedule_t *out,
                                      uint8_t initial_cache_mask) {
    if (req == NULL || out == NULL) return MOE_ERR_BAD_INPUT;
    if (req->n_experts == 0 || req->n_experts > MOE_MAX_EXPERTS)
        return MOE_ERR_BAD_INPUT;

    entry_t rem[MOE_MAX_EXPERTS];
    int nrem = 0;
    for (uint16_t i = 0; i < req->n_experts; i++) {
        if (req->experts[i].ntokens == 0) return MOE_ERR_BAD_INPUT;
        rem[nrem].eid  = (int16_t)req->experts[i].expert_id;
        rem[nrem].ntok = req->experts[i].ntokens;
        nrem++;
    }
    sort_desc(rem, nrem);

    snap_t c2, c3;
    snap_make_initial(&c2, req->cache_eid_c2);
    snap_make_initial(&c3, req->cache_eid_c3);

    out->n_tasks = 0;
    out->_pad = 0;
    out->est_makespan_cc = 0;

    int c2_cached_idx = (initial_cache_mask & INIT_CACHE_C2)
                      ? find_rem_index(rem, nrem, req->cache_eid_c2) : -1;
    int c3_cached_idx = (initial_cache_mask & INIT_CACHE_C3)
                      ? find_rem_index(rem, nrem, req->cache_eid_c3) : -1;

    if (c2_cached_idx >= 0 && c3_cached_idx >= 0 &&
        req->cache_eid_c2 == req->cache_eid_c3) {
        int16_t eid = req->cache_eid_c2;
        uint16_t ntok = rem[c2_cached_idx].ntok;
        if (ntok >= 2) {
            uint16_t c2_ntok = (uint16_t)((ntok + 1u) / 2u);
            uint16_t c3_ntok = (uint16_t)(ntok - c2_ntok);
            snap_assign_best_cached(&c2, 0, eid, c2_ntok);
            snap_assign_best_cached(&c3, 0, eid, c3_ntok);
            if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
            if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
        } else {
            snap_assign_best_cached(&c2, 0, eid, ntok);
            if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
        }
        remove_rem_index(rem, &nrem, c2_cached_idx);
    } else {
        if (c2_cached_idx >= 0) {
            snap_assign_best_cached(&c2, 0, req->cache_eid_c2,
                                    rem[c2_cached_idx].ntok);
            if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
            remove_rem_index(rem, &nrem, c2_cached_idx);
        }

        c3_cached_idx = (initial_cache_mask & INIT_CACHE_C3)
                      ? find_rem_index(rem, nrem, req->cache_eid_c3) : -1;
        if (c3_cached_idx >= 0) {
            snap_assign_best_cached(&c3, 0, req->cache_eid_c3,
                                    rem[c3_cached_idx].ntok);
            if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
            remove_rem_index(rem, &nrem, c3_cached_idx);
        }
    }

    int max_iters = MOE_MAX_EXPERTS * 4 + 8;

#ifdef MOE_SCHEDULE_FAST
    /* ─────────────────────────────────────────────────────────────────────
     * FAST PATH  (O(E) greedy, ~1000× faster than full search)
     *
     * Each step evaluates at most 3 fixed candidates instead of N_SHAPES^4:
     *   n==1    : SOLO(C2) | SOLO(C3) | SPLIT(ceil/2) | early-start
     *   both_idle: PAIR(top0→C2,top1→C3) | reversed PAIR | SPLIT(top0 ceil/2)
     *   !both_idle: SOLO(top0) on idle cluster (solo_shape or conc_shape)
     * Shape selection uses best_conc/solo_shape_s1/s3() directly.
     *
     * Mirrors fast_scheduler.py:fast_schedule() / _fast_schedule_core().
     * ───────────────────────────────────────────────────────────────────── */
    while (nrem > 0 && max_iters-- > 0) {
        apply_next_s1_prefetch_pair(&c2, &c3, rem[0].eid);
        uint32_t t2 = c2.task_end, t3 = c3.task_end;
        uint32_t now = u32_max(t2, t3);
        int both_idle = (t2 == t3);
        int n = nrem;
        int16_t  e0 = rem[0].eid;
        uint16_t n0 = rem[0].ntok;

        /* ── n == 1 : terminal expert ── */
        if (n == 1) {
            moe_shape_t s1_s = best_solo_shape_s1(n0);
            moe_shape_t s3_s = best_solo_shape_s3(n0);
            uint8_t c2c = cache_hit(&c2, e0, now);
            uint8_t c3c = cache_hit(&c3, e0, now);
            uint32_t best_ms = UINT32_MAX;
            snap_t best_sna, best_snb;
            int best_kind = 0; /* 0=C2 solo, 1=C3 solo, 2=split */

            { snap_t sn;
              snap_assign(&sn, now, s1_s, s3_s, n0, e0, c2c);
              if (sn.task_end < best_ms) { best_ms = sn.task_end; best_kind = 0; best_sna = sn; } }
            { snap_t sn;
              snap_assign(&sn, now, s1_s, s3_s, n0, e0, c3c);
              if (sn.task_end < best_ms) { best_ms = sn.task_end; best_kind = 1; best_sna = sn; } }

            if (n0 >= 2) {
                uint16_t half = (uint16_t)((n0 + 1u) / 2u);
                uint16_t rest = (uint16_t)(n0 - half);
                snap_t sna, snb;
                snap_assign(&sna, now, best_conc_shape_s1(half), best_conc_shape_s3(half), half, e0, c2c);
                snap_assign(&snb, now, best_conc_shape_s1(rest), best_conc_shape_s3(rest), rest, e0, c3c);
                if (bw_feasible(&sna, &snb)) {
                    uint32_t cost = u32_max(sna.task_end, snb.task_end);
                    if (cost < best_ms) { best_ms = cost; best_kind = 2; best_sna = sna; best_snb = snb; }
                }
            }

            /* early-start candidate (only when !both_idle) */
            if (!both_idle) {
                int idle_is_c2 = (t2 < t3);
                snap_t       *idle_cl = idle_is_c2 ? &c2 : &c3;
                const snap_t *busy_cl = idle_is_c2 ? &c3 : &c2;
                uint32_t idle_t = idle_cl->task_end;
                uint8_t cc = cache_hit(idle_cl, e0, idle_t);
                snap_t sn;
                snap_assign(&sn, idle_t, s1_s, s3_s, n0, e0, cc);
                int ok = idle_is_c2 ? bw_feasible(&sn, busy_cl)
                                    : bw_feasible(busy_cl, &sn);
                if (ok) {
                    uint32_t cost = u32_max(sn.task_end, busy_cl->task_end);
                    if (cost < best_ms) {
                        best_ms = cost;
                        best_kind = idle_is_c2 ? 0 : 1;
                        best_sna = sn;
                    }
                }
            }

            if (best_ms == UINT32_MAX) return MOE_ERR_INTERNAL;
            if (best_kind == 0) {
                c2 = best_sna;
                if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
            } else if (best_kind == 1) {
                c3 = best_sna;
                if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
            } else {
                c2 = best_sna; c3 = best_snb;
                if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
                if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
            }
            nrem = 0;
            break;
        }

        /* ── !both_idle : SOLO top0 on idle cluster ── */
        if (!both_idle) {
            int idle_is_c2 = (t2 < t3);
            snap_t       *idle_cl = idle_is_c2 ? &c2 : &c3;
            const snap_t *busy_cl = idle_is_c2 ? &c3 : &c2;
            uint32_t idle_t = idle_cl->task_end;
            uint8_t cc = cache_hit(idle_cl, e0, idle_t);
            moe_shape_t try_s1[2], try_s3[2];
            try_s1[0] = best_solo_shape_s1(n0); try_s3[0] = best_solo_shape_s3(n0);
            try_s1[1] = best_conc_shape_s1(n0); try_s3[1] = best_conc_shape_s3(n0);
            uint32_t best_ms = UINT32_MAX; snap_t best_sn;
            for (int k = 0; k < 2; k++) {
                snap_t sn;
                snap_assign(&sn, idle_t, try_s1[k], try_s3[k], n0, e0, cc);
                int ok = idle_is_c2 ? bw_feasible(&sn, busy_cl)
                                    : bw_feasible(busy_cl, &sn);
                if (!ok) continue;
                uint32_t cost = u32_max(sn.task_end, busy_cl->task_end);
                if (cost < best_ms) { best_ms = cost; best_sn = sn; }
            }
            if (best_ms == UINT32_MAX) {
                /* BW conflict: wait until now, use solo shape */
                snap_assign(&best_sn, now, try_s1[0], try_s3[0], n0, e0,
                            cache_hit(idle_cl, e0, now));
            }
            if (idle_is_c2) { c2 = best_sn; if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW; }
            else             { c3 = best_sn; if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW; }
            for (int i = 0; i+1 < nrem; i++) rem[i] = rem[i+1];
            nrem--;
            continue;
        }

        /* ── both_idle && n >= 2 : PAIR(top0,top1) two dirs + SPLIT(top0) ── */
        {
            int16_t  e1 = rem[1].eid;
            uint16_t n1 = rem[1].ntok;
            moe_shape_t s1_0 = best_conc_shape_s1(n0), s3_0 = best_conc_shape_s3(n0);
            moe_shape_t s1_1 = best_conc_shape_s1(n1), s3_1 = best_conc_shape_s3(n1);
            uint8_t c2c_0 = cache_hit(&c2, e0, now);
            uint8_t c3c_0 = cache_hit(&c3, e0, now);
            uint8_t c2c_1 = cache_hit(&c2, e1, now);
            uint8_t c3c_1 = cache_hit(&c3, e1, now);

            /* Build greedy_heuristic rem_ntok arrays */
            uint16_t gh_ntok_pair[MOE_MAX_EXPERTS]; int gh_n_pair = 0;
            uint16_t gh_ntok_split[MOE_MAX_EXPERTS]; int gh_n_split = 0;
            for (int i = 0; i < nrem; i++) {
                if (rem[i].eid != e0 && rem[i].eid != e1)
                    gh_ntok_pair[gh_n_pair++] = rem[i].ntok;
            }
            for (int i = 1; i < nrem; i++) gh_ntok_split[gh_n_split++] = rem[i].ntok;

            uint32_t best_ms = UINT32_MAX;
            snap_t best_sna, best_snb;
            int16_t best_ea = -1, best_eb = -1;

            /* PAIR dir1: top0→C2, top1→C3 */
            { snap_t sna, snb;
              snap_assign(&sna, now, s1_0, s3_0, n0, e0, c2c_0);
              snap_assign(&snb, now, s1_1, s3_1, n1, e1, c3c_1);
              if (bw_feasible(&sna, &snb)) {
                  uint32_t c = greedy_heuristic(sna.task_end, snb.task_end, gh_ntok_pair, gh_n_pair);
                  if (c < best_ms) { best_ms = c; best_sna = sna; best_snb = snb; best_ea = e0; best_eb = e1; }
              } }

            /* PAIR dir2: top1→C2, top0→C3 */
            { snap_t sna, snb;
              snap_assign(&sna, now, s1_1, s3_1, n1, e1, c2c_1);
              snap_assign(&snb, now, s1_0, s3_0, n0, e0, c3c_0);
              if (bw_feasible(&sna, &snb)) {
                  uint32_t c = greedy_heuristic(sna.task_end, snb.task_end, gh_ntok_pair, gh_n_pair);
                  if (c < best_ms) { best_ms = c; best_sna = sna; best_snb = snb; best_ea = e1; best_eb = e0; }
              } }

            /* SPLIT top0: ceil/2 on C2, rest on C3 */
            if (n0 >= 2) {
                uint16_t half = (uint16_t)((n0 + 1u) / 2u);
                uint16_t rest = (uint16_t)(n0 - half);
                snap_t sna, snb;
                snap_assign(&sna, now, best_conc_shape_s1(half), best_conc_shape_s3(half), half, e0, c2c_0);
                snap_assign(&snb, now, best_conc_shape_s1(rest),  best_conc_shape_s3(rest),  rest, e0, c3c_0);
                if (bw_feasible(&sna, &snb)) {
                    uint32_t c = greedy_heuristic(sna.task_end, snb.task_end, gh_ntok_split, gh_n_split);
                    if (c < best_ms) { best_ms = c; best_sna = sna; best_snb = snb; best_ea = e0; best_eb = e0; }
                }
            }

            if (best_ms == UINT32_MAX) {
                /* Fallback: SOLO top0 on C2 at now */
                snap_t sn;
                snap_assign(&sn, now, best_solo_shape_s1(n0), best_solo_shape_s3(n0), n0, e0, c2c_0);
                c2 = sn;
                if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
                for (int i = 0; i+1 < nrem; i++) rem[i] = rem[i+1];
                nrem--;
                continue;
            }

            c2 = best_sna; c3 = best_snb;
            if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
            if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;

            if (best_ea == best_eb) {
                /* SPLIT: only e0 consumed */
                for (int i = 0; i+1 < nrem; i++) rem[i] = rem[i+1];
                nrem--;
            } else {
                /* PAIR: remove both e_a and e_b */
                int w = 0;
                for (int i = 0; i < nrem; i++) {
                    if (rem[i].eid != best_ea && rem[i].eid != best_eb)
                        rem[w++] = rem[i];
                }
                nrem = w;
            }
        }
    }
#else   /* ── FULL SEARCH (original O(E³×N⁴)) ─────────────────────────────── */
    while (nrem > 0 && max_iters-- > 0) {
        apply_next_s1_prefetch_pair(&c2, &c3, rem[0].eid);
        uint32_t t2 = c2.task_end, t3 = c3.task_end;
        uint32_t now = u32_max(t2, t3);
        int both_idle = (t2 == t3);
        int n = nrem;
        int16_t  e0 = rem[0].eid;
        uint16_t n0 = rem[0].ntok;

        /* ─────────────────────────────────────────────────────
         *  n == 1 : terminal expert (Python n==1 branch)
         * ───────────────────────────────────────────────────── */
        if (n == 1) {
            uint32_t best_ms = UINT32_MAX;
            enum { K_C2, K_C3, K_SPLIT } kind = K_C2;
            snap_t bestA, bestB;

            uint8_t c2c = cache_hit(&c2, e0, now);
            uint8_t c3c = cache_hit(&c3, e0, now);

            /* Method A : SOLO at `now` on each side (3x3 shapes) */
            for (int s1 = 0; s1 < N_SHAPES; s1++) {
            for (int s3 = 0; s3 < N_SHAPES; s3++) {
                snap_t sn;
                snap_assign(&sn, now, (moe_shape_t)s1, (moe_shape_t)s3,
                            n0, e0, c2c);
                apply_s2_down_prefetch_single(&sn, (moe_shape_t)s3, NULL);
                uint32_t cost = sn.task_end; /* = max(sn.end, t3) since now>=t3 */
                if (cost < best_ms) { best_ms = cost; kind = K_C2; bestA = sn; }
                snap_assign(&sn, now, (moe_shape_t)s1, (moe_shape_t)s3,
                            n0, e0, c3c);
                apply_s2_down_prefetch_single(&sn, (moe_shape_t)s3, NULL);
                cost = sn.task_end;
                if (cost < best_ms) { best_ms = cost; kind = K_C3; bestA = sn; }
            }}

            /* Method A : SPLIT at `now` (full cut set per Python) */
            if (n0 >= 2) {
                uint16_t cuts[8]; int nc = 0;
                uint16_t hi = (uint16_t)((n0+1)/2), lo = (uint16_t)(n0/2);
                cuts[nc++] = hi;
                if (lo != hi) cuts[nc++] = lo;
                static const uint16_t MS[3] = {2,4,8};
                for (int i = 0; i < 3; i++) {
                    uint16_t m = MS[i];
                    if (m >= 1 && m < n0) {
                        int dup=0;
                        for (int j=0;j<nc;j++) if (cuts[j]==m){dup=1;break;}
                        if (!dup && nc<8) cuts[nc++] = m;
                        uint16_t cm = (uint16_t)(n0 - m);
                        if (cm >= 1 && cm < n0) {
                            dup=0;
                            for (int j=0;j<nc;j++) if (cuts[j]==cm){dup=1;break;}
                            if (!dup && nc<8) cuts[nc++] = cm;
                        }
                    }
                }
                for (int ci = 0; ci < nc; ci++) {
                    uint16_t cA = cuts[ci];
                    uint16_t cB = (uint16_t)(n0 - cA);
                    for (int s1a=0;s1a<N_SHAPES;s1a++)
                    for (int s1b=0;s1b<N_SHAPES;s1b++)
                    for (int s3a=0;s3a<N_SHAPES;s3a++)
                    for (int s3b=0;s3b<N_SHAPES;s3b++) {
                        snap_t a, b;
                        snap_assign(&a, now, (moe_shape_t)s1a, (moe_shape_t)s3a,
                                    cA, e0, c2c);
                        snap_assign(&b, now, (moe_shape_t)s1b, (moe_shape_t)s3b,
                                    cB, e0, c3c);
                        apply_s2_down_prefetch_pair(&a, (moe_shape_t)s3a,
                                                    &b, (moe_shape_t)s3b);
                        if (!bw_feasible(&a, &b)) continue;
                        uint32_t cost = u32_max(a.task_end, b.task_end);
                        if (cost < best_ms) {
                            best_ms = cost; kind = K_SPLIT;
                            bestA = a; bestB = b;
                        }
                    }
                }
            }

            /* Method B : early start on idle side */
            if (!both_idle) {
                int idle_is_c2 = (t2 < t3);
                const snap_t *idle_cl = idle_is_c2 ? &c2 : &c3;
                const snap_t *busy_cl = idle_is_c2 ? &c3 : &c2;
                uint32_t idle_t = idle_cl->task_end;
                uint32_t starts[12]; int ns = 0;
                starts[ns++] = idle_t;
                uint32_t pts[11];
                int np = snap_bw_change_pts(busy_cl, pts);
                for (int i = 0; i < np; i++) {
                    if (pts[i] >= idle_t && ns < 12) {
                        int dup=0;
                        for (int j=0;j<ns;j++) if (starts[j]==pts[i]){dup=1;break;}
                        if (!dup) starts[ns++] = pts[i];
                    }
                }
                for (int k = 0; k < ns; k++) {
                    uint32_t t_st = starts[k];
                    uint8_t cc = cache_hit(idle_cl, e0, t_st);
                    for (int s1=0;s1<N_SHAPES;s1++)
                    for (int s3=0;s3<N_SHAPES;s3++) {
                        snap_t sn;
                        snap_assign(&sn, t_st, (moe_shape_t)s1, (moe_shape_t)s3,
                                    n0, e0, cc);
                        apply_s2_down_prefetch_single(&sn, (moe_shape_t)s3, busy_cl);
                        int ok = idle_is_c2 ? bw_feasible(&sn, busy_cl)
                                            : bw_feasible(busy_cl, &sn);
                        if (!ok) continue;
                        uint32_t cost = u32_max(sn.task_end, busy_cl->task_end);
                        if (cost < best_ms) {
                            best_ms = cost;
                            kind = idle_is_c2 ? K_C2 : K_C3;
                            bestA = sn;
                        }
                    }
                }
            }

            if (best_ms == UINT32_MAX) return MOE_ERR_INTERNAL;

            if (kind == K_C2) {
                c2 = bestA;
                if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
            } else if (kind == K_C3) {
                c3 = bestA;
                if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
            } else { /* K_SPLIT */
                c2 = bestA; c3 = bestB;
                if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
                if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
            }
            /* drop top0 */
            for (int i=0;i+1<nrem;i++) rem[i]=rem[i+1];
            nrem--;
            continue;
        }

        /* ─────────────────────────────────────────────────────
         *  n >= 2, !both_idle : SOLO top0 on idle with early-start
         * ───────────────────────────────────────────────────── */
        if (!both_idle) {
            int idle_is_c2 = (t2 < t3);
            snap_t       *idle_cl = idle_is_c2 ? &c2 : &c3;
            const snap_t *busy_cl = idle_is_c2 ? &c3 : &c2;
            uint32_t idle_t = idle_cl->task_end;
            uint32_t starts[12]; int ns = 0;
            starts[ns++] = idle_t;
            uint32_t pts[11];
            int np = snap_bw_change_pts(busy_cl, pts);
            for (int i = 0; i < np; i++) {
                if (pts[i] >= idle_t && ns < 12) {
                    int dup=0;
                    for (int j=0;j<ns;j++) if (starts[j]==pts[i]){dup=1;break;}
                    if (!dup) starts[ns++] = pts[i];
                }
            }
            uint32_t best_ms = UINT32_MAX; snap_t best_sn;
            for (int k = 0; k < ns; k++) {
                uint32_t t_st = starts[k];
                uint8_t cc = cache_hit(idle_cl, e0, t_st);
                for (int s1=0;s1<N_SHAPES;s1++)
                for (int s3=0;s3<N_SHAPES;s3++) {
                    snap_t sn;
                    snap_assign(&sn, t_st, (moe_shape_t)s1, (moe_shape_t)s3,
                                n0, e0, cc);
                    apply_s2_down_prefetch_single(&sn, (moe_shape_t)s3, busy_cl);
                    int ok = idle_is_c2 ? bw_feasible(&sn, busy_cl)
                                        : bw_feasible(busy_cl, &sn);
                    if (!ok) continue;
                    uint32_t cost = u32_max(sn.task_end, busy_cl->task_end);
                    if (cost < best_ms) { best_ms = cost; best_sn = sn; }
                }
            }
            if (best_ms == UINT32_MAX) return MOE_ERR_INTERNAL;

            if (idle_is_c2) {
                c2 = best_sn;
                if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
            } else {
                c3 = best_sn;
                if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
            }
            for (int i=0;i+1<nrem;i++) rem[i]=rem[i+1];
            nrem--;
            continue;
        }

        /* ─────────────────────────────────────────────────────
         *  both_idle && n >= 2 : full search
         * ───────────────────────────────────────────────────── */
        cand_t best;
        cand_init(&best);

        uint8_t c2c_0 = cache_hit(&c2, e0, now);
        uint8_t c3c_0 = cache_hit(&c3, e0, now);

        /* ── PAIR(top0, topK), K = 1..min(n-1,3), both directions ── */
        int max_K = (n - 1 < 3) ? (n - 1) : 3;
        for (int K = 1; K <= max_K; K++) {
            int16_t  eK = rem[K].eid;
            uint16_t nK = rem[K].ntok;
            uint8_t c2c_K = cache_hit(&c2, eK, now);
            uint8_t c3c_K = cache_hit(&c3, eK, now);

            /* dir 1: top0 → C2 (cache c2c_0), topK → C3 (cache c3c_K) */
            for (int s1a=0;s1a<N_SHAPES;s1a++) {
                if (!c2c_0 && !c3c_K && SHAPE[s1a].alloc > MAX_BW/2) continue;
            for (int s1b=0;s1b<N_SHAPES;s1b++) {
                if (!c3c_K && !c2c_0 && SHAPE[s1b].alloc > MAX_BW/2) continue;
            for (int s3a=0;s3a<N_SHAPES;s3a++)
            for (int s3b=0;s3b<N_SHAPES;s3b++) {
                snap_t a, b;
                snap_assign(&a, now, (moe_shape_t)s1a, (moe_shape_t)s3a,
                            n0, e0, c2c_0);
                snap_assign(&b, now, (moe_shape_t)s1b, (moe_shape_t)s3b,
                            nK, eK, c3c_K);
                apply_s2_down_prefetch_pair(&a, (moe_shape_t)s3a,
                                            &b, (moe_shape_t)s3b);
                if (!bw_feasible(&a, &b)) continue;
                uint16_t rem_ntok[MOE_MAX_EXPERTS]; int16_t last_eid;
                int n_rem = rem_excluding(rem, nrem, e0, eK, rem_ntok, &last_eid);
                uint32_t cost = eval_cost(&a, &b, rem_ntok, n_rem, last_eid);
                if (cand_better(&best, cost, &a, &b)) {
                    cand_update(&best, cost, &a, &b, e0, eK, -1, 1, 1);
                }
            }}}

            /* dir 2: topK → C2 (cache c2c_K), top0 → C3 (cache c3c_0) */
            for (int s1a=0;s1a<N_SHAPES;s1a++) {
                if (!c2c_K && !c3c_0 && SHAPE[s1a].alloc > MAX_BW/2) continue;
            for (int s1b=0;s1b<N_SHAPES;s1b++) {
                if (!c3c_0 && !c2c_K && SHAPE[s1b].alloc > MAX_BW/2) continue;
            for (int s3a=0;s3a<N_SHAPES;s3a++)
            for (int s3b=0;s3b<N_SHAPES;s3b++) {
                snap_t a, b;
                snap_assign(&a, now, (moe_shape_t)s1a, (moe_shape_t)s3a,
                            nK, eK, c2c_K);
                snap_assign(&b, now, (moe_shape_t)s1b, (moe_shape_t)s3b,
                            n0, e0, c3c_0);
                apply_s2_down_prefetch_pair(&a, (moe_shape_t)s3a,
                                            &b, (moe_shape_t)s3b);
                if (!bw_feasible(&a, &b)) continue;
                uint16_t rem_ntok[MOE_MAX_EXPERTS]; int16_t last_eid;
                int n_rem = rem_excluding(rem, nrem, e0, eK, rem_ntok, &last_eid);
                uint32_t cost = eval_cost(&a, &b, rem_ntok, n_rem, last_eid);
                if (cand_better(&best, cost, &a, &b)) {
                    cand_update(&best, cost, &a, &b, eK, e0, -1, 1, 1);
                }
            }}}
        }

        /* ── PAIR(topK, topJ), K<J, n>=3 (S3 limited to {B,C}) ── */
        if (n >= 3) {
            int max_KJ = (n - 1 < 3) ? (n - 1) : 3;
            static const moe_shape_t S3_PAIR_KJ[2] = { MOE_SHAPE_B, MOE_SHAPE_C };
            for (int K = 1; K < max_KJ; K++) {
            for (int J = K + 1; J <= max_KJ; J++) {
                if (J >= n) break;
                int16_t  eK = rem[K].eid;  uint16_t nK = rem[K].ntok;
                int16_t  eJ = rem[J].eid;  uint16_t nJ = rem[J].ntok;
                uint8_t c2c_K = cache_hit(&c2, eK, now);
                uint8_t c3c_J = cache_hit(&c3, eJ, now);
                uint8_t c2c_J = cache_hit(&c2, eJ, now);
                uint8_t c3c_K = cache_hit(&c3, eK, now);

                /* both directions */
                for (int dir = 0; dir < 2; dir++) {
                    int16_t  eA, eB; uint16_t nA, nB;
                    uint8_t  ccA, ccB;
                    if (dir == 0) {
                        eA=eK; nA=nK; ccA=c2c_K;
                        eB=eJ; nB=nJ; ccB=c3c_J;
                    } else {
                        eA=eJ; nA=nJ; ccA=c2c_J;
                        eB=eK; nB=nK; ccB=c3c_K;
                    }
                    for (int s1a=0;s1a<N_SHAPES;s1a++) {
                        if (!ccA && !ccB && SHAPE[s1a].alloc > MAX_BW/2) continue;
                    for (int s1b=0;s1b<N_SHAPES;s1b++) {
                        if (!ccB && !ccA && SHAPE[s1b].alloc > MAX_BW/2) continue;
                    for (int sa=0;sa<2;sa++)
                    for (int sb=0;sb<2;sb++) {
                        snap_t a, b;
                        snap_assign(&a, now, (moe_shape_t)s1a, S3_PAIR_KJ[sa],
                                    nA, eA, ccA);
                        snap_assign(&b, now, (moe_shape_t)s1b, S3_PAIR_KJ[sb],
                                    nB, eB, ccB);
                        apply_s2_down_prefetch_pair(&a, S3_PAIR_KJ[sa],
                                                    &b, S3_PAIR_KJ[sb]);
                        if (!bw_feasible(&a, &b)) continue;
                        uint16_t rem_ntok[MOE_MAX_EXPERTS]; int16_t last_eid;
                        int n_rem = rem_excluding(rem, nrem, eK, eJ, rem_ntok, &last_eid);
                        uint32_t cost = eval_cost(&a, &b, rem_ntok, n_rem, last_eid);
                        if (cand_better(&best, cost, &a, &b)) {
                            cand_update(&best, cost, &a, &b, eA, eB, -1, 1, 1);
                        }
                    }}}
                }
            }}
        }

        /* ── SPLIT(top0) full cut set ── */
        if (n0 >= 2) {
            uint16_t cuts[8]; int nc = 0;
            uint16_t hi = (uint16_t)((n0+1)/2), lo = (uint16_t)(n0/2);
            cuts[nc++] = hi;
            if (lo != hi) cuts[nc++] = lo;
            static const uint16_t MS[3] = {2,4,8};
            for (int i = 0; i < 3; i++) {
                uint16_t m = MS[i];
                if (m >= 1 && m < n0) {
                    int dup=0;
                    for (int j=0;j<nc;j++) if (cuts[j]==m){dup=1;break;}
                    if (!dup && nc<8) cuts[nc++] = m;
                    uint16_t cm = (uint16_t)(n0 - m);
                    if (cm >= 1 && cm < n0) {
                        dup=0;
                        for (int j=0;j<nc;j++) if (cuts[j]==cm){dup=1;break;}
                        if (!dup && nc<8) cuts[nc++] = cm;
                    }
                }
            }
            for (int ci = 0; ci < nc; ci++) {
                uint16_t cA = cuts[ci];
                uint16_t cB = (uint16_t)(n0 - cA);
                for (int s1a=0;s1a<N_SHAPES;s1a++)
                for (int s1b=0;s1b<N_SHAPES;s1b++)
                for (int s3a=0;s3a<N_SHAPES;s3a++)
                for (int s3b=0;s3b<N_SHAPES;s3b++) {
                    snap_t a, b;
                    snap_assign(&a, now, (moe_shape_t)s1a, (moe_shape_t)s3a,
                                cA, e0, c2c_0);
                    snap_assign(&b, now, (moe_shape_t)s1b, (moe_shape_t)s3b,
                                cB, e0, c3c_0);
                    apply_s2_down_prefetch_pair(&a, (moe_shape_t)s3a,
                                                &b, (moe_shape_t)s3b);
                    if (!bw_feasible(&a, &b)) continue;
                    uint16_t rem_ntok[MOE_MAX_EXPERTS]; int16_t last_eid;
                    int n_rem = rem_excluding(rem, nrem, e0, -1, rem_ntok, &last_eid);
                    uint32_t cost = eval_cost(&a, &b, rem_ntok, n_rem, last_eid);
                    if (cand_better(&best, cost, &a, &b)) {
                        cand_update(&best, cost, &a, &b, e0, -1, -1, 1, 1);
                    }
                }
            }
        }

        /* ── WAIT-SINGLE-PAIR : first a 1-token cold expert, then pair two tails ── */
        if (n >= 5) {
            int max_first = (n < 4) ? n : 4;
            for (int K = 1; K < max_first; K++) {
                int16_t first_eid = rem[K].eid;
                uint16_t first_ntok = rem[K].ntok;
                if (first_ntok != 1u) continue;

                entry_t rem_after_first[MOE_MAX_EXPERTS];
                int n_after_first = rem_excluding_entries(rem, nrem, first_eid, -1,
                                                          rem_after_first);
                if (n_after_first < 4) continue;

                entry_t pair_anchor = rem_after_first[1];
                moe_shape_t s1_first = best_solo_shape_s1(first_ntok);
                moe_shape_t s3_first = best_solo_shape_s3(first_ntok);
                uint8_t c2c_first = cache_hit(&c2, first_eid, now);

                snap_t first_sn;
                snap_assign(&first_sn, now, s1_first, s3_first,
                            first_ntok, first_eid, c2c_first);
                apply_s2_down_prefetch_single(&first_sn, s3_first, NULL);
                uint32_t t_pair = first_sn.task_end;
                snap_t wait_sn;
                snap_wait(&wait_sn, t_pair);
                if (!bw_feasible(&first_sn, &wait_sn)) continue;

                int max_pair = (n_after_first < 4) ? n_after_first : 4;
                for (int cand_idx = 2; cand_idx < max_pair; cand_idx++) {
                    entry_t pair_cand = rem_after_first[cand_idx];
                    if (pair_cand.ntok != 1u) continue;

                    entry_t rem_after_pair[MOE_MAX_EXPERTS];
                    int n_after_pair = rem_excluding_entries(rem_after_first,
                                                             n_after_first,
                                                             pair_anchor.eid,
                                                             pair_cand.eid,
                                                             rem_after_pair);
                    for (int swap_pair = 0; swap_pair < 2; swap_pair++) {
                        int16_t eid_a = swap_pair ? pair_cand.eid : pair_anchor.eid;
                        uint16_t ntok_a = swap_pair ? pair_cand.ntok : pair_anchor.ntok;
                        int16_t eid_b = swap_pair ? pair_anchor.eid : pair_cand.eid;
                        uint16_t ntok_b = swap_pair ? pair_anchor.ntok : pair_cand.ntok;

                        moe_shape_t s1_a = best_conc_shape_s1(ntok_a);
                        moe_shape_t s1_b = best_conc_shape_s1(ntok_b);
                        moe_shape_t s3_a = best_conc_shape_s3(ntok_a);
                        moe_shape_t s3_b = best_conc_shape_s3(ntok_b);
                        snap_t a, b;
                        snap_assign(&a, t_pair, s1_a, s3_a, ntok_a, eid_a,
                                    cache_hit(&first_sn, eid_a, t_pair));
                        snap_assign(&b, t_pair, s1_b, s3_b, ntok_b, eid_b,
                                    cache_hit(&wait_sn, eid_b, t_pair));
                        apply_s2_down_prefetch_pair(&a, s3_a, &b, s3_b);
                        if (!bw_feasible(&a, &b)) continue;

                        uint32_t tail_sum = 0;
                        for (int ti = 1; ti < n_after_pair; ti++) {
                            tail_sum += rem_after_pair[ti].ntok;
                        }
                        uint32_t cost = (n_after_pair <= 3 && tail_sum <= EXACT_TAIL_MAX_TOKENS)
                                      ? split_hot_tail_cost(&a, &b, rem_after_pair, n_after_pair)
                                      : eval_cost_entries(&a, &b, rem_after_pair, n_after_pair);
                        if (cand_better(&best, cost, &a, &b)) {
                            cand_update(&best, cost, &a, &b,
                                        first_eid, eid_a, eid_b, 1, 1);
                            cand_set_pre_emit(&best, MOE_CLUSTER_C2, &first_sn);
                        }
                    }
                }
            }
        }

        /* ── WAIT-PAIR : SINGLE(topK) first, leave other side waiting ── */
        int max_K_wait = (n < 4) ? n : 4;
        for (int K = 1; K < max_K_wait; K++) {
            int16_t  eK = rem[K].eid;
            uint16_t nK = rem[K].ntok;
            uint8_t c2c_K = cache_hit(&c2, eK, now);
            uint8_t c3c_K = cache_hit(&c3, eK, now);

            for (int dir = 0; dir < 2; dir++) {
                uint8_t cc = (dir == 0) ? c2c_K : c3c_K;
                for (int s1=0;s1<N_SHAPES;s1++)
                for (int s3=0;s3<N_SHAPES;s3++) {
                    snap_t sn_k, wait;
                    snap_assign(&sn_k, now, (moe_shape_t)s1, (moe_shape_t)s3,
                                nK, eK, cc);
                    apply_s2_down_prefetch_single(&sn_k, (moe_shape_t)s3, NULL);
                    uint32_t t_k = sn_k.task_end;
                    snap_wait(&wait, t_k);
                    snap_t a, b;
                    if (dir == 0) { a = sn_k; b = wait; }
                    else          { a = wait; b = sn_k; }
                    if (!bw_feasible(&a, &b)) continue;
                    uint16_t rem_ntok[MOE_MAX_EXPERTS]; int16_t last_eid;
                    int n_rem = rem_excluding(rem, nrem, eK, -1, rem_ntok, &last_eid);
                    if (n_rem == 0) continue;
                    uint32_t cost = eval_cost(&a, &b, rem_ntok, n_rem, last_eid);
                    if (cand_better(&best, cost, &a, &b)) {
                        cand_update(&best, cost, &a, &b, eK, -1, -1,
                                    (uint8_t)(dir == 0), (uint8_t)(dir == 1));
                    }
                }
            }
        }

        if (best.cost == UINT32_MAX) {
            /* fallback: SOLO top0 on C2 */
            snap_t sn;
            snap_assign(&sn, now, MOE_SHAPE_B, MOE_SHAPE_B, n0, e0, c2c_0);
            apply_s2_down_prefetch_single(&sn, MOE_SHAPE_B, NULL);
            c2 = sn;
            if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
            for (int i=0;i+1<nrem;i++) rem[i]=rem[i+1];
            nrem--;
            continue;
        }

        /* Apply best candidate */
        if (best.emit_pre) {
            if (!emit_task(out, best.pre_cluster, &best.pre_emit)) return MOE_ERR_OVERFLOW;
        }
        c2 = best.c2_after;
        c3 = best.c3_after;
        if (best.emit_c2) {
            if (!emit_task(out, MOE_CLUSTER_C2, &c2)) return MOE_ERR_OVERFLOW;
        }
        if (best.emit_c3) {
            if (!emit_task(out, MOE_CLUSTER_C3, &c3)) return MOE_ERR_OVERFLOW;
        }
        /* Remove consumed experts from rem[] */
        int16_t consumed[3];
        consumed[0] = best.consumed_eid_a;
        consumed[1] = best.consumed_eid_b;
        consumed[2] = best.consumed_eid_c;
        int w = 0;
        for (int i = 0; i < nrem; i++) {
            int skip = 0;
            for (int j = 0; j < 3; j++) {
                if (consumed[j] >= 0 && rem[i].eid == consumed[j]) skip = 1;
            }
            if (!skip) rem[w++] = rem[i];
        }
        nrem = w;
    }
#endif  /* MOE_SCHEDULE_FAST */

    if (nrem > 0) return MOE_ERR_OVERFLOW;
    out->est_makespan_cc = u32_max(c2.task_end, c3.task_end);

    /* ────────────────────────────────────────────────────────
     *  Post-pass: fill prefetch_eid hints.
     *  For each task on cluster X, find the next task on the SAME
     *  cluster whose expert_id differs; that expert is the
    *  prefetch candidate.  This is advisory metadata: current schedule timing
    *  does not consume the hint as a later S1 cache hit.
     * ──────────────────────────────────────────────────────── */
    for (int i = 0; i < out->n_tasks; i++) {
        moe_task_t *ti = &out->tasks[i];
        for (int j = i + 1; j < out->n_tasks; j++) {
            if (out->tasks[j].cluster != ti->cluster) continue;
            if (out->tasks[j].expert_id != ti->expert_id) {
                ti->prefetch_eid = (int16_t)out->tasks[j].expert_id;
            }
            break;   /* stop at the first same-cluster successor */
        }
    }

    return MOE_OK;
}

static int find_req_expert(const moe_request_t *req, int16_t eid) {
    if (req == NULL || eid < 0) return -1;
    for (uint16_t i = 0; i < req->n_experts && i < MOE_MAX_EXPERTS; i++) {
        if ((int16_t)req->experts[i].expert_id == eid) return (int)i;
    }
    return -1;
}

moe_status_t moe_schedule(const moe_request_t *req, moe_schedule_t *out) {
    uint8_t valid = 0;
    uint8_t masks[4];
    int n_masks = 0;
    moe_schedule_t best;
    moe_status_t best_status = MOE_ERR_INTERNAL;
    uint8_t have_best = 0;

    if (req != NULL) {
        if (find_req_expert(req, req->cache_eid_c2) >= 0) valid |= INIT_CACHE_C2;
        if (find_req_expert(req, req->cache_eid_c3) >= 0) valid |= INIT_CACHE_C3;
    }
    masks[n_masks++] = 0u;
    if (valid & INIT_CACHE_C2) masks[n_masks++] = INIT_CACHE_C2;
    if (valid & INIT_CACHE_C3) masks[n_masks++] = INIT_CACHE_C3;
    if ((valid & (INIT_CACHE_C2 | INIT_CACHE_C3)) == (INIT_CACHE_C2 | INIT_CACHE_C3)) {
        masks[n_masks++] = INIT_CACHE_C2 | INIT_CACHE_C3;
    }

    for (int i = 0; i < n_masks; i++) {
        moe_schedule_t cand;
        moe_status_t st = moe_schedule_impl(req, &cand, masks[i]);
        if (st != MOE_OK) {
            if (!have_best && best_status == MOE_ERR_INTERNAL) best_status = st;
            continue;
        }
        if (!have_best || cand.est_makespan_cc < best.est_makespan_cc) {
            best = cand;
            best_status = MOE_OK;
            have_best = 1;
        }
    }

    if (best_status == MOE_OK && out != NULL) *out = best;
    return best_status;
}
