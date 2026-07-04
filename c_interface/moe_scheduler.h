/* moe_scheduler.h
 * --------------------------------------------------------------------------
 * Analytical greedy scheduler for HeMAiA two-cluster MoE workload.
 * Pure host-side (CVA6) function. No FP, no malloc, no OS deps.
 *
 * Pipeline-stage cycle constants (per 1 token, on a 1-cluster x 2-versacore
 * compute fabric @ 512 MAC/cc) are hard-coded in moe_scheduler.c and MUST
 * be kept in sync with Idea_Model/four_stage_scheduler.py.
 * --------------------------------------------------------------------------
 */
#ifndef MOE_SCHEDULER_H
#define MOE_SCHEDULER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── compile-time limits (sized for HeMAiA 2-cluster, top-K<=2 routing) ── */
#define MOE_MAX_EXPERTS    8     /* per request                        */
#define MOE_MAX_TASKS     32     /* upper bound for schedule length    */

/* ── Shape enum (mirrors ShapeA/B/C in four_stage_scheduler.py) ───────── */
typedef enum {
    MOE_SHAPE_A = 0,   /* M_dim=8, alloc=64,  bw_req=32   */
    MOE_SHAPE_B = 1,   /* M_dim=4, alloc=64,  bw_req=64   */
    MOE_SHAPE_C = 2    /* M_dim=2, alloc=128, bw_req=128  */
} moe_shape_t;

/* ── Cluster id (compile-time mapping) ───────────────────────────────── */
typedef enum {
    MOE_CLUSTER_C2 = 0,
    MOE_CLUSTER_C3 = 1
} moe_cluster_t;

/* ── Input: per-expert token count after routing ──────────────────────── */
typedef struct {
    uint16_t expert_id;       /* 0..N_EXPERTS-1                          */
    uint16_t ntokens;         /* number of tokens routed to this expert  */
} moe_expert_load_t;

typedef struct {
    moe_expert_load_t experts[MOE_MAX_EXPERTS];
    uint16_t  n_experts;      /* number of valid entries (<=MOE_MAX_EXPERTS) */
    int16_t   cache_eid_c2;   /* expert resident in C2 weight SRAM, -1 = none */
    int16_t   cache_eid_c3;   /* expert resident in C3 weight SRAM, -1 = none */
} moe_request_t;

/* ── Output: one scheduled task = (cluster, expert, ntoks, shapes) ────── */
typedef struct {
    moe_cluster_t cluster;    /* which cluster runs this task            */
    uint16_t  expert_id;      /* expert to run                           */
    uint16_t  ntokens;        /* slice of tokens (after possible SPLIT)  */
    moe_shape_t shape_s1;     /* shape used for stage S1 (gate+up DMA)   */
    moe_shape_t shape_s3;     /* shape used for stage S3 (down DMA)      */
    uint8_t   skip_dma_s1;    /* 1 if weight already in cluster L1       */
    uint8_t   skip_dma_s3;    /* 1 if down weight already in cluster L1  */
    int16_t   prefetch_eid;   /* expert id to prefetch during S4, -1=none */
    uint8_t   _pad;
    uint32_t  est_start_cc;   /* analytical estimate, dispatch hint only */
    uint32_t  est_end_cc;     /* analytical estimate, dispatch hint only */
} moe_task_t;

typedef struct {
    moe_task_t tasks[MOE_MAX_TASKS];
    uint16_t   n_tasks;       /* number of valid tasks                    */
    uint16_t   _pad;
    uint32_t   est_makespan_cc;  /* total estimated cycles                */
} moe_schedule_t;

/* ── Return codes ─────────────────────────────────────────────────────── */
typedef enum {
    MOE_OK              =  0,
    MOE_ERR_BAD_INPUT   = -1,  /* n_experts=0 or >MAX, or ntokens=0      */
    MOE_ERR_OVERFLOW    = -2,  /* generated more than MOE_MAX_TASKS      */
    MOE_ERR_INTERNAL    = -3   /* unreachable branch, indicates bug      */
} moe_status_t;

/* ─────────────────────────────────────────────────────────────────────────
 *  moe_schedule
 *  ------------
 *  Pure function: given the per-expert token distribution and current cache
 *  residency, fills `out` with an ordered list of cluster tasks.
 *
 *  Caller workflow on host:
 *      moe_request_t req = { ... };           // fill from router output
 *      moe_schedule_t sch;
 *      if (moe_schedule(&req, &sch) != MOE_OK) { handle_error(); }
 *      for (int i = 0; i < sch.n_tasks; i++) {
 *          dispatch_to_cluster(&sch.tasks[i]); // mailbox / SPM write
 *      }
 *
 *  The function is deterministic and reentrant. Runtime depends on how many
 *  candidate branches are enumerated for the current request. No dynamic
 *  memory is used.
 * ───────────────────────────────────────────────────────────────────────── */
moe_status_t moe_schedule(const moe_request_t *req, moe_schedule_t *out);

#ifdef __cplusplus
}
#endif
#endif /* MOE_SCHEDULER_H */
