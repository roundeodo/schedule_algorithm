/* test_moe_scheduler.c — host-side smoke test for moe_scheduler.
 * Builds standalone with: gcc -O2 -Wall -o test test_moe_scheduler.c moe_scheduler.c
 * Compares makespan estimates against a few hand-checked Python cases.
 */
#include "moe_scheduler.h"
#include <stdio.h>
#include <string.h>

static const char *shape_str(moe_shape_t s) {
    switch (s) { case MOE_SHAPE_A: return "A"; case MOE_SHAPE_B: return "B";
                 case MOE_SHAPE_C: return "C"; default: return "?"; }
}

static void run(const char *name, moe_request_t *req) {
    moe_schedule_t sch;
    moe_status_t st = moe_schedule(req, &sch);
    printf("\n=== %s ===\n", name);
    printf("status=%d  n_tasks=%u  est_makespan=%u cc\n",
           st, sch.n_tasks, sch.est_makespan_cc);
    for (int i = 0; i < sch.n_tasks; i++) {
        moe_task_t *t = &sch.tasks[i];
        printf("  [%d] cluster=%s eid=%u ntok=%u s1=%s s3=%s skipDMA=%u "
               "[%u..%u]\n",
               i, t->cluster == MOE_CLUSTER_C2 ? "C2" : "C3",
               t->expert_id, t->ntokens,
               shape_str(t->shape_s1), shape_str(t->shape_s3),
               t->skip_dma_s1, t->est_start_cc, t->est_end_cc);
    }
}

int main(void) {
    /* Case 1: single expert, 32 tokens, no cache → SPLIT or PAIR-style equiv. */
    moe_request_t r1 = {0};
    r1.experts[0].expert_id = 0; r1.experts[0].ntokens = 32;
    r1.n_experts = 1; r1.cache_eid_c2 = -1; r1.cache_eid_c3 = -1;
    run("single expert, 32 tok, no cache", &r1);

    /* Case 2: two equal experts, 16+16, no cache → PAIR */
    moe_request_t r2 = {0};
    r2.experts[0] = (moe_expert_load_t){0, 16};
    r2.experts[1] = (moe_expert_load_t){1, 16};
    r2.n_experts = 2; r2.cache_eid_c2 = -1; r2.cache_eid_c3 = -1;
    run("two experts 16+16, no cache", &r2);

    /* Case 3: hot+cold, 24+8, no cache */
    moe_request_t r3 = {0};
    r3.experts[0] = (moe_expert_load_t){5, 24};
    r3.experts[1] = (moe_expert_load_t){2,  8};
    r3.n_experts = 2; r3.cache_eid_c2 = -1; r3.cache_eid_c3 = -1;
    run("two experts 24+8, no cache", &r3);

    /* Case 4: 4 experts, 16/8/4/4, cache hint top0→C2 */
    moe_request_t r4 = {0};
    r4.experts[0] = (moe_expert_load_t){0, 16};
    r4.experts[1] = (moe_expert_load_t){1,  8};
    r4.experts[2] = (moe_expert_load_t){2,  4};
    r4.experts[3] = (moe_expert_load_t){3,  4};
    r4.n_experts = 4; r4.cache_eid_c2 = 0; r4.cache_eid_c3 = -1;
    run("4 experts 16/8/4/4, cache C2=0", &r4);

    /* Case 5: tiny — single expert 4 tokens, no cache (worst beam case) */
    moe_request_t r5 = {0};
    r5.experts[0] = (moe_expert_load_t){0, 4};
    r5.n_experts = 1; r5.cache_eid_c2 = -1; r5.cache_eid_c3 = -1;
    run("single expert 4 tok, no cache", &r5);

    return 0;
}
