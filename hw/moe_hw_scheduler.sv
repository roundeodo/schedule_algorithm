// =============================================================================
// moe_hw_scheduler.sv
// MoE Layer Hardware Scheduler
//
// Implements a simplified beam-search-derived scheduling policy for dispatching
// experts to two compute clusters (C2, C3).
//
// Design decisions:
//   ACTION SELECTION  (CART decision tree, depth 5, trained on 32826 beam-search steps):
//     Features: top0_ntok, top1_ntok, top2_ntok, (top0-top1), q_rem,
//               both_idle, c2_cache_rank, c3_cache_rank
//     Training accuracy: 77.4%  (vs 70.0% for previous hand-written rules)
//
//   SHAPE SELECTION   (CART decision tree, depth 4, trained on same dataset):
//     c2_s1 accuracy: 80.8%  (vs 63.0% hand-written)
//     c2_s3 accuracy: 84.5%
//     c3_s1 accuracy: 79.0%  (vs 66.5% hand-written)
//     c3_s3 accuracy: 81.8%
//     Hard override: cache_hit → SH_A regardless of DT output (physical constraint)
//
// Shape encoding: 2'b00=A (M8, bw_req=32 B/cc,  alloc=64,  T_s1=90,112 cc, T_s3=45,056 cc)
//                 2'b01=B (M4, bw_req=64 B/cc,  alloc=64,  T_s1=45,056 cc, T_s3=22,528 cc)
//                 2'b10=C (M2, bw_req=128 B/cc, alloc=128, T_s1=22,528 cc, T_s3=11,264 cc)
//
// Interface:
//   Input  → sorted Top-K routing result (descending token count)
//             + cluster SRAM cache state from previous layer
//             + cluster done pulses
//   Output → per-cluster assignment packets (one valid pulse each assignment)
//
// =============================================================================
`default_nettype none
`timescale 1ns / 1ps

module moe_hw_scheduler #(
  parameter int MAX_EXP   = 8,    // max experts per MoE layer (≤ 8)
  parameter int TOK_W     = 7,    // bits for token count  (0 – 127)
  parameter int EID_W     = 3,    // bits for expert ID    (0 – 7)
  parameter int SPLIT_THR = 4     // min tok0 to consider SPLIT (data: threshold=4 gives 100% accuracy)
) (
  input  logic              clk,
  input  logic              rst_n,

  // ── Top-K routing input (sorted descending by token count) ─────────────────
  // Present for one cycle when a new MoE layer begins.
  input  logic              valid_i,
  input  logic [3:0]        n_experts_i,          // 1 – MAX_EXP active experts
  input  logic [EID_W-1:0]  eid_i  [0:MAX_EXP-1], // expert IDs
  input  logic [TOK_W-1:0]  ntok_i [0:MAX_EXP-1], // token counts (descending)

  // ── Cluster cache state (what remains in SRAM from the previous layer) ─────
  // The scheduler uses this to detect cache hits and skip S1 DMA.
  input  logic              c2_cache_valid_i,
  input  logic [EID_W-1:0]  c2_cache_eid_i,
  input  logic              c3_cache_valid_i,
  input  logic [EID_W-1:0]  c3_cache_eid_i,

  // ── Cluster done feedback (one-cycle pulse when cluster finishes expert) ───
  input  logic              c2_done_i,
  input  logic              c3_done_i,

  // ── C2 assignment output ───────────────────────────────────────────────────
  output logic              c2_assign_o,    // high for one cycle when assigning
  output logic [EID_W-1:0]  c2_eid_o,
  output logic [TOK_W-1:0]  c2_ntok_o,
  output logic [1:0]        c2_shape_s1_o,  // ShapeA/B/C for SwishGLU DMA
  output logic [1:0]        c2_shape_s3_o,  // ShapeA/B/C for DownProj DMA
  output logic              c2_cached_o,    // 1 → skip S1 DMA (cache hit)

  // ── C3 assignment output ───────────────────────────────────────────────────
  output logic              c3_assign_o,
  output logic [EID_W-1:0]  c3_eid_o,
  output logic [TOK_W-1:0]  c3_ntok_o,
  output logic [1:0]        c3_shape_s1_o,
  output logic [1:0]        c3_shape_s3_o,
  output logic              c3_cached_o,

  // ── Status ─────────────────────────────────────────────────────────────────
  output logic              ready_o,   // high when idle, ready for new layer
  output logic              done_o     // high when all experts dispatched & clusters idle
);

  // ── Shape constants ──────────────────────────────────────────────────────────
  localparam logic [1:0] SH_A = 2'b00;  // M_dim=8, bw_req=32 B/cc,  alloc=64,  T_s1=90112 cc, T_s3=45056 cc
  localparam logic [1:0] SH_B = 2'b01;  // M_dim=4, bw_req=64 B/cc,  alloc=64,  T_s1=45056 cc, T_s3=22528 cc
  localparam logic [1:0] SH_C = 2'b10;  // M_dim=2, bw_req=128 B/cc, alloc=128, T_s1=22528 cc, T_s3=11264 cc

  // ── Action type encoding ────────────────────────────────────────────────────
  localparam logic [1:0] ACT_PAIR    = 2'd0;  // top-0→C2, top-1→C3
  localparam logic [1:0] ACT_SPLIT   = 2'd1;  // split top-0 between C2 and C3
  localparam logic [1:0] ACT_SNGL_C2 = 2'd2;  // top-0→C2 only
  localparam logic [1:0] ACT_SNGL_C3 = 2'd3;  // top-0→C3 only

  // ── FSM state encoding ──────────────────────────────────────────────────────
  typedef enum logic [1:0] {
    ST_IDLE   = 2'd0,  // waiting for valid_i
    ST_ASSIGN = 2'd1,  // emit next assignment(s)
    ST_WAIT   = 2'd2,  // waiting for a cluster done signal
    ST_DONE   = 2'd3   // all experts dispatched; waiting for clusters to idle
  } state_t;

  state_t state, state_nxt;

  // ── Expert queue ─────────────────────────────────────────────────────────────
  // Sorted descending by token count; top-K router guarantees this ordering.
  logic [EID_W-1:0]  q_eid  [0:MAX_EXP-1];
  logic [TOK_W-1:0]  q_ntok [0:MAX_EXP-1];
  logic [3:0]        q_head;    // index of next un-assigned expert
  logic [3:0]        q_total;   // number of experts loaded for this layer

  // Remaining count (combinational)
  wire  [3:0]        q_rem   = q_total - q_head;
  wire               q_empty = (q_head >= q_total);

  // Queue aliases for the top two entries.
  // top1 is only valid when q_rem >= 2; guard prevents out-of-range access.
  wire [EID_W-1:0]  top0_eid  = q_eid [q_head];
  wire [TOK_W-1:0]  top0_ntok = q_ntok[q_head];
  wire [3:0]        top1_idx  = (q_rem > 4'd1) ? q_head + 4'd1 : 4'd0;
  wire [EID_W-1:0]  top1_eid  = q_eid [top1_idx];
  wire [TOK_W-1:0]  top1_ntok = (q_rem > 4'd1) ? q_ntok[top1_idx] : '0;
  wire [3:0]        top2_idx  = (q_rem > 4'd2) ? q_head + 4'd2 : 4'd0;
  wire [TOK_W-1:0]  top2_ntok = (q_rem > 4'd2) ? q_ntok[top2_idx] : '0;

  // ── Cluster busy tracking ─────────────────────────────────────────────────
  logic c2_busy_r, c3_busy_r;

  // Effective busy: accounts for done pulse arriving this cycle
  wire  c2_busy_eff = c2_busy_r && !c2_done_i;
  wire  c3_busy_eff = c3_busy_r && !c3_done_i;
  wire  both_idle   = !c2_busy_eff && !c3_busy_eff;

  // ── Cache state tracking ──────────────────────────────────────────────────
  // Tracks which expert's SwishGLU weights are still in each cluster's SRAM.
  // Updated when: (a) new layer arrives with externally-reported cache state,
  //               (b) cluster finishes an expert → remember the completed eid.
  logic              c2_cache_v_r;
  logic [EID_W-1:0]  c2_cache_eid_r;
  logic              c3_cache_v_r;
  logic [EID_W-1:0]  c3_cache_eid_r;

  // ── Action decision (CART DT depth-5, trained on 32826 beam-search steps) ──
  // Features: top0_ntok, top1_ntok, top2_ntok, tok_diff=(top0-top1),
  //           q_rem, both_idle, c2_cache_rank, c3_cache_rank
  // Accuracy: 77.4% on training set (vs 70.0% hand-written)
  logic [1:0] action;
  wire [TOK_W-1:0] tok_diff = top0_ntok - top1_ntok;

  always_comb begin
    action = ACT_SNGL_C2;  // default (unreachable in normal operation)
    if (tok_diff <= 7'(5)) begin
      if (top1_ntok <= 7'(0)) begin
        if (top0_ntok <= 7'(3)) begin
          if (both_idle <= 1'(0)) begin
            action = ACT_SNGL_C2;
          end else begin
            if (top0_ntok <= 7'(2)) begin
              action = ACT_SNGL_C2;
            end else begin
              action = ACT_SPLIT;
            end
          end
        end else begin
          if (both_idle <= 1'(0)) begin
            if (top0_ntok <= 7'(4)) begin
              action = ACT_SNGL_C2;
            end else begin
              action = ACT_SPLIT;
            end
          end else begin
            action = ACT_SPLIT;
          end
        end
      end else begin
        if (top2_ntok <= 7'(4)) begin
          if (both_idle <= 1'(0)) begin
            if (top2_ntok <= 7'(0)) begin
              action = ACT_PAIR;
            end else begin
              action = ACT_SNGL_C3;
            end
          end else begin
            if (top0_ntok <= 7'(4)) begin
              action = ACT_PAIR;
            end else begin
              action = ACT_PAIR;
            end
          end
        end else begin
          if (both_idle <= 1'(0)) begin
            if (top1_ntok <= 7'(8)) begin
              action = ACT_SNGL_C3;
            end else begin
              action = ACT_SPLIT;
            end
          end else begin
            if (c3_cache_rank <= 3'(0)) begin
              action = ACT_SNGL_C2;
            end else begin
              action = ACT_PAIR;
            end
          end
        end
      end
    end else begin
      if (top2_ntok <= 7'(0)) begin
        if (both_idle <= 1'(0)) begin
          if (tok_diff <= 7'(6)) begin
            if (top0_ntok <= 7'(6)) begin
              action = ACT_SPLIT;
            end else begin
              action = ACT_SNGL_C2;
            end
          end else begin
            action = ACT_SPLIT;
          end
        end else begin
          if (tok_diff <= 7'(6)) begin
            action = ACT_SPLIT;
          end else begin
            if (top1_ntok <= 7'(0)) begin
              action = ACT_SPLIT;
            end else begin
              action = ACT_SPLIT;
            end
          end
        end
      end else begin
        if (both_idle <= 1'(0)) begin
          if (c2_cache_rank <= 3'(3)) begin
            action = ACT_SPLIT;
          end else begin
            if (q_rem <= 4'(4)) begin
              action = ACT_SNGL_C3;
            end else begin
              action = ACT_SNGL_C3;
            end
          end
        end else begin
          if (q_rem <= 4'(3)) begin
            if (top0_ntok <= 7'(8)) begin
              action = ACT_SNGL_C2;
            end else begin
              action = ACT_SPLIT;
            end
          end else begin
            if (q_rem <= 4'(5)) begin
              action = ACT_SNGL_C2;
            end else begin
              action = ACT_PAIR;
            end
          end
        end
      end
    end
  end

  // ── SPLIT point calculation ───────────────────────────────────────────────
  // C2 gets ceil(ntok/2) tokens, C3 gets floor(ntok/2) tokens.
  // Using ceil/floor (not mul-of-4 alignment) ensures balanced load:
  //   ntok=4 → (2,2); ntok=5 → (3,2); ntok=6 → (3,3); ntok=8 → (4,4).
  // Both clusters use ShapeB (alloc=64 each, 128 B/cc total = BW limit).
  logic [TOK_W-1:0] split_c2, split_c3;

  always_comb begin
    split_c2 = (top0_ntok + TOK_W'(1)) >> 1;  // ceil(ntok/2)
    split_c3 = top0_ntok - split_c2;           // floor(ntok/2)
  end

  // ── Cache hit detection ───────────────────────────────────────────────────
  // A hit means the expert's SwishGLU weights are already in the cluster's SRAM,
  // so we can skip Stage-1 DMA entirely.
  wire c2_hit_top0 = c2_cache_v_r && (c2_cache_eid_r == top0_eid);
  wire c3_hit_top0 = c3_cache_v_r && (c3_cache_eid_r == top0_eid);
  wire c3_hit_top1 = c3_cache_v_r && (c3_cache_eid_r == top1_eid);

  // ── Cache rank: 0-based position of cached expert in remaining queue ────────
  // 7 = not cached / not found in remaining queue.
  // Used as a feature by the DT to approximate cache-hit probability.
  logic [2:0] c2_cache_rank, c3_cache_rank;
  always_comb begin
    c2_cache_rank = 3'd7;
    c3_cache_rank = 3'd7;
    // Scan from back to front so the lowest-index (hottest) match wins.
    for (int i = 7; i >= 0; i--) begin
      if (c2_cache_v_r && (q_head + 4'(i)) < q_total &&
          q_eid[q_head + 4'(i)] == c2_cache_eid_r)
        c2_cache_rank = 3'(i);
      if (c3_cache_v_r && (q_head + 4'(i)) < q_total &&
          q_eid[q_head + 4'(i)] == c3_cache_eid_r)
        c3_cache_rank = 3'(i);
    end
  end

  // ── Shape selection (CART DT depth-4, trained on 32826 beam-search steps) ──
  // Hard override: cache_hit → SH_A (DMA skipped; shape physically irrelevant).
  // DT predicts shape for non-cached cases using:
  //   top0_ntok, top1_ntok, top2_ntok, tok_diff, q_rem, both_idle,
  //   c2_cache_rank, c3_cache_rank
  // Accuracies: c2_s1=80.8%, c2_s3=84.5%, c3_s1=79.0%, c3_s3=81.8%
  logic              c2_cache_hit,  c3_cache_hit;
  logic [1:0]        c2_s1_sel, c2_s3_sel;
  logic [1:0]        c3_s1_sel, c3_s3_sel;

  always_comb begin
    // Cache hits (hard physical constraint)
    c2_cache_hit = c2_hit_top0;
    c3_cache_hit = (action == ACT_PAIR) ? c3_hit_top1 : c3_hit_top0;

    // ── C2 S1 shape (DT, accuracy 80.8%) ─────────────────────────────────
    if (c2_cache_hit) begin
      c2_s1_sel = SH_A;  // cache hit: S1 DMA skipped
    end else if (top0_ntok <= 7'(12)) begin
      if (top1_ntok <= 7'(6)) begin
        if (tok_diff <= 7'(0)) begin
          c2_s1_sel = (c2_cache_rank <= 3'(3)) ? SH_C : SH_B;
        end else begin
          c2_s1_sel = (top0_ntok <= 7'(1)) ? SH_C : SH_B;
        end
      end else begin
        if (top2_ntok <= 7'(1)) begin
          c2_s1_sel = SH_A;
        end else begin
          c2_s1_sel = SH_B;
        end
      end
    end else begin
      if (q_rem <= 4'(3)) begin
        if (top2_ntok <= 7'(0)) begin
          c2_s1_sel = (top1_ntok <= 7'(2)) ? SH_A : SH_A;
        end else begin
          c2_s1_sel = (top1_ntok <= 7'(2)) ? SH_C : SH_A;
        end
      end else begin
        if (top2_ntok <= 7'(4)) begin
          c2_s1_sel = (q_rem <= 4'(6)) ? SH_C : SH_B;
        end else begin
          c2_s1_sel = (top2_ntok <= 7'(6)) ? SH_B : SH_A;
        end
      end
    end

    // ── C2 S3 shape (DT, accuracy 84.5%) ─────────────────────────────────
    if (top0_ntok <= 7'(8)) begin
      if (top1_ntok <= 7'(0)) begin
        c2_s3_sel = (top0_ntok <= 7'(2)) ? SH_C : SH_B;
      end else begin
        if (top1_ntok <= 7'(6)) begin
          c2_s3_sel = SH_B;
        end else begin
          c2_s3_sel = (top2_ntok <= 7'(1)) ? SH_A : SH_B;
        end
      end
    end else begin
      if (q_rem <= 4'(3)) begin
        if (top0_ntok <= 7'(12)) begin
          c2_s3_sel = (tok_diff <= 7'(4)) ? SH_A : SH_B;
        end else begin
          c2_s3_sel = (top2_ntok <= 7'(0)) ? SH_A : SH_A;
        end
      end else begin
        if (top2_ntok <= 7'(2)) begin
          c2_s3_sel = (q_rem <= 4'(6)) ? SH_C : SH_B;
        end else begin
          c2_s3_sel = (top2_ntok <= 7'(8)) ? SH_B : SH_A;
        end
      end
    end

    // ── C3 S1 shape (DT, accuracy 79.0%) ─────────────────────────────────
    if (c3_cache_hit) begin
      c3_s1_sel = SH_A;  // cache hit: S1 DMA skipped
    end else if (top0_ntok <= 7'(12)) begin
      if (top1_ntok <= 7'(4)) begin
        if (c2_cache_rank <= 3'(3)) begin
          c3_s1_sel = (top0_ntok <= 7'(4)) ? SH_C : SH_B;
        end else begin
          c3_s1_sel = SH_B;
        end
      end else begin
        if (top2_ntok <= 7'(0)) begin
          c3_s1_sel = (top1_ntok <= 7'(6)) ? SH_B : SH_A;
        end else begin
          c3_s1_sel = SH_B;
        end
      end
    end else begin
      if (q_rem <= 4'(3)) begin
        if (both_idle) begin
          c3_s1_sel = (top2_ntok <= 7'(0)) ? SH_A : SH_A;
        end else begin
          c3_s1_sel = (tok_diff <= 7'(1)) ? SH_A : SH_A;
        end
      end else begin
        if (!both_idle) begin  // DT: both_idle<=0.5 (NOT idle) → tok2-based
          c3_s1_sel = (top2_ntok <= 7'(2)) ? SH_C : SH_B;
        end else begin          // both_idle → n_active-based
          c3_s1_sel = (q_rem <= 4'(5)) ? SH_A : SH_B;
        end
      end
    end

    // ── C3 S3 shape (DT, accuracy 81.8%) ─────────────────────────────────
    if (top0_ntok <= 7'(12)) begin
      if (!both_idle) begin  // DT: both_idle<=0.5 (NOT idle) → c3_rank-based
        if (c3_cache_rank <= 3'(0)) begin
          c3_s3_sel = (top2_ntok <= 7'(2)) ? SH_A : SH_C;
        end else begin
          c3_s3_sel = (top0_ntok <= 7'(2)) ? SH_C : SH_B;
        end
      end else begin          // both_idle → tok1-based
        if (top1_ntok <= 7'(6)) begin
          c3_s3_sel = SH_B;
        end else begin
          c3_s3_sel = (top2_ntok <= 7'(2)) ? SH_A : SH_B;
        end
      end
    end else begin
      if (top2_ntok <= 7'(0)) begin
        if (top1_ntok <= 7'(2)) begin
          c3_s3_sel = SH_A;
        end else begin
          c3_s3_sel = (top1_ntok <= 7'(4)) ? SH_B : SH_A;
        end
      end else begin
        if (top2_ntok <= 7'(2)) begin
          c3_s3_sel = (both_idle) ? SH_B : SH_C;
        end else begin
          c3_s3_sel = (q_rem <= 4'(3)) ? SH_A : SH_B;
        end
      end
    end
  end

  // ── Assignment combinational outputs ──────────────────────────────────────
  // These are the "next" values to be registered on the ST_ASSIGN clock edge.
  logic              nx_c2_assign, nx_c3_assign;
  logic [EID_W-1:0]  nx_c2_eid,   nx_c3_eid;
  logic [TOK_W-1:0]  nx_c2_ntok,  nx_c3_ntok;
  logic [1:0]        nx_c2_s1,    nx_c3_s1;
  logic [1:0]        nx_c2_s3,    nx_c3_s3;
  logic              nx_c2_hit,   nx_c3_hit;
  logic [3:0]        q_advance;    // how many queue entries to consume

  always_comb begin
    nx_c2_assign = 1'b0;  nx_c3_assign = 1'b0;
    nx_c2_eid    = '0;    nx_c3_eid    = '0;
    nx_c2_ntok   = '0;    nx_c3_ntok   = '0;
    nx_c2_s1     = SH_A;  nx_c3_s1     = SH_A;
    nx_c2_s3     = SH_A;  nx_c3_s3     = SH_A;
    nx_c2_hit    = 1'b0;  nx_c3_hit    = 1'b0;
    q_advance    = 4'd0;

    if (!q_empty) begin
      unique case (action)
        ACT_PAIR: begin
          nx_c2_assign = 1'b1;
          nx_c2_eid    = top0_eid;  nx_c2_ntok = top0_ntok;
          nx_c2_s1     = c2_s1_sel; nx_c2_s3   = c2_s3_sel;
          nx_c2_hit    = c2_cache_hit;
          if (q_rem > 4'd1) begin
            nx_c3_assign = 1'b1;
            nx_c3_eid    = top1_eid;  nx_c3_ntok = top1_ntok;
            nx_c3_s1     = c3_s1_sel; nx_c3_s3   = c3_s3_sel;
            nx_c3_hit    = c3_cache_hit;
            q_advance    = 4'd2;
          end else begin
            q_advance    = 4'd1;
          end
        end

        ACT_SPLIT: begin
          // Both clusters process the same expert with different token slices
          nx_c2_assign = 1'b1;
          nx_c2_eid    = top0_eid;  nx_c2_ntok = split_c2;
          nx_c2_s1     = c2_s1_sel; nx_c2_s3   = c2_s3_sel;
          nx_c2_hit    = c2_cache_hit;
          nx_c3_assign = 1'b1;
          nx_c3_eid    = top0_eid;  nx_c3_ntok = split_c3;
          nx_c3_s1     = c3_s1_sel; nx_c3_s3   = c3_s3_sel;
          nx_c3_hit    = 1'b0;      // only one cluster can use the cache
          q_advance    = 4'd1;      // one expert consumed
        end

        ACT_SNGL_C2: begin
          nx_c2_assign = 1'b1;
          nx_c2_eid    = top0_eid;  nx_c2_ntok = top0_ntok;
          nx_c2_s1     = c2_s1_sel; nx_c2_s3   = c2_s3_sel;
          nx_c2_hit    = c2_cache_hit;
          q_advance    = 4'd1;
        end

        ACT_SNGL_C3: begin
          nx_c3_assign = 1'b1;
          nx_c3_eid    = top0_eid;  nx_c3_ntok = top0_ntok;
          nx_c3_s1     = c3_s1_sel; nx_c3_s3   = c3_s3_sel;
          nx_c3_hit    = c3_cache_hit;
          q_advance    = 4'd1;
        end

        default: ;
      endcase
    end
  end

  // ── FSM: next-state logic ─────────────────────────────────────────────────
  always_comb begin
    state_nxt = state;
    unique case (state)
      ST_IDLE:
        if (valid_i && n_experts_i > 4'd0) state_nxt = ST_ASSIGN;

      ST_ASSIGN:
        // After emitting assignment(s), go wait for cluster(s) to finish.
        // If nothing to assign (shouldn't happen normally), go to DONE.
        state_nxt = q_empty ? ST_DONE : ST_WAIT;

      ST_WAIT: begin
        // A done signal while there are remaining experts → go emit next assignment
        if ((c2_done_i || c3_done_i) && !q_empty)
          state_nxt = ST_ASSIGN;
        // All experts already dispatched AND both clusters now idle → done
        else if (both_idle && q_empty)
          state_nxt = ST_DONE;
      end

      ST_DONE:
        if (!c2_busy_r && !c3_busy_r) state_nxt = ST_IDLE;

      default: state_nxt = ST_IDLE;
    endcase
  end

  // ── FSM: sequential logic ─────────────────────────────────────────────────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= ST_IDLE;
      q_head         <= 4'd0;
      q_total        <= 4'd0;
      c2_busy_r      <= 1'b0;
      c3_busy_r      <= 1'b0;
      c2_cache_v_r   <= 1'b0;
      c2_cache_eid_r <= '0;
      c3_cache_v_r   <= 1'b0;
      c3_cache_eid_r <= '0;
      // Output resets
      c2_assign_o    <= 1'b0;  c3_assign_o    <= 1'b0;
      c2_eid_o       <= '0;    c3_eid_o       <= '0;
      c2_ntok_o      <= '0;    c3_ntok_o      <= '0;
      c2_shape_s1_o  <= SH_A;  c3_shape_s1_o  <= SH_A;
      c2_shape_s3_o  <= SH_A;  c3_shape_s3_o  <= SH_A;
      c2_cached_o    <= 1'b0;  c3_cached_o    <= 1'b0;

    end else begin
      // Default: deassert single-cycle assignment pulses
      c2_assign_o <= 1'b0;
      c3_assign_o <= 1'b0;

      // ── Cluster done handling (any state) ──────────────────────────────
      // Clear busy flag; record completed expert as the new cache occupant.
      // (Its SwishGLU weights remain in SRAM Region_SWISH until overwritten.)
      if (c2_done_i) begin
        c2_busy_r      <= 1'b0;
        c2_cache_v_r   <= 1'b1;
        c2_cache_eid_r <= c2_eid_o;   // expert that just finished
      end
      if (c3_done_i) begin
        c3_busy_r      <= 1'b0;
        c3_cache_v_r   <= 1'b1;
        c3_cache_eid_r <= c3_eid_o;
      end

      // ── State transitions ────────────────────────────────────────────────
      state <= state_nxt;

      unique case (state)

        // ── IDLE ────────────────────────────────────────────────────────────
        ST_IDLE: begin
          if (valid_i && n_experts_i > 4'd0) begin
            // Load expert queue from sorted Top-K input
            q_total <= n_experts_i;
            q_head  <= 4'd0;
            for (int i = 0; i < MAX_EXP; i++) begin
              q_eid [i] <= eid_i [i];
              q_ntok[i] <= ntok_i[i];
            end
            // Initialise cache from external state report
            c2_cache_v_r   <= c2_cache_valid_i;
            c2_cache_eid_r <= c2_cache_eid_i;
            c3_cache_v_r   <= c3_cache_valid_i;
            c3_cache_eid_r <= c3_cache_eid_i;
            // Clusters must be idle at new-layer start
            c2_busy_r <= 1'b0;
            c3_busy_r <= 1'b0;
          end
        end

        // ── ASSIGN ──────────────────────────────────────────────────────────
        ST_ASSIGN: begin
          if (!q_empty) begin
            // Latch assignment outputs
            if (nx_c2_assign) begin
              c2_assign_o   <= 1'b1;
              c2_eid_o      <= nx_c2_eid;
              c2_ntok_o     <= nx_c2_ntok;
              c2_shape_s1_o <= nx_c2_s1;
              c2_shape_s3_o <= nx_c2_s3;
              c2_cached_o   <= nx_c2_hit;
              c2_busy_r     <= 1'b1;
              // Invalidate cache entry when we start using (or overwriting) it
              if (nx_c2_hit) c2_cache_v_r <= 1'b0;
            end
            if (nx_c3_assign) begin
              c3_assign_o   <= 1'b1;
              c3_eid_o      <= nx_c3_eid;
              c3_ntok_o     <= nx_c3_ntok;
              c3_shape_s1_o <= nx_c3_s1;
              c3_shape_s3_o <= nx_c3_s3;
              c3_cached_o   <= nx_c3_hit;
              c3_busy_r     <= 1'b1;
              if (nx_c3_hit) c3_cache_v_r <= 1'b0;
            end
            // Advance queue head
            q_head <= q_head + q_advance;
          end
        end

        // ── WAIT ────────────────────────────────────────────────────────────
        // When a cluster finishes and there are remaining experts, return to
        // ST_ASSIGN.  The action decision will issue SINGLE_C2 or SINGLE_C3
        // to the now-free cluster.  If both finish simultaneously, both_idle
        // will be true and we will PAIR or SPLIT the next experts.
        ST_WAIT: ; // busy/cache updates handled unconditionally above

        // ── DONE ────────────────────────────────────────────────────────────
        ST_DONE: ; // wait for c2_busy_r && c3_busy_r to clear (done above)

        default: ;
      endcase
    end
  end

  // ── Status outputs ────────────────────────────────────────────────────────
  assign ready_o = (state == ST_IDLE);
  assign done_o  = (state == ST_DONE) && !c2_busy_r && !c3_busy_r;

endmodule
`default_nettype wire
