// =============================================================================
// tb_moe_hw_scheduler.sv
// Self-checking testbench for moe_hw_scheduler.sv
//
// Tests:
//   1. Single expert  (SINGLE_C2, ShapeC)
//   2. Two balanced   (PAIR, ShapeB concurrent)
//   3. Symmetric SPLIT (tok0=16, SPLIT expected, ShapeB S1, ShapeC S3)
//   4. Initial cache  (first expert already cached → c2_cached=1)
//   5. Sequential     (4 experts, verify ordering and cache update)
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module tb_moe_hw_scheduler;

  // ── DUT parameters ──────────────────────────────────────────────────────────
  localparam int MAX_EXP   = 8;
  localparam int TOK_W     = 7;
  localparam int EID_W     = 3;
  localparam int SPLIT_THR = 4;

  // ── Shape encoding constants (mirrors DUT) ───────────────────────────────────
  localparam logic [1:0] SH_A = 2'b00;
  localparam logic [1:0] SH_B = 2'b01;
  localparam logic [1:0] SH_C = 2'b10;

  // ── Clock / reset ────────────────────────────────────────────────────────────
  logic clk = 0;
  always #5 clk = ~clk;   // 100 MHz

  logic rst_n;

  // ── DUT I/O ─────────────────────────────────────────────────────────────────
  logic              valid_i;
  logic [3:0]        n_experts_i;
  logic [EID_W-1:0]  eid_i  [0:MAX_EXP-1];
  logic [TOK_W-1:0]  ntok_i [0:MAX_EXP-1];

  logic              c2_cache_valid_i;
  logic [EID_W-1:0]  c2_cache_eid_i;
  logic              c3_cache_valid_i;
  logic [EID_W-1:0]  c3_cache_eid_i;

  logic              c2_done_i;
  logic              c3_done_i;

  logic              c2_assign_o, c3_assign_o;
  logic [EID_W-1:0]  c2_eid_o,   c3_eid_o;
  logic [TOK_W-1:0]  c2_ntok_o,  c3_ntok_o;
  logic [1:0]        c2_shape_s1_o, c2_shape_s3_o;
  logic [1:0]        c3_shape_s1_o, c3_shape_s3_o;
  logic              c2_cached_o, c3_cached_o;
  logic              ready_o, done_o;

  // ── DUT instantiation ────────────────────────────────────────────────────────
  moe_hw_scheduler #(
    .MAX_EXP   (MAX_EXP),
    .TOK_W     (TOK_W),
    .EID_W     (EID_W),
    .SPLIT_THR (SPLIT_THR)
  ) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .valid_i          (valid_i),
    .n_experts_i      (n_experts_i),
    .eid_i            (eid_i),
    .ntok_i           (ntok_i),
    .c2_cache_valid_i (c2_cache_valid_i),
    .c2_cache_eid_i   (c2_cache_eid_i),
    .c3_cache_valid_i (c3_cache_valid_i),
    .c3_cache_eid_i   (c3_cache_eid_i),
    .c2_done_i        (c2_done_i),
    .c3_done_i        (c3_done_i),
    .c2_assign_o      (c2_assign_o),
    .c2_eid_o         (c2_eid_o),
    .c2_ntok_o        (c2_ntok_o),
    .c2_shape_s1_o    (c2_shape_s1_o),
    .c2_shape_s3_o    (c2_shape_s3_o),
    .c2_cached_o      (c2_cached_o),
    .c3_assign_o      (c3_assign_o),
    .c3_eid_o         (c3_eid_o),
    .c3_ntok_o        (c3_ntok_o),
    .c3_shape_s1_o    (c3_shape_s1_o),
    .c3_shape_s3_o    (c3_shape_s3_o),
    .c3_cached_o      (c3_cached_o),
    .ready_o          (ready_o),
    .done_o           (done_o)
  );

  // ── Helpers ──────────────────────────────────────────────────────────────────
  int pass_cnt = 0;
  int fail_cnt = 0;

  task automatic tick(input int n = 1);
    repeat (n) @(posedge clk); #1;
  endtask

  task automatic reset_dut();
    rst_n            = 0;
    valid_i          = 0;
    n_experts_i      = 0;
    c2_cache_valid_i = 0;
    c3_cache_valid_i = 0;
    c2_cache_eid_i   = 0;
    c3_cache_eid_i   = 0;
    c2_done_i        = 0;
    c3_done_i        = 0;
    for (int i = 0; i < MAX_EXP; i++) begin
      eid_i[i]  = i[EID_W-1:0];
      ntok_i[i] = 0;
    end
    tick(2);
    rst_n = 1;
    tick(1);
  endtask

  // Present one layer; set up expert list and optional cache.
  task automatic present_layer(
    input int n_exp,
    input int tok [0:7],
    input int c2_cache = -1,
    input int c3_cache = -1
  );
    n_experts_i      = n_exp[3:0];
    c2_cache_valid_i = (c2_cache >= 0) ? 1'b1 : 1'b0;
    c2_cache_eid_i   = (c2_cache >= 0) ? c2_cache[EID_W-1:0] : '0;
    c3_cache_valid_i = (c3_cache >= 0) ? 1'b1 : 1'b0;
    c3_cache_eid_i   = (c3_cache >= 0) ? c3_cache[EID_W-1:0] : '0;
    for (int i = 0; i < MAX_EXP; i++) begin
      eid_i[i]  = i[EID_W-1:0];
      ntok_i[i] = (i < n_exp) ? tok[i][TOK_W-1:0] : '0;
    end
    valid_i = 1'b1;
    tick(1);
    valid_i = 1'b0;
  endtask

  // Wait for an assignment output (up to max_cycles).
  // Returns which cluster(s) fired.
  task automatic wait_assign(
    output logic got_c2,
    output logic got_c3,
    input  int   max_cycles = 10
  );
    got_c2 = 0; got_c3 = 0;
    for (int i = 0; i < max_cycles; i++) begin
      if (c2_assign_o || c3_assign_o) begin
        got_c2 = c2_assign_o;
        got_c3 = c3_assign_o;
        return;
      end
      tick(1);
    end
    $display("  ERROR: no assignment within %0d cycles", max_cycles);
    fail_cnt++;
  endtask

  task automatic check(
    input string  label,
    input logic   cond
  );
    if (cond) begin
      $display("  PASS: %s", label);
      pass_cnt++;
    end else begin
      $display("  FAIL: %s", label);
      fail_cnt++;
    end
  endtask

  // ── Test 1: Single expert → SINGLE_C2, ShapeC ────────────────────────────────
  task automatic test_single_expert();
    logic got_c2, got_c3;
    $display("\n── Test 1: Single expert (E0, 6 tokens) ────────────────────────");
    reset_dut();
    present_layer(1, '{6, 0, 0, 0, 0, 0, 0, 0});
    wait_assign(got_c2, got_c3);
    check("C2 assigned",           got_c2 == 1'b1);
    check("C3 not assigned",       got_c3 == 1'b0);
    check("C2 eid = 0",            c2_eid_o == 3'd0);
    check("C2 ntok = 6",           c2_ntok_o == TOK_W'(6));
    check("C2 S1 = ShapeC (solo)", c2_shape_s1_o == SH_C);
    check("C2 S3 = ShapeC",        c2_shape_s3_o == SH_C);
    check("C2 not cached",         c2_cached_o == 1'b0);
    // Simulate cluster done
    tick(2); c2_done_i = 1; tick(1); c2_done_i = 0;
    tick(2);
    check("done_o asserted",       done_o == 1'b1);
  endtask

  // ── Test 2: Two balanced experts → PAIR, ShapeB S1 ──────────────────────────
  task automatic test_pair();
    logic got_c2, got_c3;
    $display("\n── Test 2: Two balanced experts (E0=8, E1=6) → PAIR ────────────");
    reset_dut();
    present_layer(2, '{8, 6, 0, 0, 0, 0, 0, 0});
    wait_assign(got_c2, got_c3);
    check("C2 assigned",               got_c2 == 1'b1);
    check("C3 assigned",               got_c3 == 1'b1);
    check("C2 eid = 0",                c2_eid_o == 3'd0);
    check("C3 eid = 1",                c3_eid_o == 3'd1);
    check("C2 S1 = ShapeB (concurrent)",  c2_shape_s1_o == SH_B);
    check("C3 S1 = ShapeB (concurrent)",  c3_shape_s1_o == SH_B);
    check("C2 S3 = ShapeB (concurrent)",  c2_shape_s3_o == SH_B);  // PAIR→S3 concurrent→ShapeB
    check("C3 S3 = ShapeB (concurrent)",  c3_shape_s3_o == SH_B);
    // Simulate done
    tick(3); c2_done_i = 1; c3_done_i = 1; tick(1);
    c2_done_i = 0; c3_done_i = 0; tick(2);
    check("done_o asserted",           done_o == 1'b1);
  endtask

  // ── Test 3: Hot expert → SPLIT ───────────────────────────────────────────────
  task automatic test_split();
    logic got_c2, got_c3;
    $display("\n── Test 3: Hot single expert (E0=16) → SPLIT ───────────────────");
    reset_dut();
    present_layer(1, '{16, 0, 0, 0, 0, 0, 0, 0});
    wait_assign(got_c2, got_c3);
    check("C2 assigned",            got_c2 == 1'b1);
    check("C3 assigned",            got_c3 == 1'b1);
    check("Same eid (SPLIT)",       c2_eid_o == c3_eid_o);
    check("Tokens sum to 16",       (c2_ntok_o + c3_ntok_o) == TOK_W'(16));
    check("C2 ntok aligned to 4",  (c2_ntok_o & TOK_W'(3)) == 0);
    // SPLIT with tok=16: split=8+8 both > 4, SPLIT action→S3 concurrent→ShapeB
    check("C2 S3 = ShapeB (SPLIT concurrent)", c2_shape_s3_o == SH_B);
    check("C3 not cached",         c3_cached_o == 1'b0);
    // done
    tick(3); c2_done_i = 1; c3_done_i = 1; tick(1);
    c2_done_i = 0; c3_done_i = 0; tick(2);
    check("done_o asserted",       done_o == 1'b1);
  endtask

  // ── Test 4: Initial cache hit ─────────────────────────────────────────────────
  task automatic test_cache_hit();
    logic got_c2, got_c3;
    $display("\n── Test 4: Initial cache hit (E0 cached in C2) ─────────────────");
    reset_dut();
    // E0 already in C2 SRAM from previous layer
    present_layer(1, '{8, 0, 0, 0, 0, 0, 0, 0}, .c2_cache=0);
    wait_assign(got_c2, got_c3);
    check("C2 assigned",           got_c2 == 1'b1);
    check("C2 cached = 1",         c2_cached_o == 1'b1);
    check("C2 S1 = ShapeA (cache-dontcare)", c2_shape_s1_o == SH_A);
    // done
    tick(3); c2_done_i = 1; tick(1); c2_done_i = 0; tick(2);
    check("done_o asserted",       done_o == 1'b1);
  endtask

  // ── Test 5: PAIR small → both ShapeB S3 (concurrent DMA safe) ─────────────
  task automatic test_sym_small_pair();
    logic got_c2, got_c3;
    $display("\n── Test 5: Small PAIR (E0=4, E1=4) → PAIR, ShapeB S3 ──────────────");
    reset_dut();
    present_layer(2, '{4, 4, 0, 0, 0, 0, 0, 0});
    wait_assign(got_c2, got_c3);
    check("Both assigned (PAIR)",   got_c2 && got_c3);
    // PAIR action → S3 concurrent → ShapeB for both
    check("C2 S3 = ShapeB (PAIR)",  c2_shape_s3_o == SH_B);
    check("C3 S3 = ShapeB (PAIR)",  c3_shape_s3_o == SH_B);
    // done
    tick(3); c2_done_i = 1; c3_done_i = 1; tick(1);
    c2_done_i = 0; c3_done_i = 0; tick(2);
    check("done_o asserted",        done_o == 1'b1);
  endtask

  // ── Test 6: Four experts, sequential assignment ────────────────────────────────
  task automatic test_four_experts();
    logic got_c2, got_c3;
    $display("\n── Test 6: Four experts {20,15,10,5} → PAIR then PAIR ──────────");
    reset_dut();
    present_layer(4, '{20, 15, 10, 5, 0, 0, 0, 0});
    // First PAIR: top-0→C2, top-1→C3
    wait_assign(got_c2, got_c3);
    check("Round-1 C2 eid=0",       c2_eid_o == 3'd0);
    check("Round-1 C3 eid=1",       c3_eid_o == 3'd1);
    // Simulate C2 finishes first
    tick(5); c2_done_i = 1; tick(1); c2_done_i = 0;
    // Should assign next expert to C2
    wait_assign(got_c2, got_c3);
    check("Round-2 C2 assigned",    got_c2 == 1'b1);
    check("Round-2 C3 not",         got_c3 == 1'b0);
    check("Round-2 C2 eid=2",       c2_eid_o == 3'd2);
    check("Round-2 C2 solo ShapeC", c2_shape_s1_o == SH_C);  // SINGLE→ShapeC S1
    // Simulate both done
    tick(5); c2_done_i = 1; c3_done_i = 1; tick(1);
    c2_done_i = 0; c3_done_i = 0;
    // Should assign E3 to C2 (or C3)
    wait_assign(got_c2, got_c3);
    check("Round-3 one assigned",   got_c2 || got_c3);
    tick(5); c2_done_i = 1; c3_done_i = 1; tick(1);
    c2_done_i = 0; c3_done_i = 0; tick(3);
    check("Final done_o",           done_o == 1'b1);
  endtask

  // ── Main ─────────────────────────────────────────────────────────────────────
  initial begin
    $dumpfile("tb_moe_hw_scheduler.vcd");
    $dumpvars(0, tb_moe_hw_scheduler);

    test_single_expert();
    test_pair();
    test_split();
    test_cache_hit();
    test_sym_small_pair();
    test_four_experts();

    $display("\n═══════════════════════════════════════════════════");
    $display("  Results: %0d PASS  /  %0d FAIL", pass_cnt, fail_cnt);
    $display("═══════════════════════════════════════════════════");
    if (fail_cnt == 0)
      $display("  ALL TESTS PASSED");
    else
      $display("  SOME TESTS FAILED");
    $finish;
  end

  // Timeout guard
  initial begin
    #50000;
    $display("TIMEOUT");
    $finish;
  end

endmodule
`default_nettype wire
