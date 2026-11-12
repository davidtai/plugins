//! The expert bank's slot rows at the budgets it opens with, MLX-free: each phase's whole rows past its windows and the
//! read pool's staging (src/streamed/ling_source.zig `billBank`). The arithmetic RQ's plans took per request in the
//! model plugin; as a quant plugin the bank opens once at the host's budget (`open_bank`), so only this part stays.

const std = @import("std");

/// The bank's terms beside its slot rows: a row of every routed layer, a phase's windows, the read pool's staging past
/// its default workers, and a layer's experts.
pub const Bank = struct { row_bytes: u64, prompt_windows: u64, decode_windows: u64, staging: u64, n_experts: u32 };
/// The phases' budgets for the slot rows and their terms (the host's `abi.BankBudget`).
pub const Budget = struct { prompt: u64, decode: u64 };
pub const Rows = struct { prompt: u32, decode: u32 };

/// The whole rows `budget` holds past `fixed` bytes of a phase's own terms (its windows and the staging).
pub fn rowsIn(bank: Bank, budget: u64, fixed: u64) u64 {
    return (budget -| fixed) / bank.row_bytes;
}

/// Each phase's whole rows past its windows and the staging; the prompt at least one row, decode at least the
/// prompt's, both at most a layer's experts.
pub fn rowsOf(bank: Bank, budget: Budget) error{StreamedRowsDoNotFit}!Rows {
    const n: u64 = bank.n_experts;
    const p = rowsIn(bank, budget.prompt, bank.prompt_windows + bank.staging);
    const d = rowsIn(bank, budget.decode, bank.decode_windows + bank.staging);
    if (p == 0) return error.StreamedRowsDoNotFit;
    return .{ .prompt = @intCast(@min(p, n)), .decode = @intCast(@min(@max(d, p), n)) };
}

// ── tests ──

const testing = std.testing;
const mib: u64 = 1 << 20;

/// Ling-3.1-flash 2.60bpw's bank (68 routed layers of 4,947,456 B a record, 5 prompt windows of 48 records, one at
/// decode, 118,489,088 B of staging: the served bank's terms).
const served: Bank = .{
    .row_bytes = 336_427_008,
    .prompt_windows = 5 * 48 * 2 * 4_947_456,
    .decode_windows = 48 * 2 * 4_947_456,
    .staging = 118_489_088,
    .n_experts = 512,
};

test "bank rows: each phase's whole rows past its windows and staging, decode at least the prompt's, both at most 512" {
    const r = try rowsOf(served, .{ .prompt = 69_998 * mib, .decode = 76_329 * mib });
    // (budget - windows - staging) / row, rounded down.
    try testing.expectEqual(@as(u32, @intCast((69_998 * mib - served.prompt_windows - served.staging) / served.row_bytes)), r.prompt);
    try testing.expectEqual(@as(u32, @intCast((76_329 * mib - served.decode_windows - served.staging) / served.row_bytes)), r.decode);
    try testing.expect(r.decode >= r.prompt);
    // A decode budget below the prompt's keeps the prompt's rows; a budget past every expert stops at 512.
    const low = try rowsOf(served, .{ .prompt = 69_998 * mib, .decode = 0 });
    try testing.expectEqual(low.prompt, low.decode);
    const huge = try rowsOf(served, .{ .prompt = 1 << 50, .decode = 1 << 50 });
    try testing.expectEqual(Rows{ .prompt = 512, .decode = 512 }, huge);
}

test "bank rows: a prompt budget that holds no row past its windows and staging is refused" {
    try testing.expectError(error.StreamedRowsDoNotFit, rowsOf(served, .{ .prompt = served.prompt_windows + served.staging + served.row_bytes - 1, .decode = 1 << 40 }));
}
