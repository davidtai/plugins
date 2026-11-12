//! PF_DFIN's whole-call rule, MLX-free: a wide routed prompt call takes turboexl3's fused finish (the down GEMM's
//! epilogue) only at a mean of at least `dfin_call_min` rows an expert, every other call the served chain, the same
//! bytes either way (src/ling_math.zig `dfinTakes`); and its counts, the `PF_DFIN calls:` line.
const std = @import("std");

/// PF_DFIN's whole-call rule (src/ling_math.zig `chunks()`): a wide routed prompt call of `rows` rows (one layer's routed
/// call over whole spans under LAYER_MAJOR, or a chunk-loop span) takes turboexl3's fused finish when its mean rows an
/// expert, rows * topk / experts, is at least `call_min`, in integers rows * topk >= call_min * experts; a `call_min` of 0
/// takes it at every call. Under it the call keeps the served chain. The host decides from `rows` alone (no read-back of
/// the routes), once a call: every chunk of the call takes the same side.
pub fn dfinCallTakes(rows: u64, topk: u32, experts: u32, call_min: u32) bool {
    return call_min == 0 or rows * topk >= @as(u64, call_min) * experts;
}

/// PF_DFIN's served cutoff: 80 rows an expert, the first step of turboexl3's TIME ladder (ling-v1-routed, rows an
/// expert) where the plain fused finish was at or under the served chain at both n 40 and 32 (at 8 to 32 rows an expert
/// it was slower). Main kept it after the cutoff records (2026-11-09: CALL_MIN 0 was no faster at 1,120, 2,042 and
/// 4,094 tokens).
pub const dfin_call_min: u16 = 80;

/// The two sides of a routed call PF_DFIN chooses for (src/ling_math.zig): a wide prompt call (`chunks()`, each of its
/// chunks on the side its rows R take) or a decode-width call (`down()`: a cold-row chunk or a decode wave's group, at
/// most 48 pairs, each by the same rule from its own rows)
pub const DfinSide = enum { wide, decode_width };

/// Where the counts line is printed (plugin.zig): after a prompt forward (its rows), or at a decode step (every 64
/// decode steps, the `[stream-stats]` cadence, and decode's last when the next prompt begins)
pub const DfinAt = union(enum) { prompt: u32, decode: u32 };

/// PF_DFIN's counts: the turboexl3 routed calls it chose an ending for, by side and ending (0 the served chain, 1 the
/// fused finish). src/ling_math.zig counts each call as it is issued. Host counters only: no launch, no byte changes.
pub const DfinCalls = struct {
    calls: [2][2]u64 = .{ .{ 0, 0 }, .{ 0, 0 } },

    /// A call on `side` at PF_DFIN's choice `fused` (the fused finish, or the served chain)
    pub fn count(c: *DfinCalls, side: DfinSide, fused: bool) void {
        c.calls[@intFromEnum(side)][@intFromBool(fused)] += 1;
    }

    /// The counts line into `buf`: `[turboexl3_ling] PF_DFIN calls:`, where (`prompt rows=R` or `decode steps=N`), the
    /// calls since load by side and ending (`wide_fused`, `wide_served`, `decode_width_fused`, `decode_width_served`),
    /// then the rule both sides took at `call_min` rows an expert
    pub fn line(c: *const DfinCalls, buf: []u8, at: DfinAt, call_min: u16) []const u8 {
        const w = c.calls[@intFromEnum(DfinSide.wide)];
        const d = c.calls[@intFromEnum(DfinSide.decode_width)];
        var where_buf: [32]u8 = undefined;
        const where: []const u8 = switch (at) {
            .prompt => |r| std.fmt.bufPrint(&where_buf, "prompt rows={d}", .{r}) catch "prompt",
            .decode => |n| std.fmt.bufPrint(&where_buf, "decode steps={d}", .{n}) catch "decode",
        };
        return std.fmt.bufPrint(buf, "[turboexl3_ling] PF_DFIN calls: {s} wide_fused={d} wide_served={d} decode_width_fused={d} decode_width_served={d} (turboexl3 routed calls since load; each side by the whole-call rule from its call's rows, the fused finish at {d} rows an expert and up)", .{ where, w[1], w[0], d[1], d[0], call_min }) catch "[turboexl3_ling] PF_DFIN calls: the line does not fit its buffer";
    }
};

// ── tests ──

const testing = std.testing;

test "PF_DFIN's whole-call rule at Ling's top-8 over 512 experts: 5,119 rows keeps the served chain, 5,120 and 16,384 take the fused finish, 1,120 keeps the served chain; call_min 0 takes it at every call" {
    const n = dfin_call_min;
    try testing.expectEqual(@as(u16, 80), n);
    // either side of a mean of 80 rows an expert (40,952 and 40,960 routed rows against 80 * 512 = 40,960)
    try testing.expect(!dfinCallTakes(5119, 8, 512, n));
    try testing.expect(dfinCallTakes(5120, 8, 512, n));
    // a layer-major call (16,384 rows: 256 an expert; with its tail, 18,432) and a short call (1,120 rows: 17.5)
    try testing.expect(dfinCallTakes(16384, 8, 512, n));
    try testing.expect(dfinCallTakes(16384 + 2048, 8, 512, n));
    try testing.expect(!dfinCallTakes(1120, 8, 512, n));
    // the rule's own boundary at another top-k and expert count: 80 * 256 / 4 = 5,120 rows
    try testing.expect(!dfinCallTakes(5119, 4, 256, n) and dfinCallTakes(5120, 4, 256, n));
    // 0: every call, whatever its rows
    for ([_]u64{ 1, 17, 1120, 5119, 5120, 16384 }) |r| try testing.expect(dfinCallTakes(r, 8, 512, 0));
}

test "PF_DFIN's counts (DfinCalls): each call counted on its side by its ending; the line the A/B greps, `] PF_DFIN calls:`, gives where, the four counts and the cutoff" {
    var c: DfinCalls = .{};
    c.count(.wide, true);
    c.count(.wide, true);
    c.count(.wide, false);
    c.count(.decode_width, false);
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_ling] PF_DFIN calls: prompt rows=18380 wide_fused=2 wide_served=1 decode_width_fused=0 decode_width_served=1 (turboexl3 routed calls since load; each side by the whole-call rule from its call's rows, the fused finish at 80 rows an expert and up)", c.line(&buf, .{ .prompt = 18380 }, dfin_call_min));
    // the counts keep growing (since load), and a decode line names its step
    c.count(.decode_width, true);
    const later = c.line(&buf, .{ .decode = 64 }, dfin_call_min);
    try testing.expect(std.mem.startsWith(u8, later, "[turboexl3_ling] PF_DFIN calls: decode steps=64 wide_fused=2 wide_served=1 decode_width_fused=1 decode_width_served=1 ("));
    // a buffer too small for the line gives a line that says so, never a part of one
    var small: [16]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_ling] PF_DFIN calls: the line does not fit its buffer", c.line(&small, .{ .prompt = 1 }, 80));
}
