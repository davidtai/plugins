//! copied_from: src/sdk/weights/streamed/sdk_ext/expert/lookahead.zig@bc6c5f2c102c1f08c11d04c3ca13e17a99dd78b4
//! Host half of the expert streamer's lookahead (the lookahead4 lane's
//! Selector and pre-read list). The router barrier also evaluates the next
//! routed layer's own gate on this layer's router input, giving M rows of
//! per-expert scores; per row the top-K experts by score are kept iff their
//! score is at least the row's routed-top-k-th score minus tau, the union is ordered by
//! (best rank over rows, then score), residents are dropped, and the first
//! `budget` become speculative whole-record reads. A score tie breaks by expert
//! id (the Python lane leaves tied experts in numpy's argpartition order).
//! Pure: it chooses records and reads nothing.
const std = @import("std");
const mlx = @import("sdk").mlx;
const expert_policy = @import("policy.zig");
const LayerPolicy = expert_policy.LayerPolicy;
/// The widest per-row candidate list a selector keeps (its `k`).
pub const max_k = 12;
/// turboexl3_ling: 8 (was 4): a decode call's read-ahead up to `Lookahead.budget_per_row` records a row (the IO pool's
/// 16 staging slots hold 2 x 8).
pub const max_budget = 8;
/// Verify rows per layer call.
pub const max_rows = 8;
/// Certain misses handed to one pre-read; the re-validation candidate cap.
pub const max_candidates = 64;
/// turboexl3_ling: the most expert groups a group-limited pick takes (`limitGroups`; its router keeps a group mask in a
/// 32-bit word).
pub const max_groups = 32;
/// turboexl3_ling: the group-limited pick's engagement line prints once per process.
var groups_logged = false;
const Entry = struct { rank: u8, score: f32, expert: u16 };
/// Expert `e` is resident under `p`: the residency `select` skips (one layer's policy).
pub fn residentIn(p: *const LayerPolicy, e: u16) bool {
    return p.slotOf(e) != null;
}
/// A selector for an arch routing `routed_top_k` experts per row: tau is measured from each row's routed-top-k-th score,
/// and a selector keeps at least that many candidates per row (`k`, up to `max_k`).
pub fn SelectorOf(comptime routed_top_k: u32) type {
    comptime if (routed_top_k == 0 or routed_top_k > max_k) @compileError("lookahead: routed_top_k outside 1..max_k");
    const min_k = routed_top_k;
    return struct {
        const Selector = @This();
        n_experts: u32,
        k: u32,
        /// inf keeps each row's plain top-K.
        tau: f32,
        budget: u32,
        /// turboexl3_ling: Ling's group limit (`limitGroups`; 0 = off): a row's candidates come only from its best
        /// `topk_group` of `n_group` equal expert groups, as Ling's router (`sigmoid_bias_grouped`) keeps them.
        n_group: u32 = 0,
        topk_group: u32 = 0,
        seen: std.DynamicBitSetUnmanaged,
        pub fn init(a: std.mem.Allocator, n_experts: u32, k: u32, tau: f32, budget: u32) !Selector {
            if (k < min_k or k > max_k or k > n_experts or budget == 0 or budget > max_budget or std.math.isNan(tau) or tau == -std.math.inf(f32))
                return error.InvalidSelector;
            return .{ .n_experts = n_experts, .k = k, .tau = tau, .budget = budget, .seen = try std.DynamicBitSetUnmanaged.initEmpty(a, n_experts) };
        }
        pub fn deinit(self: *Selector, a: std.mem.Allocator) void {
            self.seen.deinit(a);
            self.* = undefined;
        }
        /// turboexl3_ling: pick as Ling's router does, from the best `topk_group` of `n_group` equal groups (each scored
        /// by its two best experts), then the top-k by score. Residency only: it chooses which records are read ahead.
        pub fn limitGroups(self: *Selector, n_group: u32, topk_group: u32) !void {
            if (n_group < 2 or n_group > max_groups or self.n_experts % n_group != 0 or topk_group == 0 or topk_group > n_group or
                topk_group * (self.n_experts / n_group) < self.k) return error.InvalidSelector;
            self.n_group = n_group;
            self.topk_group = topk_group;
        }
        /// The records to read ahead for the next layer: `scores` is rows x
        /// n_experts (row-major f32), `next` that layer's residency. Returns the
        /// first `out.len` non-resident experts of the ordered union.
        pub fn select(self: *Selector, scores: []const f32, next: *const LayerPolicy, out: []u16) []u16 {
            return self.selectWith(scores, next, residentIn, out, null);
        }
        /// `select` with the residency asked of `resident(ctx, expert)` (the next layer's experts over several
        /// policies: one per bank layer).
        pub fn selectWith(self: *Selector, scores: []const f32, ctx: anytype, comptime resident: fn (@TypeOf(ctx), u16) bool, out: []u16, ranks: ?[]u8) []u16 {
            std.debug.assert(scores.len % self.n_experts == 0 and scores.len / self.n_experts <= max_rows);
            var entries: [max_rows * max_k]Entry = undefined;
            const ordered = self.order(scores, &entries);
            var n: usize = 0;
            for (ordered) |en| {
                if (self.seen.isSet(en.expert)) continue;
                self.seen.set(en.expert);
                if (resident(ctx, en.expert)) continue;
                if (ranks) |r| r[n] = en.rank;
                out[n] = en.expert;
                n += 1;
                if (n == out.len) break;
            }
            for (ordered) |en| self.seen.unset(en.expert);
            return out[0..n];
        }
        /// Per row the top-k by (score desc, expert asc), kept iff score >= the
        /// row's routed-top-k-th score - tau (f32); then a stable sort by (rank, score desc),
        /// so equal keys keep row-major order, as the lane's lexsort does.
        fn order(self: *const Selector, scores: []const f32, entries: *[max_rows * max_k]Entry) []Entry {
            const e_n = self.n_experts;
            const rows = scores.len / e_n;
            const finite = std.math.isFinite(self.tau);
            const grouped = self.n_group > 0;
            if (grouped and !groups_logged) {
                groups_logged = true;
                std.log.info("[lookahead] turboexl3_ling: read-ahead picks by Ling's router rule engaged: the best {d} of {d} expert groups, then the top {d}", .{ self.topk_group, self.n_group, self.k });
            }
            const gsz = if (grouped) e_n / self.n_group else e_n;
            var n: usize = 0;
            for (0..rows) |r| {
                const row = scores[r * e_n ..][0..e_n];
                const keep: u32 = if (grouped) keptGroups(row, self.n_group, self.topk_group) else 0;
                var top: [max_k]u16 = undefined;
                var cnt: usize = 0;
                for (row, 0..) |v, ei| {
                    const e: u16 = @intCast(ei);
                    if (grouped and keep & (@as(u32, 1) << @intCast(ei / gsz)) == 0) continue;
                    if (cnt == self.k and !better(v, e, row[top[cnt - 1]], top[cnt - 1])) continue;
                    var i = if (cnt < self.k) cnt else cnt - 1;
                    if (cnt < self.k) cnt += 1;
                    while (i > 0 and better(v, e, row[top[i - 1]], top[i - 1])) : (i -= 1) top[i] = top[i - 1];
                    top[i] = e;
                }
                const threshold: f32 = if (finite) row[top[routed_top_k - 1]] - self.tau else -std.math.inf(f32);
                for (top[0..cnt], 0..) |e, rank| {
                    const v = row[e];
                    if (finite and !(v >= threshold)) continue;
                    entries[n] = .{ .rank = @intCast(rank), .score = v, .expert = e };
                    n += 1;
                }
            }
            const out = entries[0..n];
            std.sort.insertion(Entry, out, {}, struct {
                fn less(_: void, a: Entry, b: Entry) bool {
                    return if (a.rank != b.rank) a.rank < b.rank else a.score > b.score;
                }
            }.less);
            return out;
        }
        /// The layer call's certain misses: its unique routed experts not resident
        /// in `layer`, in route order, at most `out.len` (they are its plan's misses).
        pub fn certainMisses(self: *Selector, ids: []const u16, layer: *const LayerPolicy, out: []u16) []u16 {
            var n: usize = 0;
            for (ids) |e| {
                if (self.seen.isSet(e)) continue;
                self.seen.set(e);
                if (layer.slotOf(e) == null and n < out.len) {
                    out[n] = e;
                    n += 1;
                }
            }
            for (ids) |e| self.seen.unset(e);
            return out[0..n];
        }
    };
}
fn better(v: f32, e: u16, w: f32, f: u16) bool {
    return v > w or (v == w and e < f);
}
/// turboexl3_ling: the groups Ling's router keeps (`router_group_limit.metal`), a bit each: every group scored by the sum
/// of its two best scores (a tie counts twice), then `topk` rounds of the best remaining group, the lower index on a tie.
pub fn keptGroups(row: []const f32, n_group: u32, topk: u32) u32 {
    var gs: [max_groups]f32 = undefined;
    const gsz = row.len / n_group;
    for (gs[0..n_group], 0..) |*s, g| {
        var m1 = -std.math.inf(f32);
        var m2 = -std.math.inf(f32);
        for (row[g * gsz ..][0..gsz]) |v| {
            if (v > m1) {
                m2 = m1;
                m1 = v;
            } else if (v > m2) m2 = v;
        }
        s.* = m1 + m2;
    }
    var keep: u32 = 0;
    for (0..topk) |_| {
        var best = -std.math.inf(f32);
        var pick: u32 = n_group - 1;
        for (gs[0..n_group], 0..) |v, g| if (v > best) {
            best = v;
            pick = @intCast(g);
        };
        keep |= @as(u32, 1) << @intCast(pick);
        gs[pick] = -std.math.inf(f32);
    }
    return keep;
}
// ── Tests ──
