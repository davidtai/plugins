//! turboexl3_ling: Ling-3.1-flash's exllamav3 (EXL3) streaming packs, as a quant plugin. mlx-serve's own bailing_hybrid
//! serves the pack (its forward: KDA, MLA, the grouped MoE, the head); this plugin loads the trunk's trellis linears the
//! pack stores and runs them on turboexl3's Metal kernels, and streams the routed experts of layers 4..71 from the pack's
//! `experts.bin` (`open_bank`: the expert streamer in src/streamed/, module `mlx_stream`, with Ling's bank source
//! `ling_source` and its routed math on turboexl3, src/ling_math.zig). The template is turboexl3_ds41.
//!
//! The streamer reaches this root as `mlx_host`: MLX, the host's log, the environment, the process globals, and the
//! plugin modules its Ling source calls (`ling_math`, `ling_combine`, `ling_ran`, `bank_rows`, `ling_router`).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.quant;
const t3 = @import("turboexl3");
const tkit = @import("turboexl3_kit");
const stream = @import("mlx_stream");
const shims = @import("src/host_shims.zig");
const clamps = @import("src/clamps.zig");
pub const config = @import("config.zig");

// ── what the streamer reads through `mlx_host` ──

pub const mlx = kit.mlx;
pub const log = @import("src/host/log.zig");
pub const env = @import("src/host/env.zig");
pub const shared = @import("src/host/shared.zig");
pub const ling_math = @import("src/ling_math.zig");
pub const ling_combine = @import("src/ling_combine.zig");
pub const ling_ran = @import("src/ling_ran.zig");
pub const bank_rows = @import("src/bank_rows.zig");
/// ROUTER_FUSED's handoff (the model plugin's router launch scored the bank's read-ahead with the next layer's gate) has
/// no caller here: mlx-serve's bailing_hybrid runs the router. The bank scores its read-ahead from the gates itself
/// (experts.zig `scoresOf`, src/ling_predictor.zig), as it did before ROUTER_FUSED.
pub const ling_router = struct {
    pub fn takeScores(layer: u32, rows: u32) ?mlx.mlx_array {
        _ = layer;
        _ = rows;
        return null;
    }
};

/// Shown at install and in Settings.
pub const version = "0.2.0";

pub const plugin = kit.sdk.Plugin{ .name = "turboexl3_ling", .kind = .quant, .options = &.{} };

pub const table = abi.QuantTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .validate = &config.validate,
    .bill = &config.bill,
    .open = &open,
    .close = &close,
    .load_tensor = &loadTensor,
    .load_bank = &loadBank,
    .release = &release,
    .matmul = &matmul,
    .moe = &moe,
    .swiglu = &swiglu,
    .bank_rows = &bankRows,
    .open_bank = &openBank,
};

const Verdict = abi.Verdict;
const A = mlx.mlx_array;
const gpa = std.heap.c_allocator;
/// The one codebook `validate` admits.
const dec: t3.format.Decode = .mul1;
const missing = Verdict.refuse(.failed, "MissingWeight");
const oom = Verdict.refuse(.failed, "turboexl3_ling: out of memory");
/// turboexl3's prompt kernels and the bank's run on MLX's matrix units (NAX): an M5-class GPU.
const no_nax = Verdict.refuse(.unsupported, "turboexl3_ling: an M5-class GPU (NAX) is needed (turboexl3's prompt kernels run on its matrix units)");

fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    const h = kit.host.Host.of(ctx.host) catch |e| return Verdict.named(.unsupported, @errorName(e));
    // The pack again (validate's process may not be this one): the codebook, the bank's files, the clamps.
    config.checkPack(ctx.block_json.slice(), ctx.config_json.slice(), ctx.dir) catch |e| return Verdict.named(.unsupported, @errorName(e));
    // Kernels from the server's cache, launched through its launcher; MLX's matrix units only where its NAX gate holds.
    tkit.bind(h.t);
    // The streamer's events, slot buffers and launches, and its log lines, through the server's table.
    if (!shims.install(ctx.host)) return Verdict.refuse(.unsupported, "turboexl3_ling: the server's host table is older than this plugin's kit");
    log.use(h);
    if (!tkit.nax.available()) return no_nax;
    // turboexl3's configuration as the served stack sets it (src/ling_math.zig `configure`): the banked decode's grouped
    // texts on t4, the one-row table, the fused routed finish (PF_DFIN), the router's rows launch.
    ling_math.configure() catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn close(_: *abi.LoadCtx) callconv(.c) void {}

/// `<base>.trellis` (I16 [in/16, out/16, n]) beside `<base>.suh` (f16 [in]) and `<base>.svh` (f16 [out]), as exllamav3
/// stores a linear (KDA's joined qkv_proj and the head included).
fn loadTensor(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const p = (partsOf(ctx, name.slice()) catch return Verdict.refuse(.failed, "turboexl3_ling: a name too long")) orelse return missing;
    if (!shaped(p, in_dim)) return not_shaped;
    out.* = .{ .w = share(p[0]), .scales = share(p[1]), .biases = share(p[2]) };
    return .ok;
}

const not_shaped = Verdict.refuse(.unsupported, "turboexl3_ling: not an EXL3 linear of the expected input width (trellis [in/16, out/16, n], suh f16 [in], svh f16 [out])");

/// `p` (trellis, suh, svh) as exllamav3 stores a linear of `in_dim` inputs (0: any).
fn shaped(p: [3]A, in_dim: u32) bool {
    const t = mlx.getShape(p[0]);
    const u = mlx.getShape(p[1]);
    const v = mlx.getShape(p[2]);
    if (t.len != 3 or u.len != 1 or v.len != 1 or u[0] != t[0] * 16 or v[0] != t[1] * 16) return false;
    if (in_dim != 0 and u[0] != in_dim) return false;
    return mlx.mlx_array_dtype(p[1]) == .float16 and mlx.mlx_array_dtype(p[2]) == .float16;
}

/// The routed experts `<prefix>.<proj>` as one resident bank (the MTP head's, when mlx-serve loads Ling's head): the
/// pack's per-expert linears `<prefix>.<e>.<proj>` stacked on a new first axis, the trellis read as u16 (the same bits).
fn loadBank(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const full = name.slice();
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return missing;
    const v = stackParts(ctx, full[0..dot], full[dot..], in_dim, out);
    if (!v.isOk()) return v;
    // Every expert the layer's router scores (`<layer>.mlp.gate.weight` [experts, hidden]): an expert missing from the
    // pack would end the stack early, and a route past its end reads another tensor's bytes.
    const routed = routerRows(ctx, full[0..dot]);
    if (routed == null or routed.? != mlx.getShape(arr(out.w))[0]) {
        release(out);
        return Verdict.refuse(.unsupported, "turboexl3_ling: the bank holds another count of experts than its router scores");
    }
    return .ok;
}

/// The experts the router beside `<layer>.mlp.experts` scores; null without one.
fn routerRows(ctx: *const abi.LoadCtx, experts: []const u8) ?c_int {
    if (!std.mem.endsWith(u8, experts, ".experts")) return null;
    var buf: [192]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "{s}.gate.weight", .{experts[0 .. experts.len - ".experts".len]}) catch return null;
    const g: A = .{ .ctx = ctx.source.tensor(name).ctx };
    if (g.ctx == null) return null;
    const sh = mlx.getShape(g);
    return if (sh.len == 2) sh[0] else null;
}

/// The linears `<prefix>.<i><suffix>` for i = 0.. stacked on a new first axis, the trellis viewed as u16.
fn stackParts(ctx: *const abi.LoadCtx, prefix: []const u8, suffix: []const u8, in_dim: u32, out: *abi.Weight) Verdict {
    var parts: [3]std.ArrayList(A) = .{ .empty, .empty, .empty };
    defer for (&parts) |*l| l.deinit(gpa);
    var e: usize = 0;
    while (true) : (e += 1) {
        var buf: [192]u8 = undefined;
        const each = std.fmt.bufPrint(&buf, "{s}.{d}{s}", .{ prefix, e, suffix }) catch return Verdict.refuse(.failed, "turboexl3_ling: a name too long");
        const p = (partsOf(ctx, each) catch return Verdict.refuse(.failed, "turboexl3_ling: a name too long")) orelse break;
        if (!shaped(p, in_dim)) return not_shaped;
        for (&parts, p) |*l, a| l.append(gpa, a) catch return oom;
    }
    if (e == 0) return missing;
    var stacked: [3]A = @splat(.{ .ctx = null });
    for (&stacked, &parts) |*s, *l| s.* = stack(ctx, l.items) catch |err| {
        for (stacked) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        }
        return Verdict.named(.failed, @errorName(err));
    };
    var trellis = mlx.mlx_array_new();
    const viewed = mlx.mlx_view(&trellis, stacked[0], .uint16, .{ .ctx = ctx.stream.ctx }) == 0 and mlx.mlx_array_eval(trellis) == 0;
    _ = mlx.mlx_array_free(stacked[0]);
    stacked[0] = trellis;
    if (!viewed) {
        for (stacked) |x| _ = mlx.mlx_array_free(x);
        return Verdict.refuse(.failed, "turboexl3_ling: the bank's trellis does not view as u16");
    }
    out.* = .{ .w = .{ .ctx = stacked[0].ctx }, .scales = .{ .ctx = stacked[1].ctx }, .biases = .{ .ctx = stacked[2].ctx } };
    return .ok;
}

/// `base`'s trellis, suh and svh as the source holds them (borrowed); null when the pack has no such linear.
fn partsOf(ctx: *const abi.LoadCtx, base: []const u8) !?[3]A {
    var out: [3]A = undefined;
    for (&out, [_][]const u8{ "trellis", "suh", "svh" }) |*a, part| {
        var buf: [192]u8 = undefined;
        a.* = .{ .ctx = ctx.source.tensor(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, part })).ctx };
        if (a.ctx == null) return null;
    }
    return out;
}

/// `xs` stacked on a new first axis and evaluated (each evaluated first: a stack waiting on hundreds of queued disk
/// reads times out).
fn stack(ctx: *const abi.LoadCtx, xs: []const A) !A {
    for (xs) |x| try mlx.check(mlx.mlx_array_eval(x));
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_stack_axis(&o, vec, 0, .{ .ctx = ctx.stream.ctx }));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// A new handle on `a`.
fn share(a: A) abi.Array {
    var h = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&h, a);
    return .{ .ctx = h.ctx };
}

fn release(w: *abi.Weight) callconv(.c) void {
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
    };
    w.* = .{};
}

fn arr(a: abi.Array) A {
    return .{ .ctx = a.ctx };
}

fn projOf(w: *const abi.Weight) t3.Proj {
    return .{ .trellis = arr(w.w), .suh = arr(w.scales), .svh = arr(w.biases) };
}

/// Every trunk linear the pack stores in EXL3 (KDA's qkv / f / g / o projections, MLA's, the dense layers', the shared
/// experts' and the head): turboexl3's `linear` (DEC-ONE at decode rows on the one-row table, the fused GEMM chain at
/// prompt rows).
fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = t3.linear(st, arr(x), arr(w.w), arr(w.scales), arr(w.biases), dec) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "turboexl3_ling: routed experts run through swiglu");
}

/// Resident routed experts (a format bank: the MTP head's) on turboexl3's Ling SwiGLU. `Swiglu.limit` is a clamp on the
/// gate's input (GLM's); Ling-3.1 clamps after the SiLU and its MTP layer clamps nothing, so a clamped call is refused.
fn swiglu(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.Swiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (inds.ctx == null) return Verdict.refuse(.unsupported, "turboexl3_ling: a dense SwiGLU runs the caller's chain");
    if (opts.limit != 0) return Verdict.refuse(.unsupported, "turboexl3_ling: a gate-input clamp (Ling-3.1 clamps after the SiLU)");
    const bank: t3.Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = t3.moeLing(st, arr(x), bank, arr(inds), arr(scores), dec, 0) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

// ── the streamed bank (`experts.bin`) ──

const io = std.Io.Threaded.global_single_threaded.io();
const LingSource = stream.ling_source;

/// The bank's geometry: the host's (`BankGeometry`, every trunk layer) past the pack's dense layers, with each routed
/// layer's clamp after the SiLU (src/clamps.zig) and the router's groups (the read-ahead's predictor picks by them).
/// `limits` holds `n_layers - first` entries.
fn geometry(geo: *const abi.BankGeometry, l: config.Ling, limits: []c_int) !LingSource.Geometry {
    if (geo.n_layers != l.layers or l.first >= geo.n_layers) return error.LingBankGeometry;
    const n_moe = geo.n_layers - l.first;
    if (limits.len != n_moe) return error.LingBankGeometry;
    for (limits, l.first..) |*o, li| o.* = @intCast(clamps.routed(@intCast(li)));
    return .{ .hidden_size = geo.hidden, .moe_intermediate_size = geo.inter, .n_experts_per_tok = geo.top_k, .n_routed_experts = geo.n_experts, .n_layers = n_moe, .limits = limits, .n_group = l.n_group, .topk_group = l.topk_group };
}

fn budgetOf(b: abi.BankBudget) LingSource.Budget {
    // TODO(box): the host's room is priced by bailing_hybrid's prefill_bytes (the engine's arithmetic, not memfit's). The
    // served plugin's bill (src/bill.zig, memfit v3) gave the bank 214 / 236 rows at 18K; re-fit the decode term's
    // give-back (turboexl3_ds41's `budgetOf` adds its own) from the new shape's [mem] peaks before the A/B.
    return .{ .prompt = b.prompt, .decode = b.decode };
}

fn bankRefusal(e: anyerror) Verdict {
    return if (e == error.StreamedRowsDoNotFit) Verdict.refuse(.too_large, "StreamedRowsDoNotFit") else Verdict.named(.failed, @errorName(e));
}

fn bankRows(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.BankRows) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    const l = config.lingOf(ctx.config_json.slice()) orelse return Verdict.refuse(.unsupported, "turboexl3_ling: not a Ling config");
    const limits = gpa.alloc(c_int, geo.n_layers -| l.first) catch return oom;
    defer gpa.free(limits);
    const g = geometry(geo, l, limits) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const b = LingSource.billOf(gpa, io, ctx.model_dir.slice(), g, budgetOf(budget)) catch |e| return bankRefusal(e);
    out.* = .{ .prompt = b.rows.prompt, .decode = b.rows.decode, .row_bytes = b.row_bytes };
    return .ok;
}

/// The bank the host holds (`abi.Bank.ctx`): the source and the predictor's gates it borrows.
const Held = struct {
    src: *LingSource.Source,
    /// Each routed layer's router as [experts, hidden] (the host's gates are [hidden, experts] views) and its bias.
    gates: []LingSource.Gate,

    fn deinit(self: *Held) void {
        self.src.deinit();
        for (self.gates) |g| _ = mlx.mlx_array_free(g.w);
        gpa.free(self.gates);
        gpa.destroy(self);
    }
};

/// The load line's account of the one bank a process streams.
var about: [160]u8 = undefined;

fn openBank(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.Bank) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    const held = openHeld(ctx, geo, budget) catch |e| return bankRefusal(e);
    const src = held.src;
    const line = std.fmt.bufPrint(&about, "{d} prompt / {d} decode slot rows per layer ({d} B a row)", .{ src.bill.rows.prompt, src.bill.rows.decode, src.bill.row_bytes }) catch "";
    out.* = .{
        .ctx = held,
        .close = &Bank.close,
        .about = .of(line),
        .decode_ids = @intCast(LingSource.decode_route_ids),
        .rows = &Bank.rows,
        .settle = &Bank.settle,
        .phase = &Bank.phase,
        .read_ahead = &Bank.readAhead,
    };
    return .ok;
}

fn openHeld(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget) !*Held {
    const l = config.lingOf(ctx.config_json.slice()) orelse return error.LingBankGeometry;
    const host_gates = geo.gates orelse return error.LingBankNoGates;
    const limits = try gpa.alloc(c_int, geo.n_layers -| l.first);
    defer gpa.free(limits);
    const g = try geometry(geo, l, limits);
    const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
    // The predictor reads each routed layer's router as [experts, hidden] (`Math.predictorScores`: x @ w^T); the host
    // hands the trunk's [hidden, experts] view. Residency only: the scores pick which records are read ahead.
    // TODO(box): the host's gate is bf16 where the served plugin's was the pack's raw router; the read-ahead's picks may
    // move (never an output byte). Check the [stream-stats] hit rate against 87046444's.
    const gates = try gpa.alloc(LingSource.Gate, g.n_layers);
    var made: usize = 0;
    errdefer {
        for (gates[0..made]) |x| _ = mlx.mlx_array_free(x.w);
        gpa.free(gates);
    }
    for (gates, host_gates[l.first..geo.n_layers]) |*o, x| {
        var w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose(&w, arr(x.w), s));
        o.* = .{ .w = w, .bias = arr(x.bias) };
        made += 1;
    }
    // PF_DFIN's whole-call rule reads the model's routing before the bank's first routed call (its warm-up).
    ling_math.routing = .{ .topk = g.n_experts_per_tok, .experts = g.n_routed_experts };
    const src = try LingSource.Source.open(gpa, io, ctx.model_dir.slice(), s, g, gates, l.first, budgetOf(budget));
    errdefer src.deinit();
    const held = try gpa.create(Held);
    held.* = .{ .src = src, .gates = gates };
    log.info("[turboexl3_ling] streamed experts: {d} prompt / {d} decode slot rows per layer of {d} ({d} B a row of every layer)\n", .{ src.bill.rows.prompt, src.bill.rows.decode, g.n_routed_experts, src.bill.row_bytes });
    return held;
}

/// `abi.Bank`'s entries over the source.
const Bank = struct {
    fn of(ctx: ?*anyopaque) *Held {
        return @ptrCast(@alignCast(ctx.?));
    }
    fn done(r: anyerror!void) Verdict {
        r catch |e| return Verdict.named(.failed, @errorName(e));
        return .ok;
    }
    fn close(ctx: ?*anyopaque) callconv(.c) void {
        of(ctx).deinit();
    }
    fn rows(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        const y = of(ctx).src.rows(layer, arr(x), arr(idx), arr(wts)) catch |e| return Verdict.named(.failed, @errorName(e));
        out.* = .{ .ctx = y.ctx };
        return .ok;
    }
    fn settle(ctx: ?*anyopaque) callconv(.c) Verdict {
        return done(of(ctx).src.settle());
    }
    /// The host tells no request with the phase, so decode takes the rows the bank opened with (`Planner.none`; RQ's
    /// per-request plan needs a host hook: the core-hooks plan's B rows).
    fn phase(ctx: ?*anyopaque, prompt: bool) callconv(.c) Verdict {
        const r = of(ctx).src.setPhase(if (prompt) .prompt else .decode, LingSource.Planner.none);
        _ = r catch |e| return Verdict.named(.failed, @errorName(e));
        return .ok;
    }
    fn readAhead(ctx: ?*anyopaque, layer: u32, ids: [*]const u16, n: usize) callconv(.c) Verdict {
        return done(of(ctx).src.readAhead(layer, ids[0..n]));
    }
};

pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return .{ .files = &.{.{ .name = "config.json", .bytes = "{\"model_type\":\"bailing_hybrid\",\"num_hidden_layers\":2,\"first_k_dense_replace\":1}" }}, .block = "{\"quant_method\":\"exl3\",\"codebook\":\"mul1\"}" };
}
