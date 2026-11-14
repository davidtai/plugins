//! The routed experts' op backend: `MlxOps` builds mlx-c graphs on one stream (the inference thread); intermediates are
//! freed by `reset`, state leaves the scope only through `keep`.
//! Python scalars are weak in MLX (`to_array(v, other.dtype)`); callers pass
//! the dtype explicitly through `scalar`, never an f32 default.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");

pub const Dtype = mlx.mlx_dtype;
pub const max_dims = 8;

pub const Shape = struct {
    n: u8 = 0,
    d: [max_dims]c_int = @splat(0),

    pub fn of(dims: []const c_int) Shape {
        var s: Shape = .{ .n = @intCast(dims.len) };
        @memcpy(s.d[0..dims.len], dims);
        return s;
    }

    pub fn slice(self: *const Shape) []const c_int {
        return self.d[0..self.n];
    }

    pub fn dim(self: Shape, axis: c_int) c_int {
        return self.d[normAxis(axis, self.n)];
    }

};

pub fn normAxis(axis: c_int, n: u8) usize {
    return @intCast(if (axis < 0) axis + @as(c_int, n) else axis);
}

/// A point in a backend's scope (`mark`). `resetTo` frees every tracked array
/// built after it; arrays held through `keep` are separate handles and survive.
/// A mark is spent by `reset` or by a `resetTo` to an earlier mark.
pub const Mark = struct { n: usize };

/// Frees `list[from..]` and truncates the list: the one release path of
/// `reset` (from 0) and `resetTo` (from a mark).
fn freeFrom(comptime E: type, list: *std.ArrayList(E), from: usize, comptime free: fn (E) void) void {
    std.debug.assert(from <= list.items.len);
    for (list.items[from..]) |a| free(a);
    list.shrinkRetainingCapacity(from);
}

/// The compiled regions a model builds at construction (`prepareTape`), one
/// closure per region and context (the trunk's config, the draft head's).
pub const Region = enum { attn_core, qkv_prep, out_prep, gate_prefix, moe_combine, hc_attn_prep, hc_ffn_prep, hc_post, seg2, seg3, draft_kv, markov_step, confidence, shared_mid };
const n_regions = std.meta.fieldNames(Region).len;
const contexts_per_region = 2;

// ── MLX backend ──

pub const MlxOps = struct {
    pub const T = mlx.mlx_array;

    gpa: std.mem.Allocator,
    s: mlx.mlx_stream,
    live: std.ArrayList(mlx.mlx_array) = .empty,
    /// The host's waits on the GPU so far: every evaluation that blocks (`evalAll`) and every read of an array's values
    /// (`hostIds`, `hostU32`, `hostF32`, `hostBool`, `hostArgmax`); `asyncEval` waits for nothing, nor do `readU32` and
    /// `readBool` (they read what an earlier eval produced).
    waits: u64 = 0,
    /// `nn.silu` / `nn.softplus` are `mx.compile(shapeless=True)` regions in
    /// the Python oracle; the same closures are compiled here (a compiled
    /// graph is not output-preserving against the op chain).
    silu_fn: mlx.mlx_closure = .{ .ctx = null },
    softplus_fn: mlx.mlx_closure = .{ .ctx = null },
    stream_box: *mlx.mlx_stream,
    /// Compiled trunk regions, one per (region type, construction context).
    tapes: std.ArrayList(TapeEntry) = .empty,
    /// Each region's closures by context, filled by `prepareTape`.
    regions: [n_regions][contexts_per_region]RegionSlot = @splat(@splat(.{})),
    /// A region's tracing context: borrows the owner's stream and closures.
    is_child: bool = false,

    const RegionSlot = struct { ctx: ?*const anyopaque = null, compiled: mlx.mlx_closure = .{ .ctx = null } };

    const TapeEntry = struct {
        key: usize,
        ctx: *const anyopaque,
        compiled: mlx.mlx_closure,
        payload: *anyopaque,
        free: *const fn (*anyopaque, std.mem.Allocator) void,
    };

    pub fn init(gpa: std.mem.Allocator, s: mlx.mlx_stream) !MlxOps {
        const box = try gpa.create(mlx.mlx_stream);
        errdefer gpa.destroy(box);
        box.* = s;
        var g: MlxOps = .{ .gpa = gpa, .s = s, .stream_box = box };
        g.silu_fn = try compileUnary(siluBody, box);
        errdefer _ = mlx.mlx_closure_free(g.silu_fn);
        g.softplus_fn = try compileUnary(softplusBody, box);
        return g;
    }

    pub fn deinit(g: *MlxOps) void {
        g.reset();
        g.live.deinit(g.gpa);
        if (g.is_child) return;
        for (g.tapes.items) |t| {
            _ = mlx.mlx_closure_free(t.compiled);
            t.free(t.payload, g.gpa);
        }
        g.tapes.deinit(g.gpa);
        _ = mlx.mlx_closure_free(g.silu_fn);
        _ = mlx.mlx_closure_free(g.softplus_fn);
        g.gpa.destroy(g.stream_box);
    }

    pub fn evalAll(g: *MlxOps, xs: []const T) !void {
        g.waits += 1;
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    fn freeArray(a: mlx.mlx_array) void {
        _ = mlx.mlx_array_free(a);
    }

    /// Free every intermediate built since the last reset.
    pub fn reset(g: *MlxOps) void {
        freeFrom(T, &g.live, 0, freeArray);
    }

    /// Return MLX's cached (freed) buffers to the system: the phase boundaries' release (the prompt
    /// fence, the phase change, construction), owned by the backend like every other allocation.
    pub fn clearCache(_: *MlxOps) void {
        _ = mlx.mlx_clear_cache();
    }

    /// The current scope point (a wave's start).
    pub fn mark(g: *const MlxOps) Mark {
        return .{ .n = g.live.items.len };
    }

    /// Free every array tracked since `m` (a wave's intermediates, after its
    /// eval); what the caller kept survives.
    pub fn resetTo(g: *MlxOps, m: Mark) void {
        freeFrom(T, &g.live, m.n, freeArray);
    }

    /// A reference that outlives `reset` (cache state, outputs); the caller
    /// frees it with `release`.
    pub fn keep(_: *MlxOps, x: T) T {
        var out = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&out, x);
        return out;
    }

    pub fn release(_: *MlxOps, x: T) void {
        _ = mlx.mlx_array_free(x);
    }

    /// Take ownership of an array built outside the backend (freed by `reset`).
    pub fn adopt(g: *MlxOps, x: T) !T {
        return g.track(x);
    }

    /// The bytes of an array made by `hostArray` (its own unified-memory buffer), writable in place: DRAFT_AHEAD's
    /// persistent inputs, written only while no committed command reads them.
    pub fn hostBytes(_: *MlxOps, x: T) ![]u8 {
        const p = mlx.mlx_array_data_uint8(x) orelse return error.MlxNoData;
        return @constCast(p)[0 .. mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x)];
    }

    /// A copy of host bytes as an array (`mx.array(numpy)`); freed by `reset`.
    pub fn hostArray(g: *MlxOps, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
        const a = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (a.ctx == null) return error.MlxError;
        return g.track(a);
    }

    fn track(g: *MlxOps, a: T) !T {
        g.live.append(g.gpa, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        return a;
    }

    pub fn dtypeOf(_: *MlxOps, x: T) Dtype {
        return mlx.mlx_array_dtype(x);
    }

    pub fn shapeOf(_: *MlxOps, x: T) Shape {
        return Shape.of(mlx.getShape(x));
    }

    fn op1(g: *MlxOps, comptime f: anytype, a: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, a, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    fn op2(g: *MlxOps, comptime f: anytype, a: T, b: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, a, b, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// A 0-d constant of `dt` holding `v` rounded as MLX rounds a Python
    /// scalar (f64 -> f32 -> dt).
    pub fn scalar(g: *MlxOps, v: f64, dt: Dtype) !T {
        const shape = [_]c_int{};
        const a = switch (dt) {
            .float32 => mlx.mlx_array_new_float(@floatCast(v)),
            .bool_ => mlx.mlx_array_new_bool(v != 0),
            .int32 => mlx.mlx_array_new_int(@intFromFloat(v)),
            .bfloat16 => blk: {
                const bits = bf16Bits(@floatCast(v));
                break :blk mlx.mlx_array_new_data(&bits, &shape, 0, .bfloat16);
            },
            .float16 => blk: {
                const h: f16 = @floatCast(@as(f32, @floatCast(v)));
                break :blk mlx.mlx_array_new_data(&h, &shape, 0, .float16);
            },
            else => return error.UnsupportedScalar,
        };
        if (a.ctx == null) return error.MlxError;
        return g.track(a);
    }

    pub fn ones(g: *MlxOps, shape: []const c_int, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_ones(&r, shape.ptr, shape.len, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn zeros(g: *MlxOps, shape: []const c_int, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_zeros(&r, shape.ptr, shape.len, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn astype(g: *MlxOps, x: T, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_astype(&r, x, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn add(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_add, a, b);
    }
    pub fn sub(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_subtract, a, b);
    }
    pub fn mul(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_multiply, a, b);
    }
    pub fn maximum(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_maximum, a, b);
    }
    pub fn minimum(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_minimum, a, b);
    }
    pub fn matmul(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_matmul, a, b);
    }

    pub fn rsqrt(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_rsqrt, x);
    }
    pub fn exp(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_exp, x);
    }
    /// `mx.log` (GLM-5.3's MTP lane: the typical test's entropy of a sampled row).
    pub fn log(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_log, x);
    }
    pub fn sigmoid(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_sigmoid, x);
    }
    pub fn cos(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_cos, x);
    }
    pub fn sin(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_sin, x);
    }
    pub fn abs(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_abs, x);
    }
    /// `.T`: every axis reversed.
    pub fn transpose(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_transpose, x);
    }

    pub fn clip(g: *MlxOps, x: T, lo: T, hi: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_clip(&r, x, lo, hi, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn reshape(g: *MlxOps, x: T, shape: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_reshape(&r, x, shape.ptr, shape.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn slice(g: *MlxOps, x: T, start: []const c_int, stop: []const c_int, strides: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_slice(&r, x, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn concat(g: *MlxOps, xs: []const T, axis: c_int) !T {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_concatenate_axis(&r, vec, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn take(g: *MlxOps, x: T, idx: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_take_axis(&r, x, idx, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    fn reduce(g: *MlxOps, comptime f: anytype, x: T, axis: c_int, keepdims: bool) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, x, axis, keepdims, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }
    pub fn sum(g: *MlxOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(mlx.mlx_sum_axis, x, axis, keepdims);
    }
    pub fn mean(g: *MlxOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(mlx.mlx_mean_axis, x, axis, keepdims);
    }
    pub fn max(g: *MlxOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(mlx.mlx_max_axis, x, axis, keepdims);
    }

    /// `mx.softmax` (precise = False).
    pub fn softmax(g: *MlxOps, x: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_softmax_axis(&r, x, axis, false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `nn.silu`: the compiled `x * sigmoid(x)`.
    pub fn silu(g: *MlxOps, x: T) !T {
        return g.applyUnary(g.silu_fn, x);
    }

    /// `mx.async_eval(xs)`: the GPU starts on `xs`; nothing waits.
    pub fn asyncEval(_: *MlxOps, xs: []const T) !void {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
    }

    /// The routing barrier (`mx.eval(indices)` + `tolist`): evaluates the integer ids `x` and copies
    /// them, row-major, into `out` (x's size). Read in place (`readInPlace`).
    pub fn hostIds(g: *MlxOps, x: T, out: []u16) ![]const u16 {
        g.waits += 1;
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostIdsSize;
        // One producer dtype: the router's int32 indices (the arm's stand-in emits the same).
        std.debug.assert(mlx.mlx_array_dtype(x) == .int32);
        return readInPlace(i32, u16, x, mlx.mlx_array_data_int32(x), out);
    }

    /// Copies the evaluated `x`'s values, row-major, into `out`, reading its buffer through its own
    /// strides: an array an eval already produced is read with no GPU round trip (a `reshape(-1)`
    /// is a new array, and evaluating it is a fresh eval: a commit and a wait).
    fn readInPlace(comptime S: type, comptime D: type, x: T, data: ?[*]const S, out: []D) ![]const D {
        const nd = mlx.mlx_array_ndim(x);
        copyStrided(S, D, data orelse return error.MlxNoData, mlx.mlx_array_shape(x)[0..nd], mlx.mlx_array_strides(x)[0..nd], out);
        return out;
    }

    pub fn hostF32(g: *MlxOps, x: T, out: []f32) ![]const f32 {
        g.waits += 1;
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .float32);
        return readInPlace(f32, f32, x, mlx.mlx_array_data_float32(x), out);
    }

    fn compileUnary(comptime body: fn (mlx.mlx_array, mlx.mlx_stream) ?mlx.mlx_array, box: *mlx.mlx_stream) !mlx.mlx_closure {
        const Cb = struct {
            fn call(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                const s: *const mlx.mlx_stream = @ptrCast(@alignCast(payload.?));
                var x = mlx.mlx_array_new();
                if (mlx.mlx_vector_array_get(&x, input, 0) != 0) return -1;
                defer _ = mlx.mlx_array_free(x);
                const y = body(x, s.*) orelse return -1;
                const out = [_]mlx.mlx_array{y};
                res.* = mlx.mlx_vector_array_new_data(&out, 1);
                _ = mlx.mlx_array_free(y);
                return 0;
            }
        };
        const raw = mlx.mlx_closure_new_func_payload(&Cb.call, @ptrCast(box), null);
        defer _ = mlx.mlx_closure_free(raw);
        var compiled = mlx.mlx_closure{ .ctx = null };
        try mlx.check(mlx.mlx_compile(&compiled, raw, true));
        return compiled;
    }

    fn siluBody(x: mlx.mlx_array, s: mlx.mlx_stream) ?mlx.mlx_array {
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        if (mlx.mlx_sigmoid(&sig, x, s) != 0) return null;
        var y = mlx.mlx_array_new();
        if (mlx.mlx_multiply(&y, x, sig, s) != 0) {
            _ = mlx.mlx_array_free(y);
            return null;
        }
        return y;
    }

    fn softplusBody(x: mlx.mlx_array, s: mlx.mlx_stream) ?mlx.mlx_array {
        // `mx.logaddexp(x, 0)`: the Python int is weak, so 0 takes x's dtype.
        const zero = switch (mlx.mlx_array_dtype(x)) {
            .float32 => mlx.mlx_array_new_float(0),
            else => return null,
        };
        defer _ = mlx.mlx_array_free(zero);
        var y = mlx.mlx_array_new();
        if (mlx.mlx_logaddexp(&y, x, zero, s) != 0) {
            _ = mlx.mlx_array_free(y);
            return null;
        }
        return y;
    }
};

/// The SDK's strided host copy (sdk_ext.ops), the name this package's code reads.
pub const copyStrided = sdk_ext.ops.copyStrided;

/// f32 -> bf16 bits, round to nearest even (what MLX's host conversion does).
pub fn bf16Bits(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return @intCast((u >> 16) | 0x40);
    const rounding: u32 = 0x7FFF + ((u >> 16) & 1);
    return @intCast((u +% rounding) >> 16);
}
