//! copied_from: src/sdk/weights/streamed/sdk/root.zig@bc6c5f2c102c1f08c11d04c3ca13e17a99dd78b4, trimmed for turboexl3_ling
//! as turboexl3_ds41 trims its own: the expert streamer's `sdk` as a quant plugin's bank reads it (MLX, the host's log,
//! the file helpers, the pack peeks, the memory ledgers, the quant modes and the comptime checks). The module-owned arch
//! SDK (arch, spec, weights, acceptance, testing) is the host's, never a bank's.

/// The profile probes (mlx-serve's `-Dplugin-profile`), never built into this plugin.
pub const plugin_profile = false;

pub const mlx = @import("mlx_host").mlx;
pub const log = @import("mlx_host").log;
pub const io_util = @import("io_util.zig");

const peek = @import("peek.zig");
pub const Diag = peek.Diag;
pub const GroupPeek = peek.GroupPeek;

/// Process and box memory readings (the kernel's ledgers).
pub const memory = @import("memory.zig");
pub const QuantMode = @import("quant_mode.zig").QuantMode;

