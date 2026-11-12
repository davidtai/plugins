//! The expert streamer as this plugin builds it (module `mlx_stream`, over its own `sdk` in streamed/sdk/, the plugin's
//! root as `mlx_host`): Ling-3.1's bank source (`ling_source`: the EXL3 bank at Ling's geometry, every routed call on
//! turboexl3 through the plugin's src/ling_math.zig), which `open_bank` opens.

pub const sdk = @import("sdk");
/// Ling-3.1's EXL3 bank (streamed/ling_source.zig).
pub const ling_source = @import("streamed/ling_source.zig");
/// The event shim's ABI version the streamer checks at its first event (src/host_shims.zig serves it).
pub const event_abi = @import("streamed/sdk_ext/expert/event.zig").abi_version;
