use std::{
    ffi::{CStr, c_void},
    os::raw::c_char,
};

use crate::HipError;

/// Narrow HIP runtime surface exported by `src/hipblaslt_shim.cpp`.
///
/// The Rust side never includes HIP headers or replicates their ABI: streams
/// are opaque handles, pointers are untyped, and every entry point reports the
/// raw C `int` status where [`SUCCESS`] means success. Callers translate the
/// status through [`crate::error::check_hip`].
pub(crate) mod hip {
    use std::ffi::c_void;

    /// Status code returned by every HIP runtime entry point.
    pub(crate) type Status = i32;

    /// Status value returned by successful HIP runtime calls.
    pub(crate) const SUCCESS: Status = 0;

    /// Opaque handle to a HIP stream created through [`stream_create`].
    ///
    /// Stored as an integer so contexts holding a stream stay `Send` and
    /// `Sync`; the handle is only ever dereferenced inside the shim.
    #[derive(Clone, Copy)]
    pub(crate) struct Stream {
        handle: usize,
    }

    impl Stream {
        pub(crate) fn null() -> Self {
            Self { handle: 0 }
        }

        pub(crate) fn as_c_void(self) -> *mut c_void {
            self.handle as *mut c_void
        }
    }

    unsafe extern "C" {
        fn mote_hip_get_device_count(count: *mut i32) -> Status;
        fn mote_hip_set_device(ordinal: i32) -> Status;
        fn mote_hip_malloc(pointer: *mut *mut c_void, bytes: usize) -> Status;
        fn mote_hip_free(pointer: *mut c_void) -> Status;
        fn mote_hip_memcpy_host_to_device(
            destination: *mut c_void,
            source: *const c_void,
            bytes: usize,
        ) -> Status;
        fn mote_hip_memcpy_device_to_host(
            destination: *mut c_void,
            source: *const c_void,
            bytes: usize,
        ) -> Status;
        fn mote_hip_stream_create(stream: *mut *mut c_void) -> Status;
        fn mote_hip_stream_destroy(stream: *mut c_void) -> Status;
        fn mote_hip_stream_synchronize(stream: *mut c_void) -> Status;
        fn mote_hip_mem_get_info(free_bytes: *mut usize, total_bytes: *mut usize) -> Status;
    }

    pub(crate) fn get_device_count(count: &mut i32) -> Status {
        unsafe { mote_hip_get_device_count(count) }
    }

    pub(crate) fn set_device(ordinal: i32) -> Status {
        unsafe { mote_hip_set_device(ordinal) }
    }

    pub(crate) fn malloc(pointer: &mut *mut c_void, bytes: usize) -> Status {
        unsafe { mote_hip_malloc(pointer, bytes) }
    }

    pub(crate) fn free(pointer: *mut c_void) -> Status {
        unsafe { mote_hip_free(pointer) }
    }

    pub(crate) fn memcpy_host_to_device(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> Status {
        unsafe { mote_hip_memcpy_host_to_device(destination, source, bytes) }
    }

    pub(crate) fn memcpy_device_to_host(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> Status {
        unsafe { mote_hip_memcpy_device_to_host(destination, source, bytes) }
    }

    /// Creates a stream on the current device; on failure `stream` is left
    /// null.
    pub(crate) fn stream_create(stream: &mut Stream) -> Status {
        let mut pointer = std::ptr::null_mut();
        let status = unsafe { mote_hip_stream_create(&mut pointer) };
        if status == SUCCESS {
            stream.handle = pointer as usize;
        }
        status
    }

    pub(crate) fn stream_destroy(stream: Stream) -> Status {
        unsafe { mote_hip_stream_destroy(stream.as_c_void()) }
    }

    pub(crate) fn stream_synchronize(stream: Stream) -> Status {
        unsafe { mote_hip_stream_synchronize(stream.as_c_void()) }
    }

    /// Writes the current device's free and total memory in bytes.
    pub(crate) fn mem_get_info(free_bytes: &mut usize, total_bytes: &mut usize) -> Status {
        unsafe { mote_hip_mem_get_info(free_bytes, total_bytes) }
    }
}

unsafe extern "C" {
    /// Launch shim exported by `src/rms_norm.hip` (compiled by `hipcc`).
    ///
    /// One block of 256 threads normalizes one row: F16 input/weight/output
    /// with F32 accumulation and a shared-memory reduction, so hidden sizes
    /// that are not multiples of 256 (e.g. 513) are handled exactly.
    fn mote_hip_rms_norm_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: u64,
        hidden_size: u64,
        epsilon: f32,
        stream: *mut c_void,
    ) -> hip::Status;
}

/// Enqueues the RMSNorm kernel on `stream`, borrowing it as an opaque value.
///
/// Returns the raw HIP status; callers translate it through
/// [`crate::error::check_hip`].
pub(crate) fn rms_norm_f16(
    input: *const c_void,
    weight: *const c_void,
    output: *mut c_void,
    rows: u64,
    hidden_size: u64,
    epsilon: f32,
    stream: hip::Stream,
) -> hip::Status {
    unsafe {
        mote_hip_rms_norm_f16(
            input,
            weight,
            output,
            rows,
            hidden_size,
            epsilon,
            stream.as_c_void(),
        )
    }
}

unsafe extern "C" {
    /// Launch shim exported by `src/fused_add_rms_norm.hip` (compiled by
    /// `hipcc`).
    ///
    /// One block of 256 threads fuses the residual add and the RMS norm for
    /// one row: F16 input/residual/weight/output with F32 accumulation and a
    /// shared-memory reduction, so hidden sizes that are not multiples of
    /// 256 (e.g. 513) are handled exactly. The statistics pass only reads
    /// `input` and the original `residual`; after the reduction a second
    /// pass re-reads both to reproduce the same un-rounded F32 sums, writes
    /// the normalized row, and overwrites `residual` with the F16-rounded
    /// sums. `input` and `output` may alias; `output` must not alias
    /// `residual`.
    fn mote_hip_fused_add_rms_norm_f16(
        input: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: u64,
        hidden_size: u64,
        epsilon: f32,
        stream: *mut c_void,
    ) -> hip::Status;
}

/// Enqueues the fused residual-add + RMSNorm kernel on `stream`, borrowing
/// it as an opaque value.
///
/// `residual` is passed mutably: the kernel reads the original values in
/// both passes and overwrites the buffer with the F16-rounded
/// `input + residual` sums in the second one. `input` and `output` may
/// alias; `output` must not alias `residual`. Returns the raw HIP status;
/// callers translate it through [`crate::error::check_hip`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_add_rms_norm_f16(
    input: *const c_void,
    residual: *mut c_void,
    weight: *const c_void,
    output: *mut c_void,
    rows: u64,
    hidden_size: u64,
    epsilon: f32,
    stream: hip::Stream,
) -> hip::Status {
    unsafe {
        mote_hip_fused_add_rms_norm_f16(
            input,
            residual,
            weight,
            output,
            rows,
            hidden_size,
            epsilon,
            stream.as_c_void(),
        )
    }
}

unsafe extern "C" {
    /// Launch shim exported by `src/rope.hip` (compiled by `hipcc`).
    ///
    /// One block of 256 threads rotates one (token, head) vector: F16
    /// input/output with F32 cos/sin rows of `rotary_dim / 2` values per
    /// token, the leading `rotary_dim` dimensions rotated, and the tail
    /// passed through unchanged. `layout` selects the pairing convention
    /// ([`ROPE_LAYOUT_HALF_SPLIT`] / [`ROPE_LAYOUT_INTERLEAVED`]); the shim
    /// rejects unknown values before launching.
    fn mote_hip_rope_f16(
        input: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        output: *mut c_void,
        tokens: u64,
        heads: u64,
        head_dim: u64,
        rotary_dim: u64,
        layout: u32,
        stream: *mut c_void,
    ) -> hip::Status;
}

/// Layout selector for the RoPE shim: pair the first and second halves of
/// the rotary dimensions (`mote_types::RopeLayout::HalfSplit`).
pub(crate) const ROPE_LAYOUT_HALF_SPLIT: u32 = 0;

/// Layout selector for the RoPE shim: pair adjacent rotary dimensions
/// (`mote_types::RopeLayout::Interleaved`).
pub(crate) const ROPE_LAYOUT_INTERLEAVED: u32 = 1;

/// Enqueues the RoPE kernel on `stream`, borrowing it as an opaque value.
///
/// `layout` must be one of [`ROPE_LAYOUT_HALF_SPLIT`] /
/// [`ROPE_LAYOUT_INTERLEAVED`]; unknown values are rejected by the shim.
/// Returns the raw HIP status; callers translate it through
/// [`crate::error::check_hip`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_f16(
    input: *const c_void,
    cos: *const c_void,
    sin: *const c_void,
    output: *mut c_void,
    tokens: u64,
    heads: u64,
    head_dim: u64,
    rotary_dim: u64,
    layout: u32,
    stream: hip::Stream,
) -> hip::Status {
    unsafe {
        mote_hip_rope_f16(
            input,
            cos,
            sin,
            output,
            tokens,
            heads,
            head_dim,
            rotary_dim,
            layout,
            stream.as_c_void(),
        )
    }
}

unsafe extern "C" {
    /// Launch shim exported by `src/quantized_linear.hip`.
    ///
    /// Each block computes one output element from an F16 activation row and
    /// one GGML-compatible Q4_0 weight row, accumulating in F32 and storing
    /// F16. The weight matrix is `[output_features, input_features]` and the
    /// result is `input * weights^T`.
    fn mote_hip_quantized_linear_q4_0_f16(
        input: *const c_void,
        weights: *const c_void,
        output: *mut c_void,
        rows: u64,
        output_features: u64,
        input_features: u64,
        stream: *mut c_void,
    ) -> hip::Status;
}

/// Enqueues the Q4_0 weight-only linear kernel on `stream`.
pub(crate) fn quantized_linear_q4_0_f16(
    input: *const c_void,
    weights: *const c_void,
    output: *mut c_void,
    rows: u64,
    output_features: u64,
    input_features: u64,
    stream: hip::Stream,
) -> hip::Status {
    unsafe {
        mote_hip_quantized_linear_q4_0_f16(
            input,
            weights,
            output,
            rows,
            output_features,
            input_features,
            stream.as_c_void(),
        )
    }
}

unsafe extern "C" {
    fn mote_hipblaslt_create(handle: *mut *mut c_void) -> i32;
    fn mote_hipblaslt_destroy(handle: *mut c_void);
    fn mote_hipblaslt_last_error(handle: *mut c_void) -> *const c_char;
    fn mote_hipblaslt_matmul_f16_f32(
        handle: *mut c_void,
        m: u64,
        n: u64,
        k: u64,
        lhs: *const c_void,
        rhs: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

pub(crate) struct HipBlasLt {
    handle: usize,
}

impl HipBlasLt {
    pub(crate) fn new() -> Result<Self, HipError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { mote_hipblaslt_create(&mut handle) };
        if status == 0 {
            Ok(Self {
                handle: handle as usize,
            })
        } else {
            Err(HipError::BlasLt {
                operation: "handle creation",
                status,
                message: "hipblasLtCreate failed".into(),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_f16_f32(
        &mut self,
        m: u64,
        n: u64,
        k: u64,
        lhs: *const c_void,
        rhs: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: hip::Stream,
    ) -> Result<(), HipError> {
        let handle = self.handle as *mut c_void;
        let status = unsafe {
            mote_hipblaslt_matmul_f16_f32(
                handle,
                m,
                n,
                k,
                lhs,
                rhs,
                output,
                workspace,
                workspace_bytes,
                stream.as_c_void(),
            )
        };
        if status == 0 {
            return Ok(());
        }

        let message = unsafe {
            let pointer = mote_hipblaslt_last_error(handle);
            if pointer.is_null() {
                "unknown hipBLASLt error".into()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(HipError::BlasLt {
            operation: "F16/F32 matmul",
            status,
            message,
        })
    }
}

impl Drop for HipBlasLt {
    fn drop(&mut self) {
        unsafe { mote_hipblaslt_destroy(self.handle as *mut c_void) };
    }
}
