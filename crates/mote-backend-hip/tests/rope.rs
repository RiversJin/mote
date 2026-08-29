#![cfg(feature = "rocm")]

//! Standalone GPU integration tests for the native HIP RoPE op.
//!
//! Expected values come from `mote_reference::rope_f32`: the F16 input is
//! quantized first and widened back to F32 so the oracle sees the exact bit
//! patterns the GPU reads, while the F32 cache is shared verbatim; the oracle
//! output is re-quantized to F16 before comparison with the GPU readback.
//! The default geometry `[tokens=2, heads=3, head_dim=70]` with
//! `rotary_dim=66` deliberately covers both the rotated prefix and the
//! non-rotary tail, under both pairing conventions.

use half::f16;
use mote_backend_hip::{HipContext, HipError, rope_f16};
use mote_core::Tensor;
use mote_reference::rope_f32;
use mote_types::{DType, Encoding, Layout, RopeLayout, Shape, TensorDesc};

const TOKENS: usize = 2;
const HEADS: usize = 3;
const HEAD_DIM: usize = 70;
const ROTARY_DIM: usize = 66;
const FREQUENCY_COUNT: usize = ROTARY_DIM / 2;

#[test]
fn hip_rope_f16_matches_reference_half_split() {
    assert_matches_reference(&[TOKENS], RopeLayout::HalfSplit, false);
}

#[test]
fn hip_rope_f16_matches_reference_interleaved() {
    assert_matches_reference(&[TOKENS], RopeLayout::Interleaved, false);
}

#[test]
fn hip_rope_f16_matches_reference_for_flattened_leading_dims() {
    // Rank 2: the leading dims are empty, so a single token shares one cache
    // row. Rank 4: two leading dims flatten into four tokens.
    for layout in [RopeLayout::HalfSplit, RopeLayout::Interleaved] {
        assert_matches_reference(&[], layout, false);
        assert_matches_reference(&[2, 2], layout, false);
    }
}

#[test]
fn hip_rope_f16_in_place_matches_reference() {
    // The kernel is race-free when output aliases input; the public API must
    // not reject it.
    assert_matches_reference(&[TOKENS], RopeLayout::HalfSplit, true);
    assert_matches_reference(&[TOKENS], RopeLayout::Interleaved, true);
}

#[test]
fn empty_tensors_return_without_launching() {
    let context = HipContext::new(0).unwrap();
    let (_, cos, sin, _) = fixtures(&context, ROTARY_DIM);

    // Zero tokens: the cache is empty too.
    let input = context.empty(f16_desc(&[0, HEADS, HEAD_DIM])).unwrap();
    let output = context.empty(f16_desc(&[0, HEADS, HEAD_DIM])).unwrap();
    let empty_cache = context.empty(f32_desc(&[0, FREQUENCY_COUNT])).unwrap();
    rope_f16(
        &context,
        &input,
        &empty_cache,
        &empty_cache,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());

    // Zero heads: the cache keeps its shape but is never read.
    let input = context.empty(f16_desc(&[TOKENS, 0, HEAD_DIM])).unwrap();
    let output = context.empty(f16_desc(&[TOKENS, 0, HEAD_DIM])).unwrap();
    rope_f16(
        &context,
        &input,
        &cos,
        &sin,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());

    // Zero head dimension degenerates to an empty rotary prefix.
    let input = context.empty(f16_desc(&[TOKENS, HEADS, 0])).unwrap();
    let output = context.empty(f16_desc(&[TOKENS, HEADS, 0])).unwrap();
    let empty_cache = context.empty(f32_desc(&[TOKENS, 0])).unwrap();
    rope_f16(
        &context,
        &input,
        &empty_cache,
        &empty_cache,
        &output,
        0,
        RopeLayout::Interleaved,
    )
    .unwrap();
    assert!(context.read_bytes(&output).unwrap().is_empty());
}

#[test]
fn odd_rotary_dim_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, cos, sin, output) = fixtures(&context, ROTARY_DIM);

    let error = rope_f16(
        &context,
        &input,
        &cos,
        &sin,
        &output,
        ROTARY_DIM + 1,
        RopeLayout::HalfSplit,
    )
    .unwrap_err();
    match error {
        HipError::InvalidRotaryDim {
            rotary_dim,
            head_dim,
        } => {
            assert_eq!(rotary_dim, ROTARY_DIM + 1);
            assert_eq!(head_dim, HEAD_DIM);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn oversized_rotary_dim_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, cos, sin, output) = fixtures(&context, ROTARY_DIM);

    // Even, but past the end of the head vector.
    let error = rope_f16(
        &context,
        &input,
        &cos,
        &sin,
        &output,
        HEAD_DIM + 2,
        RopeLayout::Interleaved,
    )
    .unwrap_err();
    match error {
        HipError::InvalidRotaryDim {
            rotary_dim,
            head_dim,
        } => {
            assert_eq!(rotary_dim, HEAD_DIM + 2);
            assert_eq!(head_dim, HEAD_DIM);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn cache_shape_mismatches_are_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, cos, sin, output) = fixtures(&context, ROTARY_DIM);

    // Wrong frequency count.
    let wrong_cos = context
        .empty(f32_desc(&[TOKENS, FREQUENCY_COUNT - 1]))
        .unwrap();
    let error = rope_f16(
        &context,
        &input,
        &wrong_cos,
        &sin,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap_err();
    assert!(
        matches!(error, HipError::ShapeMismatch { tensor: "cos", .. }),
        "{error:?}"
    );

    // Wrong (extra) leading dimension.
    let wrong_sin = context
        .empty(f32_desc(&[TOKENS, 1, FREQUENCY_COUNT]))
        .unwrap();
    let error = rope_f16(
        &context,
        &input,
        &cos,
        &wrong_sin,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap_err();
    assert!(
        matches!(error, HipError::ShapeMismatch { tensor: "sin", .. }),
        "{error:?}"
    );
}

#[test]
fn output_shape_mismatch_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, cos, sin, _) = fixtures(&context, ROTARY_DIM);

    let wrong_output = context
        .empty(f16_desc(&[TOKENS, HEADS, HEAD_DIM + 1]))
        .unwrap();
    let error = rope_f16(
        &context,
        &input,
        &cos,
        &sin,
        &wrong_output,
        ROTARY_DIM,
        RopeLayout::Interleaved,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::ShapeMismatch {
                tensor: "output",
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn rank_one_input_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (_, cos, sin, _) = fixtures(&context, ROTARY_DIM);

    let input = context.empty(f16_desc(&[HEADS * HEAD_DIM])).unwrap();
    let output = context.empty(f16_desc(&[HEADS * HEAD_DIM])).unwrap();
    let error = rope_f16(
        &context,
        &input,
        &cos,
        &sin,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::RankTooSmall {
                tensor: "input",
                minimum: 2,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn cache_dtype_mismatch_is_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, cos, sin, output) = fixtures(&context, ROTARY_DIM);

    let wrong_cos = context.empty(f16_desc(&[TOKENS, FREQUENCY_COUNT])).unwrap();
    let error = rope_f16(
        &context,
        &input,
        &wrong_cos,
        &sin,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::DTypeMismatch {
                tensor: "cos",
                expected: DType::F32,
                ..
            }
        ),
        "{error:?}"
    );
    assert_ne!(cos.desc().encoding(), wrong_cos.desc().encoding());

    let wrong_sin = context.empty(f16_desc(&[TOKENS, FREQUENCY_COUNT])).unwrap();
    let error = rope_f16(
        &context,
        &input,
        &cos,
        &wrong_sin,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            HipError::DTypeMismatch {
                tensor: "sin",
                expected: DType::F32,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn tensors_from_another_context_are_rejected() {
    let context = HipContext::new(0).unwrap();
    let (input, cos, sin, _) = fixtures(&context, ROTARY_DIM);

    let stranger = HipContext::new(0).unwrap();
    let output = stranger
        .empty(f16_desc(&[TOKENS, HEADS, HEAD_DIM]))
        .unwrap();
    let error = rope_f16(
        &context,
        &input,
        &cos,
        &sin,
        &output,
        ROTARY_DIM,
        RopeLayout::HalfSplit,
    )
    .unwrap_err();
    assert!(matches!(error, HipError::WrongStorage), "{error:?}");
}

/// Runs the GPU op for `leading_dims ++ [HEADS, HEAD_DIM]` and compares the
/// readback against `mote_reference::rope_f32`. With `in_place`, the input
/// tensor doubles as the output.
fn assert_matches_reference(leading_dims: &[usize], layout: RopeLayout, in_place: bool) {
    let context = HipContext::new(0).unwrap();

    let shape: Vec<usize> = leading_dims
        .iter()
        .copied()
        .chain([HEADS, HEAD_DIM])
        .collect();
    let cache_dims: Vec<usize> = leading_dims
        .iter()
        .copied()
        .chain([FREQUENCY_COUNT])
        .collect();
    let input_len: usize = shape.iter().product();
    let cache_len: usize = cache_dims.iter().product();
    let tokens: usize = leading_dims.iter().product();

    // Quantize to F16 first, then widen back to F32: the oracle must see the
    // exact bit patterns the GPU reads. The cache stays F32 end to end.
    let input: Vec<f16> = (0..input_len)
        .map(|index| f16::from_f32(input_element(index)))
        .collect();
    let cos: Vec<f32> = (0..cache_len).map(cos_element).collect();
    let sin: Vec<f32> = (0..cache_len).map(sin_element).collect();
    let input_f32: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
    let expected = rope_f32(
        &input_f32, &cos, &sin, tokens, HEADS, HEAD_DIM, ROTARY_DIM, layout,
    )
    .unwrap();

    let input_tensor = context
        .from_bytes(f16_desc(&shape), &encode_f16(&input))
        .unwrap();
    let cos_tensor = context
        .from_bytes(f32_desc(&cache_dims), &encode_f32(&cos))
        .unwrap();
    let sin_tensor = context
        .from_bytes(f32_desc(&cache_dims), &encode_f32(&sin))
        .unwrap();

    let readback = if in_place {
        rope_f16(
            &context,
            &input_tensor,
            &cos_tensor,
            &sin_tensor,
            &input_tensor,
            ROTARY_DIM,
            layout,
        )
        .unwrap();
        decode_f16(&context.read_bytes(&input_tensor).unwrap())
    } else {
        let output = context.empty(f16_desc(&shape)).unwrap();
        rope_f16(
            &context,
            &input_tensor,
            &cos_tensor,
            &sin_tensor,
            &output,
            ROTARY_DIM,
            layout,
        )
        .unwrap();
        decode_f16(&context.read_bytes(&output).unwrap())
    };

    assert_eq!(readback.len(), input_len);
    for (index, (raw, expected)) in readback.iter().zip(expected.iter()).enumerate() {
        // Both sides rotate in F32 and round once to F16; the tolerance
        // absorbs a few F16 ulps plus fma-contraction noise in the kernel.
        let actual = raw.to_f32();
        let expected = f16::from_f32(*expected).to_f32();
        let tolerance = 3e-3_f32.max(expected.abs() * 3e-3);
        assert!(
            (actual - expected).abs() <= tolerance,
            "mismatch at {index}: expected {expected}, got {actual}"
        );
        // The tail past `rotary_dim` is a verbatim copy of the quantized
        // input, so it must survive bit-exact.
        if index % HEAD_DIM >= ROTARY_DIM {
            assert_eq!(
                raw.to_bits(),
                input[index].to_bits(),
                "tail element {index} changed"
            );
        }
    }
}

/// F16 input, F32 cache, and F16 output with the default geometry and a cache
/// sized for `rotary_dim`.
fn fixtures(context: &HipContext, rotary_dim: usize) -> (Tensor, Tensor, Tensor, Tensor) {
    let input: Vec<f16> = (0..TOKENS * HEADS * HEAD_DIM)
        .map(|index| f16::from_f32(input_element(index)))
        .collect();
    let cache_len = TOKENS * (rotary_dim / 2);
    let cos: Vec<f32> = (0..cache_len).map(cos_element).collect();
    let sin: Vec<f32> = (0..cache_len).map(sin_element).collect();

    let input = context
        .from_bytes(f16_desc(&[TOKENS, HEADS, HEAD_DIM]), &encode_f16(&input))
        .unwrap();
    let cos = context
        .from_bytes(f32_desc(&[TOKENS, rotary_dim / 2]), &encode_f32(&cos))
        .unwrap();
    let sin = context
        .from_bytes(f32_desc(&[TOKENS, rotary_dim / 2]), &encode_f32(&sin))
        .unwrap();
    let output = context.empty(f16_desc(&[TOKENS, HEADS, HEAD_DIM])).unwrap();
    (input, cos, sin, output)
}

fn input_element(index: usize) -> f32 {
    ((index * 37 + 5) % 19) as f32 / 6.0 - 1.5
}

fn cos_element(index: usize) -> f32 {
    ((index * 11 + 3) % 13) as f32 / 12.0
}

fn sin_element(index: usize) -> f32 {
    ((index * 7 + 1) % 17) as f32 / 16.0
}

fn encode_f16(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_ne_bytes())
        .collect()
}

fn decode_f16(bytes: &[u8]) -> Vec<f16> {
    let (words, remainder) = bytes.as_chunks::<2>();
    assert!(remainder.is_empty());
    words
        .iter()
        .copied()
        .map(|word| f16::from_bits(u16::from_ne_bytes(word)))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn f16_desc(dims: &[usize]) -> TensorDesc {
    TensorDesc::new(
        Shape::new(dims),
        Encoding::Plain(DType::F16),
        Layout::Contiguous,
    )
    .unwrap()
}

fn f32_desc(dims: &[usize]) -> TensorDesc {
    TensorDesc::new(
        Shape::new(dims),
        Encoding::Plain(DType::F32),
        Layout::Contiguous,
    )
    .unwrap()
}
