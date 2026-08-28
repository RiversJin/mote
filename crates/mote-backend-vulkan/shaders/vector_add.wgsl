struct FloatBuffer {
    values: array<f32>,
}

@group(0) @binding(0)
var<storage, read> lhs: FloatBuffer;

@group(0) @binding(1)
var<storage, read> rhs: FloatBuffer;

@group(0) @binding(2)
var<storage, read_write> output: FloatBuffer;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index < arrayLength(&output.values) {
        output.values[index] = lhs.values[index] + rhs.values[index];
    }
}
