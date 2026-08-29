struct MatmulParams
{
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]]
ConstantBuffer<MatmulParams> params;

[[vk::binding(0, 0)]]
StructuredBuffer<half> lhs;

[[vk::binding(1, 0)]]
StructuredBuffer<half> rhs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> output;

using namespace linalg;

typealias MatrixA = CoopMat<
    half,
    MemoryScope.Subgroup,
    16,
    16,
    CoopMatMatrixUse::MatrixA>;
typealias MatrixB = CoopMat<
    half,
    MemoryScope.Subgroup,
    16,
    16,
    CoopMatMatrixUse::MatrixB>;
typealias Accumulator = CoopMat<
    float,
    MemoryScope.Subgroup,
    16,
    16,
    CoopMatMatrixUse::MatrixAccumulator>;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 groupId : SV_GroupID)
{
    uint row = groupId.y * 16;
    uint col = groupId.x * 64;
    Accumulator accumulator0 = Accumulator(0.0f);
    Accumulator accumulator1 = Accumulator(0.0f);
    Accumulator accumulator2 = Accumulator(0.0f);
    Accumulator accumulator3 = Accumulator(0.0f);

    for (uint inner = 0; inner < params.k; inner += 16)
    {
        let a = MatrixA.Load<CoopMatMatrixLayout::RowMajor>(
            lhs,
            row * params.k + inner,
            params.k);
        let b0 = MatrixB.Load<CoopMatMatrixLayout::RowMajor>(
            rhs,
            inner * params.n + col,
            params.n);
        let b1 = MatrixB.Load<CoopMatMatrixLayout::RowMajor>(
            rhs,
            inner * params.n + col + 16,
            params.n);
        let b2 = MatrixB.Load<CoopMatMatrixLayout::RowMajor>(
            rhs,
            inner * params.n + col + 32,
            params.n);
        let b3 = MatrixB.Load<CoopMatMatrixLayout::RowMajor>(
            rhs,
            inner * params.n + col + 48,
            params.n);
        accumulator0 = coopMatMulAdd<float, false>(a, b0, accumulator0);
        accumulator1 = coopMatMulAdd<float, false>(a, b1, accumulator1);
        accumulator2 = coopMatMulAdd<float, false>(a, b2, accumulator2);
        accumulator3 = coopMatMulAdd<float, false>(a, b3, accumulator3);
    }

    accumulator0.Store<CoopMatMatrixLayout::RowMajor>(
        output,
        row * params.n + col,
        params.n);
    accumulator1.Store<CoopMatMatrixLayout::RowMajor>(
        output,
        row * params.n + col + 16,
        params.n);
    accumulator2.Store<CoopMatMatrixLayout::RowMajor>(
        output,
        row * params.n + col + 32,
        params.n);
    accumulator3.Store<CoopMatMatrixLayout::RowMajor>(
        output,
        row * params.n + col + 48,
        params.n);
}
