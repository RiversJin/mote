struct MatmulParams
{
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]]
ConstantBuffer<MatmulParams> params;

[[vk::binding(0, 0)]]
StructuredBuffer<float> lhs;

[[vk::binding(1, 0)]]
StructuredBuffer<float> rhs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> output;

static const uint TILE = 16;
groupshared float lhsTile[TILE][TILE];
groupshared float rhsTile[TILE][TILE];

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID)
{
    uint row = groupId.y * TILE + localId.y;
    uint col = groupId.x * TILE + localId.x;
    float sum = 0.0f;

    for (uint tile = 0; tile < (params.k + TILE - 1) / TILE; ++tile)
    {
        uint lhsCol = tile * TILE + localId.x;
        uint rhsRow = tile * TILE + localId.y;
        lhsTile[localId.y][localId.x] =
            row < params.m && lhsCol < params.k
                ? lhs[row * params.k + lhsCol]
                : 0.0f;
        rhsTile[localId.y][localId.x] =
            rhsRow < params.k && col < params.n
                ? rhs[rhsRow * params.n + col]
                : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint inner = 0; inner < TILE; ++inner)
        {
            sum += lhsTile[localId.y][inner] * rhsTile[inner][localId.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < params.m && col < params.n)
    {
        output[row * params.n + col] = sum;
    }
}
