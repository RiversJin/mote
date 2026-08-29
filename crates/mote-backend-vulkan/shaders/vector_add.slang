[[vk::binding(0, 0)]]
StructuredBuffer<float> lhs;

[[vk::binding(1, 0)]]
StructuredBuffer<float> rhs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> output;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    if (index < output.getCount())
    {
        output[index] = lhs[index] + rhs[index];
    }
}
