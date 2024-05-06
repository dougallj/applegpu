#include <metal_stdlib>
using namespace metal;

kernel void
add_arrays(
	uint3 index [[thread_position_in_grid]],
	uint3 tpg [[threads_per_grid]],
	uint sgidx [[thread_index_in_simdgroup]],
	device uint4 *output0 [[buffer(0)]],
	device uint4 *output1 [[buffer(1)]],
	device uint4 *output2 [[buffer(2)]],
	device uint4 *output3 [[buffer(3)]],
	device uint4 *output4 [[buffer(4)]],
	device uint4 *output5 [[buffer(5)]],
	device uint4 *output6 [[buffer(6)]],
	device uint4 *output7 [[buffer(7)]]
)
{
	uint4 q = output0[index.x];
	uint4 r = output1[index.x];
	int result = index.x;

	#define REP1    result += 1; result ^= 1;
	#define REP2    REP1 REP1
	#define REP3    REP2 REP2
	#define REP4    REP3 REP3
	#define REP5    REP4 REP4
	#define REP6    REP5 REP5
	#define REP7    REP6 REP6

	REP7

	uint64_t outputp = (uint64_t) output0;
	output0[index.x] = uint4(0, result, outputp, 10);
	output1[index.x] = uint4(1, result, outputp, 11);
	output2[index.x] = uint4(2, result, outputp, 12);
	output3[index.x] = uint4(3, result, outputp, 13);
	output4[index.x] = q;
	output5[index.x] = r;
	output6[index.x] = uint4(6, tpg.x, tpg.y, 16);
	output7[index.x] = uint4(7, tpg.z, outputp, sgidx);
}
