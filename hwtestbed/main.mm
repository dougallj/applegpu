#include <stdio.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/time.h>

#import <MetalKit/MetalKit.h>

int main(int argc, char **argv) {
	struct timeval tval_before, tval_after, tval_result;

	gettimeofday(&tval_before, NULL);

	size_t count = argc >= 2 ? atoi(argv[1]) : 32;
	size_t outputBufferSize = argc >= 3 ? atoi(argv[2]) : (32 * sizeof(uint32_t)) * count;

	char *buffer0_filename = argc >= 4 ? argv[3] : NULL;
	char *buffer1_filename = argc >= 5 ? argv[4] : NULL;

	long long expected_sum = 0;

	gettimeofday(&tval_after, NULL);
	timersub(&tval_after, &tval_before, &tval_result);
	gettimeofday(&tval_before, NULL);

	NSError *error = nil;

	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (!device)
		return 1;

	error = nil;
	id<MTLLibrary> library = [device newLibraryWithFile:@"compute.metallib" error:&error];
	if (!library)
		return 1;

	id<MTLFunction> kernelFunction = [library newFunctionWithName:@"add_arrays"];
	if (!kernelFunction)
		return 1;

	error = nil;
	id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
	if (!computePipelineState)
		return 1;

	id<MTLCommandQueue> commandQueue = [device newCommandQueue];
	if (!commandQueue)
		return 1;

#define NUM_BUFFERS 8
	id<MTLBuffer> outputBuffers[NUM_BUFFERS];
	for (int i = 0; i < NUM_BUFFERS; i++) {
		outputBuffers[i] = [device newBufferWithLength:outputBufferSize options:MTLResourceStorageModeShared];
		memset(outputBuffers[i].contents, 0, outputBufferSize);
	}

	if (buffer0_filename) {
		FILE *f = fopen(buffer0_filename, "rb");
		if (f) {
			size_t replaced = fread(outputBuffers[0].contents, 1, outputBufferSize, f);
			fclose(f);
		} else {
			fprintf(stderr, "failed to open %s\n", buffer0_filename);
		}
	}

	if (buffer1_filename) {
		FILE *f = fopen(buffer1_filename, "rb");
		if (f) {
			size_t replaced = fread(outputBuffers[1].contents, 1, outputBufferSize, f);
			fclose(f);
		} else {
			fprintf(stderr, "failed to open %s\n", buffer0_filename);
		}
	}

	long long sum;

	id<MTLCommandBuffer> commandBuffer = commandQueue.commandBuffer;
	id<MTLComputeCommandEncoder> encoder = commandBuffer.computeCommandEncoder;
	
	[encoder setComputePipelineState:computePipelineState];


	for (int i = 0; i < NUM_BUFFERS; i++) {
		[encoder setBuffer:outputBuffers[i] offset:0 atIndex:i];
	}

	[encoder setThreadgroupMemoryLength:0x100 atIndex:0];

	MTLSize threadgroupsPerGrid = MTLSizeMake(count, 1, 1);

	NSUInteger threads = computePipelineState.maxTotalThreadsPerThreadgroup;
	if (count < threads)
		threads = count;

	MTLSize threadsPerThreadgroup = MTLSizeMake(threads, 1, 1);
	
	[encoder dispatchThreads:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
	[encoder endEncoding];

	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];

	printf("[\n");
	for (int r = 0; r < 31; r++) {
		printf("[");
		assert(r/4 < NUM_BUFFERS);
		for (int simd = 0; simd < count; simd++) {
			uint32_t *outputs = (uint32_t*)outputBuffers[r/4].contents;
			printf("0x%x,", outputs[(r & 3) + (simd * 4)]);
		}
		printf("],\n");
	}
	printf("]\n");

	FILE *f = fopen("buffer0.bin", "wb");
	if (f) {
		size_t replaced = fwrite(outputBuffers[0].contents, 1, outputBufferSize, f);
		fclose(f);
	}

	return 0;
}

