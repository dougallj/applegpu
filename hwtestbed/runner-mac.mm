#include "runner.h"

#include <cstring>
#include <Metal/Metal.h>

#if !__has_feature(objc_arc)
	#error Please compile with -fobjc-arc
#endif

static MTLSize MTLSizeMakeFromArray(const uint32_t(&array)[3]) {
	return MTLSizeMake(array[0], array[1], array[2]);
}

static void reportError(std::string* err, const char* action, NSError* nserr = nullptr) {
	if (!err)
		return;
	if (nserr)
		*err = [[NSString stringWithFormat:@"%s: %@", action, [nserr localizedDescription]] UTF8String];
	else
		*err = action;
}

class MacRunner : public Runner {
	dispatch_data_t metallib;
	id<MTLDevice> dev;
	id<MTLCommandQueue> queue;
public:
	MacRunner(id<MTLDevice> dev, dispatch_data_t metallib): dev(dev), metallib(metallib) {
		queue = [dev newCommandQueue];
	}
	Buffer create_buffer(size_t size, std::string* err) override {
		@autoreleasepool {
			id<MTLBuffer> buf = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
			Buffer res;
			res.cpu_pointer = [buf contents];
			res.gpu_handle = (__bridge_retained void*)buf;
			res.size = size;
			return res;
		}
	}
	void destroy_buffer(const Buffer &buffer) override {
		(void)(__bridge_transfer id)buffer.gpu_handle;
	}
	Shader* create_compute_shader_from_file(const char* filename, std::string* error) override {
		return reinterpret_cast<Shader*>(strdup(filename));
	}
	Shader* create_compute_shader(void* data, size_t size, std::string* err) override {
		if (err)
			*err = "Compute shader from data is unsupported on macOS";
		return nullptr;
	}
	void destroy_shader(Shader *shader) override {
		free(shader);
	}

	id<MTLComputePipelineState> late_create_compute_shader(const char* path, const char** description, NSError** err) {
		// All our shaders look like the same shader from Metal's perspective, since we're hackily modiying binary archives to make them.
		// MTLLibraries cache their pipelines, so we need a new MTLLibrary for every run.
		MTLBinaryArchiveDescriptor* adesc = [MTLBinaryArchiveDescriptor new];
		MTLComputePipelineDescriptor* pdesc;
		id<MTLLibrary> lib = [dev newLibraryWithData:metallib error:err];
		if (!lib) { *description = "Failed to create metallib"; return nullptr; }
		id<MTLFunction> func = [lib newFunctionWithName:@"add_arrays"];
		if (!func) { *description = "Function missing from metallib"; return nullptr; }
		[adesc setUrl:[NSURL fileURLWithPath:[NSString stringWithCString:path encoding:NSUTF8StringEncoding]]];
		id<MTLBinaryArchive> archive = [dev newBinaryArchiveWithDescriptor:adesc error:err];
		if (!archive) { *description = "Failed to create binary archive"; return nullptr; }
		pdesc = [MTLComputePipelineDescriptor new];
		[pdesc setBinaryArchives:@[archive]];
		[pdesc setComputeFunction:func];
		id<MTLComputePipelineState> pipeline;
		pipeline = [dev newComputePipelineStateWithDescriptor:pdesc
		                                              options:MTLPipelineOptionFailOnBinaryArchiveMiss
		                                           reflection:nil
		                                                error:err];
		if (!pipeline) { *description = "Failed to create pipeline"; return nullptr; }
		return pipeline;
	}

	bool run_compute_shader(ComputeRun& run, std::string* error) override {
		@autoreleasepool {
			NSError* nserr = nullptr;
			const char* action = nullptr;
			if (!run.shader) {
				reportError(error, "No shader specified");
				return false;
			}
			id<MTLComputePipelineState> pipe = late_create_compute_shader(reinterpret_cast<const char*>(run.shader), &action, &nserr);
			if (!pipe) {
				reportError(error, action, nserr);
				return false;
			}
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:pipe];
			if (run.threadgroup_memory_size)
				[enc setThreadgroupMemoryLength:run.threadgroup_memory_size atIndex:0];
			for (size_t i = 0; i < run.num_buffers; i++) {
				if (run.buffers[i].gpu_handle) {
					[enc setBuffer:(__bridge id<MTLBuffer>)run.buffers[i].gpu_handle offset:0 atIndex:i];
				}
			}
			[enc dispatchThreadgroups:MTLSizeMakeFromArray(run.threadgroups_per_grid)
			    threadsPerThreadgroup:MTLSizeMakeFromArray(run.threads_per_threadgroup)];
			[enc endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			if (!cb || !enc) {
				reportError(error, "Failed to create command buffer and encoder", nullptr);
				return false;
			} else if ([cb status] == MTLCommandBufferStatusError) {
				reportError(error, "Command buffer failed", [cb error]);
				return false;
			}
			run.nanoseconds_elapsed = static_cast<uint64_t>(([cb GPUEndTime] - [cb GPUStartTime]) * 1000000000ull);
		}
		return true;
	}
};

Runner* Runner::make(std::string* err) {
	@autoreleasepool {
		NSURL* url = [NSURL fileURLWithPath:[[[NSProcessInfo processInfo] arguments] objectAtIndex:0]];
		url = [url URLByDeletingLastPathComponent];
		NSData* nsmetallib = [NSData dataWithContentsOfURL:[url URLByAppendingPathComponent:@"compute.metallib"]];
		if (!nsmetallib) {
			reportError(err, "Failed to get shader metallib");
			return nullptr;
		}
		dispatch_data_t metallib = dispatch_data_create([nsmetallib bytes], [nsmetallib length], nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

		MacRunner* res = nullptr;
		NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
		if ([devices count] < 1) {
			reportError(err, "No metal devices available");
		} else {
			res = new MacRunner([devices objectAtIndexedSubscript:0], metallib);
		}
		return res;
	}
}
