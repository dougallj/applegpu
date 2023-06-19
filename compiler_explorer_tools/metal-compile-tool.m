#import <Metal/Metal.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#if !__has_feature(objc_arc)
	#error Please compile with -fobjc-arc
#endif

static void printUsageAndExit(const char* argv0) {
	fprintf(stderr, "Usage: %s --output binary_archive.bin shader.metal\n", argv0);
	exit(EXIT_FAILURE);
}

static void dieIfError(NSError* err, const char* msg, ...) {
	if (err) {
		va_list va;
		va_start(va, msg);
		vfprintf(stderr, msg, va);
		va_end(va);
		fprintf(stderr, ": %s\n", [[err localizedDescription] UTF8String]);
		exit(EXIT_FAILURE);
	}
}

enum Options {
	OPTION_NO_FAST_MATH = 128,
	OPTION_TARGET_FORMAT,
};

MTLPixelFormat getFormat(const char* name) {
	static const struct {
		const char* name;
		MTLPixelFormat fmt;
	} formats[] = {
		{"rgba8unorm",  MTLPixelFormatRGBA8Unorm},
		{"rgba16float", MTLPixelFormatRGBA16Float},
		{"rgba16uint",  MTLPixelFormatRGBA16Uint},
		{"rgba32float", MTLPixelFormatRGBA32Float},
		{"rgba32uint",  MTLPixelFormatRGBA32Uint},
		{"r32uint",     MTLPixelFormatR32Uint},
	};
	for (size_t i = 0; i < sizeof(formats)/sizeof(*formats); i++) {
		if (0 == strcasecmp(name, formats[i].name))
			return formats[i].fmt;
	}
	fprintf(stderr, "Unrecognized texture format %s.  Supported formats:\n", name);
	for (size_t i = 0; i < sizeof(formats)/sizeof(*formats); i++) {
		fprintf(stderr, "  %s\n", formats[i].name);
	}
	exit(EXIT_FAILURE);
}

static const struct option longOpts[] = {
	{"output",        required_argument, NULL, 'o'},
	{"gpu",           required_argument, NULL, 'g'},
	{"no-fast-math",  no_argument,       NULL, OPTION_NO_FAST_MATH},
	{"target-format", required_argument, NULL, OPTION_TARGET_FORMAT},
	{"function",      required_argument, NULL, 'f'},
	{NULL, 0, NULL, 0}
};

int main(int argc, char* argv[]) {
	if (argc <= 1) {
		printUsageAndExit(argv[0]);
	}
	BOOL fastMath = YES;
	const char* output_name = NULL;
	const char* gpu = NULL;
	const char* target_function_name = NULL;
	MTLPixelFormat targetFmt = MTLPixelFormatRGBA8Unorm;
	int c;
	while ((c = getopt_long(argc, argv, "o:g:f:", longOpts, NULL)) > 0) {
		switch (c) {
			case 'o':
				output_name = optarg;
				break;
			case 'g':
				gpu = optarg;
				break;
			case OPTION_NO_FAST_MATH:
				fastMath = NO;
				break;
			case OPTION_TARGET_FORMAT:
				targetFmt = getFormat(optarg);
				break;
			case 'f':
				target_function_name = optarg;
				break;
			case '?':
				if (!optopt) {
				} else if (strchr("og", optopt)) {
					fprintf(stderr, "Option %c requires an argument!\n", optopt);
				} else {
					fprintf(stderr, "Unrecognized argument: %c\n", optopt);
				}
				printUsageAndExit(argv[0]);
		}
	}
	if (!output_name) {
		fprintf(stderr, "No output file");
		printUsageAndExit(argv[0]);
	}
	if (optind >= argc) {
		fprintf(stderr, "Need a file to compile!\n");
		printUsageAndExit(argv[0]);
	}
	id<MTLDevice> dev;
	if (gpu) {
		for (id<MTLDevice> check in MTLCopyAllDevices()) {
			if (0 == strncasecmp(gpu, [[check name] UTF8String], strlen(gpu))) {
				dev = check;
				break;
			}
		}
	} else {
		dev = MTLCreateSystemDefaultDevice();
		if (!dev) {
			NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
			if ([devs count] > 0) { dev = [devs objectAtIndex:0]; }
		}
	}
	if (!dev) {
		fprintf(stderr, "Failed to get GPU %s.  Available GPUs:\n", gpu);
		for (id<MTLDevice> gpu in MTLCopyAllDevices()) {
			fprintf(stderr, "  %s\n", [[gpu name] UTF8String]);
		}
		return EXIT_FAILURE;
	}
	NSData* shaderData;
	NSError* err;
	if (strcmp(argv[optind], "-") == 0) {
		shaderData = [[NSFileHandle fileHandleWithStandardInput] readDataToEndOfFile];
	} else {
		shaderData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[optind]] options:0 error:&err];
		dieIfError(err, "Failed to read %s", argv[optind]);
	}
	id<MTLLibrary> lib = [dev newLibraryWithData:dispatch_data_create([shaderData bytes], [shaderData length], dispatch_get_main_queue(), ^{}) error:nil];
	if (!lib) {
		NSString* shader = [[NSString alloc] initWithData:shaderData encoding:NSUTF8StringEncoding];
		MTLCompileOptions* options = [MTLCompileOptions new];
		[options setFastMathEnabled:fastMath];
		lib = [dev newLibraryWithSource:shader options:options error:&err];
	}
	dieIfError(err, "Failed to compile shaders");
	id<MTLFunction> vs, fs, cs;
	for (NSString* name in [lib functionNames]) {
		id<MTLFunction> fn = [lib newFunctionWithName:name];
		if (target_function_name && strcmp(target_function_name, [name UTF8String]) != 0) {
			continue;
		}

		if (!fn) {
			fprintf(stderr, "Failed to make function %s\n", [name UTF8String]);
			return EXIT_FAILURE;
		}
		switch ([fn functionType]) {
			case MTLFunctionTypeVertex:
				if (vs) {
					fprintf(stderr, "Only one vertex shader is allowed! (Got both %s and %s)\n", [[vs name] UTF8String], [name UTF8String]);
					return EXIT_FAILURE;
				}
				vs = fn;
				break;
			case MTLFunctionTypeFragment:
				if (fs) {
					fprintf(stderr, "Only one fragment shader is allowed! (Got both %s and %s)\n", [[fs name] UTF8String], [name UTF8String]);
					return EXIT_FAILURE;
				}
				fs = fn;
				break;
			case MTLFunctionTypeKernel:
				if (cs) {
					fprintf(stderr, "Only one compute shader is allowed! (Got both %s and %s)\n", [[cs name] UTF8String], [name UTF8String]);
					return EXIT_FAILURE;
				}
				cs = fn;
				break;
			default:
				fprintf(stderr, "Function %s is of unsupported type %lu\n", [name UTF8String], (unsigned long)[fn functionType]);
				return EXIT_FAILURE;
		}
	}
	if (!vs && !fs && !cs) {
		fprintf(stderr, "No shaders found\n");
		return EXIT_FAILURE;
	}
	id<MTLBinaryArchive> arc = [dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
	dieIfError(err, "Failed to make binary archive");
	if (vs || fs) {
		MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
		[desc setVertexFunction:vs];
		[desc setFragmentFunction:fs];
		[[desc colorAttachments][0] setPixelFormat:targetFmt];
		[arc addRenderPipelineFunctionsWithDescriptor:desc error:&err];
		if (err) {
			if (vs && fs) {
				dieIfError(err, "Failed to add render pipeline with %s and %s", [[vs name] UTF8String], [[fs name] UTF8String]);
			} else {
				dieIfError(err, "Failed to add render pipeline with %s", [[(vs ? vs : fs) name] UTF8String]);
			}
		}
	}
	if (cs) {
		MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
		[desc setComputeFunction:cs];
		[arc addComputePipelineFunctionsWithDescriptor:desc error:&err];
		dieIfError(err, "Failed to add render pipeline with %s", [[cs name] UTF8String]);
	}
	[arc serializeToURL:[NSURL fileURLWithPath:[NSString stringWithCString:output_name encoding:NSUTF8StringEncoding]]
	              error:&err];
	dieIfError(err, "Failed to write %s", output_name);
	return 0;
}
