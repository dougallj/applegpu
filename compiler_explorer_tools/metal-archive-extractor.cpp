#include <cstdlib>
#include <cstdio>
#include <getopt.h>
#include <mach-o/dyld.h>
#include <mach-o/fat.h>
#include <mach-o/nlist.h>
#include <vector>

#define FAT_MAGIC_METAL 0xcbfebabe
#define FAT_CIGAM_METAL 0xbebafecb

struct Buffer {
	void* data;
	size_t size;
	bool inBounds(void* ptr) { return static_cast<char*>(ptr) - static_cast<char*>(data) < size; }
	bool endInBounds(void* ptr) { return static_cast<char*>(ptr) - static_cast<char*>(data) <= size; }
	template <typename T = void>
	T* offsetPtr(size_t offset) const { return reinterpret_cast<T*>(static_cast<char*>(data) + offset);}
	operator bool() { return data; }
};

enum class GPUMachineType {
	AppleGPU = 0x1000013,
	AMDGPU   = 0x1000014,
	IntelGPU = 0x1000015,
	AIR64    = 0x1000017,
};

static const char* getMachineTypeName(Buffer buffer) {
	mach_header_64* mach_header = static_cast<mach_header_64*>(buffer.data);
	switch (static_cast<GPUMachineType>(mach_header->cputype)) {
		case GPUMachineType::AppleGPU: return "Apple GPU";
		case GPUMachineType::AMDGPU:   return "AMD GPU";
		case GPUMachineType::IntelGPU: return "Intel GPU";
		default: return "Unknown Target";
	}
}

static bool isGPUType(uint32_t machineType) {
	switch (static_cast<GPUMachineType>(machineType)) {
		case GPUMachineType::AppleGPU:
		case GPUMachineType::AMDGPU:
		case GPUMachineType::IntelGPU:
			return true;
		default:
			return false;
	}
}

static uint32_t bswapIfNecessary(bool necessary, uint32_t value) {
	return necessary ? OSSwapInt32(value) : value;
}

static Buffer findGPU(Buffer buffer) {
	fat_header* header = static_cast<fat_header*>(buffer.data);
	mach_header_64* mach_header = static_cast<mach_header_64*>(buffer.data);
	if (!buffer.inBounds(header + 1) || !buffer.inBounds(mach_header + 1)) {
		fprintf(stderr, "File too small for fat header!\n");
		return {};
	}
	bool swap;
	if (header->magic == FAT_MAGIC_METAL) {
		swap = false;
	} else if (header->magic == FAT_CIGAM_METAL) {
		swap = true;
	} else if (header->magic == MH_MAGIC_64) {
		if (isGPUType(static_cast<mach_header_64*>(buffer.data)->cputype)) {
			return buffer;
		} else {
			fprintf(stderr, "File isn't a GPU binary\n");
			return {};
		}
	} else if (header->magic == MH_CIGAM_64) {
		if (isGPUType(OSSwapInt32(static_cast<mach_header_64*>(buffer.data)->cputype))) {
			return buffer;
		} else {
			fprintf(stderr, "File isn't a GPU binary\n");
			return {};
		}
	} else {
		fprintf(stderr, "Bad Header Magic %08x\n", header->magic);
		return {};
	}
	uint32_t narch = bswapIfNecessary(swap, header->nfat_arch);
	fat_arch* archs = reinterpret_cast<fat_arch*>(header + 1);
	if (!buffer.inBounds(archs + narch)) {
		fprintf(stderr, "File too small for header!\n");
		return {};
	}
	for (uint32_t i = 0; i < narch; i++) {
		if (!isGPUType(bswapIfNecessary(swap, archs[i].cputype))) {
			continue;
		}
		uint32_t offset = bswapIfNecessary(swap, archs[i].offset);
		uint32_t subsize = bswapIfNecessary(swap, archs[i].size);
		if (buffer.size < offset + subsize) {
			fprintf(stderr, "Fat header referenced out of bounds area!\n");
			return {};
		}
		return { static_cast<char*>(buffer.data) + offset, subsize };
	}
	fprintf(stderr, "Couldn't find any GPU binaries in fat header!\n");
	return {};
}

static bool isDylib(Buffer buffer) {
	mach_header_64* mach_header = static_cast<mach_header_64*>(buffer.data);
	return buffer.inBounds(mach_header + 1) && mach_header->filetype == MH_GPU_DYLIB;
}

static void* findCommand(Buffer buffer, void* ctx, void* (*isMyCommand)(void* ctx, load_command*)) {
	mach_header_64* header = static_cast<mach_header_64*>(buffer.data);
	if (!buffer.inBounds(header + 1)) {
		fprintf(stderr, "File too small for mach-o header!\n");
		return nullptr;
	}
	if (header->magic == MH_CIGAM_64) {
		fprintf(stderr, "Non-native-endian mach-o files currently unsupported\n");
		return nullptr;
	} else if (header->magic != MH_MAGIC_64) {
		fprintf(stderr, "Bad mach-o magic\n");
		return nullptr;
	}
	load_command* lc = reinterpret_cast<load_command*>(header + 1);
	for (uint32_t i = 0; i < header->ncmds; i++, lc = reinterpret_cast<load_command*>(reinterpret_cast<char*>(lc) + lc->cmdsize)) {
		if (!buffer.inBounds(lc + 1) || !buffer.inBounds(reinterpret_cast<char*>(lc) + lc->cmdsize)) {
			fprintf(stderr, "Load commands went out of bounds!\n");
			return {};
		}
		if (void* res = isMyCommand(ctx, lc)) {
			return res;
		}
	}
	return nullptr;
}

struct FindSectionCtx {
	Buffer buffer;
	const char* segment;
	const char* section;
};

static void* findSectionHelper(void* vctx, load_command* cmd) {
	FindSectionCtx* ctx = static_cast<FindSectionCtx*>(vctx);
	if (cmd->cmdsize < sizeof(segment_command_64) || cmd->cmd != LC_SEGMENT_64) { return nullptr; }
	segment_command_64* segment = reinterpret_cast<segment_command_64*>(cmd);
	if (segment->cmdsize < segment->nsects * sizeof(section_64) + sizeof(segment_command_64)) {
		fprintf(stderr, "Segments %.16s is too small for its section list\n", segment->segname);
		return nullptr;
	}
	section_64* sections = reinterpret_cast<section_64*>(segment + 1);
	for (uint32_t i = 0; i < segment->nsects; i++) {
		if (0 != strncmp(ctx->segment, sections[i].segname,  sizeof(sections[i].segname ))) { continue; }
		if (0 != strncmp(ctx->section, sections[i].sectname, sizeof(sections[i].sectname))) { continue; }
		if (!ctx->buffer.inBounds(ctx->buffer.offsetPtr(sections[i].offset + sections[i].size))) {
			fprintf(stderr, "Section %s,%s is out of bounds!\n", ctx->segment, ctx->section);
			return nullptr;
		}
		return &sections[i];
	}
	return nullptr;
}

static section_64* findSection(Buffer buffer, const char* segment, const char* section) {
	FindSectionCtx ctx { buffer, segment, section };
	return static_cast<section_64*>(findCommand(buffer, &ctx, findSectionHelper));
}

static void* isSymtab(void* ctx, load_command* cmd) {
	if (cmd->cmdsize >= sizeof(symtab_command) && cmd->cmd == LC_SYMTAB) {
		return cmd;
	} else {
		return nullptr;
	}
}

static bool findSymtab(Buffer buffer, const char* segment, const char* sectionName,
                       section_64** section, symtab_command** cmd, char** strings, nlist_64** symbols)
{
	*section = findSection(buffer, segment, sectionName);
	if (!*section) { return false; }
	*cmd = reinterpret_cast<symtab_command*>(findCommand(buffer, nullptr, isSymtab));
	*strings = buffer.offsetPtr<char>((*cmd)->stroff);
	if (!buffer.endInBounds(*strings + (*cmd)->strsize)) {
		fprintf(stderr, "String table is out of bounds!\n");
		return false;
	}
	*symbols = buffer.offsetPtr<nlist_64>((*cmd)->symoff);
	if (!buffer.inBounds(*symbols + (*cmd)->nsyms)) {
		fprintf(stderr, "Symbol table is out of bounds!\n");
		return false;
	}
	return true;
}

static Buffer findSymbol(Buffer buffer, const char* segment, const char* sectionName, const char* symbol) {
	section_64* section;
	symtab_command* cmd;
	char* strings;
	nlist_64* symbols;
	if (!findSymtab(buffer, segment, sectionName, &section, &cmd, &strings, &symbols)) {
		return {};
	}
	uint64_t begin = 0;
	uint64_t end = 0;
	for (uint32_t i = 0; i < cmd->nsyms; i++) {
		uint32_t stroff = symbols[i].n_un.n_strx;
		if (stroff >= cmd->strsize) {
			fprintf(stderr, "Symbol %d's name is out of bounds\n", i);
			continue;
		}
		if (0 == strncmp(strings + stroff, symbol, cmd->strsize - stroff)) {
			if (symbols[i].n_value - section->addr > section->size) {
				fprintf(stderr, "Symbol %s is not in %s,%s\n", symbol, segment, sectionName);
				return {};
			} else {
				begin = symbols[i].n_value;
				end = section->size + section->addr;
				break;
			}
		}
	}
	if (begin == end) { return {}; }
	// Assume the symbol ends with the next one
	for (uint32_t i = 0; i < cmd->nsyms; i++) {
		if (symbols[i].n_value > begin && symbols[i].n_value < end) {
			end = symbols[i].n_value;
		}
	}
	return { buffer.offsetPtr(begin - section->addr + section->offset), end - begin };
}

struct GPUFunction {
	const char* name;
	uint64_t offset;
};

static std::vector<GPUFunction> findAllSymbols(Buffer buffer, const char* segment, const char* sectionName) {
	section_64* section;
	symtab_command* cmd;
	char* strings;
	nlist_64* symbols;
	if (!findSymtab(buffer, segment, sectionName, &section, &cmd, &strings, &symbols)) {
		return {};
	}
	std::vector<GPUFunction> out;
	for (uint32_t i = 0; i < cmd->nsyms; i++) {
		uint32_t stroff = symbols[i].n_un.n_strx;
		uint32_t limit = cmd->strsize - stroff;
		if (stroff >= cmd->strsize || strnlen(strings + stroff, limit) == limit) {
			fprintf(stderr, "Symbol %d's name is out of bounds\n", i);
			continue;
		}
		if (symbols[i].n_value - section->addr < section->size) {
			out.push_back({ strings + stroff, symbols[i].n_value - section->addr });
		}
	}
	return out;
}

static FILE* openOrDie(const char* filename, const char* mode) {
	FILE* output = fopen(filename, mode);
	if (!output) {
		fprintf(stderr, "Failed to open %s: %s\n", filename, strerror(errno));
		exit(EXIT_FAILURE);
	}
	return output;
}

static void printUsageAndExit(const char* argv0) {
	fprintf(stderr, "Usage: %s [--extract-vertex vertex.bin] [--extract-fragment fragment.bin] [--extract-compute compute.bin] binary_archive.bin\n"
	                "Usage: %s [--list-shaders] [--extract-prolog-shader prolog.bin] [--extract-main-shader main.bin] [--extract-named-shader shader_name --output named.bin] agx_shader_archive.bin\n",
	        argv0, argv0);
	exit(EXIT_FAILURE);
}

static void dumpSection(Buffer buffer, const char* filename, const char* segmentName, const char* sectionName) {
	FILE* file = strcmp(filename, "-") == 0 ? stdout : openOrDie(filename, "wb");
	if (section_64* section = findSection(buffer, segmentName, sectionName)) {
		if (!fwrite(buffer.offsetPtr(section->offset), section->size, 1, file)) {
			fprintf(stderr, "Failed to write to %s: %s\n", filename, strerror(ferror(file)));
			exit(EXIT_FAILURE);
		}
	}
	if (file != stdout) { fclose(file); }
}

static void dumpSymbol(Buffer buffer, const char* filename, const char* segmentName, const char* sectionName, const char* symbolName) {
	FILE* file = strcmp(filename, "-") == 0 ? stdout : openOrDie(filename, "wb");
	if (Buffer symbol = findSymbol(buffer, segmentName, sectionName, symbolName)) {
		if (!fwrite(symbol.data, symbol.size, 1, file)) {
			fprintf(stderr, "Failed to write to %s: %s\n", filename, strerror(ferror(file)));
			exit(EXIT_FAILURE);
		}
	}
	if (file != stdout) { fclose(file); }
}

enum Options {
	OPTION_NAMED_SHADER = 128,
	OPTION_OUTPUT,
};

static constexpr option longOpts[] = {
	{"extract-vertex",         required_argument, nullptr, 'v'},
	{"extract-fragment",       required_argument, nullptr, 'f'},
	{"extract-compute",        required_argument, nullptr, 'c'},
	{"extract-dylib",          required_argument, nullptr, 'd'},
	{"extract-prolog-shader",  required_argument, nullptr, 'p'},
	{"extract-main-shader",    required_argument, nullptr, 'm'},
	{"list-shaders",           no_argument,       nullptr, 'l'},
	{"extract-named-shader",   required_argument, nullptr, OPTION_NAMED_SHADER},
	{"output",                 required_argument, nullptr, OPTION_OUTPUT},
	{nullptr, 0, nullptr, 0}
};

int main(int argc, char* argv[]) {
	if (argc <= 1) {
		printUsageAndExit(argv[0]);
	}
	const char* vertex_out = nullptr;
	const char* fragment_out = nullptr;
	const char* compute_out = nullptr;
	const char* dylib_out = nullptr;
	const char* prolog_out = nullptr;
	const char* main_out = nullptr;
	const char* extract_name = nullptr;
	const char* other_out = nullptr;
	bool cmd_list = false;
	int c;
	while ((c = getopt_long(argc, argv, "v:f:c:d:p:m:l", longOpts, nullptr)) > 0) {
		switch (c) {
			case 'v':
				vertex_out = optarg;
				break;
			case 'f':
				fragment_out = optarg;
				break;
			case 'c':
				compute_out = optarg;
				break;
			case 'd':
				dylib_out = optarg;
				break;
			case 'p':
				prolog_out = optarg;
				break;
			case 'm':
				main_out = optarg;
				break;
			case 'l':
				cmd_list = true;
				break;
			case OPTION_NAMED_SHADER:
				extract_name = optarg;
				break;
			case OPTION_OUTPUT:
				other_out = optarg;
				break;
			case '?':
				if (!optopt) {
				} else if (strchr("vfcpm", optopt)) {
					fprintf(stderr, "Option %c requires an argument!\n", optopt);
				} else {
					fprintf(stderr, "Unrecognized argument: %c\n", optopt);
				}
				printUsageAndExit(argv[0]);
		}
	}
	bool extract_section = vertex_out || fragment_out || compute_out || dylib_out;
	bool extract_subsection = prolog_out || main_out || extract_name;
	if (extract_section && extract_subsection) {
		fprintf(stderr, "Can't combine archive extract options and agx extract options!\n");
		printUsageAndExit(argv[0]);
	}
	if (extract_name && !other_out) {
		fprintf(stderr, "--extract-named-shader requires an output declared with --output!\n");
		printUsageAndExit(argv[0]);
	}
	if (optind >= argc) {
		fprintf(stderr, "Need a file to extract!\n");
		printUsageAndExit(argv[0]);
	}

	FILE* input = strcmp(argv[optind], "-") == 0 ? stdin : openOrDie(argv[optind], "rb");
	Buffer buffer { malloc(4096), 0 };
	while (true) {
		size_t amt = buffer.size < 4096 ? 4096 : buffer.size;
		size_t read = fread(buffer.offsetPtr(buffer.size), 1, amt, input);
		buffer.size += read;
		if (read < amt) { break; }
		buffer.data = realloc(buffer.data, buffer.size * 2);
	}
	if (input != stdin) { fclose(input); }

	Buffer gpu = findGPU(buffer);
	if (!gpu) { return EXIT_FAILURE; }
	if (cmd_list) {
		for (GPUFunction symbol : findAllSymbols(gpu, "__TEXT", "__text"))
			printf("0x%llx %s\n", symbol.offset, symbol.name);
	} else if (extract_section) {
		if (vertex_out)   { dumpSection(gpu, vertex_out,   "__TEXT", "__vertex"  ); }
		if (fragment_out) { dumpSection(gpu, fragment_out, "__TEXT", "__fragment"); }
		if (compute_out)  { dumpSection(gpu, compute_out,  "__TEXT", "__compute" ); }
		// Contained program also uses `__TEXT,__text` section, so make extra sure this is a dylib before trying to extract
		if (dylib_out && isDylib(gpu)) {
			dumpSection(gpu, dylib_out, "__TEXT", "__text" );
		}
	} else if (extract_subsection) {
		if (prolog_out)   { dumpSymbol(gpu, prolog_out, "__TEXT", "__text", "_agc.main.constant_program"); }
		if (main_out)     { dumpSymbol(gpu, main_out,   "__TEXT", "__text", "_agc.main"); }
		if (extract_name) { dumpSymbol(gpu, other_out,  "__TEXT", "__text", extract_name); }
	} else {
		printf("Binary for %s\n", getMachineTypeName(gpu));
		if (findSection(gpu, "__TEXT", "__vertex")) {
			printf("Contains a vertex shader.\n");
		}
		if (findSection(gpu, "__TEXT", "__fragment")) {
			printf("Contains a fragment shader.\n");
		}
		if (findSection(gpu, "__TEXT", "__compute")) {
			printf("Contains a compute shader.\n");
		}
		if (findSymbol(gpu, "__TEXT", "__text", "_agc.main.constant_program")) {
			printf("Contains an AGX prolog shader\n");
		}
		if (findSymbol(gpu, "__TEXT", "__text", "_agc.main")) {
			printf("Contains an AGX shader\n");
		}
	}

	free(buffer.data);
	return 0;
}
