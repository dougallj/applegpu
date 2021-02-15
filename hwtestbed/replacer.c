/*
 * Copyright (c) 2021 Dougall Johnson
 * Copyright (c) 2020 Asahi Linux contributors
 * Copyright (c) 1998-2014 Apple Computer, Inc. All rights reserved.
 * Copyright (c) 2005 Apple Computer, Inc. All rights reserved.
 *
 * IOKit prototypes and stub implementations from upstream IOKitLib sources.
 * DYLD_INTERPOSE macro from dyld source code.  All other code in the file is
 * by Asahi Linux contributors.
 *
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <assert.h>

#include <mach/mach.h>
#include <IOKit/IOKitLib.h>
#include <libkern/OSCacheControl.h>

#include <stdbool.h>
#include <mach/mach.h>

enum agx_alloc_type {
	AGX_ALLOC_REGULAR = 0,
	AGX_ALLOC_MEMMAP = 1,
	AGX_ALLOC_CMDBUF = 2,
	AGX_NUM_ALLOC,
};

static const char *agx_alloc_types[AGX_NUM_ALLOC] = { "mem", "map", "cmd" };

struct agx_allocation {
	enum agx_alloc_type type;
	size_t size;

	/* Index unique only up to type */
	unsigned index;

	/* If CPU mapped, CPU address. NULL if not mapped */
	void *map;

	/* If type REGULAR, mapped GPU address */
	uint64_t gpu_va;
};

struct agx_map_header {
	uint32_t unk0; // cc c3 68 01
	uint32_t unk1; // 01 00 00 00
	uint32_t unk2; // 01 00 00 00
	uint32_t unk3; // 28 05 00 80
	uint32_t unk4; // cd c3 68 01
	uint32_t unk5; // 01 00 00 00 
	uint32_t unk6; // 00 00 00 00
	uint32_t unk7; // 80 07 00 00
	uint32_t nr_entries_1;
	uint32_t nr_entries_2;
	uint32_t unka; // 0b 00 00 00
	uint32_t padding[4];
} __attribute__((packed));

struct agx_map_entry {
	uint32_t unkAAA; // 20 00 00 00
	uint32_t unk2; // 00 00 00 00 
	uint32_t unk3; // 00 00 00 00
	uint32_t unk4; // 00 00 00 00
	uint32_t unk5; // 00 00 00 00
	uint32_t unk6; // 00 00 00 00 
	uint32_t unkBBB; // 01 00 00 00
	uint32_t unk8; // 00 00 00 00
	uint32_t unk9; // 00 00 00 00
	uint32_t unka; // ff ff 01 00 
	uint32_t index;
	uint32_t padding[5];
} __attribute__((packed));

enum agx_selector {
	AGX_SELECTOR_SET_API = 0x7,
	AGX_SELECTOR_ALLOCATE_MEM = 0xA,
	AGX_SELECTOR_CREATE_CMDBUF = 0xF,
	AGX_SELECTOR_SUBMIT_COMMAND_BUFFERS = 0x1E,
	AGX_SELECTOR_GET_VERSION = 0x23,
	AGX_NUM_SELECTORS = 0x30
};

static const char *selector_table[AGX_NUM_SELECTORS] = {
	"unk0",
	"unk1",
	"unk2",
	"unk3",
	"unk4",
	"unk5",
	"unk6",
	"SET_API",
	"unk8",
	"unk9",
	"ALLOCATE_MEM",
	"unkB",
	"unkC",
	"unkD",
	"unkE",
	"CREATE_CMDBUF",
	"unk10",
	"unk11",
	"unk12",
	"unk13",
	"unk14",
	"unk15",
	"unk16",
	"unk17",
	"unk18",
	"unk19",
	"unk1A",
	"unk1B",
	"unk1C",
	"unk1D",
	"SUBMIT_COMMAND_BUFFERS",
	"unk1F",
	"unk20",
	"unk21",
	"unk22",
	"GET_VERSION",
	"unk24",
	"unk25",
	"unk26",
	"unk27",
	"unk28",
	"unk29",
	"unk2A",
	"unk2B",
	"unk2C",
	"unk2D",
	"unk2E",
	"unk2F"
};

static inline const char *
wrap_selector_name(uint32_t selector)
{
	return (selector < AGX_NUM_SELECTORS) ? selector_table[selector] : "unk??";
}

struct agx_create_cmdbuf_resp {
	void *map;
	uint32_t size;
	uint32_t id;
} __attribute__((packed));


unsigned MAP_COUNT = 0;
#define MAX_MAPPINGS 4096
struct agx_allocation mappings[MAX_MAPPINGS];

static void
dump_mappings(void)
{
	for (unsigned i = 0; i < MAP_COUNT; ++i) {
		if (!mappings[i].map || !mappings[i].size)
			continue;

		char *name = NULL;
		assert(mappings[i].type < AGX_NUM_ALLOC);
		asprintf(&name, "%s_%llx_%u.bin", agx_alloc_types[mappings[i].type], mappings[i].gpu_va, mappings[i].index);
		FILE *fp = fopen(name, "wb");
		fwrite(mappings[i].map, 1, mappings[i].size, fp);
		fclose(fp);
	}
}

void agx_disassemble(void *code, size_t maxlen, FILE *fp);

/* Apple macro */

#define DYLD_INTERPOSE(_replacment,_replacee) \
	__attribute__((used)) static struct{ const void* replacment; const void* replacee; } _interpose_##_replacee \
	__attribute__ ((section ("__DATA,__interpose"))) = { (const void*)(unsigned long)&_replacment, (const void*)(unsigned long)&_replacee };

mach_port_t metal_connection = 0;

kern_return_t
wrap_IOConnectCallMethod(
	mach_port_t	 connection,		// In
	uint32_t	 selector,		// In
	const uint64_t	*input,			// In
	uint32_t	 inputCnt,		// In
	const void	*inputStruct,		// In
	size_t		 inputStructCnt,	// In
	uint64_t	*output,		// Out
	uint32_t	*outputCnt,		// In/Out
	void		*outputStruct,		// Out
	size_t		*outputStructCntP)	// In/Out
{
	/* Heuristic guess which connection is Metal, skip over I/O from everything else */
	bool bail = false;

	if (metal_connection == 0) {
		if (selector == AGX_SELECTOR_SET_API)
			metal_connection = connection;
		else
			bail = true;
	} else if (metal_connection != connection)
		bail = true;

	if (bail)
		return IOConnectCallMethod(connection, selector, input, inputCnt, inputStruct, inputStructCnt, output, outputCnt, outputStruct, outputStructCntP);

	/* Check the arguments make sense */
	assert((input != NULL) == (inputCnt != 0));
	assert((inputStruct != NULL) == (inputStructCnt != 0));
	assert((output != NULL) == (outputCnt != 0));
	assert((outputStruct != NULL) == (outputStructCntP != 0));

	/* Dump inputs */
	switch (selector) {
	case AGX_SELECTOR_SET_API:
		assert(input == NULL && output == NULL && outputStruct == NULL);
		assert(inputStruct != NULL && inputStructCnt == 16);
		assert(((uint8_t *) inputStruct)[15] == 0x0);
		break;

	case AGX_SELECTOR_SUBMIT_COMMAND_BUFFERS:
		assert(output == NULL && outputStruct == NULL);
		assert(inputStructCnt == 40);
		assert(inputCnt == 1);

		bool fs = false;

		for (unsigned i = 0; i < MAP_COUNT; ++i) {
			unsigned offset = fs ? 0x1380 : 0x13c0;

			if (!mappings[i].gpu_va && !(mappings[i].type)) {
				FILE *f = fopen("replace.bin", "rb");
				if (f) {
					size_t replaced = fread(mappings[i].map + offset, 1, 1024, f);
					fclose(f);

					sys_dcache_flush(mappings[i].map + offset, replaced);
				}
			}
		}
	}

	/* Invoke the real method */
	kern_return_t ret = IOConnectCallMethod(connection, selector, input, inputCnt, inputStruct, inputStructCnt, output, outputCnt, outputStruct, outputStructCntP);


	/* Track allocations for later analysis (dumping, disassembly, etc) */
	switch (selector) {
	case AGX_SELECTOR_CREATE_CMDBUF: {
		 assert(inputCnt == 2);
		assert((*outputStructCntP) == 0x10);
		uint64_t *inp = (uint64_t *) input;
		assert(inp[1] == 1 || inp[1] == 0);
		uint64_t *ptr = (uint64_t *) outputStruct;
		uint32_t *words = (uint32_t *) (ptr + 1);
		unsigned mapping = MAP_COUNT++;
		assert(mapping < MAX_MAPPINGS);
		mappings[mapping] = (struct agx_allocation) {
			.index = words[1],
			.map = (void *) *ptr,
			.size = words[0],
			.type = inp[1] ? AGX_ALLOC_CMDBUF : AGX_ALLOC_MEMMAP
		};
		break;
	}
	
	case AGX_SELECTOR_ALLOCATE_MEM: {
		assert((*outputStructCntP) == 0x50);
		uint64_t *iptrs = (uint64_t *) inputStruct;
		uint64_t *ptrs = (uint64_t *) outputStruct;
		uint64_t gpu_va = ptrs[0];
		uint64_t cpu = ptrs[1];
		uint64_t cpu_fixed_1 = iptrs[6];
		uint64_t cpu_fixed_2 = iptrs[7]; /* xxx what's the diff? */
		if (cpu && cpu_fixed_1)
			assert(cpu == cpu_fixed_1);
		else if (cpu == 0)
			cpu = cpu_fixed_1;
		uint64_t size = ptrs[4];
		unsigned mapping = MAP_COUNT++;
		//printf("allocate gpu va %llx, cpu %llx, 0x%llx bytes (%u)\n", gpu_va, cpu, size, mapping);
		assert(mapping < MAX_MAPPINGS);
		mappings[mapping] = (struct agx_allocation) {
			.type = AGX_ALLOC_REGULAR,
			.size = size,
			.index = iptrs[3] >> 32ull,
			.gpu_va = gpu_va,
			.map = (void *) cpu,
		};
	}

	default:
		break;
	}

	return ret;
}

kern_return_t
wrap_IOConnectCallAsyncMethod(
        mach_port_t      connection,            // In
        uint32_t         selector,              // In
        mach_port_t      wakePort,              // In
        uint64_t        *reference,             // In
        uint32_t         referenceCnt,          // In
        const uint64_t  *input,                 // In
        uint32_t         inputCnt,              // In
        const void      *inputStruct,           // In
        size_t           inputStructCnt,        // In
        uint64_t        *output,                // Out
        uint32_t        *outputCnt,             // In/Out
        void            *outputStruct,          // Out
        size_t          *outputStructCntP)      // In/Out
{
	//printf("async call method! connection %X, selector %s\n", connection, wrap_selector_name(selector));
	return IOConnectCallAsyncMethod(connection, selector, wakePort, reference, referenceCnt, input, inputCnt, inputStruct, inputStructCnt, output, outputCnt, outputStruct, outputStructCntP);
}

kern_return_t
wrap_IOConnectCallStructMethod(
        mach_port_t      connection,            // In
        uint32_t         selector,              // In
        const void      *inputStruct,           // In
        size_t           inputStructCnt,        // In
        void            *outputStruct,          // Out
        size_t          *outputStructCntP)       // In/Out
{
	return wrap_IOConnectCallMethod(connection, selector, NULL, 0, inputStruct, inputStructCnt, NULL, NULL, outputStruct, outputStructCntP);
}

kern_return_t
wrap_IOConnectCallAsyncStructMethod(
        mach_port_t      connection,            // In
        uint32_t         selector,              // In
        mach_port_t      wakePort,              // In
        uint64_t        *reference,             // In
        uint32_t         referenceCnt,          // In
        const void      *inputStruct,           // In
        size_t           inputStructCnt,        // In
        void            *outputStruct,          // Out
        size_t          *outputStructCnt)       // In/Out
{
    return wrap_IOConnectCallAsyncMethod(connection,   selector, wakePort,
                                    reference,    referenceCnt,
                                    NULL,         0,
                                    inputStruct,  inputStructCnt,
                                    NULL,         NULL,
                                    outputStruct, outputStructCnt);
}

kern_return_t
wrap_IOConnectCallScalarMethod(
        mach_port_t      connection,            // In
        uint32_t         selector,              // In
        const uint64_t  *input,                 // In
        uint32_t         inputCnt,              // In
        uint64_t        *output,                // Out
        uint32_t        *outputCnt)             // In/Out
{
    return wrap_IOConnectCallMethod(connection, selector,
                               input,      inputCnt,
                               NULL,       0,
                               output,     outputCnt,
                               NULL,       NULL);
}

kern_return_t
wrap_IOConnectCallAsyncScalarMethod(
        mach_port_t      connection,            // In
        uint32_t         selector,              // In
        mach_port_t      wakePort,              // In
        uint64_t        *reference,             // In
        uint32_t         referenceCnt,          // In
        const uint64_t  *input,                 // In
        uint32_t         inputCnt,              // In
        uint64_t        *output,                // Out
        uint32_t        *outputCnt)             // In/Out
{
    return wrap_IOConnectCallAsyncMethod(connection, selector, wakePort,
                                    reference,  referenceCnt,
                                    input,      inputCnt,
                                    NULL,       0,
                                    output,    outputCnt,
                                    NULL,      NULL);
}

DYLD_INTERPOSE(wrap_IOConnectCallMethod, IOConnectCallMethod);
DYLD_INTERPOSE(wrap_IOConnectCallAsyncMethod, IOConnectCallAsyncMethod);
DYLD_INTERPOSE(wrap_IOConnectCallStructMethod, IOConnectCallStructMethod);
DYLD_INTERPOSE(wrap_IOConnectCallAsyncStructMethod, IOConnectCallAsyncStructMethod);
DYLD_INTERPOSE(wrap_IOConnectCallScalarMethod, IOConnectCallScalarMethod);
DYLD_INTERPOSE(wrap_IOConnectCallAsyncScalarMethod, IOConnectCallAsyncScalarMethod);
