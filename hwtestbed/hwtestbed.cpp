#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <mutex>
#include <thread>

#include "runner.h"
#include "protocol.h"

struct BufferInfo {
	uint32_t index;
	uint32_t size;
};

struct RequestExecuteCompute {
	uint32_t threadgroups_per_grid[3];
	uint32_t threads_per_threadgroup[3];
};

struct RequestHandler {
	std::string err;
	void* tmp = nullptr;
	size_t tmpsize = 0;

	constexpr static uint32_t MAX_BUFFERS = 16;
	uint32_t tgsm = 0;
	Runner::Shader* shader = nullptr;
	bool wantsResult[MAX_BUFFERS] = {};
	Runner::Buffer buffers[MAX_BUFFERS] = {};
	Runner* runner = nullptr;

	RequestHandler(Runner* runner): runner(runner) {}

	~RequestHandler() {
		reset();
		free(tmp);
	}

	void freeBuffer(Runner::Buffer& buffer) {
		if (buffer.gpu_handle) {
			runner->destroy_buffer(buffer);
			buffer = {};
		}
	}

	void freeShader() {
		if (shader) {
			runner->destroy_shader(shader);
			shader = nullptr;
		}
	}

	void reset() {
		for (auto& buffer : buffers) {
			freeBuffer(buffer);
		}
		freeShader();
		memset(wantsResult, 0, sizeof(wantsResult));
		err.clear();
		tgsm = 0;
	}

	/// Only allow error reporting if there isn't already an errror
	std::string* newErr() {
		if (err.empty())
			return &err;
		return nullptr;
	}

	void setNewErr(const char* msg) {
		if (err.empty())
			err = msg;
	}

	// For some reason C stdio doesn't handle EINTR for us...
	size_t fread(void* ptr, size_t size, size_t nitems, FILE* stream) {
		size_t res;
		while ((res = ::fread(ptr, size, nitems, stream)) == 0 && errno == EINTR)
			;
		// fprintf(stderr, "Read %zd * %zd bytes: ", size, res);
		// for (size_t i = 0; i < size * res; i++)
		// 	fprintf(stderr, "%02x ", reinterpret_cast<unsigned char*>(ptr)[i]);
		// fprintf(stderr, "\n");
		return res;
	}

	size_t fwrite(const void* ptr, size_t size, size_t nitems, FILE* stream) {
		size_t res;
		while ((res = ::fwrite(ptr, size, nitems, stream)) == 0 && errno == EINTR)
			;
		// fprintf(stderr, "Wrote %zd * %zd bytes: ", size, res);
		// for (size_t i = 0; i < size * res; i++)
		// 	fprintf(stderr, "%02x ", reinterpret_cast<const unsigned char*>(ptr)[i]);
		// fprintf(stderr, "\n");
		return res;
	}

	bool readData(uint32_t* size, FILE* file) {
		if (fread(size, sizeof(*size), 1, file) < 1)
			return false;
		ensureSize(*size);
		if (fread(tmp, *size, 1, file) < 1)
			return false;
		return true;
	}

	void ensureSize(size_t needed) {
		if (tmpsize < needed) {
			tmp = realloc(tmp, needed);
			tmpsize = needed;
		}
	}

	void dummyRead(uint32_t size, FILE* file) {
		ensureSize(size);
		fread(tmp, size, 1, file);
	}

	void sendOp(HWTestBedResponse op, FILE* file) {
		uint8_t u8op = static_cast<uint8_t>(op);
		fwrite(&u8op, sizeof(u8op), 1, file);
	}

	void run(FILE* in, FILE* out) {
		while (true) {
			uint8_t op;
			if (fread(&op, sizeof(op), 1, stdin) < 1)
				break;
			switch (static_cast<HWTestBedRequest>(op)) {
			case HW_TEST_BED_REQUEST_BEGIN_COMPUTE:
				reset();
				break;
			case HW_TEST_BED_REQUEST_SET_COMPUTE_SHADER_DATA: {
				freeShader();
				uint32_t size;
				if (!readData(&size, in))
					return;
				if (!(shader = runner->create_compute_shader(tmp, size, newErr())))
					setNewErr("Failed to create compute shader");
				break;
			}
			case HW_TEST_BED_REQUEST_SET_COMPUTE_SHADER_FILE: {
				freeShader();
				uint32_t size;
				if (!readData(&size, in))
					return;
				ensureSize(size + 1);
				static_cast<char*>(tmp)[size] = '\0';
				if (!(shader = runner->create_compute_shader_from_file(static_cast<const char*>(tmp), newErr())))
					setNewErr("Failed to create compute shader");
				break;
			}
			case HW_TEST_BED_REQUEST_SET_BUFFER_DATA:
			case HW_TEST_BED_REQUEST_SET_BUFFER_RESULT: {
				BufferInfo info;
				if (fread(&info, sizeof(info), 1, in) < 1)
					return;
				if (info.index >= MAX_BUFFERS) {
					setNewErr("Buffer index too large");
					if (op == HW_TEST_BED_REQUEST_SET_BUFFER_DATA)
						dummyRead(info.size, in);
					break;
				}
				Runner::Buffer& buffer = buffers[info.index];
				Runner::Buffer oldBuffer = buffer;
				if (op == HW_TEST_BED_REQUEST_SET_BUFFER_RESULT) {
					wantsResult[info.index] = true;
				}
				if (buffer.size < info.size)
					buffer = {};
				if (!buffer.cpu_pointer) {
					buffer = runner->create_buffer(info.size, newErr());
					if (buffer.cpu_pointer && op == HW_TEST_BED_REQUEST_SET_BUFFER_RESULT) {
						if (oldBuffer.cpu_pointer)
							memcpy(buffer.cpu_pointer, oldBuffer.cpu_pointer, oldBuffer.size);
						memset(static_cast<char*>(buffer.cpu_pointer) + oldBuffer.size, 0, info.size - oldBuffer.size);
					}
					freeBuffer(oldBuffer);
				}
				if (buffer.cpu_pointer) {
					if (op == HW_TEST_BED_REQUEST_SET_BUFFER_DATA) {
						if (fread(buffer.cpu_pointer, info.size, 1, in) < 1)
							return;
					}
				} else {
					setNewErr("Failed to create buffer");
					if (op == HW_TEST_BED_REQUEST_SET_BUFFER_DATA)
						dummyRead(info.size, in);
				}
				break;
			}

			case HW_TEST_BED_REQUEST_SET_COMPUTE_TGSM:
				if (fread(&tgsm, sizeof(tgsm), 1, in) < 1)
					return;
				break;

			case HW_TEST_BED_REQUEST_EXECUTE_COMPUTE: {
				RequestExecuteCompute req;
				if (fread(&req, sizeof(req), 1, in) < 1)
					return;
				Runner::ComputeRun run;
				run.buffers = buffers;
				run.num_buffers = MAX_BUFFERS;
				run.shader = shader;
				run.threadgroup_memory_size = tgsm;
				memcpy(run.threads_per_threadgroup, req.threads_per_threadgroup, sizeof(req.threads_per_threadgroup));
				memcpy(run.threadgroups_per_grid, req.threadgroups_per_grid, sizeof(req.threadgroups_per_grid));
				if (err.empty())
					if (!runner->run_compute_shader(run, &err))
						setNewErr("Failed to run shader");
				sendOp(HW_TEST_BED_RESPONSE_BEGIN, out);
				sendOp(HW_TEST_BED_RESPONSE_TIME, out);
				fwrite(&run.nanoseconds_elapsed, sizeof(run.nanoseconds_elapsed), 1, out);
				if (err.empty()) {
					for (uint32_t i = 0; i < MAX_BUFFERS; i++) {
						if (!wantsResult[i])
							continue;
						sendOp(HW_TEST_BED_RESPONSE_BUFFER_DATA, out);
						BufferInfo info;
						info.index = i;
						info.size = static_cast<uint32_t>(buffers[i].size);
						fwrite(&info, sizeof(info), 1, out);
						fwrite(buffers[i].cpu_pointer, info.size, 1, out);
					}
				} else {
					sendOp(HW_TEST_BED_RESPONSE_ERROR, out);
					uint32_t size = static_cast<uint32_t>(err.size() + 1);
					fwrite(&size, sizeof(size), 1, out);
					fwrite(err.c_str(), size, 1, out);
				}
				sendOp(HW_TEST_BED_RESPONSE_END, out);
				fflush(out);
				reset();
				break;
			}

			default:
				fprintf(stderr, "Unexpected opcode %d\n", op);
				return;
			}
		}
	}
};

int main(int argc, const char* argv[]) {
	std::string err;

	Runner* runner = Runner::make(&err);
	if (!runner) {
		fprintf(stderr, "Failed to create runner: %s\n", err.c_str());
		return EXIT_FAILURE;
	}

	RequestHandler handler(runner);
	handler.run(stdin, stdout);

	delete runner;
}
