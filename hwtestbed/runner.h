#ifndef RUNNER_H
#define RUNNER_H

#include <cstdlib>
#include <string>

class Runner;

class Runner {
public:
	struct Buffer {
		void* cpu_pointer = nullptr;
		void* gpu_handle = nullptr;
		size_t size = 0;
	};
	struct Shader;
	struct ComputeRun {
		const Shader* shader;
		const Buffer* buffers;
		size_t num_buffers;
		uint32_t threadgroup_memory_size;
		uint32_t threadgroups_per_grid[3];
		uint32_t threads_per_threadgroup[3];
		uint64_t nanoseconds_elapsed;
	};
	virtual ~Runner() = default;
	virtual Buffer create_buffer(size_t size, std::string* err) = 0;
	virtual void destroy_buffer(const Buffer& buffer) = 0;
	virtual Shader* create_compute_shader_from_file(const char* filename, std::string* err) = 0;
	virtual Shader* create_compute_shader(void* data, size_t size, std::string* err) = 0;
	virtual void destroy_shader(Shader* shader) = 0;
	virtual bool run_compute_shader(ComputeRun& run, std::string* error) = 0;
	static Runner* make(std::string* err);
};

#endif // RUNNER_H
