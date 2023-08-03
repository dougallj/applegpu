import disassemble
import subprocess
import os
import sys
import tempfile
import struct

toolsDir = os.path.join(os.path.dirname(__file__), 'compiler_explorer_tools')
compileTool = os.path.join(toolsDir, 'metal-compile-tool')
archiveExtractor = os.path.join(toolsDir, 'metal-archive-extractor')

def read_shader_archive(archiveName):
	output = []
	for shaderType in ('vertex', 'fragment', 'compute'):
		shader = subprocess.check_output([archiveExtractor, '--extract-' + shaderType, '-', archiveName])
		if shader:
			output.append((shaderType, shader))
	return output

if __name__ == '__main__':
	if not os.path.exists(compileTool):
		subprocess.run(['make', '-C', toolsDir])
	if len(sys.argv) == 1:
		print(f"Usage: python3 {sys.argv[0]} file.metal")
		exit(1)
	shaders = None
	if len(sys.argv) == 2 and os.path.exists(sys.argv[1]):
		with open(sys.argv[1], "rb") as file:
			# If the file is a mach-o file, assume it's a compiled shader binary
			magic = struct.unpack("<I", file.read(4))[0]
			if magic in (0xcbfebabe, 0xbebafecb, 0xfeedfacf, 0xcffaedfe):
				shaders = read_shader_archive(sys.argv[1])
	if not shaders:
		with tempfile.TemporaryDirectory() as tmpdirname:
			filename = os.path.join(tmpdirname, 'shader.bin')
			subprocess.check_output([compileTool, '-o', filename] + sys.argv[1:])
			shaders = read_shader_archive(filename)
	for shaderType, shader in shaders:
		list = subprocess.check_output([archiveExtractor, '--list-shaders', '-'], input=shader).decode("utf-8").split("\n")
		sort_order = {"_agc.main.constant_program": 1, "_agc.main": 2}
		list.sort(key=lambda x: sort_order.get(x, 0))
		found = False
		for name in list:
			if not name.strip():
				continue
			data = subprocess.check_output([archiveExtractor, '--extract-named-shader', name, '--output', '-', '-'], input=shader)
			# Trim excess traps (compiler uses them for padding)
			while len(data) > 2 and data[-2] == 8 and data[-1] == 0:
				data = data[:-2]
			if not any(byte != 0 for byte in data):
				continue
			found = True
			friendly_names = {
				"_agc.main.constant_program": "shader prolog",
				"_agc.main": "shader",
			}
			friendly_name = friendly_names.get(name, name)
			print(f"{shaderType} {friendly_name}:")
			disassemble.disassemble(data)
			print()
		if found:
			print()
