import subprocess
import os
import struct
import metallib_replacer

class HWTestBedError(Exception):
	def __init__(self, message):
		self.message = message

class HWTestBedResponse:
	def __init__(self):
		self.buffers = {}
		self.time = 0

	def set_buffer(self, index, buffer):
		self.buffers[index] = buffer

class HWTestBedRequest:
	def __init__(self, shader=None, buffers=[], responses=[], num_tg=(1, 1, 1), tg_size=(1, 1, 1), tgsm=0):
		self.buffers = {}
		self.requests = {}
		self.num_tg = num_tg
		self.tg_size = tg_size
		self.tgsm = tgsm
		self.shader = shader
		for (index, buffer) in buffers:
			self.set_buffer(index, buffer)
		for (index, size) in responses:
			self.request_result(index, size)

	def set_shader(self, shader):
		self.shader = shader

	def set_buffer(self, index, buffer):
		self.buffers[index] = buffer

	def request_result(self, index, size):
		self.requests[index] = size

	def set_tgsm_size(self, size):
		self.tgsm = size

class HWTestBed:
	_binDir = os.path.join(os.path.dirname(__file__), 'hwtestbed')
	_toolsDir = os.path.join(os.path.dirname(__file__), 'compiler_explorer_tools')
	_hwtestbed = os.path.join(_binDir, 'hwtestbed')
	_metallib = os.path.join(_binDir, 'compute.metallib')
	_compileTool = os.path.join(_toolsDir, 'metal-compile-tool')

	_RESPONSE_BEGIN = 1
	_RESPONSE_END   = 2
	_RESPONSE_ERROR = 3
	_RESPONSE_TIME  = 4
	_RESPONSE_BUFFER_DATA = 5

	def __init__(self, tmpfilename, replacer=None):
		if not os.path.exists(self._hwtestbed):
			subprocess.run(['make', '-C', self._binDir])
		self.process = subprocess.Popen([self._hwtestbed], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
		self.request = self.process.stdin
		self.response = self.process.stdout
		self.tmpfilename = tmpfilename.encode('utf-8')
		self.replacer = replacer

	def _req(self, opcode, *args):
		self.request.write(struct.pack('=B' + 'I' * len(args), opcode, *args))

	def _req_begin_compute(self):
		self._req(1)
	def _req_set_cs(self, bfile):
		self._req(3, len(bfile))
		self.request.write(bfile)
	def _req_set_buffer_data(self, index, buffer):
		self._req(4, index, len(buffer))
		self.request.write(buffer)
	def _req_set_buffer_result(self, index, size):
		self._req(5, index, size)
	def _req_execute_compute(self, num_tg, tg_size):
		num_tg = tuple(num_tg) + (1, 1, 1)
		tg_size = tuple(tg_size) + (1, 1, 1)
		self._req(6, *num_tg[:3], *tg_size[:3])
	def _req_set_tgsm(self, size):
		self._req(7, size)

	def _read_response(self, size):
		data = self.response.read(size)
		if len(data) < size:
			raise Exception('hwtestbed process died')
		return data
	def _read_response_opcode(self):
		return self._read_response(1)[0]

	def _process_shader(self, shader):
		if not self.replacer:
			if not os.path.exists(self._compileTool):
				subprocess.run(['make', '-C', self._toolsDir])
			subprocess.run([self._compileTool, '-o', self.tmpfilename.decode('utf-8'), self._metallib])
			with open(self.tmpfilename, 'rb') as file:
				self.replacer = metallib_replacer.MetallibReplacer(file.read())
		return self.replacer.replace('__TEXT,__compute', '_agc.main', shader)

	def run(self, request):
		self._req_begin_compute()
		self._req_set_cs(self.tmpfilename)
		if request.shader:
			shader = self._process_shader(request.shader)
			with open(self.tmpfilename, 'wb') as file:
				file.write(shader)
		for (index, size) in request.requests.items():
			self._req_set_buffer_result(index, size)
		for (index, buffer) in request.buffers.items():
			self._req_set_buffer_data(index, buffer)
		self._req_set_tgsm(request.tgsm)
		self._req_execute_compute(request.num_tg, request.tg_size)
		self.request.flush()
		opcode = self._read_response_opcode()
		if opcode != self._RESPONSE_BEGIN:
			raise Exception(f'hwtestbed desync, got opcode {opcode} expecting BEGIN')
		response = HWTestBedResponse()
		error = None
		while True:
			opcode = self._read_response_opcode()
			if opcode == self._RESPONSE_BUFFER_DATA:
				(index, size) = struct.unpack('=II', self._read_response(8))
				response.set_buffer(index, self._read_response(size))
			elif opcode == self._RESPONSE_ERROR:
				size = struct.unpack('=I', self._read_response(4))[0]
				error = HWTestBedError(self._read_response(size).decode('utf-8'))
			elif opcode == self._RESPONSE_TIME:
				response.time = struct.unpack('=Q', self._read_response(8))[0] / 1000000000
			elif opcode == self._RESPONSE_END:
				break
			else:
				raise Exception(f'hwtestbed desync, got opcode {opcode}')
		if error:
			raise error
		return response

	def close(self):
		self.request.close()
		self.response.close()
		self.process.wait(timeout=1)

	def __del__(self):
		self.close()

if __name__ == '__main__':
	import argparse
	import assemble
	import tempfile
	parser = argparse.ArgumentParser(description='Run AGX shader asm')
	parser.add_argument('instructions', nargs='*', help='Shader instructions')
	parser.add_argument('-i', '--input', help='Path to input file')
	parser.add_argument('-t', '--tempdir', help='Directory to place temporary files')
	parser.add_argument('-r', '--registers', action='append', help='Comma-separated list of custom input registers (up to 32).  Numbers with decimals will be interpreted as floats.  Specify multiple times to run multiple threads.')
	parser.add_argument('-b', '--binary', action='store_true', help='Treat code as binary instead of assembly')
	parser.add_argument('--tgsm', type=int, default=0x100, help='Set the amount of threadgroup shared memory')
	args = parser.parse_args()
	if args.input and args.instructions:
		parser.error('Please supply instructions with an input file or on the command line but not both')
	elif not args.input and not args.instructions:
		parser.error('An input file or instructions are required')
	if args.input:
		code = assemble.assemble_file(args.input, args.binary)
	else:
		if args.binary:
			code = bytes.fromhex(' '.join(args.instructions))
		else:
			code = assemble.assemble_multiline(' '.join(args.instructions).split(';'))
	before = bytes.fromhex(
		'f2791004'         # get_sr       r30.cache, sr80 (thread_position_in_grid.x)
		'9e7bfc02009c0102' # imadd        r30_r31.cache, r30.discard, 128, u14
		'0e7dfee219000000' # iadd         r31, r31.discard, u15
		'05010c0730c8f200' # device_load  0, i32, xyzw, r0_r1_r2_r3, r30_r31, 0, unsigned, lsl 2
		'05211c0730c8f200' # device_load  0, i32, xyzw, r4_r5_r6_r7, r30_r31, 1, unsigned, lsl 2
		'05412c0730c8f200' # device_load  0, i32, xyzw, r8_r9_r10_r11, r30_r31, 2, unsigned, lsl 2
		'05613c0730c8f200' # device_load  0, i32, xyzw, r12_r13_r14_r15, r30_r31, 3, unsigned, lsl 2
		'05814c0730c8f200' # device_load  0, i32, xyzw, r16_r17_r18_r19, r30_r31, 4, unsigned, lsl 2
		'05a15c0730c8f200' # device_load  0, i32, xyzw, r20_r21_r22_r23, r30_r31, 5, unsigned, lsl 2
		'05c16c0730c8f200' # device_load  0, i32, xyzw, r24_r25_r26_r27, r30_r31, 6, unsigned, lsl 2
		'05e17c0730c8f200' # device_load  0, i32, xyzw, r28_r29_r30_r31, r30_r31, 7, unsigned, lsl 2
		'3800'             # wait         0
	)
	after = bytes.fromhex(
		'f2791004'         # get_sr       r30.cache, sr80 (thread_position_in_grid.x)
		'9e7bfc02009c0102' # imadd        r30_r31.cache, r30.discard, 128, u14
		'0e7dfee219000000' # iadd         r31, r31.discard, u15
		'45010c0730c8f200' # device_store 0, i32, xyzw, r0_r1_r2_r3, r30_r31, 0, unsigned, lsl 2, 0
		'45211c0730c8f200' # device_store 0, i32, xyzw, r4_r5_r6_r7, r30_r31, 1, unsigned, lsl 2, 0
		'45412c0730c8f200' # device_store 0, i32, xyzw, r8_r9_r10_r11, r30_r31, 2, unsigned, lsl 2, 0
		'45613c0730c8f200' # device_store 0, i32, xyzw, r12_r13_r14_r15, r30_r31, 3, unsigned, lsl 2, 0
		'45814c0730c8f200' # device_store 0, i32, xyzw, r16_r17_r18_r19, r30_r31, 4, unsigned, lsl 2, 0
		'45a15c0730c8f200' # device_store 0, i32, xyzw, r20_r21_r22_r23, r30_r31, 5, unsigned, lsl 2, 0
		'45c16c0730c8f200' # device_store 0, i32, xyzw, r24_r25_r26_r27, r30_r31, 6, unsigned, lsl 2, 0
		'45e17c0730c8f200' # device_store 0, i32, xyzw, r28_r29_r30_r31, r30_r31, 7, unsigned, lsl 2, 0
		'8800'             # stop
	)
	shader = b''.join((before, code, after))
	registers = args.registers
	if not registers:
		registers = [
			', '.join(str(x) for x in range(8)),
			', '.join(str(x) for x in range(8, 16)),
			', '.join(str(x) for x in range(-8, 0)),
			', '.join(str(x / 8) for x in range(8)),
			', '.join(str(x / 8) for x in range(8, 16)),
			', '.join(str(x / 8) for x in range(-8, 0)),
		]
	ibuf = bytearray(len(registers) * 128)
	for invocation, reglist in enumerate(registers):
		for idx, register in enumerate(reglist.split(',')[:32]):
			offset = invocation * 128 + idx * 4
			register = register.strip()
			if '.' in register:
				data = struct.pack('=f', float(register))
			else:
				data = struct.pack('=i' if register.startswith('-') else '=I', int(register, 0))
			ibuf[offset:offset+4] = data

	def seems_float_ish(num):
		exponent = ((num >> 23) & 0xff) - 127
		return 64 >= exponent >= -64

	def print_buffer(idx, data):
		elems = []
		floats = []
		header = False
		for i in range(len(data)//4):
			(elem, f32) = struct.unpack('=If', data[i * 4 : i * 4 + 4] * 2)
			elems.append(elem)
			floats.append(f32)
			if i % 4 == 3:
				if any(x != 0 for x in elems):
					if not header:
						print(f'Buffer {idx}:')
						header = True
					line = ' '.join(f'{x:08x}' for x in elems)
					if any(seems_float_ish(x) for x in elems):
						line = line.ljust(35)
						line += ' ('
						line += ' '.join(f'{floats[x]:<8.6g}' if seems_float_ish(elems[x]) else ' ' * 8 for x in range(len(elems)))
						line += ')'
					print(f'  {(i-3)*4:3x}: {line}')
				elems = []
				floats = []



	with tempfile.TemporaryDirectory(dir=args.tempdir) as tempdir:
		testbed = HWTestBed(os.path.join(tempdir, 'compute.metallib'))
		request = HWTestBedRequest(
			shader = shader,
			buffers = [
				(7, ibuf)
			],
			responses = [
				(0, 4096),
				(1, 4096),
				(7, len(ibuf)),
			],
			tg_size = (len(registers), 1, 1),
			tgsm = args.tgsm
		)
		try:
			result = testbed.run(request)
			obuf = result.buffers[7]
			nregs = 30 # 30 and 31 are used for addressing
			changed = [False] * 30

			for thread in range(len(registers)):
				for register in range(nregs):
					idata = 0
					pos = thread * 128 + register * 4
					idata = struct.unpack('=I', ibuf[pos:pos + 4])[0]
					opos = thread * 128 + register * 4
					odata = struct.unpack('=I', obuf[pos:pos + 4])[0]
					if idata != odata:
						changed[register] = True
			if any(changed):
				id_len = len(str(len(registers) - 1))
				for thread, reglist in enumerate(registers):
					regs = []
					for register in range(nregs):
						if not changed[register]:
							continue
						pos = thread * 128 + register * 4
						(u32, s32, f32) = struct.unpack('=Iif', obuf[pos:pos + 4] * 3)
						extra = f'{f32:<10.8g}' if seems_float_ish(u32) else f'{s32:<10d}'
						regs.append(f'r{register}: {u32:08x} ({extra})')
					regs.append('input: ' + reglist)
					print(f'Thread {thread:{id_len}}: ' + ', '.join(regs))

			print_buffer(0, result.buffers[0])
			print_buffer(1, result.buffers[1])
		except HWTestBedError as err:
			print('Failed to run shader: ' + err.message)
