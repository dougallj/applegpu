import fma
import os

from srgb import SRGB_TABLE

MAX_OPCODE_LEN = 12

ABS_FLAG = 'abs'
NEGATE_FLAG = 'neg'
SIGN_EXTEND_FLAG = 'sx'
CACHE_FLAG = 'cache'
DISCARD_FLAG = 'discard'

OPERAND_FLAGS = [
	ABS_FLAG,
	NEGATE_FLAG,
	SIGN_EXTEND_FLAG,
	CACHE_FLAG,
	DISCARD_FLAG,
]

CACHE_HINT = '$'

SR_NAMES = {
	0:  'threadgroup_position_in_grid.x',
	1:  'threadgroup_position_in_grid.y',
	2:  'threadgroup_position_in_grid.z',
	4:  'threads_per_threadgroup.x',
	5:  'threads_per_threadgroup.y',
	6:  'threads_per_threadgroup.z',
	8:  'dispatch_threads_per_threadgroup.x',
	9:  'dispatch_threads_per_threadgroup.y',
	10: 'dispatch_threads_per_threadgroup.z',
	48: 'thread_position_in_threadgroup.x',
	49: 'thread_position_in_threadgroup.y',
	50: 'thread_position_in_threadgroup.z',
	51: 'thread_index_in_threadgroup',
	52: 'thread_index_in_simdgroup',
	53: 'simdgroup_index_in_threadgroup',
	56: 'active_thread_index_in_quadgroup',
	58: 'active_thread_index_in_simdgroup',
        # In fragment shaders. Invert for front facing
	62: 'backfacing',
	63: 'is_active_thread', # compare to zero for simd/quad_is_helper_thread
        # 80, 81 also used for calculating fragcoord.xy
	80: 'thread_position_in_grid.x',
	81: 'thread_position_in_grid.y',
	82: 'thread_position_in_grid.z',
}

def opcode_to_number(opcode):
	n = 0
	for i, c in enumerate(opcode[:MAX_OPCODE_LEN]):
		n |= c << (8 * i)
	return n

def sign_extend(v, bits):
	v &= (1 << bits) - 1
	if v & (1 << (bits-1)):
		v |= (-1 << bits)
	return v

class Operand:
	pass

def _add_flags(base, flags):
	parts = [base]
	for i in flags:
		if 'APPLEGPU_CRYPTIC' in os.environ and i == 'cache':
			parts[0] = CACHE_HINT + parts[0]
		else:
			parts.append(i)
	return '.'.join(parts)


class RegisterTuple(Operand):
	def __init__(self, registers, flags=None):
		self.registers = list(registers)
		if flags is None:
			self.flags = []
		else:
			self.flags = list(flags)

	def __str__(self):
		return _add_flags('_'.join(map(str, self.registers)), self.flags)

	def __repr__(self):
		return 'RegisterTuple(%r)' % self.registers

	def __repr__(self):
		return 'RegisterTuple(%r)' % self.registers

	def __getitem__(self, i):
		return self.registers[i]

	def get_with_flags(self, i):
		r = self[i]
		return r.__class__(r.n, flags=self.flags)

	def __len__(self):
		return len(self.registers)

	def get_bit_size(self):
		raise NotImplementedError('get_bit_size')

	def set_thread(self, corestate, thread, result):
		raise NotImplementedError('set_thread')

	def get_thread(self, corestate, thread):
		raise NotImplementedError('get_thread')

class Immediate(Operand):
	# TODO: how should we handle bit_size?
	def __init__(self, value, bit_size=16, flags=None):
		self.value = value
		self._bit_size = bit_size
		if flags is None:
			self.flags = []
		else:
			self.flags = list(flags)

	def get_bit_size(self):
		return self._bit_size

	def get_thread(self, corestate, thread):
		return self.value

	def __str__(self):
		return '.'.join([str(self.value)] + self.flags)

	def __repr__(self):
		if self.flags:
			return 'Immediate(%r, flags=%r)' % (self.value, self.flags)
		return 'Immediate(%r)' % self.value

class RelativeOffset(Immediate):
	def __str__(self):
		base = getattr(self, 'base', None)
		if base is not None:
			v = '0x%X' % (base + self.value,)
		elif self.value >= 0:
			v = 'pc+%d' % (self.value,)
		else:
			v = 'pc-%d' % (-self.value,)
		return '.'.join([v] + self.flags)

	def __repr__(self):
		if self.flags:
			return 'RelativeOffset(%r, flags=%r)' % (self.value, self.flags)
		return 'RelativeOffset(%r)' % self.value

class Register(Operand):
	def __init__(self, n, flags=None):
		self.n = n
		if flags is None:
			self.flags = []
		else:
			self.flags = list(flags)

	def _str(self, names):
		return _add_flags(names[self.n], self.flags)

	def _repr(self, clsname):
		if self.flags:
			return '%s(%d, flags=%r)' % (clsname, self.n, self.flags)
		return '%s(%d)' % (clsname, self.n)

class BaseReg(Register):
	pass

class Reg16(BaseReg):
	def __str__(self):
		return self._str(reg16_names)

	def __repr__(self):
		return self._repr('Reg16')

	def get_bit_size(self):
		return 16

	def set_thread(self, corestate, thread, result):
		corestate.set_reg16(self.n, thread, result)

	def get_thread(self, corestate, thread):
		return corestate.get_reg16(self.n, thread)

class Reg32(BaseReg):
	def __str__(self):
		return self._str(reg32_names)

	def __repr__(self):
		return self._repr('Reg32')

	def get_bit_size(self):
		return 32

	def set_thread(self, corestate, thread, result):
		corestate.set_reg32(self.n, thread, result)

	def get_thread(self, corestate, thread):
		return corestate.get_reg32(self.n, thread)

class Reg64(BaseReg):
	def __str__(self):
		return self._str(reg64_names)

	def __repr__(self):
		return self._repr('Reg64')

	def get_bit_size(self):
		return 64

	def set_thread(self, corestate, thread, result):
		corestate.set_reg64(self.n, thread, result)

	def get_thread(self, corestate, thread):
		return corestate.get_reg64(self.n, thread)

class BaseUReg(Register):
	pass

class UReg16(BaseUReg):
	def __str__(self):
		return self._str(ureg16_names)

	def __repr__(self):
		return self._repr('UReg16')

	def get_thread(self, corestate, thread):
		return corestate.uniforms.get_reg16(self.n)

	def get_bit_size(self):
		return 16

class UReg32(BaseUReg):
	def __str__(self):
		return self._str(ureg32_names)

	def __repr__(self):
		return self._repr('UReg32')

	def get_thread(self, corestate, thread):
		return corestate.uniforms.get_reg32(self.n)

	def get_bit_size(self):
		return 32

class UReg64(BaseUReg):
	def __str__(self):
		return self._str(ureg64_names)

	def __repr__(self):
		return self._repr('UReg64')

	def get_thread(self, corestate, thread):
		return corestate.uniforms.get_reg64(self.n)

	def get_bit_size(self):
		return 64

class SReg32(Register):
	def __str__(self):
		name = 'sr%d' % (self.n)
		if self.n in SR_NAMES:
			name += ' (' + SR_NAMES[self.n] + ')'
		return name

	def __repr__(self):
		return self._repr('SReg32')

	def get_bit_size(self):
		return 32

class TextureState(Register):
	def __str__(self):
		return 'ts%d' % (self.n)

	def __repr__(self):
		return self._repr('TextureState')

	def get_bit_size(self):
		return 32 # ?

class SamplerState(Register):
	def __str__(self):
		return 'ss%d' % (self.n)

	def __repr__(self):
		return self._repr('SamplerState')

	def get_bit_size(self):
		return 32 # ?

class CF(Register):
	def __str__(self):
		return 'cf%d' % (self.n)

	def __repr__(self):
		return self._repr('CF')

	def get_bit_size(self):
		return 32 # ?

ureg16_names = []
ureg32_names = []
ureg64_names = []

for _i in range(256):
	ureg16_names.append('u%dl' % _i)
	ureg16_names.append('u%dh' % _i)
	ureg32_names.append('u%d' % _i)
	ureg64_names.append('u%d_u%d' % (_i, _i + 1))

reg16_names = []
reg32_names = []
reg64_names = []
reg96_names = []
reg128_names = []
for _i in range(128):
	reg16_names.append('r%dl' % _i)
	reg16_names.append('r%dh' % _i)
	reg32_names.append('r%d' % _i)
	# TODO: limit? can cross r31-r32 boundary?
	reg64_names.append('r%d_r%d' % (_i, _i + 1))
	reg96_names.append('r%d_r%d_r%d' % (_i, _i + 1, _i + 2))
	reg128_names.append('r%d_r%d_r%d_r%d' % (_i, _i + 1, _i + 2, _i + 3))


# TODO: is this the right number?
ts_names = []
ss_names = []
cf_names = []
for _i in range(256):
	ts_names.append('ts%d' % _i)
	ss_names.append('ss%d' % _i)
	cf_names.append('cf%d' % _i)



registers_by_name = {}

for _namelist, _c in [
	(reg16_names, Reg16),
	(reg32_names, Reg32),
	(reg64_names, Reg64),
	(ureg16_names, UReg16),
	(ureg32_names, UReg32),
	(ureg64_names, UReg64),
	(ts_names, TextureState),
	(ss_names, SamplerState),
	(cf_names, CF),
]:
	for _i, _name in enumerate(_namelist):
		registers_by_name[_name] = (_c, _i)


def try_parse_register(s):
	flags = []
	if s.startswith(CACHE_HINT):
		s = s[1:]
		flags.append(CACHE_FLAG)
	parts = s.split('.')
	if parts[0] not in registers_by_name:
		return None
	for i in parts[1:]:
		if i not in OPERAND_FLAGS:
			return None
		flags.append(i)

	c, n = registers_by_name[parts[0]]

	return c(n, flags=flags)

def try_parse_register_tuple(s):
	flags = []
	if s.startswith(CACHE_HINT):
		s = s[1:]
		flags.append(CACHE_FLAG)
	parts = s.split('.')
	regs = [try_parse_register(i) for i in parts[0].split('_')]
	if not all(isinstance(r, Reg32) for r in regs) and not all(isinstance(r, Reg16) for r in regs):
		return None
	if any(i.flags for i in regs):
		return None
	if not all(regs[i].n + 1 == regs[i+1].n for i in range(len(regs)-1)):
		return None

	for i in parts[1:]:
		if i not in OPERAND_FLAGS:
			return None
		flags.append(i)

	return RegisterTuple(regs, flags=flags)

SIMD_WIDTH = 32

class AsmInstruction:
	def __init__(self, mnem, operands=None):
		self.mnem = mnem
		self.operands = list(operands)

	def __str__(self):
		operands = ', '.join(filter(None, (str(i) for i in self.operands)))
		return self.mnem.ljust(16) + ' ' + operands

	def __repr__(self):
		return 'AsmInstruction(%r, %r)' % (self.mnem, self.operands)

class AddressSpace:
	def __init__(self):
		self.mappings = []

	def map(self, address, size):
		# TODO: check for overlap
		self.mappings.append((address, [0] * size))

	def set_byte(self, address, value):
		for start, values in self.mappings:
			if start < address and address - start < len(values):
				values[address - start] = value
				return
		assert False, 'bad address %x' % address

	def get_byte(self, address):
		for start, values in self.mappings:
			if start < address and address - start < len(values):
				return values[address - start]
		assert False, 'bad address %x' % address

	def get_u16(self, address):
		return self.get_byte(address) | (self.get_byte(address + 1) << 8)

	def get_u32(self, address):
		return self.get_u16(address) | (self.get_u16(address + 2) << 16)

class Uniforms:
	def __init__(self):
		self.reg16s = [0] * 256

	def get_reg16(self, regid):
		return self.reg16s[regid]

	def set_reg32(self, regid, value):
		self.reg16s[regid * 2] = value & 0xFFFF
		self.reg16s[regid * 2 + 1] = (value >> 16) & 0xFFFF

	def get_reg32(self, regid):
		return self.reg16s[regid * 2] | (self.reg16s[regid * 2 + 1] << 16)

	def get_reg64(self, regid):
		return self.get_reg32(regid) | (self.get_reg32(regid + 1) << 32)

	def set_reg64(self, regid, value):
		self.set_reg32(regid, value & 0xFFFFFFFF)
		self.set_reg32(regid + 1, (value >> 32) & 0xFFFFFFFF)

class CoreState:
	def __init__(self, num_registers=8, uniforms=None, device_memory=None):
		self.reg16s = [[0] * SIMD_WIDTH for i in range(num_registers * 2)]
		self.pc = 0
		self.exec = [True] * SIMD_WIDTH
		if uniforms is None:
			uniforms = Uniforms()
		self.uniforms = uniforms
		self.device_memory = device_memory

	def get_reg16(self, regid, thread):
		return self.reg16s[regid][thread]

	def set_reg16(self, regid, thread, value):
		self.reg16s[regid][thread] = value & 0xFFFF

	def get_reg32(self, regid, thread):
		return self.reg16s[regid * 2][thread] | (self.reg16s[regid * 2 + 1][thread] << 16)

	def set_reg32(self, regid, thread, value):
		self.reg16s[regid * 2][thread] = value & 0xFFFF
		self.reg16s[regid * 2 + 1][thread] = (value >> 16) & 0xFFFF

	def get_reg64(self, regid, thread):
		return self.get_reg32(regid, thread) | (self.get_reg32(regid + 1, thread) << 32)

	def set_reg64(self, regid, thread, value):
		self.set_reg32(regid, thread, value & 0xFFFFFFFF)
		self.set_reg32(regid + 1, thread, (value >> 32) & 0xFFFFFFFF)


class InstructionDesc:
	documentation_skip = False
	def __init__(self, name, size=2, length_bit_pos=15):
		self.name = name

		self.mask = 0
		self.bits = 0
		self.fields = []
		self.ordered_operands = []
		self.operands = {}
		self.constants = []

		self.merged_fields = []

		self.fields_mask = 0

		assert isinstance(size, (int, tuple))
		self.sizes = (size, size) if isinstance(size, int) else size
		self.length_bit_pos = length_bit_pos


	def matches(self, instr):
		instr = self.mask_instr(instr)
		return (instr & self.mask) == self.bits

	def add_raw_field(self, start, size, name):
		# collision check
		mask = ((1 << size) - 1) << start
		assert (self.mask & mask) == 0, name
		assert (self.fields_mask & mask) == 0
		for _, _, existing_name in self.fields:
			assert existing_name != name, name

		self.fields_mask |= mask
		self.fields.append((start, size, name))

	def add_merged_field(self, name, subfields):
		pairs = []
		shift = 0
		for start, size, subname in subfields:
			self.add_raw_field(start, size, subname)
			pairs.append((subname, shift))
			shift += size
		self.merged_fields.append((name, pairs))

	def add_field(self, start, size, name):
		self.add_merged_field(name, [(start, size, name)])

	def add_suboperand(self, operand):
		# a "suboperand" is an operand which does not appear in the operand list,
		# but is used by other operands. currently unused.
		for start, size, name in operand.fields:
			self.add_field(start, size, name)
		for name, subfields in operand.merged_fields:
			self.add_merged_field(name, subfields)
		self.operands[operand.name] = operand

	def add_operand(self, operand):
		self.add_suboperand(operand)
		self.ordered_operands.append(operand)

	def add_constant(self, start, size, value):
		mask = (1 << size) - 1
		assert (value & ~mask) == 0
		assert (self.mask & (mask << start)) == 0
		self.mask |= mask << start
		self.bits |= value << start

		self.constants.append((start, size, value))

	def decode_raw_fields(self, instr):
		instr = self.mask_instr(instr)
		assert self.matches(instr)
		fields = []
		for start, size, name in self.fields:
			fields.append((name, (instr >> start) & ((1 << size) - 1)))
		return fields

	def decode_remainder(self, instr):
		instr = self.mask_instr(instr)
		assert self.matches(instr)
		instr &= ~self.mask
		for start, size, name in self.fields:
			instr &= ~(((1 << size) - 1) << start)
		if self.sizes[0] != self.sizes[1]:
			instr &= ~(1 << self.length_bit_pos)
		return instr

	def patch_raw_fields(self, encoded, fields):
		lookup = {name: (start, size) for start, size, name in self.fields}
		for name, value in fields.items():
			start, size = lookup[name]
			mask = (1 << size) - 1
			assert (value & ~mask) == 0
			encoded = (encoded & ~(mask << start)) | (value << start)

		if self.sizes[0] != self.sizes[1]:
			encoded &= ~(1 << self.length_bit_pos)
			if (encoded & (0xFFFF << (self.sizes[0] * 8))) != 0:
				# use long encoding
				encoded |= (1 << self.length_bit_pos)

		assert self.matches(encoded)
		encoded = self.mask_instr(encoded)

		return encoded

	def encode_raw_fields(self, fields):
		assert sorted(lookup.keys()) == sorted(name for start, size, name in self.fields)
		return self.patch_raw_fields(self.bits, fields)

	def patch_fields(self, encoded, fields):
		mf_lookup = dict(self.merged_fields)
		size_lookup = {name: size for start, size, name in self.fields}

		raw_fields = {}
		for name, value in fields.items():
			for subname, shift in mf_lookup[name]:
				mask = (1 << size_lookup[subname]) - 1
				raw_fields[subname] = (value >> shift) & mask

		return self.patch_raw_fields(encoded, raw_fields)

	def encode_fields(self, fields):
		if sorted(fields.keys()) != sorted(name for name, subfields in self.merged_fields):
			print(sorted(fields.keys()))
			print(sorted(name for name, subfields in self.merged_fields))
			assert False

		return self.patch_fields(self.bits, fields)

	def to_bytes(self, instr):
		return bytes((instr >> (i*8)) & 0xFF for i in range(self.decode_size(instr)))

	def decode_fields(self, instr):
		raw = dict(self.decode_raw_fields(instr))
		fields = []
		for name, subfields in self.merged_fields:
			value = 0
			for subname, shift in subfields:
				value |= raw[subname] << shift
			fields.append((name, value))
		return fields

	def decode_operands(self, instr):
		instr = self.mask_instr(instr)
		fields = dict(self.decode_fields(instr))
		return self.fields_to_operands(fields)

	def fields_to_operands(self, fields):
		ordered_operands = []
		for o in self.ordered_operands:
			ordered_operands.append(o.decode(fields))
		return ordered_operands

	def decode_size(self, instr):
		return self.sizes[(instr >> self.length_bit_pos) & 1]

	def mask_instr(self, instr):
		return instr & ((1 << (self.decode_size(instr) * 8)) - 1)

	def decode_mnem(self, instr):
		instr = self.mask_instr(instr)
		assert self.matches(instr)
		return self.fields_to_mnem(dict(self.decode_fields(instr)))


	def fields_to_mnem_base(self, fields):
		return self.name
	def fields_to_mnem_suffix(self, fields):
		return ''

	def fields_to_mnem(self, fields):
		return self.fields_to_mnem_base(fields) + self.fields_to_mnem_suffix(fields)

	def map_to_alias(self, mnem, operands):
		return mnem, operands

	def disassemble(self, n, pc=None):
		mnem = self.decode_mnem(n)
		operands = self.decode_operands(n)
		mnem, operands = self.map_to_alias(mnem, operands)
		for operand in operands:
			if isinstance(operand, RelativeOffset):
				operand.base = pc
		return AsmInstruction(mnem, operands)

	def fields_for_mnem(self, mnem, operand_strings):
		if self.name == mnem:
			return {}

	def rewrite_operands_strings(self, mnem, opstrs):
		return opstrs

documentation_operands = []

def document_operand(cls):
	documentation_operands.append(cls)
	return cls

class OperandDesc:
	def __init__(self, name=None):
		self.name = name
		self.fields = []
		self.merged_fields = []

	def add_field(self, start, size, name):
		self.fields.append((start, size, name))

	def add_merged_field(self, name, subfields):
		self.merged_fields.append((name, subfields))

	def decode(self, fields):
		return '<TODO>'

	def get_bit_size(self, fields):
		r = self.decode(fields)
		return r.get_bit_size()



def add_dest_hint_modifier(reg, bits):
	if bits & 1:
		reg.flags.append(CACHE_FLAG)
	return reg

def add_hint_modifier(reg, bits):
	if bits == 0b10:
		reg.flags.append(CACHE_FLAG)
		return reg
	elif bits == 0b11:
		reg.flags.append(DISCARD_FLAG)
		return reg
	else:
		assert bits == 0b01
		return reg

def decode_float_immediate(n):
	sign = -1.0 if n & 0x80 else 1.0
	e = (n & 0x70) >> 4
	f = n & 0xF
	if e == 0:
		return sign * f / 64.0
	else:
		return sign * float(0x10 | f) * (2.0 ** (e - 7))

# okay, this is very lazy
float_immediate_lookup = {str(decode_float_immediate(i)): i for i in range(0x100)}

def add_float_modifier(r, modifier):
	if modifier & 1:
		r.flags.append(ABS_FLAG)
	if modifier & 2:
		r.flags.append(NEGATE_FLAG)
	return r

class AbstractDstOperandDesc(OperandDesc):
	def set_thread(self, fields, corestate, thread, result):
		r = self.decode(fields)
		r.set_thread(corestate, thread, result)

class AbstractSrcOperandDesc(OperandDesc):
	def evaluate_thread(self, fields, corestate, thread):
		r = self.decode(fields)
		return r.get_thread(corestate, thread)

class ImplicitR0LDesc(AbstractDstOperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_field(7, 1, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		r = Reg16(0)
		return add_dest_hint_modifier(r, flags)

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg and isinstance(reg, Reg16) and reg.n == 0:
			flags = 0
			if CACHE_FLAG in reg.flags:
				flags |= 1
			fields[self.name + 't'] = flags
			return
		raise Exception('invalid ImplicitR0LDesc %r' % (opstr,))


@document_operand
class ALUDstDesc(AbstractDstOperandDesc):
	def __init__(self, name, bit_off_ex):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(9, 6, self.name),
			(bit_off_ex, 2, self.name + 'x')
		])
		self.add_field(7, 2, self.name + 't')

	def _allow64(self):
		return False

	def _allow32(self):
		return True

	def _paired(self):
		return False

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		if flags & 2 and self._allow32():
			if (value & 1) and self._allow64():
				assert not self._paired()
				r = Reg64(value >> 1)
			else:
				if self._paired():
					r = RegisterTuple(Reg32((value >> 1) + i) for i in range(2))
				else:
					r = Reg32(value >> 1)
		else:
			if self._paired():
				r = RegisterTuple(Reg16(value + i) for i in range(2))
			else:
				r = Reg16(value)

		return add_dest_hint_modifier(r, flags)

	def encode(self, fields, operand):
		if self._paired() and isinstance(operand, RegisterTuple):
			# TODO: validate
			operand = operand.get_with_flags(0)
		flags = 0
		value = 0
		if isinstance(operand, BaseReg):
			if isinstance(operand, Reg16):
				value = operand.n
			elif isinstance(operand, Reg32):
				if not self._allow32():
					print('WARNING: encoding invalid 32-bit register')
				value = operand.n << 1
				flags |= 2
			else:
				assert isinstance(operand, Reg64)
				if not self._allow64():
					print('WARNING: encoding invalid 64-bit register')
				value = (operand.n << 1) | 1
				flags |= 2

			if CACHE_FLAG in operand.flags:
				flags |= 1
		else:
			raise Exception('invalid ALUDstDesc %r' % (operand,))

		fields[self.name + 't'] = flags
		fields[self.name] = value

	pseudocode = '''
	{name}(value, flags, max_size=32):
		cache_flag = flags & 1
		if flags & 2 and value & 1 and max_size >= 64:
			return Reg64Reference(value >> 1, cache=cache_flag)
		elif flags & 2 and max_size >= 32:
			return Reg32Reference(value >> 1, cache=cache_flag)
		else:
			return Reg16Reference(value, cache=cache_flag)
	'''

	def encode_string(self, fields, opstr):
		if self._paired():
			regs = try_parse_register_tuple(opstr)
			if regs and len(regs) == 2:
				return self.encode(fields, regs)
			raise Exception('invalid paired ALUDstDesc %r' % (opstr,))

		reg = try_parse_register(opstr)
		if reg:
			return self.encode(fields, reg)
		raise Exception('invalid ALUDstDesc %r' % (opstr,))


class PairedALUDstDesc(ALUDstDesc):
	# converts r0 <-> r0_r1 and r0h <-> r0h_r1l
	def _paired(self):
		return True

@document_operand
class ALUDst64Desc(ALUDstDesc):
	pseudocode = '''
	{name}(value, flags):
		return ALUDst(value, flags, max_size=64)
	'''
	def _allow64(self):
		return True

class ALUDst16Desc(ALUDstDesc):
	pseudocode = '''
	{name}(value, flags):
		return ALUDst(value, flags, max_size=16)
	'''
	def _allow32(self):
		return False

@document_operand
class FloatDstDesc(ALUDstDesc):
	def __init__(self, name, bit_off_ex):
		super().__init__(name, bit_off_ex)
		self.add_field(6, 1, 'S')

	# so far this is the same, but conceptually i'd like the destination to
	# be responsible for converting the result to the correct size, which is
	# a very different operation for floats.
	pseudocode = '''
	{name}(value, flags, saturating, max_size=32):
		destination = ALUDst(value, flags, max_size=max_size)
		if destination.thread_bit_size == 32:
			wrapper = RoundToFloat32Wrapper(destination, flush_to_zero=True)
		else:
			wrapper = RoundToFloat16Wrapper(destination, flush_to_zero=False)

		if saturating:
			wrapper = SaturateRealWrapper(wrapper)

		return wrapper
	'''

@document_operand
class FloatDst16Desc(FloatDstDesc):
	def _allow32(self):
		return False
	pseudocode = '''
	{name}(value, flags, saturating):
		return FloatDst(value, flags, saturating, max_size=16)
	'''

@document_operand
class ALUSrcDesc(AbstractSrcOperandDesc):
	"Zero-extended 16 or 32 bit source field"

	def __init__(self, name, bit_off, bit_off_ex):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(bit_off, 6, self.name),
			(bit_off_ex, 2, self.name + 'x')
		])
		self.add_field(bit_off + 6, self._type_size(), self.name + 't')

	def _type_size(self):
		return 4

	def _allow32(self):
		return True

	def _allow64(self):
		return False

	def _paired(self):
		return False

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		return bitop_decode(flags, value, allow64)

	def decode_immediate(self, fields, value):
		return Immediate(value)

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]
		if flags == 0:
			return self.decode_immediate(fields, value)
		elif (flags & 0b1100) == 0b0100:
			value |= (flags & 1) << 8
			if flags & 2:
				return UReg32(value >> 1)
			else:
				return UReg16(value)
		elif (flags & 0b11) != 0: # in (0b0001, 0b0010, 0b0011):
			if self._allow64() and (flags & 0b1100) == 0b1100:
				assert (value & 1) == 0
				assert not self._paired()
				r = Reg64(value >> 1)
			elif self._allow32() and (flags & 0b1100) in (0b1000, 0b1100):
#				assert (value & 1) == 0
				if self._paired():
					r = RegisterTuple(Reg32((value >> 1) + i) for i in range(2))
				else:
					r = Reg32(value >> 1)
			elif (flags & 0b1100) in (0b0000, 0b1000, 0b1100):
				if self._paired():
					r = RegisterTuple(Reg16(value + i) for i in range(2))
				else:
					r = Reg16(value)
			else:
				return
			return add_hint_modifier(r, flags & 0b11)
		else:
			print('TODO: ' + format(flags, '04b'))

	def encode(self, fields, operand):
		if self._paired():
			if isinstance(operand, (Reg16,Reg32)):
				raise Exception('invalid paired operand %r' % (operand,))
			elif isinstance(operand, RegisterTuple):
				# TODO: validate
				operand = operand.get_with_flags(0)
		flags = 0
		value = 0
		if isinstance(operand, (UReg16, UReg32)):
			flags = 0b0100
			if isinstance(operand, UReg32):
				flags |= 2
				value = operand.n << 1
			else:
				value = operand.n
			flags |= (value >> 8) & 1
			value &= 0xFF
		elif isinstance(operand, BaseReg):
			if isinstance(operand, Reg16):
				flags = 0b0000
				value = operand.n
			elif isinstance(operand, Reg32):
				if not self._allow32():
					print('WARNING: encoding invalid 32-bit register')
				flags = 0b1000
				value = operand.n << 1
			else:
				flags = 0b1100
				assert isinstance(operand, Reg64)
				if not self._allow64():
					print('WARNING: encoding invalid 64-bit register')
				value = operand.n << 1

			if CACHE_FLAG in operand.flags:
				#if DISCARD_FLAG not in operand.flags
				flags |= 2
			elif DISCARD_FLAG in operand.flags:
				flags |= 3
			else:
				flags |= 1
		elif isinstance(operand, Immediate):
			flags = 0
			if not 0 <= operand.value < 256:
				raise Exception('out of range immediate %r' % (operand,))
			value = operand.value
		else:
			raise Exception('invalid ALUSrcDesc %r' % (operand,))

		fields[self.name + 't'] = flags
		fields[self.name] = value

	pseudocode = '''
	{name}(value, flags, max_size=32):
		if flags == 0b0000:
			return BroadcastImmediateReference(value)

		if flags >> 2 == 0b01:
			ureg = value | (flags & 1) << 8
			if flags & 0b10:
				if max_size < 32:
					UNDEFINED()
				return BroadcastUReg32Reference(ureg >> 1)
			else:
				return BroadcastUReg16Reference(ureg)

		if flags & 0b11 == 0b00: UNDEFINED()

		cache_flag   = (flags & 0b11) == 0b10
		discard_flag = (flags & 0b11) == 0b11

		if flags >> 2 == 0b11 and max_size >= 64:
			if value & 1: UNDEFINED()
			return Reg64Reference(value >> 1, cache=cache_flag, discard=discard_flag)

		if flags >> 2 >= 0b10 and max_size >= 32:
			if flags >> 2 != 0b10: UNDEFINED()
			if value & 1: UNDEFINED()
			return Reg32Reference(value >> 1, cache=cache_flag, discard=discard_flag)

		if max_size >= 16:
			if flags >> 2 != 0b00: UNDEFINED()
			return Reg16Reference(value, cache=cache_flag, discard=discard_flag)
	'''

	def encode_string(self, fields, opstr):
		if self._paired():
			regs = try_parse_register_tuple(opstr)
			if regs and len(regs) == 2:
				return self.encode(fields, regs)
		else:
			reg = try_parse_register(opstr)
			if reg:
				return self.encode(fields, reg)

		value = try_parse_integer(opstr)

		if value is None:
			raise Exception('invalid ALUSrcDesc %r' % (opstr,))

		self.encode(fields, Immediate(value))

def try_parse_integer(opstr):
	if opstr in float_immediate_lookup:
		return float_immediate_lookup[opstr]

	try:
		base = 10
		if '0b' in opstr:
			base = 2
		elif '0x' in opstr:
			base = 16
		return int(opstr, base)
	except ValueError:
		return None

assert try_parse_integer('11') == 11
assert try_parse_integer('0b11') == 3
assert try_parse_integer('0x11') == 17
assert try_parse_integer('-11') == -11
assert try_parse_integer('-0b11') == -3
assert try_parse_integer('-0x11') == -17

class ALUSrc64Desc(ALUSrcDesc):
	"Zero-extended 16, 32 or 64 bit source field"

	def _allow64(self):
		return True

class ALUSrc16Desc(ALUSrcDesc):
	"Zero-extended 16 bit source field"

	def _allow32(self):
		return False


@document_operand
class MulSrcDesc(ALUSrcDesc):
	"Sign-extendable 16 or 32 bit source field"

	def __init__(self, name, bit_off, bit_off_ex):
		super().__init__(name, bit_off, bit_off_ex)
		self.add_field(bit_off + 10, 1, self.name + 's')


	def decode(self, fields):
		r = super().decode(fields)
		if not isinstance(r, Register):
			return r

		if fields[self.name + 's'] & 1:
			r.flags.append(SIGN_EXTEND_FLAG)
		return r

	def evaluate_thread(self, fields, corestate, thread):
		r = self.decode(fields)

		value = r.get_thread(corestate, thread)
		size = r.get_bit_size()
		if SIGN_EXTEND_FLAG in r.flags:
			value = sign_extend(value, size)
		return value

	def encode(self, fields, operand):
		super().encode(fields, operand)
		sx = 0
		if isinstance(operand, Register):
			if SIGN_EXTEND_FLAG in operand.flags:
				sx = 1
		fields[self.name + 's'] = sx

	pseudocode = '''
	{name}(value, flags, sx):
		source = ALUSrc(value, flags, max_size=32)
		if sx:
			# Note: 8-bit immediates have already been zero-extended to 16-bit,
			# so do not get sign extended.
			return SignExtendWrapper(source, source.thread_bit_size)
		else:
			return source
	'''

@document_operand
class AddSrcDesc(MulSrcDesc):
	"Sign-extendable 16, 32 or 64 bit source field"

	pseudocode = '''
	{name}(value, flags, sx):
		source = ALUSrc(value, flags, max_size=64)
		if sx:
			# Note: 8-bit immediates have already been zero-extended to 16-bit,
			# so do not get sign extended.
			return SignExtendWrapper(source, source.thread_bit_size)
		else:
			return source
	'''

	def _allow64(self):
		return True

@document_operand
class CmpselSrcDesc(AbstractSrcOperandDesc):
	documentation_extra_arguments = ['Dt']

	def __init__(self, name, bit_off, bit_off_ex):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(bit_off, 6, self.name),
			(bit_off_ex, 2, self.name + 'x')
		])
		self.add_field(bit_off + 6, 3, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		is32 = ((fields['Dt'] & 2) != 0)

		if flags == 0b100:
			return Immediate(value)
		elif flags in (0b001, 0b010, 0b011):
			if is32:
				assert (value & 1) == 0
				r = Reg32(value >> 1)
			else:
				r = Reg16(value)
			return add_hint_modifier(r, flags & 0b11)
		elif flags in (0b110, 0b111):
			if is32:
				assert (value & 1) == 0
				return UReg32(value >> 1)
			else:
				return UReg16(value)
		else:
			print('TODO: ' + format(flags, '04b'))

	pseudocode = '''
	{name}(value, flags, destination_flags):
		if flags == 0b100:
			return BroadcastImmediateReference(value)

		if flags >> 1 == 0b11:
			ureg = value | (flags & 1) << 8
			if destination_flags & 2:
				if ureg & 1: UNDEFINED()
				return BroadcastUReg32Reference(ureg >> 1)
			else:
				return BroadcastUReg16Reference(ureg)

		if flags >> 2 == 1: UNDEFINED()
		if flags & 0b11 == 0b00: UNDEFINED()

		cache_flag   = (flags & 0b11) == 0b10
		discard_flag = (flags & 0b11) == 0b11

		if destination_flags & 2:
			if value & 1: UNDEFINED()
			return Reg32Reference(value >> 1, cache=cache_flag, discard=discard_flag)
		else:
			return Reg16Reference(value, cache=cache_flag, discard=discard_flag)
	'''
	def encode(self, fields, operand):
		flags = 0
		value = 0
		if isinstance(operand, (UReg16, UReg32)):
			flags = 0b110
			if isinstance(operand, UReg16):
				if fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n
			else:
				assert isinstance(operand, UReg32)
				if not fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n << 1
			flags |= (value >> 8) & 1
			value &= 0xFF
		elif isinstance(operand, BaseReg):
			if isinstance(operand, Reg16):
				if fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n
			elif isinstance(operand, Reg32):
				if not fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n << 1
			else:
				raise Exception('invalid CmpselSrcDesc %r' % (operand,))

			if CACHE_FLAG in operand.flags:
				flags = 2
			elif DISCARD_FLAG in operand.flags:
				flags = 3
			else:
				flags = 1
		elif isinstance(operand, Immediate):
			flags = 0b100
			if not 0 <= operand.value < 256:
				raise Exception('out of range immediate %r' % (operand,))
			value = operand.value
		else:
			raise Exception('invalid CmpselSrcDesc %r' % (operand,))

		fields[self.name + 't'] = flags
		fields[self.name] = value

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg:
			return self.encode(fields, reg)

		value = try_parse_integer(opstr)
		if value is None:
			raise Exception('invalid CmpselSrcDesc %r' % (opstr,))

		self.encode(fields, Immediate(value))

@document_operand
class FloatSrcDesc(ALUSrcDesc):
	def __init__(self, name, bit_off, bit_off_ex, bit_off_m=None):
		super().__init__(name, bit_off, bit_off_ex)
		if bit_off_m is None:
			bit_off_m = bit_off + 6 + self._type_size()
		self.add_field(bit_off_m, 2, self.name + 'm')

	def decode_immediate(self, fields, value):
		return Immediate(decode_float_immediate(value))

	def decode(self, fields):
		r = super().decode(fields)
		return add_float_modifier(r, fields[self.name + 'm'])

	def evaluate_thread_float(self, fields, corestate, thread):
		o = self.decode(fields)

		if isinstance(o, Immediate):
			r = fma.f64_to_u64(o.value)
		else:
			bits = o.get_thread(corestate, thread)
			bit_size = o.get_bit_size()
			if bit_size == 16:
				r = fma.f16_to_f64(bits, ftz=False)
			elif bit_size == 32:
				r = fma.f32_to_f64(bits, ftz=True)
			else:
				raise NotImplementedError()

		if ABS_FLAG in o.flags:
			r &= ~(1 << 63)
		if NEGATE_FLAG in o.flags:
			r ^= (1 << 63)

		return r


	pseudocode = '''
	{name}(value, flags, modifier, max_size=32):
		source = ALUSrc(value, flags, max_size)

		if source.is_immediate:
			float = BroadcastRealReference(decode_float_immediate(source))

		elif source.thread_bit_size == 16:
			float = Float16ToRealWrapper(source, flush_to_zero=False)
		elif source.thread_bit_size == 32:
			float = Float32ToRealWrapper(source, flush_to_zero=True)

		if modifier & 0b01: float = FloatAbsoluteValueWrapper(float)
		if modifier & 0b10: float = FloatNegateWrapper(float)

		return float
	'''

	def encode(self, fields, operand):
		super().encode(fields, operand)
		m = 0
		if isinstance(operand, Register):
			if ABS_FLAG in operand.flags:
				m |= 1
			if NEGATE_FLAG in operand.flags:
				m |= 2
		fields[self.name + 'm'] = m


class PairedFloatSrcDesc(FloatSrcDesc):
	# converts r0 <-> r0_r1 and r0h <-> r0h_r1l
	# TODO: not clear is uniform registers supported
	def _paired(self):
		return True

helper_pseudocode = '''

decode_float_immediate(value):
	sign = (value & 0x80) >> 7
	exponent = (value & 0x70) >> 4
	fraction = value & 0xF

	if exponent == 0:
		result = fraction / 64.0
	else:
		fraction = 16.0 + fraction
		exponent -= 7
		result = fraction * (2.0 ** exponent)

	if sign != 0:
		result = -result

	return result

'''

@document_operand
class FloatSrc16Desc(FloatSrcDesc):
	pseudocode = '''
	{name}(value, flags, modifier):
		return FloatSrcDesc(value, flags, modifier, max_size=16)
	'''
	def _type_size(self):
		return 3


class TruthTableDesc(OperandDesc):
	documentation_skip = True

	def __init__(self, name):
		super().__init__(name)

		self.add_field(26, 1, self.name + '0')
		self.add_field(27, 1, self.name + '1')
		self.add_field(38, 1, self.name + '2')
		self.add_field(39, 1, self.name + '3')

	def decode(self, fields):
		return ''.join(str(fields[self.name + str(i)]) for i in range(4))

	def encode_string(self, fields, opstr):
		if not all(i in '01' for i in opstr) or len(opstr) != 4:
			raise Exception('invalid TruthTable %r' % (opstr,))
		for i in range(4):
			fields['tt' + str(i)] = int(opstr[i])

class FieldDesc(OperandDesc):
	def __init__(self, name, x, size=None):
		super().__init__(name)
		if isinstance(x, list):
			subfields = x
			self.size = sum(size for start, size, name in subfields)
			self.add_merged_field(self.name, subfields)
		else:
			start = x
			assert isinstance(start, int)
			assert isinstance(size, int)
			self.size = size
			self.add_field(start, size, self.name)

class IntegerFieldDesc(FieldDesc):
	documentation_skip = True # (because it is what it is)

	def encode_string(self, fields, opstr):
		value = try_parse_integer(opstr)

		if value is None:
			raise Exception('invalid IntegerFieldDesc %r' % (opstr,))

		if value < 0 or value >= (1 << self.size):
			value &= (1 << self.size) - 1
			print('WARNING: encoding out of range IntegerFieldDesc %r (0-%d) as %d' % (opstr, (1 << self.size) - 1, value))

		fields[self.name] = value

class BinaryDesc(IntegerFieldDesc):
	def decode(self, fields):
		return '0b' + format(fields[self.name], '0' + str(self.size) + 'b')

class ImmediateDesc(IntegerFieldDesc):
	def decode(self, fields):
		return fields[self.name]

	def evaluate_thread(self, fields, corestate, thread):
		return fields[self.name]

@document_operand
class Reg32Desc(FieldDesc):
	pseudocode = '''
	{name}(value):
		return Reg32Reference(value)
	'''
	def decode(self, fields):
		return Reg32(fields[self.name])

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg and isinstance(reg, Reg32):
			fields[self.name] = reg.n
		else:
			raise Exception('invalid Reg32Desc %r' % (opstr,))


class EnumDesc(FieldDesc):
	documentation_skip = True

	def __init__(self, name, start, size, values):
		super().__init__(name, start, size)
		self.values = values

	def decode(self, fields):
		v = fields[self.name]
		return self.values.get(v, v)

	def encode_string(self, fields, opstr):
		for k, v in self.values.items():
			if v == opstr:
				fields[self.name] = k
				return

		v = try_parse_integer(opstr)
		if v is not None:
			fields[self.name] = v
			return

		raise Exception('invalid enum %r (%r)' % (opstr, list(self.values.values())))

class ShiftDesc(OperandDesc):
	documentation_no_name = True

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field('s', [
			(39, 1, 's1'),
			(52, 2, 's2'),
		])

	def decode(self, fields):
		shift = fields['s']
		return 'lsl %d' % (shift) if shift else ''

	def encode_string(self, fields, opstr):
		if opstr == '':
			s = 0
		elif opstr.startswith('lsl '):
			s = try_parse_integer(opstr[4:])
			if s is None:
				raise Exception('invalid ShiftDesc %r' % (opstr,))
		else:
			raise Exception('invalid ShiftDesc %r' % (opstr,))
		fields['s'] = s

class MaskDesc(OperandDesc):
	documentation_no_name = True

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(38, 2, self.name + '1'),
			(50, 2, self.name + '2'),
			(63, 1, self.name + '3'),
		])

	def decode(self, fields):
		mask = fields[self.name]
		return 'mask 0x%X' % ((1 << mask) - 1) if mask else ''

	def encode_string(self, fields, opstr):
		if opstr == '':
			fields[self.name] = 0
			return

		if opstr.startswith('mask '):
			mask = try_parse_integer(opstr[len('mask '):])
			b = format(mask + 1, 'b')
			if b.count('1') == 1:
				m = len(b) - 1
				if 0 < m <= 32:
					if m == 32:
						m = 0
					fields[self.name] = m
					return

		raise Exception('invalid MaskDesc %r' % (opstr,))


class BranchOffsetDesc(FieldDesc):
	'''Signed offset in bytes from start of jump instruction (must be even)'''

	documentation_skip = True

	def decode(self, fields):
		v = fields[self.name]
		#assert (v & 1) == 0
		v = sign_extend(v, self.size)
		return RelativeOffset(v)

	def encode_string(self, fields, opstr):
		if opstr.startswith('pc'):
			s = opstr.replace(' ','')
			if s.startswith(('pc-', 'pc+')):
				value = try_parse_integer(s[2:])
				if value is not None:
					masked = value & ((1 << self.size) - 1)
					print('value', value, hex(masked))
					if value != sign_extend(masked, self.size):
						raise Exception('out of range BranchOffsetDesc %r' % (opstr,))
					fields[self.name] = masked
					print(fields)
					return

		# TODO: labels, somehow
		raise Exception('invalid BranchOffsetDesc %r' % (opstr,))


class StackAdjustmentDesc(OperandDesc):
	# maybe memory index desc?
	documentation_no_name = True

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(20, 4, self.name + '1'),
			(32, 4, self.name + '2'),
			(56, 8, self.name + '3'),
		])

	def decode(self, fields):
		return Immediate(sign_extend(fields[self.name], 16))

	def encode_string(self, fields, opstr):
		value = try_parse_integer(opstr)

		if value is None:
			raise Exception('invalid StackAdjustmentDesc %r' % (opstr,))

		masked = value & 0xFFFF
		if sign_extend(masked, 16) != value:
			raise Exception('invalid StackAdjustmentDesc %r (out of range)' % (opstr,))

		fields[self.name] = masked

class StackReg32Desc(OperandDesc):
	# TODO: merge logic with Reg32Desc
	def __init__(self, name, parts):
		super().__init__(name)
		self.add_merged_field(self.name, parts)

	def decode(self, fields):
		return Reg32(fields[self.name] >> 1)

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg and isinstance(reg, Reg32):
			fields[self.name] = reg.n << 1
		else:
			raise Exception('invalid StackReg32Desc %r' % (opstr,))

class BaseConditionDesc(OperandDesc):
	def __init__(self, cc_off=13, cc_n_off=8):
		super().__init__('cc')

		self.add_field(cc_off, 3, 'cc')
		if cc_n_off is not None:
			self.add_field(cc_n_off, 1, 'ccn')

	def encode_string(self, fields, opstr):
		for k, v in self.encodings.items():
			if v == opstr:
				cond = k
				break
		else:
			raise Exception('invalid condition %r' % (opstr,))
		fields['cc'] = cond & 0b111
		if len(self.fields) == 2:
			fields['ccn'] = (cond >> 3)
		elif cond > 0b111:
			raise Exception('invalid condition %r (no ccn fields)' % (opstr,))

	def decode(self, fields):
		v = fields['cc'] | (fields.get('ccn', 0) << 3)
		return self.encodings.get(v, v)

@document_operand
class IConditionDesc(BaseConditionDesc):
	pseudocode = '''
	{name}(value, n=0):
		sign_extend   = (value & 0b100) != 0
		condition     =  value & 0b011
		invert_result = (n != 0)

		if condition == 0b00:
			return IntEqualityComparison(sign_extend, invert_result)
		if condition == 0b01:
			return IntLessThanComparison(sign_extend, invert_result)
		if condition == 0b10:
			return IntGreaterThanComparison(sign_extend, invert_result)
	'''
	def __init__(self, cc_off=13, cc_n_off=8):
		super().__init__(cc_off, cc_n_off)

		self.encodings = {
			0b0000: 'ueq',
			0b0001: 'ult',
			0b0010: 'ugt',
			0b0100: 'seq',
			0b0101: 'slt',
			0b0110: 'sgt',

			0b1000: 'nueq',
			0b1001: 'ugte',
			0b1010: 'ulte',
			0b1100: 'nseq',
			0b1101: 'sgte',
			0b1110: 'slte',
		}


@document_operand
class FConditionDesc(BaseConditionDesc):
	pseudocode = '''
	{name}(condition, n=0):
		invert_result = (n != 0)

		if condition == 0b000:
			return FloatEqualityComparison(invert_result)
		if condition == 0b001:
			return FloatLessThanComparison(invert_result)
		if condition == 0b010:
			return FloatGreaterThanComparison(invert_result)
		if condition == 0b011:
			return FloatLessThanNanLosesComparison(invert_result)
		if condition == 0b101:
			return FloatLessThanOrEqualComparison(invert_result)
		if condition == 0b110:
			return FloatGreaterThanOrEqualComparison(invert_result)
		if condition == 0b111:
			return FloatGreaterThanNanLosesComparison(invert_result)
	'''

	def __init__(self, cc_off=13, cc_n_off=8):
		super().__init__(cc_off, cc_n_off)

		self.encodings = {
			0b000: 'eq',
			0b001: 'lt',
			0b010: 'gt',
			0b011: 'ltn',
			0b101: 'gte',
			0b110: 'lte',
			0b111: 'gtn',

			0b1000: 'neq',
			0b1001: 'nlt',
			0b1011: 'nltn', # unobserved
			0b1010: 'ngt',
			0b1101: 'ngte',
			0b1110: 'nlte',
			0b1111: 'ngtn', # unobserved
		}



class MemoryShiftDesc(OperandDesc):
	documentation_skip = True

	def __init__(self, name):
		super().__init__(name)
		self.add_field(42, 2, self.name)

	def decode(self, fields):
#		effective_shift = fields['s']
#		if effective_shift == 3:
#			effective_shift = 2
#
#		bit_packed = fields['F'] in (8, 12, 13)
#		if bit_packed:
#			effective_shift = 2
#
#		effective_shift += {
#			1: 2, # i16
#			2: 4, # i32
#			3: 2, # i16?
#			6: 2, # unorm16
#			7: 2, # unorm16
#		}.get(fields['F'], 1)
#		return 'lsl %d' % (effective_shift) if effective_shift else ''

		shift = fields[self.name]
		return 'lsl %d' % (shift) if shift else ''

	def encode_string(self, fields, opstr):
		if opstr == '':
			s = 0
		elif opstr.startswith('lsl '):
			s = try_parse_integer(opstr[4:])
			if s is None:
				raise Exception('invalid MemoryShiftDesc %r' % (opstr,))
		else:
			raise Exception('invalid MemoryShiftDesc %r' % (opstr,))
		fields[self.name] = s


@document_operand
class MemoryIndexDesc(OperandDesc):
	pseudocode = '''
	{name}(value, flags):
		if flags != 0:
			return BroadcastImmediateReference(sign_extend(value, 16))
		else:
			if value & 1: UNDEFINED()
			if value >= 0x100: UNDEFINED()
			return Reg32Reference(value >> 1)
	'''

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(20, 4, self.name + 'l'),
			(32, 4, self.name + 'h'),
			(56, 8, self.name + 'x'),
		])
		self.add_field(24, 1, self.name + 't')

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		if flags:
			return sign_extend(value, 16)
		else:
			assert (value & 1) == 0
			assert value < 0x100
			return Reg32(value >> 1)

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if r is not None:
			if isinstance(r, Reg32):
				fields[self.name + 't'] = 0
				fields[self.name] = r.n << 1
				return

		v = try_parse_integer(opstr)
		if v is not None:
			assert 0 <= v < 0x100
			fields[self.name + 't'] = 1
			fields[self.name] = v
			return

		raise Exception('invalid MemoryIndexDesc %r' % (opstr,))

#@document_operand
class ThreadgroupIndexDesc(OperandDesc):
#	pseudocode = '''
#	{name}(value, flags):
#		if flags != 0:
#			return BroadcastImmediateReference(sign_extend(value, 16))
#		else:
#			if value & 1: UNDEFINED()
#			if value >= 0x100: UNDEFINED()
#			return Reg32Reference(value >> 1)
#	'''

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(28, 6, self.name),
			(48, 10, self.name + 'x')
		])
		self.add_field(34, 1, self.name + 't')

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		if flags:
			return sign_extend(value, 16)
		else:
			assert value < 0x100
			return Reg16(value & 0xFF)

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if r is not None:
			if isinstance(r, Reg16):
				fields[self.name + 't'] = 0
				fields[self.name] = r.n << 1
				return

		v = try_parse_integer(opstr)
		if v is not None:
			assert -0x8000 <= v < 0x8000
			fields[self.name + 't'] = 1
			fields[self.name] = v & 0xFFFF
			return

		raise Exception('invalid ThreadgroupIndexDesc %r' % (opstr,))

class AsyncMemoryRegDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(9, 6, self.name),
		])
		self.add_field(8, 1, self.name + 't')

	def decode_impl(self, fields):
		value = fields[self.name]
		flags = fields[self.name + 't']
		if flags:
			return UReg64(value >> 1)
		else:
			return Reg64(value >> 1)

	def decode(self, fields):
		return self.decode_impl(fields)

class AsyncMemoryBaseDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(16, 4, self.name + 'l'),
			(36, 4, self.name + 'h'),
		])
		self.add_field(25, 1, self.name + 't')

	def decode_impl(self, fields):
		value = fields[self.name]
		# Reg64(value) = address?
		# Reg32(value+2) = length
		reg_count = 5 if fields['F'] else 3
		flags = fields[self.name + 't']
		if flags:
			return RegisterTuple((UReg32(value+i) for i in range(reg_count)))
		else:
			return RegisterTuple((Reg32(value+i) for i in range(reg_count)))

	def decode(self, fields):
		return self.decode_impl(fields)

@document_operand
class MemoryBaseDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(16, 4, self.name + 'l'),
			(36, 4, self.name + 'h'),
		])
		self.add_field(27, 1, self.name + 't')

	pseudocode = '''
	{name}(value, flags):
		if value & 1: UNDEFINED()
		if flags != 0:
			return UReg64Reference(value >> 1)
		else:
			return Reg64Reference(value >> 1)
	'''

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		assert (value & 1) == 0
		if flags:
			return UReg64(value >> 1)
		else:
			return Reg64(value >> 1)

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if not isinstance(r, (Reg64, UReg64)):
			raise Exception('invalid MemoryBaseDesc %r' % (opstr,))

		fields[self.name + 't'] = 1 if isinstance(r, UReg64) else 0
		fields[self.name] = r.n << 1

class SampleMaskDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(42, 6, self.name),
			(56, 2, self.name + 'x'),
		])
		self.add_field(22, 2, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		if flags == 0b0:
			return Immediate(value)
		elif flags == 0b1:
			return Reg16(value)
		else:
			assert(0)

	def encode_string(self, fields, opstr):
		assert(0)

class MemoryRegDesc(OperandDesc):
	def __init__(self, name, off=10, offx=40, offt=49):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		self.add_field(offt, 1, self.name + 't')

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]

		count = bin(fields['mask']).count('1')

		if flags == 0b0:
			return RegisterTuple(Reg16(value + i) for i in range(count))
		else:
			return RegisterTuple(Reg32((value >> 1) + i) for i in range(count))

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		regs = [try_parse_register(i) for i in opstr.split('_')]
		if regs and all(isinstance(r, Reg32) for r in regs):
			flags = 1
			value = regs[0].n << 1
		elif regs and all(isinstance(r, Reg16) for r in regs):
			flags = 0
			value = regs[0].n
		else:
			raise Exception('invalid MemoryRegDesc %r' % (opstr,))

		for i in range(1, len(regs)):
			if regs[i].n != regs[i-1].n + 1:
				raise Exception('invalid MemoryRegDesc %r (must be consecutive)' % (opstr,))

		if not 0 < len(regs) <= 4:
			raise Exception('invalid MemoryRegDesc %r (1-4 values)' % (opstr,))

		#fields['mask'] = (1 << len(regs)) - 1
		fields[self.name] = value
		fields[self.name + 't'] = flags


class ThreadgroupMemoryRegDesc(OperandDesc):
	# TODO: exactly the same as MemoryRegDesc except for the offsets?
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(9, 6, self.name),
			(60, 2, self.name + 'x'),
		])
		self.add_field(8, 1, self.name + 't')

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']

		value = fields[self.name]

		count = bin(fields['mask']).count('1')

		if flags == 0b0:
			return RegisterTuple(Reg16(value + i) for i in range(count))
		else:
			return RegisterTuple(Reg32((value >> 1) + i) for i in range(count))

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		regs = [try_parse_register(i) for i in opstr.split('_')]
		if regs and all(isinstance(r, Reg32) for r in regs):
			flags = 0b1
			value = regs[0].n << 1
		elif regs and all(isinstance(r, Reg16) for r in regs):
			flags = 0b0
			value = regs[0].n
		else:
			raise Exception('invalid ThreadgroupMemoryRegDesc %r' % (opstr,))

		for i in range(1, len(regs)):
			if regs[i].n != regs[i-1].n + 1:
				raise Exception('invalid ThreadgroupMemoryRegDesc %r (must be consecutive)' % (opstr,))

		if not 0 < len(regs) <= 4:
			raise Exception('invalid ThreadgroupMemoryRegDesc %r (1-4 values)' % (opstr,))

		fields['mask'] = (1 << len(regs)) - 1
		fields[self.name] = value
		fields[self.name + 't'] = flags

#@document_operand
class ThreadgroupMemoryBaseDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(16, 6, self.name),
			(58, 2, self.name + 'x'),
		])
		self.add_field(22, 2, self.name + 't')

#	pseudocode = '''
#	{name}(value, flags):
#		if value & 1: UNDEFINED()
#		if flags != 0:
#			return UReg64Reference(value >> 1)
#		else:
#			return Reg64Reference(value >> 1)
#	'''

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		if flags == 0b00:
			return Reg16(value)
		elif flags == 0b10:
			return Immediate(0)
		else:
			return UReg16(value | ((flags >> 1) << 8))

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

#	def encode_string(self, fields, opstr):
#		r = try_parse_register(opstr)
#		if not isinstance(r, Reg16): # (Reg64, UReg64)):
#			raise Exception('invalid ThreadgroupMemoryBaseDesc %r' % (opstr,))
#
#		fields[self.name + 't'] = 0 if isinstance(r, Reg16) else 0b10
#		fields[self.name] = r.n << 1

class SReg32Desc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(start, 6, self.name),
			(start_ex, 2, self.name + 'x'),
		])

	pseudocode = '''
	{name}(value):
		return SpecialRegister(value)
	'''
	def decode(self, fields):
		return SReg32(fields[self.name])

	def encode_string(self, fields, opstr):
		s = opstr
		if ' (' in s and ')' in s:
			s = s.split(' (')[0]
		if s.startswith('sr'):
			try:
				v = int(s[2:])
			except ValueError:
				raise Exception('invalid SReg32Desc %r' % (opstr,))
			if 0 <= v < 256:
				fields[self.name] = v
				return
		raise Exception('invalid SReg32Desc %r' % (opstr,))



class ExReg32Desc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)

		# TODO: this ignores the low bit. Kinda confusing?
		self.add_merged_field(self.name, [
			(start, 5, self.name),
			(start_ex, 2, self.name + 'x'),
		])

	def decode(self, fields):
		v = fields[self.name]
		return Reg32(v)


class ExReg64Desc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)

		# TODO: this ignores the low bit. Kinda confusing?
		self.add_merged_field(self.name, [
			(start, 5, self.name),
			(start_ex, 2, self.name + 'x'),
		])

	def decode(self, fields):
		v = fields[self.name]
		return Reg64(v)



class ExReg16Desc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)

		self.add_merged_field(self.name, [
			(start, 6, self.name),
			(start_ex, 2, self.name + 'x'),
		])

	def decode(self, fields):
		v = fields[self.name]
		return Reg16(v)



instruction_descriptors = []
_instruction_descriptor_names = set()
def register(cls):
	assert cls.__name__ not in _instruction_descriptor_names, 'duplicate %r' % (cls.__name__,)
	_instruction_descriptor_names.add(cls.__name__)
	instruction_descriptors.append(cls())
	return cls

class MaskedInstructionDesc(InstructionDesc):
	def exec(self, instr, corestate):
		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				self.exec_thread(instr, corestate, thread)

	def exec_thread(self, instr, corestate, thread):
		assert False, "TODO"

@register
class MovImm16InstructionDesc(MaskedInstructionDesc):
	documentation_begin_group = 'Move Instructions'
	documentation_name = 'Move 16-bit Immediate'
	def __init__(self):
		super().__init__('mov_imm', size=(4, 6))

		self.add_constant(0, 7, 0b1100010)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_constant(8, 1, 0) # TODO: this is within dst

		self.add_operand(ImmediateDesc('imm16', 16, 16))

	def fields_for_mnem(self, mnem, operand_strings):
		if (mnem == self.name and
			len(operand_strings) and isinstance(try_parse_register(operand_strings[0]), Reg16) and
			try_parse_integer(operand_strings[1]) is not None):
			return {}

	pseudocode = '''
	D.broadcast_to_active(imm16)
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['imm16'].evaluate_thread(fields, corestate, thread)
		self.operands['D'].set_thread(fields, corestate, thread, a)

@register
class MovImm32InstructionDesc(MaskedInstructionDesc):
	documentation_name = 'Move 32-bit Immediate'
	def __init__(self):
		super().__init__('mov_imm', size=(6, 8))
		self.add_constant(0, 7, 0b1100010)

		self.add_operand(ALUDstDesc('D', 60))
		self.add_constant(8, 1, 1) # TODO: this is within dst

		self.add_operand(ImmediateDesc('imm32', 16, 32))

	def fields_for_mnem(self, mnem, operand_strings):
		# TODO: should have "ALUDst32OnlyDesc" / "ALUDst16OnlyDesc" that fall through to other mnemonics
		# this kind of fallthrough would allow "mov" mnemonic to move from SR too.

		if (mnem == self.name and
			len(operand_strings) and isinstance(try_parse_register(operand_strings[0]), Reg32) and
			try_parse_integer(operand_strings[1]) is not None):
			return {}

	pseudocode = '''
	D.broadcast_to_active(imm32)
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['imm32'].evaluate_thread(fields, corestate, thread)
		self.operands['D'].set_thread(fields, corestate, thread, a)

@register
class MovFromSrInstructionDesc(MaskedInstructionDesc):
	#documentation_begin_group = 'Miscellaneous Instructions'
	documentation_name = 'Move From Special Register'
	def __init__(self):
		super().__init__('get_sr', size=4)

		self.add_constant(0, 7, 0b1110010)
		self.add_constant(15, 1, 0)
		self.add_operand(ALUDstDesc('D', 28))
		self.add_operand(SReg32Desc('SR', 16, 26))

	pseudocode = '''
	for each active thread:
		D[thread] = SR.read(thread)
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		if fields['SR'] == 80:
			self.operands['D'].set_thread(fields, corestate, thread, thread)
		else:
			assert False, 'TODO'


def icompare_thread(desc, fields, corestate, thread):
	a = desc.operands['A'].evaluate_thread(fields, corestate, thread)
	b = desc.operands['B'].evaluate_thread(fields, corestate, thread)

	a_size = desc.operands['A'].get_bit_size(fields)
	b_size = desc.operands['B'].get_bit_size(fields)

	cc = fields['cc']

	if cc & 0b100:
		a = sign_extend(a, a_size)
		b = sign_extend(b, b_size)

	if (cc & 0b11) == 0b00:
		comparison = (a == b)
	elif (cc & 0b11) == 0b01:
		comparison = (a < b)
	elif (cc & 0b11) == 0b10:
		comparison = (a > b)
	else:
		# TODO
		raise NotImplementedError()
		comparison = False

	if fields.get('ccn', False):
		comparison = not comparison

	return comparison

def fcompare_thread(desc, fields, corestate, thread):
	a64 = desc.operands['A'].evaluate_thread_float(fields, corestate, thread)
	b64 = desc.operands['B'].evaluate_thread_float(fields, corestate, thread)

	a = fma.u64_to_f64(a64)
	b = fma.u64_to_f64(b64)

	cc = fields['cc']
	if cc == 0b000:
		comparison = (a == b)
	elif cc == 0b001:
		comparison = (a < b)
	elif cc == 0b010:
		comparison = (a > b)
	elif cc == 0b101:
		comparison = (a >= b)
	elif cc == 0b110:
		comparison = (a <= b)
	else:
		# TODO
		raise NotImplementedError()
		comparison = False

	if fields.get('ccn', False):
		comparison = not comparison

	return comparison


class SaturatableInstructionDesc(MaskedInstructionDesc):
	def fields_to_mnem_suffix(self, fields):
		if fields['S']:
			return '.sat'
		return ''

	def fields_for_mnem(self, mnem, operand_strings):
		S = 0
		if mnem.endswith('.sat'):
			mnem = mnem[:-4]
			S = 1
		fields = self.fields_for_mnem_base(mnem)
		if fields is not None:
			fields['S'] = S
		return fields

	def fields_for_mnem_base(self, mnem):
		if mnem == self.name: return {}

class ISaturatableInstructionDesc(SaturatableInstructionDesc):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.add_field(6, 1, 'S')

	def saturate(self, result, dest_size, signed):
		if signed:
			minimum = -(1 << (dest_size-1))
			maximum = (1 << (dest_size-1)) - 1
		else:
			minimum = 0
			maximum = (1 << dest_size) - 1
		if result < minimum:
			result = minimum
		elif result > maximum:
			result = maximum
		return result



@register
class IAddInstructionDesc(ISaturatableInstructionDesc):
	documentation_begin_group = 'Integer Arithmetic Instructions'

	documentation_name = 'Integer Add or Subtract'

	def __init__(self):
		super().__init__('iadd', size=8)
		self.add_constant(0, 6, 0b001110)
		self.add_constant(15, 1, 0)

		self.add_field(27, 1, 'N')

		self.add_operand(ALUDst64Desc('D', 44))
		self.add_operand(AddSrcDesc('A', 16, 42))
		self.add_operand(AddSrcDesc('B', 28, 40))
		self.add_operand(ShiftDesc('shift'))

	def fields_to_mnem_base(self, fields):
		return 'isub' if fields['N'] else 'iadd'

	def fields_for_mnem_base(self, mnem):
		if mnem == 'isub': return {'N': 1}
		if mnem == 'iadd': return {'N': 0}

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		saturating = (S == 1 and shift == 0 and A.thread_bit_size <= 32 and
		              B.thread_bit_size <= 32 and D.thread_bit_size <= 32)

		if N == 1:
			b = -b

		if shift < 5:
			b <<= shift
		else:
			b = 0

		result = a + b

		if saturating:
			signed = (As == 1 or Bs == 1)
			result = saturate_integer(result, D.thread_bit_size, signed)

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)
		b = self.operands['B'].evaluate_thread(fields, corestate, thread)
		shift = fields['s']

		a_size = self.operands['A'].get_bit_size(fields)
		b_size = self.operands['B'].get_bit_size(fields)
		dest_size = self.operands['D'].get_bit_size(fields)

		saturate = (fields['S'] and shift == 0 and a_size <= 32 and b_size <= 32 and dest_size <= 32)

		if fields['N']:
			b = -b

		if shift < 5:
			b <<= shift
		else:
			b = 0

		result = a + b

		if saturate:
			signed = fields['As'] or fields['Bs']
			result = self.saturate(result, dest_size, signed)

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class IMAddInstructionDesc(ISaturatableInstructionDesc):
	documentation_name = 'Integer Multiply-Add or Subtract'

	def __init__(self):
		super().__init__('imadd', size=8)
		self.add_constant(0, 6, 0b011110)
		self.add_constant(15, 1, 0)

		self.add_field(27, 1, 'N')

		self.add_operand(ALUDst64Desc('D', 60))
		self.add_operand(MulSrcDesc('A', 16, 58))
		self.add_operand(MulSrcDesc('B', 28, 56))
		self.add_operand(AddSrcDesc('C', 40, 54))
		self.add_operand(ShiftDesc('shift'))

	def fields_to_mnem_base(self, fields):
		return 'imsub' if fields['N'] else 'imadd'

	def fields_for_mnem_base(self, mnem):
		if mnem == 'imsub': return {'N': 1}
		if mnem == 'imadd': return {'N': 0}

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]
		c = C[thread]

		saturating = (S == 1 and shift == 0 and C.thread_bit_size <= 32 and
		              D.thread_bit_size <= 32)

		if N == 1:
			c = -c

		if shift < 5:
			c <<= shift
		else:
			c = 0

		result = a * b + c

		if saturating:
			signed = (As == 1 or Bs == 1 or Cs == 1)
			result = saturate_integer(result, D.thread_bit_size, signed)

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)
		b = self.operands['B'].evaluate_thread(fields, corestate, thread)
		c = self.operands['C'].evaluate_thread(fields, corestate, thread)

		shift = fields['s']

		a_size = self.operands['A'].get_bit_size(fields)
		b_size = self.operands['B'].get_bit_size(fields)
		c_size = self.operands['C'].get_bit_size(fields)
		dest_size = self.operands['D'].get_bit_size(fields)

		saturate = (fields['S'] and shift == 0 and c_size <= 32 and dest_size <= 32)

		if fields['N']:
			c = -c

		if shift < 5:
			c <<= shift
		else:
			c = 0

		result = a * b + c

		if saturate:
			signed = fields['As'] or fields['Bs'] or fields['Cs']
			result = self.saturate(result, dest_size, signed)

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class ConvertInstructionDesc(InstructionDesc):
	def __init__(self):
		# TODO
		super().__init__('convert', size=6)
		self.add_constant(0, 7, 0b0111110)
		self.add_constant(15, 1, 1)

		self.add_constant(38, 2, 0)
		self.add_constant(42, 2, 0) # extra mode bits

		# TODO: this should probably be a ALUSrcDesc - seems it can come from a register
		# not sure when you'd want that though?
		self.add_operand(EnumDesc('mode', 16, 6, {
			0: 'u8_to_f',
			1: 's8_to_f',
			4: 'f_to_u16',
			5: 'f_to_s16',
			6: 'u16_to_f',
			7: 's16_to_f',
			8: 'f_to_u32',
			9: 'f_to_s32',
			10: 'u32_to_f',
			11: 's32_to_f',
		}))
		self.add_constant(22, 4, 0) # flags for the mode (0 = immediate)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(ALUSrcDesc('src', 28, 40))

		# TODO: confirm this (based on observed values only, Metal does
		# not allow setting round mode). If this is true, 2 and 3 are
		# likely round to negative/positive infinity respectively?
		self.add_operand(EnumDesc('round', 26, 2, {
			0: 'rtz', # round to zero
			1: 'rte' # round to nearest
		}))





class BaseShiftInstructionDesc(MaskedInstructionDesc):
	def __init__(self, name):
		super().__init__(name, size=8)
		self.add_constant(0, 7, 0x2E)

		self.add_operand(ALUDstDesc('D', 60))
		self.add_operand(ALUSrcDesc('A', 16, 58))
		self.add_operand(ALUSrcDesc('B', 28, 56))

	pseudocode_template = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		shift_amount = (b & 0x7F)

		{expr}

		D[thread] = result
	'''

class BaseBitfieldInstructionDesc(BaseShiftInstructionDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_operand(ALUSrcDesc('C', 40, 54))
		self.add_operand(MaskDesc('m'))

	pseudocode_template = '''
	for each active thread:
		a = A[thread]
		b = B[thread]
		c = C[thread]

		shift_amount = (c & 0x7F)

		if m == 0:
			mask = 0xFFFFFFFF
		else:
			mask = (1 << m) - 1

		{expr}

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)
		b = self.operands['B'].evaluate_thread(fields, corestate, thread)
		c = self.operands['C'].evaluate_thread(fields, corestate, thread)
		m = fields['m']

		shift_amount = (c & 0x7F)

		mask = (1 << m) - 1 if m else 0xFFFFFFFF

		result = self.bitfield_operation(a, b, shift_amount, mask)

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class BitfieldInsertInstructionDesc(BaseBitfieldInstructionDesc):
	documentation_begin_group = 'Shift/Bitfield Instructions'

	documentation_name = 'Bitfield Insert/Shift Left'

	def __init__(self):
		super().__init__('bfi')
		self.add_constant(15, 1, 0)
		self.add_constant(26, 2, 0)

	pseudocode = BaseBitfieldInstructionDesc.pseudocode_template.format(
		expr='result = (a & ~(mask << shift_amount)) | ((b & mask) << shift_amount)'
	)

	def bitfield_operation(self, a, b, shift_amount, mask):
		# TODO:
		# possible alias: shl (shift left) if m is 0 and a is 0
		return (a & ~(mask << shift_amount)) | ((b & mask) << shift_amount)


@register
class BitfieldExtractInstructionDesc(BaseBitfieldInstructionDesc):
	documentation_name = 'Bitfield Extract and Insert Low/Shift Right'

	def __init__(self):
		super().__init__('bfeil')
		self.add_constant(15, 1, 1)
		self.add_constant(26, 2, 0)

	pseudocode = BaseBitfieldInstructionDesc.pseudocode_template.format(
		expr='result = (a & ~mask) | ((b >> shift_amount) & mask)'
	)

	def bitfield_operation(self, a, b, shift_amount, mask):
		# TODO:
		# possible alias: bfe (bit field extract) if a = 0
		# possible alias: shr if a = 0 and m = 0
		return (a & ~mask) | ((b >> shift_amount) & mask)


@register
class ExtractInstructionDesc(BaseBitfieldInstructionDesc):
	documentation_name = 'Extract From Register Pair'

	def __init__(self):
		super().__init__('extr')
		self.add_constant(15, 1, 0)
		self.add_constant(26, 2, 1)

	pseudocode = BaseBitfieldInstructionDesc.pseudocode_template.format(
		expr='result = (((b << 32) | a) >> shift_amount) & mask'
	)

	def bitfield_operation(self, a, b, shift_amount, mask):
		# TODO:
		# possible alias: ror (rotate right) if a = b and m = 0
		# possible alias: shr64 (64-bit shift right) if m = 0
		return (((b << 32) | a) >> shift_amount) & mask



@register
class ShlhiInstructionDesc(BaseBitfieldInstructionDesc):
	documentation_name = 'Shift Left High and Insert'

	def __init__(self):
		super().__init__('shlhi')
		self.add_constant(15, 1, 0)
		self.add_constant(26, 2, 2)

	pseudocode = BaseBitfieldInstructionDesc.pseudocode_template.format(
		expr='''
		shifted_mask = mask << max(shift_amount-32, 0)
		result = (((b << shift_amount) >> 32) & shifted_mask) | (a & ~shifted_mask)
		'''.strip()
	)

	def bitfield_operation(self, a, b, shift_amount, mask):
		# shlhi (shift left high, insert)
		shifted_mask = mask << max(shift_amount-32, 0)
		return (((b << shift_amount) >> 32) & shifted_mask) | (a & ~shifted_mask)


@register
class ShrhiInstructionDesc(BaseBitfieldInstructionDesc):
	documentation_name = 'Shift Right High and Insert'

	def __init__(self):
		super().__init__('shrhi')
		self.add_constant(15, 1, 1)
		self.add_constant(26, 2, 2)

	pseudocode = BaseBitfieldInstructionDesc.pseudocode_template.format(
		expr='''
		shifted_mask = (mask << 32) >> min(shift_amount, 32)
		result = (((b << 32) >> shift_amount) & shifted_mask) | (a & ~shifted_mask)
		'''.strip()
	)

	def bitfield_operation(self, a, b, shift_amount, mask):
		# shlhi (shift left high, insert)
		shifted_mask = (mask << 32) >> min(shift_amount, 32)
		return (((b << 32) >> shift_amount) & shifted_mask) | (a & ~shifted_mask)


@register
class ArithmeticShiftRightInstructionDesc(BaseShiftInstructionDesc):
	documentation_name = 'Arithmetic Shift Right'

	def __init__(self):
		super().__init__('asr')
		self.add_constant(15, 1, 1)
		self.add_constant(26, 2, 1)

	pseudocode = BaseShiftInstructionDesc.pseudocode_template.format(
		expr='''
		result = sign_extend(a, A.thread_bit_size) >> shift_amount
		'''.strip()
	)

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)
		b = self.operands['B'].evaluate_thread(fields, corestate, thread)

		a_size = self.operands['A'].get_bit_size(fields)

		shift_amount = (b & 0x7F)

		result = sign_extend(a, a_size) >> (b & 0x7F)

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class ArithmeticShiftRightHighInstructionDesc(BaseShiftInstructionDesc):
	documentation_name = 'Arithmetic Shift Right High'

	def __init__(self):
		super().__init__('asrh')
		self.add_constant(15, 1, 1)
		self.add_constant(26, 2, 3)

	pseudocode = BaseShiftInstructionDesc.pseudocode_template.format(
		expr='''
		result = (sign_extend(a, A.thread_bit_size) << 32) >> shift_amount
		'''.strip()
	)

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)
		b = self.operands['B'].evaluate_thread(fields, corestate, thread)

		a_size = self.operands['A'].get_bit_size(fields)

		shift_amount = (b & 0x7F)

		result = (sign_extend(a, a_size) << 32) >> (b & 0x7F)

		self.operands['D'].set_thread(fields, corestate, thread, result)

class IUnaryInstructionDesc(MaskedInstructionDesc):
	def __init__(self, name):
		super().__init__(name, size=6)
		self.add_constant(0, 7, 0b0111110)
		self.add_constant(15, 1, 0)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(ALUSrcDesc('A', 16, 42))

		self.add_constant(28, 10, 0)


@register
class BitopInstructionDesc(MaskedInstructionDesc):
	documentation_begin_group = 'Bit Manipulation Instructions'

	documentation_name = 'Bitwise Operation'

	def __init__(self):
		# TODO: is there a short encoding?
		# TODO: break into mnemonics/aliases?

		super().__init__('bitop', size=6)

		self.add_constant(0, 7, 0x7E)
		self.add_constant(15, 1, 0)

		self.add_operand(TruthTableDesc('tt'))
		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(ALUSrcDesc('A', 16, 42))
		self.add_operand(ALUSrcDesc('B', 28, 40))

		self.binary_aliases = {
			'0001': 'and',
			'0111': 'or',
			'0110': 'xor',
			'1110': 'nand',
			'1000': 'nor',
		}
		self.unary_aliases = {
			'1010': 'not',
			'0101': 'mov',
		}
		self.aliases = set(self.binary_aliases.values()) | set(self.unary_aliases.values())

	def fields_for_mnem(self, mnem, operand_strings):
		if mnem in ('bitop', 'not') or mnem in self.aliases:
			return {}

	def rewrite_operands_strings(self, mnem, operand_strings):
		for k, v in self.binary_aliases.items():
			if v == mnem:
				return [k] + operand_strings

		for k, v in self.unary_aliases.items():
			if v == mnem:
				return [k] + operand_strings + ['0']

		assert mnem == 'bitop'
		return operand_strings

	def map_to_alias(self, mnem, operands):
		tt = operands[0]
		alias = self.binary_aliases.get(tt)
		if alias:
			return alias, operands[1:]

		if str(operands[3]) == '0':
			alias = self.unary_aliases.get(tt)
			if alias:
				return alias, operands[1:3]

		if tt in ('0011', '1100'):
			return 'bitop_mov_a', operands

		return mnem, operands

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		if tt0 == tt1 and tt2 == tt3 and tt0 != tt2:
			UNDEFINED()
			result = a
		else:
			result = 0
			if tt0: result |= ~a & ~b
			if tt1: result |=  a & ~b
			if tt2: result |= ~a &  b
			if tt3: result |=  a &  b

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)
		b = self.operands['B'].evaluate_thread(fields, corestate, thread)

		tt = tuple(fields['tt%d' % i] for i in range(4))

		# tt0 == tt1 and tt2 == tt3 and tt0 != tt2  # (0011/1100)
		if tt == (0, 0, 1, 1) or tt == (1, 1, 0, 0):
			result = a
		else:
			result = 0
			if tt[0]: result |= ~a & ~b
			if tt[1]: result |=  a & ~b
			if tt[2]: result |= ~a &  b
			if tt[3]: result |=  a &  b

		self.operands['D'].set_thread(fields, corestate, thread, result)


@register
class BitReverseInstructionDesc(IUnaryInstructionDesc):
	documentation_name = 'Reverse Bits'

	def __init__(self):
		super().__init__('bitrev')
		self.add_constant(38, 2, 0b00)
		self.add_constant(26, 2, 0b01)

	pseudocode = '''
	for each active thread:
		a = A[thread]

		result = 0

		i = 0
		while i < 32:
			if a & (1 << i):
				result |= (1 << (31-i))

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)

		result = 0
		for i in range(32):
			if a & (1 << i):
				result |= (1 << (31-i))

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class PopCountInstructionDesc(IUnaryInstructionDesc):
	documentation_name = 'Population Count'

	def __init__(self):
		super().__init__('popcount')
		self.add_constant(38, 2, 0b00)
		self.add_constant(26, 2, 0b10)

	pseudocode = '''
	for each active thread:
		a = A[thread]

		result = 0

		i = 0
		while i < 32:
			if a & (1 << i):
				result += 1

		D[thread] = result
	'''
	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)

		result = 0
		for i in range(32):
			if a & (1 << i):
				result += 1

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class FindFirstSetInstructionDesc(IUnaryInstructionDesc):
	documentation_name = 'Find First Set'
	# find leading set-bit (clz(x) = 31 - fls(x))

	def __init__(self):
		super().__init__('ffs')
		self.add_constant(38, 2, 0b00)
		self.add_constant(26, 2, 0b11)

	pseudocode = '''
	for each active thread:
		a = A[thread]

		result = -1

		i = 31
		while i >= 0:
			if a & (1 << i):
				result = i
				break
			i -= 1

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a = self.operands['A'].evaluate_thread(fields, corestate, thread)

		result = -1
		for i in range(32, -1, -1):
			if a & (1 << i):
				result = i
				break

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class UnknownIUnaryInstructionDesc(IUnaryInstructionDesc):
	documentation_skip = True

	def __init__(self):
		super().__init__('iunop')
		self.add_operand(BinaryDesc('op1', 38, 2))
		self.add_operand(BinaryDesc('op2', 26, 2))


class FSaturatableInstructionDesc(SaturatableInstructionDesc):
	def saturate_and_set_thread_result(self, fields, corestate, thread, result64):
		if fields['S']:
			result64 = fma.saturate64(result64)

		dest_size = self.operands['D'].get_bit_size(fields)

		result = fma.f64_to_f16(result64, ftz=False) if dest_size == 16 else fma.f64_to_f32(result64, ftz=True)

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class FMAdd32InstructionDesc(FSaturatableInstructionDesc):
	documentation_begin_group = 'Floating-Point Arithmetic'

	documentation_name =  'Floating-Point Fused Multiply-Add'

	def __init__(self):
		super().__init__('fmadd32', size=(6, 8))
		self.add_constant(0, 6, 0b111010)
		self.add_operand(FloatDstDesc('D', 60))

		self.add_operand(FloatSrcDesc('A', 16, 58))
		self.add_operand(FloatSrcDesc('B', 28, 56))
		self.add_operand(FloatSrcDesc('C', 40, 54))

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]
		c = C[thread]

		result = fused_multiply_add(a, b, c)

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a64 = self.operands['A'].evaluate_thread_float(fields, corestate, thread)
		b64 = self.operands['B'].evaluate_thread_float(fields, corestate, thread)
		c64 = self.operands['C'].evaluate_thread_float(fields, corestate, thread)

		result64 = fma.bfma64(a64, b64, c64, rounding=fma.ROUND_TO_ODD)

		self.saturate_and_set_thread_result(fields, corestate, thread, result64)

@register
class FMAdd16InstructionDesc(FSaturatableInstructionDesc):
	documentation_name = 'Half Precision Floating-Point Fused Multiply-Add'

	def __init__(self):
		super().__init__('fmadd16', size=(6,8))
		self.add_constant(0, 6, 0b110110)

		# TODO: what if dest isn't 16-bit?
		self.add_operand(FloatDst16Desc('D', 60))

		self.add_operand(FloatSrc16Desc('A', 16, 58))
		self.add_operand(FloatSrc16Desc('B', 28, 56))
		self.add_operand(FloatSrc16Desc('C', 40, 54))

	pseudocode = FMAdd32InstructionDesc.pseudocode

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a64 = self.operands['A'].evaluate_thread_float(fields, corestate, thread)
		b64 = self.operands['B'].evaluate_thread_float(fields, corestate, thread)
		c64 = self.operands['C'].evaluate_thread_float(fields, corestate, thread)

		result64 = fma.bfma64(a64, b64, c64, rounding=fma.ROUND_TO_ODD)

		self.saturate_and_set_thread_result(fields, corestate, thread, result64)

class FBinaryInstructionDesc(FSaturatableInstructionDesc):
	def __init__(self, name, opcode):
		super().__init__(name, size=(4,6))
		self.add_constant(0, 6, opcode)

		self.add_operand(FloatDstDesc('D', 44))

		self.add_operand(FloatSrcDesc('A', 16, 42))
		self.add_operand(FloatSrcDesc('B', 28, 40))

class F16BinaryInstructionDesc(FSaturatableInstructionDesc):
	def __init__(self, name, opcode):
		super().__init__(name, size=(4,6))
		self.add_constant(0, 6, opcode)

		# TODO: what if dest isn't 16-bit?
		self.add_operand(FloatDst16Desc('D', 44))

		self.add_operand(FloatSrc16Desc('A', 16, 42))
		self.add_operand(FloatSrc16Desc('B', 28, 40))

@register
class FAdd32InstructionDesc(FBinaryInstructionDesc):
	documentation_name = 'Floating-Point Add'

	def __init__(self):
		super().__init__('fadd32', 0b101010)

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		result = fused_multiply_add(a, 1.0, b)

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a64 = self.operands['A'].evaluate_thread_float(fields, corestate, thread)
		b64 = self.operands['B'].evaluate_thread_float(fields, corestate, thread)

		result64 = fma.bfma64(a64, fma.F64_ONE, b64, rounding=fma.ROUND_TO_ODD)

		self.saturate_and_set_thread_result(fields, corestate, thread, result64)

@register
class FAdd16InstructionDesc(F16BinaryInstructionDesc):
	documentation_name = 'Half Precision Floating-Point Add'

	def __init__(self):
		super().__init__('fadd16', 0b100110)

	pseudocode = FAdd32InstructionDesc.pseudocode

	def exec_thread(self, instr, corestate, thread):
		# TODO: test
		fields = dict(self.decode_fields(instr))

		a64 = self.operands['A'].evaluate_thread_float(fields, corestate, thread)
		b64 = self.operands['B'].evaluate_thread_float(fields, corestate, thread)

		result64 = fma.bfma64(a64, fma.F64_ONE, b64, rounding=fma.ROUND_TO_ODD)

		self.saturate_and_set_thread_result(fields, corestate, thread, result64)

@register
class FMul32InstructionDesc(FBinaryInstructionDesc):
	documentation_name = 'Floating-Point Multiply'

	def __init__(self):
		super().__init__('fmul32', 0b011010)

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		result = fused_multiply_add(a, b, 0.0)

		D[thread] = result
	'''

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		a64 = self.operands['A'].evaluate_thread_float(fields, corestate, thread)
		b64 = self.operands['B'].evaluate_thread_float(fields, corestate, thread)

		result64 = fma.bfma64(a64, b64, 0, rounding=fma.ROUND_TO_ODD)

		self.saturate_and_set_thread_result(fields, corestate, thread, result64)

@register
class FMul16InstructionDesc(F16BinaryInstructionDesc):
	documentation_name = 'Half Precision Floating-Point Multiply'

	def __init__(self):
		super().__init__('fmul16', 0b010110)

	pseudocode = FMul32InstructionDesc.pseudocode

	def exec_thread(self, instr, corestate, thread):
		# TODO: test
		fields = dict(self.decode_fields(instr))

		a64 = self.operands['A'].evaluate_thread_float(fields, corestate, thread)
		b64 = self.operands['B'].evaluate_thread_float(fields, corestate, thread)

		result64 = fma.bfma64(a64, b64, 0, rounding=fma.ROUND_TO_ODD)

		self.saturate_and_set_thread_result(fields, corestate, thread, result64)

class FUnaryInstructionDesc(FSaturatableInstructionDesc):
	def __init__(self, name):
		super().__init__(name, size=(4, 6))
		self.add_constant(0, 6, 0b001010)

		self.add_operand(FloatDstDesc('D', 44))
		self.add_operand(FloatSrcDesc('A', 16, 42))

		self.add_constant(34, 8, 0)

	pseudocode_template = '''
	for each active thread:
		D[thread] = {name}(A[thread])
	'''
@register
class FloorInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('floor')
		self.add_constant(28, 6, 0b000000)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='floor')


@register
class CeilInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('ceil')
		self.add_constant(28, 6, 0b010000)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='ceil')

@register
class TruncInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('trunc')
		self.add_constant(28, 6, 0b100000)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='trunc')

@register
class RintInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('rint')
		self.add_constant(28, 6, 0b110000)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='rint')

@register
class ReciprocalInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('rcp')
		self.add_constant(28, 6, 0b001000)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='reciprocal')

@register
class RsqrtInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('rsqrt')
		self.add_constant(28, 6, 0b001001)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='rsqrt')

@register
class RsqrtSpecialInstructionDesc(FUnaryInstructionDesc):
	documentation_html = '''
	<p>
	<code>rsqrt_special</code> can be used to implement fast <code>sqrt</code> as
	<code>rsqrt_special(x) * x</code>, by handling special-cases differently.
	</p>
	'''
	def __init__(self):
		super().__init__('rsqrt_special')
		self.add_constant(28, 6, 0b000001)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='rsqrt_special')

@register
class SinPt1InstructionDesc(FUnaryInstructionDesc):
	documentation_html = '''
	<p>
	<code>sin_pt_1</code> is used together with <code>sin_pt_2</code> and
	supporting ALU to compute the sine function. sin_pt_1 takes an angle
	around the circle in the interval [0, 4) and produces an intermediate
	result. This intermediate result is then passed to sin_pt_2, and the
	two results are multipled to give sin. The argument reduction to [0, 4)
	can be computed with a few ALU instructions: <code>reduce(x) = 4
	fract(x / tau)</code>, where <code>tau</code> is the circle constant
	formerly known as twice pi. Calculating cosine follows from the
	identity <code>cos(x) = sin(x + tau/4)</code>. After multipling by
	<code>1/tau</code>, the bias become 1/4 which can be added in the same
	cycle via a fused multiply-add. Tangent should be lowered to a division
	of sine and cosine.
	</p>
	'''

	def __init__(self):
		super().__init__('sin_pt_1')
		self.add_constant(28, 6, 0b001010)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='sin_pt_1')

@register
class SinPt2InstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('sin_pt_2')
		self.add_constant(28, 6, 0b001110)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='sin_pt_2')

@register
class Log2InstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('log2')
		self.add_constant(28, 6, 0b001100)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='log2')


@register
class Exp2InstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('exp2')
		self.add_constant(28, 6, 0b001101)

	pseudocode = FUnaryInstructionDesc.pseudocode_template.format(name='exp2')

@register
class DfdxInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('dfdx')
		self.add_constant(28, 6, 0b000100)
		self.add_field(46, 1, 'kill') # Kill helper invocations

@register
class DfdyInstructionDesc(FUnaryInstructionDesc):
	def __init__(self):
		super().__init__('dfdy')
		self.add_constant(28, 6, 0b000110)
		self.add_field(46, 1, 'kill') # Kill helper invocations

@register
class UnknownFUnaryInstructionDesc(FUnaryInstructionDesc):
	documentation_skip = True

	def __init__(self):
		super().__init__('funop')
		self.add_operand(BinaryDesc('op', 28, 6))

		# TODO: 0b001010 and 0b001110 are used to implement sin/cos/tan



@register
class RetInstructionDesc(InstructionDesc):
	documentation_begin_group = 'Flow Control Instructions'
	def __init__(self):
		super().__init__('ret', size=2)
		self.add_constant(0, 7, 0b0010100)
		self.add_operand(Reg32Desc('reg32', 9, 7))


@register
class StopInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('stop', size=2)
		self.add_constant(0, 16, 0x88)

	pseudocode = '''
	end_execution()
	'''

@register
class TrapInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('trap', size=2)
		self.add_constant(0, 16, 0x08)

@register
class CallRegInstructionDesc(InstructionDesc):
	# THEORY: Call-to-register appears to be done in a loop:
	#
	#     push_exec r0l, 2
	#  c: call r15
	#     branch?  c
	#     pop_exec r0l, 2
	#
	# Conditions of branch are unknown, but it's plausible
	# "call" masks execution to all threads with the same value,
	# and "ret" or the "branch?" then performs something like an
	# "else", repeating until all unhandled threads are handled?

	def __init__(self):
		super().__init__('call', size=2)
		self.add_constant(0, 7, 0b0000100)
		self.add_operand(Reg32Desc('reg32', 9, 7))

@register
class JumpIncompleteInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('jmp_incomplete', size=4)
		self.add_constant(0, 16, 0)
		self.add_constant(24, 8, 0)

		self.add_operand(BranchOffsetDesc('off', 16, 8))

@register
class JmpAnyInstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('jmp_exec_any', size=6)
		self.add_constant(0, 16, 0b1100000000000000)
		self.add_operand(BranchOffsetDesc('off', 16, 32))

	pseudocode = '''
	if any(exec_mask):
		next_pc = pc + sign_extend(off, 32)
	'''

@register
class JmpNoneInstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('jmp_exec_none', size=6)
		self.add_constant(0, 16, 0b1100000000100000)
		self.add_operand(BranchOffsetDesc('off', 16, 32))

	pseudocode = '''
	if not any(exec_mask):
		next_pc = pc + sign_extend(off, 32)
	'''

@register
class CallInstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('call', size=6)
		self.add_constant(0, 16, 0xC010)
		self.add_operand(BranchOffsetDesc('off', 16, 32))

	# TODO: check this only sets r1 on active threads
	pseudocode = '''
	next_pc = pc + sign_extend(off, 32)

	for each active thread:
		r1 = pc + 6
	'''

@register
class PopExecInstructionDesc(MaskedInstructionDesc):
	documentation_begin_group = 'Execution Mask Stack Instructions'
	def __init__(self):
		super().__init__('pop_exec', size=6)
		self.add_constant(0, 7, 0x52)
		self.add_constant(9, 2, 3) # op
		self.add_constant(13, 35, 0)

		self.add_operand(ImplicitR0LDesc('D'))
		self.add_operand(ImmediateDesc('n', 11, 2))

	pseudocode = '''
	for each thread:
		v = D[thread]
		v -= n
		if v < 0:
			v = 0
		D[thread] = v
		exec_mask[thread] = (v == 0)
	'''

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))

		for thread in range(SIMD_WIDTH):
			v = corestate.get_reg16(0, thread)
			v = max(0, v - fields['n'])
			corestate.set_reg16(0, thread, v)

		corestate.exec = [corestate.get_reg16(0, thread) == 0 for thread in range(SIMD_WIDTH)]


class FCmpMaskInstructionDesc(InstructionDesc):
	def __init__(self, name):
		super().__init__(name, size=6)
		self.add_constant(0, 7, 0x42)

		self.add_operand(ImplicitR0LDesc('D'))

		self.add_operand(FConditionDesc())

		self.add_operand(FloatSrcDesc('A', 16, 42))
		self.add_operand(FloatSrcDesc('B', 28, 40))

		self.add_constant(44, 2, 0)

		self.add_operand(ImmediateDesc('n', 11, 2)) # push count

	def compare_thread(self, fields, corestate, thread):
		return fcompare_thread(self, fields, corestate, thread)

class ICmpMaskInstructionDesc(InstructionDesc):
	def __init__(self, name):
		super().__init__(name, size=6)
		self.add_constant(0, 7, 0x52)

		self.add_operand(ImplicitR0LDesc('D'))

		self.add_operand(IConditionDesc())

		self.add_operand(ALUSrcDesc('A', 16, 42))
		self.add_operand(ALUSrcDesc('B', 28, 40))

		self.add_constant(26, 2, 0)
		self.add_constant(38, 2, 0)
		self.add_constant(44, 2, 0)

		self.add_operand(ImmediateDesc('n', 11, 2)) # push count


	def compare_thread(self, fields, corestate, thread):
		return icompare_thread(self, fields, corestate, thread)

@register
class IfICmpInstructionDesc(ICmpMaskInstructionDesc):
	def __init__(self):
		super().__init__('if_icmp')
		self.add_constant(9, 2, 0)

	pseudocode = '''
	for each thread:
		v = D[thread]
		if v != 0:
			v += n
		elif not cc.compare(A[thread], B[thread]):
			v = 1
		D[thread] = v
		exec_mask[thread] = (v == 0)
	'''

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		for thread in range(SIMD_WIDTH):
			v = corestate.get_reg16(0, thread)
			if v != 0:
				v += fields['n']
			elif not self.compare_thread(fields, corestate, thread):
				v = 1
			corestate.set_reg16(0, thread, v)
		corestate.exec = [corestate.get_reg16(0, thread) == 0 for thread in range(SIMD_WIDTH)]

@register
class IfFCmpInstructionDesc(FCmpMaskInstructionDesc):
	def __init__(self):
		super().__init__('if_fcmp')
		self.add_constant(9, 2, 0)

	def map_to_alias(self, mnem, operands):
		# unconditional
		if str(operands[1]) == 'eq' and str(operands[2]) == '0.0' and str(operands[3]) == '0.0':
			if str(operands[4]) == '0':
				return 'update_exec', [operands[0]]
			return 'push_exec', [operands[0], operands[4]]

		return mnem, operands

	pseudocode = IfICmpInstructionDesc.pseudocode

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		for thread in range(SIMD_WIDTH):
			v = corestate.get_reg16(0, thread)
			if v != 0:
				v += fields['n']
			elif not self.compare_thread(fields, corestate, thread):
				v = 1
			corestate.set_reg16(0, thread, v)
		corestate.exec = [corestate.get_reg16(0, thread) == 0 for thread in range(SIMD_WIDTH)]


@register
class WhileICmpInstructionDesc(ICmpMaskInstructionDesc):
	def __init__(self):
		super().__init__('while_icmp')
		self.add_constant(9, 2, 2)

	pseudocode = '''
	for each thread:
		v = D[thread]
		if v < n:
			if cc.compare(A[thread], B[thread]):
				v = 0
			else:
				v = n
		D[thread] = v
		exec_mask[thread] = (v == 0)
	'''

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		for thread in range(SIMD_WIDTH):
			v = corestate.get_reg16(0, thread)
			if v < fields['n']:
				if self.compare_thread(fields, corestate, thread):
					v = 0
				else:
					v = fields['n']
			corestate.set_reg16(0, thread, v)
		corestate.exec = [corestate.get_reg16(0, thread) == 0 for thread in range(SIMD_WIDTH)]


@register
class WhileFCmpInstructionDesc(FCmpMaskInstructionDesc):
	def __init__(self):
		super().__init__('while_fcmp')
		self.add_constant(9, 2, 2)

	pseudocode = WhileICmpInstructionDesc.pseudocode

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		for thread in range(SIMD_WIDTH):
			v = corestate.get_reg16(0, thread)
			if v < fields['n']:
				if self.compare_thread(fields, corestate, thread):
					v = 0
				else:
					v = fields['n']
			corestate.set_reg16(0, thread, v)
		corestate.exec = [corestate.get_reg16(0, thread) == 0 for thread in range(SIMD_WIDTH)]

@register
class ElseICmpInstructionDesc(ICmpMaskInstructionDesc):
	def __init__(self):
		super().__init__('else_icmp')
		self.add_constant(9, 2, 1)

	def map_to_alias(self, mnem, operands):
		# unconditional
		if str(operands[1]) == 'eq' and str(operands[2]) == '0.0' and str(operands[3]) == '0.0':
			return 'else_exec', [operands[0], operands[4]]

		return mnem, operands

	pseudocode = '''
	for each thread:
		v = D[thread]
		if v == 0:
			v = n
		elif v == 1:
			if cc.compare(A[thread], B[thread]):
				v = 0
			else:
				v = 1
		D[thread] = v
		exec_mask[thread] = (v == 0)
	'''

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		for thread in range(SIMD_WIDTH):
			v = corestate.get_reg16(0, thread)
			if v == 0:
				v = fields['n']
			elif v == 1:
				if self.compare_thread(fields, corestate, thread):
					v = 0
				else:
					v = 1
			corestate.set_reg16(0, thread, v)
		corestate.exec = [corestate.get_reg16(0, thread) == 0 for thread in range(SIMD_WIDTH)]

@register
class ElseFCmpInstructionDesc(FCmpMaskInstructionDesc):
	def __init__(self):
		super().__init__('else_fcmp')
		self.add_constant(9, 2, 1)

	pseudocode = ElseICmpInstructionDesc.pseudocode

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		for thread in range(SIMD_WIDTH):
			v = corestate.get_reg16(0, thread)
			if v == 0:
				v = fields['n']
			elif v == 1:
				if self.compare_thread(fields, corestate, thread):
					v = 0
				else:
					v = 1
			corestate.set_reg16(0, thread, v)
		corestate.exec = [corestate.get_reg16(0, thread) == 0 for thread in range(SIMD_WIDTH)]



@register
class ICmpselInstructionDesc(MaskedInstructionDesc):
	documentation_begin_group = 'Select Instructions'

	documentation_name = 'Integer Compare and Select'

	def __init__(self):
		super().__init__('icmpsel', size=(8, 10))
		self.add_constant(0, 7, 0x12)

		self.add_operand(IConditionDesc(61, None))

		self.add_operand(ALUDstDesc('D', 76))

		self.add_operand(ALUSrcDesc('A', 16, 74))
		self.add_operand(ALUSrcDesc('B', 28, 72))

		self.add_operand(CmpselSrcDesc('X', 40, 70))
		self.add_operand(CmpselSrcDesc('Y', 52, 68))

	pseudocode = '''
	for each active thread:
		if cc.compare(A[thread], B[thread]):
			D[thread] = X[thread]
		else:
			D[thread] = Y[thread]
	'''
	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		comparison = icompare_thread(self, fields, corestate, thread)
		if comparison:
			result = self.operands['X'].evaluate_thread(fields, corestate, thread)
		else:
			result = self.operands['Y'].evaluate_thread(fields, corestate, thread)

		self.operands['D'].set_thread(fields, corestate, thread, result)

@register
class FCmpselInstructionDesc(MaskedInstructionDesc):
	documentation_name = 'Floating-Point Compare and Select'

	def __init__(self):
		super().__init__('fcmpsel', size=(8, 10))
		self.add_constant(0, 7, 0x02)

		self.add_operand(FConditionDesc(61, None))

		self.add_operand(ALUDstDesc('D', 76))

		self.add_operand(FloatSrcDesc('A', 16, 74))
		self.add_operand(FloatSrcDesc('B', 28, 72))

		self.add_operand(CmpselSrcDesc('X', 40, 70))
		self.add_operand(CmpselSrcDesc('Y', 52, 68))

	pseudocode = ICmpselInstructionDesc.pseudocode

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		comparison = fcompare_thread(self, fields, corestate, thread)
		if comparison:
			result = self.operands['X'].evaluate_thread(fields, corestate, thread)
		else:
			result = self.operands['Y'].evaluate_thread(fields, corestate, thread)

		self.operands['D'].set_thread(fields, corestate, thread, result)



@register
class ICmpBallotInstructionDesc(InstructionDesc):
	documentation_begin_group = 'SIMD Group and Quad Group Instructions'

	def __init__(self):
		super().__init__('icmp_ballot', size=8)
		self.add_constant(0, 7, 0b0110010)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(IConditionDesc(61, 47))
		self.add_operand(ALUSrcDesc('A', 16, 42))
		self.add_operand(ALUSrcDesc('B', 28, 40))

		self.add_constant(26, 2, 0)
		self.add_constant(38, 2, 0)
		self.add_constant(48, 13, 1)

	pseudocode = '''
	result = 0

	for each active thread:
		a = A[thread]
		b = B[thread]

		if cc.compare(a, b):
			result |= 1 << thread

	D.broadcast_to_active(result)
	'''

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		result = 0
		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread] and self.compare_thread(fields, corestate, thread):
				result |= (1 << thread)

		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				self.operands['D'].set_thread(fields, corestate, thread, result)

	def compare_thread(self, fields, corestate, thread):
		return icompare_thread(self, fields, corestate, thread)


@register
class ICmpQuadBallotInstructionDesc(InstructionDesc):
	# NOTE: followed by "& 15" for a real quad_ballot
	def __init__(self):
		super().__init__('icmp_quad_ballot', size=8)
		self.add_constant(0, 7, 0b0110010)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(IConditionDesc(61, 47))
		self.add_operand(ALUSrcDesc('A', 16, 42))
		self.add_operand(ALUSrcDesc('B', 28, 40))

		self.add_constant(26, 2, 0)
		self.add_constant(38, 2, 0)
		self.add_constant(48, 13, 0)

	def compare_thread(self, fields, corestate, thread):
		return icompare_thread(self, fields, corestate, thread)

@register
class FCmpBallotInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('fcmp_ballot', size=8)
		self.add_constant(0, 7, 0b0100010)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(FConditionDesc(61, 47))
		self.add_operand(FloatSrcDesc('A', 16, 42))
		self.add_operand(FloatSrcDesc('B', 28, 40))

		self.add_constant(48, 13, 1)

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		result = 0
		for thread in range(SIMD_WIDTH):
			# TODO: is this predicated?
			if corestate.exec[thread] and self.compare_thread(fields, corestate, thread):
				result |= (1 << thread)
		print('result: %x' % (result,))
		for thread in range(SIMD_WIDTH):
			# TODO: is this predicated?
			if corestate.exec[thread]:
				self.operands['D'].set_thread(fields, corestate, thread, result)

	pseudocode = ICmpBallotInstructionDesc.pseudocode

	def compare_thread(self, fields, corestate, thread):
		return True

@register
class FCmpQuadBallotInstructionDesc(InstructionDesc):
	# NOTE: followed by "& 15" for a real quad_ballot
	def __init__(self):
		super().__init__('fcmp_quad_ballot', size=8)
		self.add_constant(0, 7, 0b0100010)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(FConditionDesc(61, 47))
		self.add_operand(FloatSrcDesc('A', 16, 42))
		self.add_operand(FloatSrcDesc('B', 28, 40))

		self.add_constant(48, 13, 0)



class BaseSimdShuffleInstructionDesc(MaskedInstructionDesc):
	def __init__(self, name):
		super().__init__(name, size=6)

		self.add_constant(0, 7, 0b01101111)
		self.add_constant(15, 1, 0)

		self.add_operand(ALUDstDesc('D', 44))
		self.add_operand(ALUSrcDesc('A', 16, 42))
		self.add_operand(ALUSrc16Desc('B', 28, 40))

@register
class SimdShuffleInstructionDesc(BaseSimdShuffleInstructionDesc):
	def __init__(self):
		# TODO: might be more accurate to call this "simd_broadcast", as it implements
		# that behaviour completely? Maybe just when the argument is a constant?

		super().__init__('simd_shuffle')

		self.add_constant(47, 1, 0b0)
		self.add_constant(38, 2, 0b00)
		self.add_constant(26, 2, 0b01)

	pseudocode = '''
	quad_values = []

	for each quad:
		quad_index = 0

		for each thread in quad:
			# NOTE: this is not execution masked, meaning any inactive thread can make
			# simd_broadcast from the whole quad undefined (although it works fine if
			# B is an immediate)

			quad_index |= B[thread] & 3

		quad_values.append(A[quad.start + quad_index])

	for each active thread:
		b = B[thread]

		if b < 32:
			result = quad_values[index >> 2]
		else:
			result = A[index]

		D[thread] = result
	'''

	def exec(self, instr, corestate):
		# Metal's "simd_shuffle" is implemented like so:
		#
		#    quad_start = target_index & 0x1C
		#    result = shuffle(r11, quad_start + 0)
		#    if quad_index + 1 == target_index: result = shuffle(r11, quad_start + 1)
		#    if quad_index + 2 == target_index: result = shuffle(r11, quad_start + 2)
		#    if quad_index + 3 == target_index: result = shuffle(r11, quad_start + 3)
		#
		# i.e. the low 2 bits should be the same - the next 3 bits select which quad
		# to pull from.

		fields = dict(self.decode_fields(instr))

		quad_values = []
		for quad_start in range(0, SIMD_WIDTH, 4):
			quad_index = 0
			for thread in range(quad_start, quad_start + 4):
				# this _isn't_ execution masked, which is kind of crazy, as any inactive thread
				# can make simd_broadcast from the whole quad undefined. Fortunately it works
				# fine if 'B' is an immediate.
				quad_index |= self.operands['B'].evaluate_thread(fields, corestate, thread) & 3
			quad_values.append(self.operands['A'].evaluate_thread(fields, corestate, quad_start + quad_index))

		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				index = self.operands['B'].evaluate_thread(fields, corestate, thread)
				if index < SIMD_WIDTH:
					result = quad_values[index >> 2]
				else:
					result = self.operands['A'].evaluate_thread(fields, corestate, thread)
				self.operands['D'].set_thread(fields, corestate, thread, result)

class OperandAccessor:
	def __init__(self, operand, fields, corestate, write_only=False, read_only=False):
		self.operand = operand
		self.fields = fields
		self.corestate = corestate
		self.write_only = write_only
		self.read_only = read_only
		if read_only:
			self.values = [self.operand.evaluate_thread(self.fields, self.corestate, i) for i in range(32)]

	def __getitem__(self, thread):
		assert not self.write_only
		if self.read_only:
			return self.values[thread]
		else:
			return self.operand.evaluate_thread(self.fields, self.corestate, thread)

	def __setitem__(self, thread, value):
		assert not self.read_only
		return self.operand.set_thread(self.fields, self.corestate, thread, value)

@register
class SimdShuffleDownInstructionDesc(BaseSimdShuffleInstructionDesc):
	def __init__(self):
		super().__init__('simd_shuffle_down')

		self.add_constant(47, 1, 0b0)
		self.add_constant(38, 2, 0b11)
		self.add_constant(26, 2, 0b01)

	# TODO: how does this work with different values in different threads?
	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		A = OperandAccessor(self.operands['A'], fields, corestate, read_only=True)
		B = OperandAccessor(self.operands['B'], fields, corestate, read_only=True)
		D = OperandAccessor(self.operands['D'], fields, corestate, write_only=True)

		quad_values = []
		for quad_start in range(0, SIMD_WIDTH, 4):
			quad_shift = 0
			for thread in range(quad_start, quad_start + 4):
				quad_shift |= B[thread] & 3
			quad_values.append([A[quad_start + ((i + quad_shift) & 3)] for i in range(4)])

		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				index = B[thread]
				if index + thread < SIMD_WIDTH:
					result = quad_values[(index + thread) >> 2][thread & 3]
				else:
					result = A[thread]
				D[thread] = result


@register
class SimdShuffleUpInstructionDesc(BaseSimdShuffleInstructionDesc):
	def __init__(self):
		super().__init__('simd_shuffle_up')

		self.add_constant(47, 1, 0b0)
		self.add_constant(38, 2, 0b10)
		self.add_constant(26, 2, 0b01)

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		A = OperandAccessor(self.operands['A'], fields, corestate, read_only=True)
		B = OperandAccessor(self.operands['B'], fields, corestate, read_only=True)
		D = OperandAccessor(self.operands['D'], fields, corestate, write_only=True)

		quad_values = []
		for quad_start in range(0, SIMD_WIDTH, 4):
			quad_shift = 0
			for thread in range(quad_start, quad_start + 4):
				quad_shift |= B[thread] & 3
			quad_values.append([A[quad_start + ((i - quad_shift) & 3)] for i in range(4)])

		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				index = B[thread]
				if thread - index >= 0:
					result = quad_values[(thread - index) >> 2][thread & 3]
				else:
					result = A[thread]
				D[thread] = result

@register
class SimdShuffleRotateUpInstructionDesc(BaseSimdShuffleInstructionDesc):
	def __init__(self):
		super().__init__('simd_shuffle_rotate_up')

		self.add_constant(47, 1, 0b1)
		self.add_constant(38, 2, 0b10)
		self.add_constant(26, 2, 0b01)

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		A = OperandAccessor(self.operands['A'], fields, corestate, read_only=True)
		B = OperandAccessor(self.operands['B'], fields, corestate, read_only=True)
		D = OperandAccessor(self.operands['D'], fields, corestate, write_only=True)

		quad_values = []
		for quad_start in range(0, SIMD_WIDTH, 4):
			quad_shift = 0
			for thread in range(quad_start, quad_start + 4):
				quad_shift |= B[thread] & 3
			quad_values.append([A[quad_start + ((i - quad_shift) & 3)] for i in range(4)])

		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				index = B[thread]
				result = quad_values[((thread - index) & 31) >> 2][thread & 3]
				D[thread] = result


@register
class SimdShuffleXorInstructionDesc(BaseSimdShuffleInstructionDesc):
	def __init__(self):
		super().__init__('simd_shuffle_xor')

		self.add_constant(47, 1, 0b0)
		self.add_constant(38, 2, 0b01)
		self.add_constant(26, 2, 0b01)

	def exec(self, instr, corestate):
		fields = dict(self.decode_fields(instr))
		A = OperandAccessor(self.operands['A'], fields, corestate, read_only=True)
		B = OperandAccessor(self.operands['B'], fields, corestate, read_only=True)
		D = OperandAccessor(self.operands['D'], fields, corestate, write_only=True)

		quad_values = []
		for quad_start in range(0, SIMD_WIDTH, 4):
			quad_xor = 0
			for thread in range(quad_start, quad_start + 4):
				quad_xor |= B[thread] & 3
			quad_values.append([A[quad_start + ((i ^ quad_xor) & 3)] for i in range(4)])

		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				xor = B[thread]

				if thread ^ xor < SIMD_WIDTH:
					result = quad_values[((thread ^ xor) & 31) >> 2][thread & 3]
				else:
					result = A[thread]
				D[thread] = result


@register
class SimdMatrixFMadd32InstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('simd_matrix_fmadd32', size=8)
		self.add_constant(0, 7, 0x6f)
		self.add_constant(15, 1, 0)
		self.add_constant(27, 1, 1)
		self.add_constant(26, 1, 1)

		self.add_operand(PairedALUDstDesc('D', 60))

		# These seem to be ALUSrc, accepting either 16-bit or 32-bit,
		# but paired (i.e. R0L -> R0, or R0 -> R0_R1).

		self.add_operand(PairedFloatSrcDesc('A', 16, 58, 52))
		self.add_operand(PairedFloatSrcDesc('B', 28, 56))
		self.add_operand(PairedFloatSrcDesc('C', 40, 54))
		self.add_constant(63, 1, 1)


@register
class SimdMatrixFMadd16InstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('simd_matrix_fmadd16', size=8)
		self.add_constant(0, 7, 0x6f)
		self.add_constant(15, 1, 0)
		self.add_constant(27, 1, 1)
		self.add_constant(26, 1, 0)

		self.add_operand(PairedALUDstDesc('D', 60))

		# These seem to be ALUSrc, accepting either 16-bit or 32-bit,
		# but paired (i.e. R0L -> R0, or R0 -> R0_R1).

		self.add_operand(PairedFloatSrcDesc('A', 16, 58, 52))
		self.add_operand(PairedFloatSrcDesc('B', 28, 56))
		self.add_operand(PairedFloatSrcDesc('C', 40, 54))
		self.add_constant(63, 1, 1)

for op1, op2, op3, name in [
	(0b0, 0b00, 0b00, 'quad_shuffle'),
	(0b0, 0b11, 0b00, 'quad_shuffle_down'),
	(0b0, 0b10, 0b00, 'quad_shuffle_up'),
	(0b1, 0b10, 0b00, 'quad_shuffle_rotate_up'),
	(0b0, 0b01, 0b00, 'quad_shuffle_xor'),
	#(0b0, 0b00, 0b01, 'simd_shuffle'),
	#(0b0, 0b01, 0b01, 'simd_shuffle_xor'),
	#(0b0, 0b10, 0b01, 'simd_shuffle_up'),
	#(0b0, 0b11, 0b01, 'simd_shuffle_down'),
	#(0b1, 0b10, 0b01, 'simd_shuffle_rotate_up'),

	# TODO: setting op1 to 1 seems to work for others too?
	#       figure out what it's doing.
	(None, None, None, 'simd_shuf_op'),
]:

	o = InstructionDesc(name, size=6)
	o.documentation_skip = True
	o.add_constant(0, 7, 0b01101111)
	o.add_constant(15, 1, 0)

	#o.add_constant(26, 2, 0b01)

	#if op1 is None:
	#o.add_operand(BinaryDesc('op1', 47, 1))
	#else:
	#	o.add_constant(47, 1, op1)

	#if op2 is None:
	if op1 is None:
		o.add_operand(BinaryDesc('op1', 47, 1))
		o.add_operand(BinaryDesc('op2', 38, 2))
		o.add_operand(BinaryDesc('op3', 26, 2))
	else:
		o.add_constant(47, 1, op1)
		o.add_constant(38, 2, op2)
		o.add_constant(26, 2, op3)

	o.add_operand(ALUDstDesc('D', 44))
	o.add_operand(ALUSrc64Desc('A', 16, 42))
	o.add_operand(ALUSrc64Desc('B', 28, 40))

	instruction_descriptors.append(o)

for op1, op2, name in [
	(0, 0b0000000000000000, 'quad_and'),
	(0, 0b0001000000000000, 'quad_or'),
	(0, 0b0010000000000000, 'quad_xor'),

	(1, 0b0000000000000000, 'quad_iadd'),
	(0, 0b0000000000000100, 'quad_fadd'),
	(0, 0b0001000000000100, 'quad_fmul'),

	(1, 0b0000000000010000, 'quad_prefix_iadd'),
	(0, 0b0000000000010100, 'quad_prefix_fadd'),
	(0, 0b0001000000010100, 'quad_prefix_fmul'),

	(1, 0b0110000000000000, 'quad_min.u'),
	(1, 0b0111000000000000, 'quad_max.u'),
	(1, 0b0010000000000000, 'quad_min.s'),
	(1, 0b0011000000000000, 'quad_max.s'),
	(0, 0b0010000000000100, 'quad_min.f'),
	(0, 0b0011000000000100, 'quad_max.f'),

	(0, 0b0000000000001000, 'simd_and'),
	(0, 0b0001000000001000, 'simd_or'),
	(0, 0b0010000000001000, 'simd_xor'),

	(1, 0b0000000000001000, 'simd_iadd'),
	(0, 0b0000000000001100, 'simd_fadd'),
	(0, 0b0001000000001100, 'simd_fmul'),

	(1, 0b0000000000011000, 'simd_prefix_iadd'),
	(0, 0b0000000000011100, 'simd_prefix_fadd'),
	(0, 0b0001000000011100, 'simd_prefix_fmul'),

	(1, 0b0110000000001000, 'simd_min.u'),
	(1, 0b0111000000001000, 'simd_max.u'),
	(1, 0b0010000000001000, 'simd_min.s'),
	(1, 0b0011000000001000, 'simd_max.s'),
	(0, 0b0010000000001100, 'simd_min.f'),
	(0, 0b0011000000001100, 'simd_max.f'),

	# generic must come last as fallback
	(None, None, 'simd_op'),
]:
	o = InstructionDesc(name, size=6)
	o.add_constant(0, 7, 0b01101111)
	o.add_constant(15, 1, 1)

	o.documentation_skip = True


	if op1 is None:
		o.add_operand(BinaryDesc('op1', 47, 1))
	else:
		o.add_constant(47, 1, op1)

	if op2 is None:
		o.add_operand(BinaryDesc('op2', 26, 16))
	else:
		o.add_constant(26, 16, op2)

	# TODO: sometimes FloatDstDesc?
	o.add_operand(ALUDstDesc('D', 44))
	o.add_operand(ALUSrc64Desc('A', 16, 42))


	instruction_descriptors.append(o)
	#print(o.matches(opcode_to_number(b'\x6F\xAC\x57\x20\x00\x80')))
	#exit(0)




for op, mnem in [
	(0b01010001, 'no_var'),
	(0b00010001, 'st_var'),
	(0b10010001, 'st_var_final'),
]:
	o = InstructionDesc(mnem, size=4)
	o.add_constant(0, 10, op)
	o.add_constant(22, 2, 2)
	# set with "no var" case
	o.add_operand(ImmediateDesc('u', 31, 1))
	o.add_operand(ExReg32Desc('r', 10, 24))
	o.add_operand(ImmediateDesc('i', [(16, 6, 'I'), (26, 2, 'Ix')]))
	instruction_descriptors.append(o)


o = InstructionDesc('writeout', size=4)
o.add_constant(0, 8, 0b01001000)
o.add_operand(ImmediateDesc('i', 8, 10)) # x: 26,2
o.add_operand(ImmediateDesc('j', 22, 2)) # x: 26,2
instruction_descriptors.append(o)


# wait for a load
@register
class WaitInstructionDesc(InstructionDesc):
	documentation_begin_group = 'Memory and Stack Instructions'
	def __init__(self):
		super().__init__('wait', size=2)
		self.add_constant(0, 8, 0b111000)
		self.add_operand(ImmediateDesc('i', 8, 1))

	pseudocode = 'wait_for_loads()'

	def exec(self, instr, corestate):
		# TODO: queue loads
		pass


MEMORY_FORMATS = {
	0: 'i8',         # size = 1
	1: 'i16',        # size = 2
	2: 'i32',        # size = 4
	3: 'f16',	 # size = 2
	4: 'u8norm',     # size = 1
	5: 's8norm',     # size = 1
	6: 'u16norm',    # size = 2
	7: 's16norm',    # size = 2
	8: 'rgb10a2',    # size = 1

	10: 'srgba8',    # size = 1

	12: 'rg11b10f',  # size = 1
	13: 'rgb9e5',    # size = 1

	# TODO: others?
}

MASK_DESCRIPTIONS = {
	a: ('x' if a & 1 else '') + ('y' if a & 2 else '') + ('z' if a & 4 else '') + ('w' if a & 8 else '') for a in range(16)
}

# TODO
# uses sr60 implicitly?
@register
class LdstTileDesc(InstructionDesc):
	documentation_html = '''
	<p>
	<code>ld_tile</code> and <code>st_tile</code> access shared memory in a
	fragment shader with pixel-relative addressing. This emulates a
	tilebuffer.

	AGX fragment shaders always run at pixel rate (sample rate shading may
	be lowered to pixel rate shading with a loop). That
	means st_tile and ld_tile need to be able to load and store specific
	samples. This is controlled by their sample mask source, which is a
	bitmask of enabled samples. To store to a particular sample N, pass
	(1 &lt;&lt; N). To store to a subset of samples { S1, ..., Sn },
	pass Σ_i=1...n (1 &lt;&lt; N). To store to all samples, pass 0xFF as the
	special "broadcast" value. Although MSAA 8x may not be used, this
	value might be optimized specially.

	The offset into shared memory is specified by O. This is in bytes and
	does not take into account multisampling or tile size.
	</p>'''

	def __init__(self):
		super().__init__('TODO.ld/st_tile', size=(8, 10))

		self.add_constant(0, 6, 0x09)

		# Direction of the transfer, set for tilebuffer->register,
		# clear for register->tilebuffer
		self.add_operand(ImmediateDesc('load', 6, 1))

		# If load is false, actually a source! ALUDstDesc's cache hint possibly a discard hint.
		#self.add_operand(ALUDstDesc('D', 60))
		self.add_operand(ThreadgroupMemoryRegDesc('R'))

		self.add_operand(EnumDesc('F', 24, 4, MEMORY_FORMATS))
		self.add_operand(ImmediateDesc('u0', 35, 1))
		self.add_operand(EnumDesc('mask', 36, 4, MASK_DESCRIPTIONS))

		self.add_operand(ImmediateDesc('O', [(28, 7, 'A'), (40, 2, 'Ax')]))
		self.add_operand(SampleMaskDesc('S'))
		self.add_operand(ImmediateDesc('C', [(16, 6, 'C'), (58, 2, 'Cx')]))

	def map_to_alias(self, mnem, operands):
		is_load = operands[0]
		mnem = 'ld_tile' if is_load else 'st_tile'
		return mnem, operands[1:]


class VarRegisterDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(9, 6, self.name),
			(56, 2, self.name + 'x'),
		])
		self.add_field(8, 1, self.name + 't')
		self.add_field(30, 2, 'count')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		count = fields['count']
		if count == 0: count = 4

		# not really clear how the alignment requirement works
		if flags == 0:
			t = RegisterTuple(Reg16((value) + i) for i in range(count))
		else:
			t = RegisterTuple(Reg32((value >> 1) + i) for i in range(count))

		return t

class VarTripleRegisterDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(9, 6, self.name),
			(56, 2, self.name + 'x'),
		])
		self.add_field(8, 1, self.name + 't')
		self.add_field(30, 2, 'count')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		count = fields['count']
		if count == 0: count = 4

		count *= 3

		# not really clear how the alignment requirement works
		if flags == 0:
			t = RegisterTuple(Reg16((value) + i) for i in range(count))
		else:
			t = RegisterTuple(Reg32((value >> 1) + i) for i in range(count))

		return t

class CFDesc(OperandDesc):
	def __init__(self, name, off=16, offx=58): #, offt=62):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		#self.add_field(offt, 1, self.name + 't')

	def decode(self, fields):
		#flags = fields[self.name + 't']
		value = fields[self.name]

		#if flags == 0b0:
		return CF(value)

class CFPerspectiveDesc(OperandDesc):
	def __init__(self, name, off=24, offx=60): #, offt=62):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		#self.add_field(offt, 1, self.name + 't')

	def decode(self, fields):
		#flags = fields[self.name + 't']
		
		if not fields['P']:
			return ''

		value = fields[self.name]

		#if flags == 0b0:
		return CF(value)

@register
class IterDesc(InstructionDesc):
	documentation_html = '''
	<p>The last four bytes are omitted if L=0.</p>
	<p>If forwarding is enabled, the result of the varying must be consumed directly as coordinates by a subsequent texture_sample instruction.</p>
	'''

	def __init__(self):
		super().__init__('TODO.iter', size=(4, 8))
		self.add_constant(0, 6, 0x21)
		self.add_constant(7, 1, 0)
		
		self.add_operand(VarRegisterDesc('D'))

		self.add_operand(EnumDesc('P', 6, 1, {
			0: 'no_perspective',
			1: 'perspective',
		}))

		self.add_operand(CFDesc('I', 16, 58))
		self.add_operand(CFPerspectiveDesc('J', 24, 60))

		self.add_operand(ImmediateDesc('q0', 32, 1))
		self.add_operand(EnumDesc('forwarding', 46, 1, {
			0: 'forward',
			1: 'no_forward'
		}))
		self.add_operand(EnumDesc('sample', 48, 1, {
			0: 'pixel',
			1: 'sample'
		}))
		self.add_operand(EnumDesc('centroid', 49, 1, {
			0: 'no_centroid',
			1: 'centroid'
		}))
		self.add_operand(BinaryDesc('kill', 52, 1)) # Kill helper invocations 

@register
class LoadCFDesc(InstructionDesc):
	documentation_html = '<p>The last four bytes are omitted if L=0.</p>'
	def __init__(self):
		super().__init__('TODO.ldcf', size=(4, 8))
		self.add_constant(0, 6, 0x21)
		self.add_constant(6, 2, 0b10)
		self.add_operand(VarTripleRegisterDesc('D')) # TODO: confirm extension
		self.add_operand(CFDesc('I', 16, 58))

@register
class StoreToUniformInstructionDesc(InstructionDesc):
	documentation_html = '''
	<p>
	<code>uniform_store</code> is used to initialise uniform registers.
	<code>R</code> is stored to offset <code>O</code>, which is typically an
	index in 16-bit units into the uniform registers. This is encoded like
	(and possibly is) a store to device memory, and can move one 16-bit register
	to initialise a 16-bit uniform, or two consecutive 16-bit registers to
	initialise a 32-bit uniform.
	</p>
	'''
	def __init__(self):
		super().__init__('uniform_store', size=(6, 8), length_bit_pos=47) # ?

		self.add_operand(ImmediateDesc('unk', 25, 2)) # ??

		self.add_constant(0, 7, 0b1000101)

		self.add_constant(9, 1, 0) # TODO

		self.add_constant(28, 2, 0b11) # ?
		self.add_constant(50, 2, 0) # ?

		self.add_operand(EnumDesc('F', 7, 2, MEMORY_FORMATS))
		self.add_operand(EnumDesc('mask', 52, 4, MASK_DESCRIPTIONS))

		self.add_operand(ImmediateDesc('b', 44, 3))
		self.add_operand(MemoryRegDesc('R'))
		self.add_constant(16, 4, 0)
		self.add_constant(36, 4, 0)
		self.add_constant(27, 1, 1)

		# memory index is ureg16 number - e.g. r6l = 12
		self.add_operand(MemoryIndexDesc('O'))
		self.add_operand(MemoryShiftDesc('s'))


class AsyncLoadStoreInstructionDesc(MaskedInstructionDesc):
	def __init__(self, name, bit):
		super().__init__(name, size=8)
		self.add_constant(0, 7, 0b0100101 | (bit << 6))
		self.add_operand(EnumDesc('F', 47, 1, {0:"copy_1d", 1:"copy_2d"}))
		self.add_operand(AsyncMemoryRegDesc('R'))
		self.add_operand(AsyncMemoryBaseDesc('A'))

@register
class AsyncStoreInstructionDesc(AsyncLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('async_store', 1)

@register
class AsyncLoadInstructionDesc(AsyncLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('async_load', 0)

class DeviceLoadStoreInstructionDesc(MaskedInstructionDesc):
	def __init__(self, name, bit):
		super().__init__(name, size=(6, 8), length_bit_pos=47) # ?

		self.add_operand(ImmediateDesc('g', 30, 1)) # wait group (scoreboarding)

		self.add_constant(0, 7, 0b0000101 | (bit << 6))

		self.add_operand(EnumDesc('F', [
			(7, 3, 'F'),
			(48, 1, 'Fx'),
		], None, MEMORY_FORMATS))

		self.add_operand(EnumDesc('mask', 52, 4, MASK_DESCRIPTIONS))

		self.add_operand(MemoryRegDesc('R'))
		self.add_operand(MemoryBaseDesc('A'))
		self.add_operand(MemoryIndexDesc('O'))
		self.add_operand(EnumDesc('Ou', 25, 1, {
			0: 'signed',
			1: 'unsigned',
		}))
		self.add_operand(MemoryShiftDesc('s'))


def decode_float11(n):
	# TODO: test subnormals/infinity
	e = (n >> 6) & 0x1F
	if e == 0x1F:
		if n & 0x3F:
			return fma.u32_to_f32(0x7fc00000) # nan
		else:
			return fma.u32_to_f32(0x7f800000) # inf
	implicit = 0x40
	if e == 0:
		implicit = 0
		e += 1
	f = (implicit | (n & 0x3F)) / 64.0
	return f * (2.0 ** (e - 15))

def decode_float10(n):
	# TODO: test subnormals/infinity
	e = (n >> 5) & 0x1F
	if e == 0x1F:
		if n & 0x1F:
			return fma.u32_to_f32(0x7fc00000) # nan
		else:
			return fma.u32_to_f32(0x7f800000) # inf
	implicit = 0x20
	if e == 0:
		implicit = 0
		e += 1
	f = (implicit | (n & 0x1F)) / 32.0
	return f * (2.0 ** (e - 15))

@register
class DeviceLoadInstructionDesc(DeviceLoadStoreInstructionDesc):
	documentation_html = '''
	<p>
	<code>device_load</code> initiates a load from device memory, the result
	of which may be used after a <code>wait</code>.

	The data can be unpacked from a variety of formats, or passed through as-is.

	On each thread, up to four aligned values, each up to 32-bits, can be
	read from a base address plus an offset (shifted left by the alignment,
	with an optional additional left shift of up to two).
	</p>

	<p>
	The number of values to read is described by a mask,
	such that <code>0b0001</code> indicates one value, or <code>0b1111</code>
	loads four values. Non-contiguous masks skip values in memory,
	but still write the result to contiguous registers.
	</p>

	<p>
	Non-packed formats (8, 16, and 32-bit values) are zero
	extended. All packed values are unpacked to 16-bit or 32-bit floating-point
	values, depending on the size of the register. Bit-packed formats (<code>rgb10a2</code>,
	<code>rg11b10f</code> and <code>rgb9e5</code>)</code> are supported, but ignore the optional
	shift and the mask. They always read an aligned 32-bit value, and write to the same number of
	registers. However simple packed values (<code>unorm8</code>, <code>snorm8</code>,
	<code>unorm16</code>, <code>snorm16</code> and <code>srgba8</code>) do not have this limitation.
	</p>

	<p>
	Unaligned addresses are rounded-down to the required alignment. The base address (<code>A</code>)
	is a 64-bit value from either uniform or general-purpose registers. The offset (<code>O</code>) may
	be a signed 16-bit immediate, or a signed or unsigned 32-bit general-purpose register.
	</p>
	'''
	def __init__(self):
		super().__init__('device_load', 0)
		self.add_constant(26, 1, 1) # u1
		self.add_constant(28, 2, 0) # u3
		self.add_constant(44, 3, 4) # u4
		self.add_constant(50, 2, 0) # u5

	def exec_thread(self, instr, corestate, thread):
		fields = dict(self.decode_fields(instr))

		# NOTE: the philosophy here is "make it work, then make it nice".
		# the idea being that once i know i understand the scope of the
		# i can make informed decisions about how to factor it and
		# represent it. this has a way to go before it works, and a long
		# way to go before it is nice.

		# TODO: these fields (A and O) should get their own state
		# via self.operands[name].evaluate_thread(fields, corestate, thread)

		if fields['At']:
			# TODO: test
			address = corestate.uniforms.get_reg64(fields['A'] >> 1, thread)
		else:
			address = corestate.get_reg64(fields['A'] >> 1, thread)

		if fields['Ot']:
			# TODO: test
			offset = sign_extend(fields['O'], 16)
		else:
			offset = corestate.get_reg32(fields['O'] >> 1, thread)
			if not fields['Ou']:
				offset = sign_extend(offset, 32)


		shift = fields['s']
		if shift == 3:
			shift = 2

		# TODO: we probably want a separate code path for bitpacked formats,
		# they always an aligned 32-bits, ignoring the shift, and ignoring the
		# mask to write 3 or 4 floating point results. the current handling
		# of this is a mess.

		# Note that, e.g. format 10 also converts to float, with different
		# behaviour depending on whether or not the byte was loaded from
		# start+3 (mask & 0b1000), but does not ignore the mask. There's a
		# lot going on.

		bit_packed = fields['F'] in (8, 12, 13)
		if bit_packed:
			# weird
			shift = 2

		offset <<= shift

		item_size = {
			1: 2, # i16
			2: 4, # i32
			3: 2, # i16?
			6: 2, # unorm16
			7: 2, # unorm16
		}.get(fields['F'], 1)

		address &= ~(item_size - 1)

		register = fields['R']
		if fields['Rt'] == 1:
			# 32-bit
			register >>= 1

		if fields['Rt'] != 1 and fields['F'] == 2:
			# invalid to load 32-bit values to 16-bit registers
			assert False, "illegal instruction"

		# Result registers are always contiguous (and the count is
		# typically popcount(mask)) but we can skip over memory by
		# using a non-contiguous mask.

		mask = fields['mask']
		if bit_packed:
			address &= ~3
			mask = 0xF

		values = []
		for i in range(4):
			if mask & (1 << i):
				load_address = address + (offset + i) * item_size

				if item_size == 1:
					value = corestate.device_memory.get_byte(load_address)
				elif item_size == 2:
					value = corestate.device_memory.get_u16(load_address)
				elif item_size == 4:
					value = corestate.device_memory.get_u32(load_address)
				else:
					assert False
				values.append((i, value))

		for n, value in values:
			# TODO: need to validate rounding and test edge cases on many of these
			load_real = None
			skip = False
			if fields['F'] < 4:
				load_value = value
			elif fields['F'] == 4:
				load_real = value / 255.0
			elif fields['F'] == 5:
				value = sign_extend(value, 8)
				if value < -127:
					value = -127
				load_real = value / 127.0
			elif fields['F'] == 6:
				load_real = value / 65535.0
			elif fields['F'] == 7:
				value = sign_extend(value, 16)
				if value < -32767:
					value = -32767
				load_real = value / 32767.0
			elif fields['F'] == 8:
				bits = sum(v << (i * 8) for i, v in values)
				if n == 0:
					load_real = (bits & 1023) / 1023.
				elif n == 1:
					load_real = ((bits >> 10) & 1023) / 1023.
				elif n == 2:
					load_real = ((bits >> 20) & 1023) / 1023.
				elif n == 3:
					load_real = ((bits >> 30) & 3) / 3.0
				else:
					load_real = 0
			elif fields['F'] == 10:
				load_real = 0
				if n == 3:
					load_real = value / 255.0
				else:
					load_real = SRGB_TABLE[value]
			elif fields['F'] == 12:
				# TODO: this is (sometimes) wrong rounding to half
				load_real = 0
				bits = sum(v << (i * 8) for i, v in values)

				if n == 0:
					load_real = decode_float11(bits & ((1 << 11) - 1))
				elif n == 1:
					load_real = decode_float11((bits >> 11) & ((1 << 11) - 1))
				elif n == 2:
					load_real = decode_float10((bits >> 22) & ((1 << 10) - 1))
				elif n == 3:
					skip = True
			elif fields['F'] == 13:
				load_real = 0
				bits = sum(v << (i * 8) for i, v in values)

				if n == 0:
					f = bits
				elif n == 1:
					f = (bits >> 9)
				elif n == 2:
					f = (bits >> 18)
				elif n == 3:
					skip = True

				e = (bits >> 27) - 15
				load_real = (2 ** e) * (f & 0x1FF) / 512.0
			else:
				# 14 and 15 look to be similar int32->several floats
				# not sure about the others - maybe one of them was
				# a bad instruction?
				assert False, 'TODO %d' % (fields['F'],)

			# TODO: is this true always when F >= 4? if so, might be good to change
			# the condition and/or separate it out.
			if load_real is not None:
				if fields['Rt']:
					load_value = fma.f32_to_u32(load_real)
				else:
					load_value = fma.f16_to_u16(load_real)

			# TODO: "skip" is a hack to handle bit_packed fields which write to
			# less than four registers.
			if not skip:
				if fields['Rt'] == 1:
					corestate.set_reg32(register, thread, load_value)
				else:
					corestate.set_reg16(register, thread, load_value)
				register += 1

@register
class DeviceStoreInstructionDesc(DeviceLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('device_store', 1)
		self.add_operand(ImmediateDesc('u6', 44, 1)) # store order thing?
		self.add_constant(26, 1, 1) # u1
		self.add_constant(28, 2, 0) # u3
		self.add_constant(45, 2, 2) # u4
		self.add_constant(50, 2, 0) # u5

@register
class DeviceLoadTodoInstructionDesc(DeviceLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('device_load.TODO', 0)
		self.add_operand(ImmediateDesc('u1', 26, 1))
		self.add_operand(ImmediateDesc('u3', 28, 2))
		self.add_operand(ImmediateDesc('u4', 44, 3))
		self.add_operand(ImmediateDesc('u5', 50, 2))

@register
class DeviceStoreTodoInstructionDesc(DeviceLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('device_store.TODO', 1)
		self.add_operand(ImmediateDesc('u6', 44, 1)) # store order thing?
		self.add_operand(ImmediateDesc('u1', 26, 1))
		self.add_operand(ImmediateDesc('u3', 28, 2))
		self.add_operand(ImmediateDesc('u4', 45, 2))
		self.add_operand(ImmediateDesc('u5', 50, 2))

class StackLoadStoreInstructionDesc(InstructionDesc):
	def __init__(self, name, bit):
		super().__init__(name, size=(6, 8), length_bit_pos=47)
		self.add_constant(0, 8, (bit << 7) | 0b00110101)
		#self.add_constant(16, 4, 0b0000)
		#self.add_constant(24, 2, 0b01)

		reg = MemoryRegDesc('R')
		#reg = StackReg32Desc('r', [
		#	(10, 6, 'rl'),
		#	(40, 1, 'rh'),
		#])

#		self.add_operand(ImmediateDesc('rh', 40, 1))
		if not bit:
			self.add_operand(reg)

		self.add_operand(EnumDesc('F', [(8, 2, 'F'), (50, 2, 'Fx')], None, MEMORY_FORMATS))
		self.add_operand(ImmediateDesc('i1', 26, 1))
		self.add_operand(ImmediateDesc('i2', 36, 3))
		#self.add_operand(ImmediateDesc('i3', 49, 1))
		#self.add_operand(ImmediateDesc('i4', 50, 2))

		self.add_operand(EnumDesc('mask', 52, 4, MASK_DESCRIPTIONS))


		self.add_operand(ImmediateDesc('i5', 44, 3))

		if bit:
			self.add_operand(reg)

		self.add_operand(MemoryIndexDesc('O'))

		self.add_operand(ImmediateDesc('i6', 30, 1))

@register
class StackStoreInstructionDesc(StackLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('stack_store', 1)

@register
class StackLoadInstructionDesc(StackLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('stack_load', 0)

@register
class StackGetPtrInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('stack_get_ptr', size=(6, 8), length_bit_pos=47)
		self.add_constant(0, 8, 0b00110101)
		self.add_constant(16, 4, 0b0001)
		self.add_constant(48, 2, 0b10)

		self.add_operand(ImmediateDesc('i0', 8, 2))
		self.add_operand(ImmediateDesc('i1', 26, 1))
		self.add_operand(ImmediateDesc('i2', 36, 3))
		self.add_operand(ImmediateDesc('i4', 50, 6))

		self.add_operand(ImmediateDesc('i3', 44, 3))

		self.add_operand(StackReg32Desc('R', [
			(10, 6, 'R'),
			(40, 2, 'Rx'),
		]))

		#self.add_operand(StackAdjustmentDesc('v'))
		#self.add_operand(ImmediateDesc('i5', 30, 1))

@register
class StackAdjustInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.stack_adjust', size=(6, 8), length_bit_pos=47)
		self.add_constant(0, 8, 0b10110101)
		self.add_constant(16, 4, 0b0001)
		self.add_constant(24, 2, 0b01)

		self.add_operand(ImmediateDesc('i0', 8, 2))
		self.add_operand(ImmediateDesc('i1', 26, 1))
		self.add_operand(ImmediateDesc('i2', 36, 3))
		self.add_operand(ImmediateDesc('i3', 44, 3))
		self.add_operand(ImmediateDesc('i4', 50, 6))

		self.add_operand(StackAdjustmentDesc('v'))
		#self.add_operand(MemoryIndexDesc('idx'))

class ThreadgroupLoadStoreInstructionDesc(InstructionDesc):
	def __init__(self, name, bit):
		super().__init__(name, size=(6, 8)) #(6, 8), length_bit_pos=47) # ?

		self.add_constant(0, 4, 0b1001)
		self.add_constant(5, 2, 0b01 | (bit << 1))

		# bit 7 might be cache/discard for load/store respectively?

		self.add_operand(EnumDesc('F', 24, 4, MEMORY_FORMATS))

		self.add_operand(EnumDesc('mask', 36, 4, MASK_DESCRIPTIONS))

		self.add_operand(ThreadgroupMemoryRegDesc('R'))

		self.add_operand(ThreadgroupMemoryBaseDesc('A'))
		self.add_operand(ThreadgroupIndexDesc('O'))
		#self.add_operand(MemoryShiftDesc('s'))

@register
class ThreadgroupLoadInstructionDesc(ThreadgroupLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('threadgroup_load', 1)

@register
class ThreadgroupStoreInstructionDesc(ThreadgroupLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('threadgroup_store', 0)



class SampleRegDesc(MemoryRegDesc):
	def __init__(self, name):
		super().__init__(name, off=9, offx=72, offt=8)



class SampleURegDesc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)

		# TODO: this ignores the low bit. Kinda confusing?
		self.add_merged_field(self.name, [
			(start, 5, self.name)
		])

	def decode(self, fields):
		v = fields[self.name]
		if fields['Tt'] & 1:
			return UReg64(v * 2)
		else:
			return None

SAMPLE_MASK_DESCRIPTIONS = {}
for _i in range(1, 16):
	SAMPLE_MASK_DESCRIPTIONS[_i] = ''
	if _i & 1: SAMPLE_MASK_DESCRIPTIONS[_i] += 'x'
	if _i & 2: SAMPLE_MASK_DESCRIPTIONS[_i] += 'y'
	if _i & 4: SAMPLE_MASK_DESCRIPTIONS[_i] += 'z'
	if _i & 8: SAMPLE_MASK_DESCRIPTIONS[_i] += 'w'

class TextureDesc(OperandDesc):
	def __init__(self, name, off=32, offx=78, offt=38):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		self.add_field(offt, 2, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		if flags == 0b0:
			return TextureState(value)
		elif flags == 0b01:
			return Reg16(value)
		elif flags == 0b10:
			if value == 0:
				return Immediate(0)
			else:
				return Reg16(value)
		elif flags == 0b11:
			return Reg32(value >> 1)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if isinstance(r, Reg32):
			value = r.n << 1
			flags = 1
		elif isinstance(r, TextureState):
			value = r.n
			flags = 0
		else:
			raise Exception('invalid TextureDesc %r' % (opstr,))

		fields[self.name] = value
		fields[self.name + 't'] = flags


class SamplerDesc(OperandDesc):
	def __init__(self, name, off=56, offx=92, offt=62):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		self.add_field(offt, 1, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		if flags == 0b0:
			return SamplerState(value)
		else:
			return Reg16(value)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if isinstance(r, Reg16):
			value = r.n
			flags = 1
		elif isinstance(r, SamplerState):
			value = r.n
			flags = 0
		else:
			raise Exception('invalid SamplerDesc %r' % (opstr,))

		fields[self.name] = value
		fields[self.name + 't'] = flags

TEX_TYPES = {
	0b000: 'tex_1d',
	0b001: 'tex_1d_array',
	0b010: 'tex_2d',
	0b011: 'tex_2d_array',
	0b100: 'tex_2d_ms',
	0b101: 'tex_3d',
	0b110: 'tex_cube',
	0b111: 'tex_cube_array',
}


TEX_SIZES = {
	0b000: (1, 0), # tex_1d
	0b001: (1, 1), # tex_1d_array
	0b010: (2, 0), # tex_2d
	0b011: (2, 1), # tex_2d_array
	0b100: (2, 1), # tex_2d_ms
	0b101: (3, 0), # tex_3d
	0b110: (3, 0), # tex_cube
	0b111: (3, 1), # tex_cube_array
}

class CoordsDesc(OperandDesc):
	def __init__(self, name, off=16, offx=74, offt=22):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		self.add_field(offt, 1, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		count, extra = TEX_SIZES[fields['n']]

		# not really clear how the alignment requirement works
		if extra:
			t = RegisterTuple(Reg16((value) + i) for i in range(count * 2 + 1))
		else:
			t = RegisterTuple(Reg32((value >> 1) + i) for i in range(count))

		if flags:
			t.flags.append(DISCARD_FLAG)
		return t

class SampleCompareOffDesc(OperandDesc):
	def __init__(self, name, off=80, offx=94, offt=91):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		self.add_field(offt, 1, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		has_offset = bool(flags)
		has_compare = bool(fields['compare'])

		regs = []

		if has_compare:
			regs += [Reg32(value >> 1)]
			value += 2
		if has_offset:
			regs += [Reg16(value)]
			value += 1

		return RegisterTuple(regs)

class LodDesc(OperandDesc):
	def __init__(self, name, off=16, offx=74): #, offt=22):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		#self.add_field(offt, 1, self.name + 't')

	def decode(self, fields):
		#flags = fields[self.name + 't']
		value = fields[self.name]
		if fields['lod'] == 0:
			return Immediate(0)
		elif fields['lod'] in (0b100, 0b1100):
			# Gradient descriptor
			count = TEX_SIZES[fields['n']][0]

			count *= 2
			if fields['lod'] == 0b1100:
				# ..with minimum
				count += 1

			return RegisterTuple(Reg32((value >> 1) + i) for i in range(count))
		elif fields['lod'] in [0b010, 0b001]:
			return UReg16(value)
		else:
			assert fields['lod'] in (0b110, 0b101)
			return Reg16(value)

class TextureLoadSampleBaseInstructionDesc(InstructionDesc):
	def __init__(self, name):
		super().__init__(name, size=(8, 12))

		self.add_operand(ImmediateDesc('compare', 23, 1))
		# unknowns
		self.add_operand(BinaryDesc('q2', 30, 2))
		self.add_operand(BinaryDesc('q3', 43, 5))
		self.add_operand(BinaryDesc('slot', 63, 1)) # slot to pass to wait

		# Bottom bit set with compares?
		self.add_operand(BinaryDesc('q6', 86, 5))

		self.add_operand(EnumDesc('mask', 48, 4, SAMPLE_MASK_DESCRIPTIONS))

		self.add_operand(BinaryDesc('kill', 69, 3)) # Kill helper invocations if set to 1, clear for not

		# output, typically a group of 4.
		self.add_operand(SampleRegDesc('R')) # destination/output

		self.add_operand(SampleURegDesc('U', 64, 5))

		# texture
		self.add_operand(TextureDesc('T'))

		# sampler
		self.add_operand(SamplerDesc('S'))

		self.add_operand(EnumDesc('n', 40, 3, TEX_TYPES))

		# co-ordinates
		self.add_operand(CoordsDesc('C'))

		# TODO: bit more to figure out here
		self.add_operand(EnumDesc('lod', 52, 4, 
		{
			0b0000: 'auto_lod',

			# Argument is a 16-bit uniform
			0b0001: 'auto_lod_bias',
			0b0010: 'lod_min',

			# Argument is a 16-bit register
			0b0101: 'auto_lod_bias',
			0b0110: 'lod_min',
			0b0100: 'lod_grad',
			0b1100: 'lod_grad_min',
		}))
		self.add_operand(LodDesc('D', 24, 76))

		# has offset?
		self.add_operand(SampleCompareOffDesc('O'))

@register
class TextureSampleInstructionDesc(TextureLoadSampleBaseInstructionDesc):
	documentation_html = '<p>The last four bytes are omitted if L=0.</p>'

	def __init__(self):
		super().__init__('texture_sample')
		self.add_constant(0, 8, 0x31)

@register
class TextureLoadInstructionDesc(TextureLoadSampleBaseInstructionDesc):
	documentation_html = '<p>The last four bytes are omitted if L=0.</p>'

	def __init__(self):
		super().__init__('texture_load')
		self.add_constant(0, 8, 0x71)

@register
class ThreadgroupBarriernstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('threadgroup_barrier', size=2)
		self.add_constant(0, 8, 0x68)

@register
class UnknownAfterSampling1InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.after_sampling1', size=2)
		self.add_constant(0, 16, 0x101C)

@register
class UnknownAfterSampling2InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.after_sampling2', size=2)
		self.add_constant(0, 16, 0x62CC)

@register
class Unk30C0InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unk30C0', size=6)
		self.add_constant(0, 16, 0xC030)
		self.add_operand(ImmediateDesc('imm', 16, 32))

@register
class Unk40C0InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unk40C0', size=6)
		self.add_constant(0, 8*6, 0xC040)

@register
class UnkC0000000InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unkC0000000', size=4)
		self.add_constant(0, 32, 0xC0)


@register
class Unk51InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unk51', size=4)
		self.add_constant(0, 31, 0x51)
@register
class UnkE8InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unkE800', size=2)
		self.add_constant(0, 16, 0xE8)
@register
class Unk0CInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unk0C00', size=2)
		self.add_constant(0, 16, 0x0C)

@register
class Unk28InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unk28', size=2)
		self.add_constant(0, 8, 0x28)
		self.add_operand(ImmediateDesc('imm', 8, 8))

@register
class Unk20InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.br20', size=4)
		self.add_constant(0, 16, 0x20)
		self.add_operand(BranchOffsetDesc('off', 16, 8))
		self.add_constant(24, 8, 0)

@register
class UnkB1InstructionDesc(InstructionDesc):
	documentation_html = '<p>The last four bytes are omitted if L=0.</p>'
	def __init__(self):
		super().__init__('TODO.unkB1', size=(6, 10))
		self.add_constant(0, 8, 0xB1)
		self.add_constant(16, 14, 0)
		self.add_constant(38, 3, 0)
		self.add_constant(48, 5, 0)
		self.add_constant(58, 4, 0)
		self.add_constant(68, 12, 0)

class Reg32_4_4_Desc(OperandDesc):
	def __init__(self, name, start, start2):
		super().__init__(name)

		self.add_merged_field(self.name, [
			(start, 4, self.name),
			(start2, 4, self.name + 'x'),
		])

	def decode(self, fields):
		v = fields[self.name]
		return Reg32(v >> 1)

@register
class StackAdjustTodoInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.stack_adjust2', size=(6, 8), length_bit_pos=47)
		self.add_constant(0, 8, 0b10110101)
		self.add_constant(16, 4, 0b0001)
		self.add_constant(24, 2, 0b00)

		self.add_operand(ImmediateDesc('i0', 8, 2))
		self.add_operand(ImmediateDesc('i1', 26, 1))
		self.add_operand(ImmediateDesc('i2', 36, 3))
		self.add_operand(ImmediateDesc('i3', 44, 3))
		self.add_operand(ImmediateDesc('i4', 50, 6))

		self.add_operand(Reg32_4_4_Desc('r', 20, 32))

@register
class TodoSrThingInstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('TODO.sr_thing', size=4)

		self.add_constant(0, 7, 0b1110010)
		self.add_constant(15, 1, 1)
		self.add_operand(ALUDstDesc('D', 28))
		self.add_operand(SReg32Desc('SR', 16, 26))

@register
class TodoPopExecInstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('TODO.pop_exec2', size=6)
		self.add_constant(0, 7, 0x52)
		self.add_constant(9, 2, 3) # op
		self.add_constant(13, 10, 0)
		self.add_constant(23, 1, 1)

		self.add_operand(ImplicitR0LDesc('D'))
		self.add_operand(ImmediateDesc('n', 11, 2))

@register
class Unk75InstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('TODO.unk75', size=8) # maybe: size=(6, 8), length_bit_pos=47)
		self.add_constant(0, 8, 0x75)
		self.add_constant(10, 1, 0)
		self.add_constant(47, 1, 1)

		self.add_constant(16, 4, 1)
		self.add_constant(24, 2, 1)
		self.add_constant(27, 3, 0)
		self.add_constant(31, 1, 0)

		self.add_operand(ExReg32Desc('R', 11, 40))
		self.add_operand(StackAdjustmentDesc('v'))

		self.add_operand(BinaryDesc('q1', 8, 2))
		#self.add_operand(BinaryDesc('q2', 30, 2))
		#self.add_operand(BinaryDesc('q3', 43, 5))
		#self.add_operand(BinaryDesc('kill', 69, 3))
		#self.add_operand(BinaryDesc('q5', 63, 1))
		#self.add_operand(BinaryDesc('q6', 86, 5))

		# TODO: 75 0A 10 05 10 80 12 00

@register
class Unk75AltInstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('TODO.unk75_alt', size=8) # maybe: size=(6, 8), length_bit_pos=47)
		self.add_constant(0, 8, 0x75)
		self.add_constant(10, 1, 0)
		self.add_constant(47, 1, 1)

		#self.add_constant(16, 4, 1)
		#self.add_constant(24, 2, 1)
		#self.add_constant(27, 3, 0)
		#self.add_constant(31, 1, 0)

		self.add_operand(ExReg32Desc('R', 11, 40))
		self.add_operand(StackAdjustmentDesc('v'))

		self.add_operand(BinaryDesc('q1', 8, 2))
		#self.add_operand(BinaryDesc('q2', 30, 2))
		#self.add_operand(BinaryDesc('q3', 43, 5))
		#self.add_operand(BinaryDesc('kill', 69, 3))
		#self.add_operand(BinaryDesc('q5', 63, 1))
		#self.add_operand(BinaryDesc('q6', 86, 5))

		# TODO: 75 0A 10 05 10 80 12 00


@register
class SampleMaskInstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('sample_mask', size=4)
		self.add_constant(0, 8, 0xC1)
		self.add_operand(ImmediateDesc('S', [(16, 6, 'S'), (26, 2, 'Sx')])) # Immediate sample mask
		self.add_constant(15, 1, 0)
		self.add_operand(ImmediateDesc('sample_mask_is_immediate', 23, 1))

@register
class UnkF59InstructionDesc(MaskedInstructionDesc):
	def __init__(self):
		super().__init__('TODO.unkF59', size=2)
		self.add_constant(0, 8, 0xF5)
		self.add_constant(12, 4, 0x9)
		self.add_operand(ImmediateDesc('a', 10, 2))
		self.add_operand(ImmediateDesc('b', 8, 2))

@register
class UnkF503InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unkF503', size=2)
		self.add_constant(0, 16, 0x03F5)

@register
class UnkF533InstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('TODO.unkF533', size=2)
		self.add_constant(0, 16, 0x33F5)

def get_instruction_descriptor(n):
	for o in instruction_descriptors:
		if o.matches(n):
			return o

def disassemble_n(n):
	for o in instruction_descriptors:
		if o.matches(n):
			return o.disassemble(n)

def disassemble_bytes(b):
	n = opcode_to_number(b)
	return disassemble_n(n)

