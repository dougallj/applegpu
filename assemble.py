import sys
import applegpu

# TODO: labels, file i/o

def begin_encoding(mnem, operand_strings):
	for o in applegpu.instruction_descriptors:
		fields = o.fields_for_mnem(mnem, operand_strings)
		if fields is not None:
			return o, fields
	raise Exception('unknown mnem %r' % mnem)

def assemble_line(line):
	line = line.strip()
	operands = []
	parts = line.split(None, 1)
	mnem = parts[0]
	if len(parts) == 1:
		operand_strings = []
	else:
		assert len(parts) == 2
		operand_strings = [i.strip() for i in parts[1].split(',')]

	idesc, fields = begin_encoding(mnem, operand_strings)

	operand_strings = idesc.rewrite_operands_strings(mnem, operand_strings)
	for opdesc, opstr in zip(idesc.ordered_operands, operand_strings):
		opdesc.encode_string(fields, opstr)

	for opdesc in idesc.ordered_operands[len(operand_strings):]:
		opdesc.encode_string(fields, '')

	n = idesc.encode_fields(fields)

	return idesc.to_bytes(n)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		inp = ' '.join(sys.argv[1:]).split(';')
		for line in inp:
			asm_bytes = assemble_line(line)
			print(asm_bytes.hex().ljust(32), applegpu.disassemble_bytes(asm_bytes))
	else:
		print('usage: python3 asssemble.py [instruction text separated by semicolons]')
		exit(1)
