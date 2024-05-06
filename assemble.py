import sys
import applegpu
import disassemble

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

def assemble_multiline(lines, print_asm=False):
	seen_labels = {}
	awaiting_labels = {}
	output = bytearray()
	try:
		for line in lines:
			line = line.strip()
			if ':' in line and not ' ' in line[:line.index(':')]:
				index = line.index(':')
				label = line[:index].strip()
				line = line[index+1:].strip()
				pc = seen_labels[label] = len(output)
				if label in awaiting_labels:
					for (target, offset, size, text) in awaiting_labels[label]:
						asm = assemble_line(text.replace(f'pc+{label}', f'pc+{pc-offset}'))
						if len(asm) != size:
							raise ValueError(f'Instruction "{text}" was a different size before setting the label vs after ({len(asm)} != {size})')
						output[offset:offset+size] = asm
					del awaiting_labels[label]
			if not line:
				continue
			if 'pc+' in line and not line[line.index('pc+') + 3].isdigit():
				index = line.index('pc+')
				label = line[index + 3:].partition(',')[0]
				if ' ' in label:
					raise ValueError(f'Bad label "{label}" (cannot contain spaces)')
				offset = len(output)
				if label in seen_labels:
					output.extend(assemble_line(line.replace(f'pc+{label}', f'pc-{offset-seen_labels[label]}')))
				else:
					asm = assemble_line(line.replace(f'pc+{label}', 'pc+0'))
					if not label in awaiting_labels:
						awaiting_labels[label] = []
					awaiting_labels[label].append((label, offset, len(asm), line))
					output.extend(asm)
			else:
				output.extend(assemble_line(line))
	finally:
		if print_asm:
			disassemble.disassemble(output)
	if awaiting_labels:
		missing = ', '.join(awaiting_labels.keys())
		raise ValueError(f'Missing required labels: {missing}')
	return output

def assemble_file(path, is_binary=False):
	if is_binary:
		with open(path, 'rb') as file:
			return file.read()
	else:
		with open(path, 'r') as file:
			return assemble_multiline(file)

if __name__ == '__main__':
	def printUsageAndExit():
		print('Usage: python3 asssemble.py ([instruction text separated by semicolons] | -i ifile.S [-o ofile.bin] [-quiet])')
		exit(1)
	if len(sys.argv) > 1:
		if sys.argv[1].startswith('-'):
			args = sys.argv[1:]
			ifile = None
			ofile = None
			quiet = False
			while args:
				arg = args.pop(0)
				if arg == '-i':
					ifile = args.pop(0)
				elif arg == '-o':
					ofile = args.pop(0)
				elif arg == '-quiet':
					quiet = True
				else:
					print(f'Unrecognized argument {arg}')
					printUsageAndExit()
			if not ifile:
				print('Input file required')
				printUsageAndExit()
			with open(ifile, 'r') as file:
				asm = assemble_multiline(file, print_asm=not quiet)
			if ofile:
				with open(ofile, 'wb') as file:
					file.write(asm)
		else:
			inp = ' '.join(sys.argv[1:]).split(';')
			assemble_multiline(inp, print_asm=True)
	else:
		printUsageAndExit()
