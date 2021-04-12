import applegpu
from applegpu import instruction_descriptors

class Opcode(object):
	def __init__(self, bits, byte_hints=True, hint_lines=None):
		self.hint_lines = hint_lines or []
		self.byte_hints = byte_hints
		self.bits = [None] * bits
		self.colspan = [1] * bits
		self.borders = ['left right'] * bits
		self.left ='left'
		self.right = 'right'
		self.extra_row = [''] * bits
		self.extra_colspan = [1] * bits

	def add_constant(self, offset, size, value, name=None):
		assert (value & ~((1 << size) - 1)) == 0
		for i in range(size-2):
			self.borders[offset + 1 + i] = ''
		if size >= 2:
			self.borders[offset] = self.right
			self.borders[offset + size - 1] = self.left
		for i in range(size):
			assert self.bits[offset + i] is None
			self.bits[offset + i] = (value >> i) & 1

	def add_field(self, offset, size, name):
		on_extra_row = False
		for i in range(size):
			if self.bits[offset + i] is not None:
				on_extra_row = True
				break
		if on_extra_row:
			for i in range(size):
				assert not self.extra_row[offset + i]
				self.extra_row[offset + i] = name
				self.extra_colspan[offset + i] = size
		else:
			for i in range(size):
				assert self.bits[offset + i] is None
				self.bits[offset + i] = name
				self.colspan[offset + i] = size

	def to_html(self):
		for i in range(len(self.bits)-1):
			if self.bits[i] is None and self.bits[i+1] is None:
				self.borders[i] = self.borders[i].replace(self.left, '').strip()
				self.borders[i+1] = self.borders[i+1].replace(self.right, '').strip()

		parts = []
		parts.append('<div class="opcodewrapper wrapped">')

		start = 0
		for i in [16, 32, 40, 48, 64]:
			if len(self.bits) > i and i-start > 8:
				if self.bits[i] and self.bits[i-1] and self.bits[i] == self.bits[i-1]:
					pass
				else:
					parts.append(self.to_html_line(start, i))
					start = i

		if start < len(self.bits):
			parts.append(self.to_html_line(start))

		parts.append('</div>')

		parts.append('<div class="opcodewrapper notwrapped">')
		parts.append(self.to_html_line())
		parts.append('</div>')
		return ''.join(parts)

	def to_html_line(self, line_low=None, line_high=None):
		if line_low is None:
			line_low = 0
		if line_high is None:
			line_high = len(self.bits)

		start = line_high - 1
		end = line_low - 1
		step = -1

		parts = ['<table class="opcodebits">']

		parts.append('<thead><tr>')

		for i in  range(start, end, step):
			if (i+1) % 8 == 0 and self.byte_hints:
				parts.append('<td class="%s">%d</td>' % (self.left, i))
			elif i % 8 == 0 and self.byte_hints:
				parts.append('<td class="%s">%d</td>' % (self.right, i))
			elif i in self.hint_lines:
				parts.append('<td class="%s">%d</td>' % (self.left, i))
			else:
				parts.append('<td>%d</td>' % i)
		parts.append('</tr></thead>')

		parts.append('<tbody><tr>')

		o = start
		while o > end:
			i = self.bits[o]
			css_class = self.borders[o]
			if i is None:
				css_class = ('unknown ' + css_class).strip()
			parts.append('<td colspan="%d" class="%s">' % (self.colspan[o], css_class))
			if i is None:
				parts.append('?')
			elif isinstance(i, int):
				parts.append('%d' % i)
			elif isinstance(i, str):
				parts.append('%s' % i)
			parts.append('</td>')
			o -= self.colspan[o]

		parts.append('</tr>')
		parts.append('</tbody>')

		if any(self.extra_row):
			parts.append('<tfoot>')
			parts.append('<tr>')
			o = start
			while o > end:
				i = self.extra_row[o]
				css_class = 'left right' if i else ''
				parts.append('<td colspan="%d" class="%s">' % (self.extra_colspan[o], css_class))
				parts.append('%s' % i)
				parts.append('</td>')
				o -= self.extra_colspan[o]
			parts.append('</tr>')
			parts.append('</tbody>')


		parts.append('</table>')
		return ''.join(parts)


print('''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Apple G13 GPU Architecture Reference</title>

  <style>
	body {
		padding: 20px;
		font-family: sans-serif;
	}
	table.opcodebits {
		border-collapse: collapse;
		text-align: center;
		font-size: small;
		margin-top: 6px;
		margin-bottom: 6px;
		display: inline-block;
		vertical-align: top;
		/* margin-left: auto; */
	}
	table.opcodebits thead td {
		font-size: smaller; width: 18px;
	}
	table.opcodebits tbody {
		border-left: 1px solid #000;
		border-right: 1px solid #000;
	}
	table.opcodebits tbody td {
		border-top: 1px solid #000;
		border-bottom: 1px solid #000;
	}
	table.opcodebits td.left {
		border-left: 1px solid #000;
	}
	table.opcodebits td.right {
		border-right: 1px solid #000;
	}
	pre {
		margin-left: 15px;
	}
	.unknown {
		color: #ccc;
	}
	body.wrap .notwrapped {
		display: none;
	}
	.wrapped {
		display: none;
	}
	body.wrap .wrapped {
		display: block;
	}

	div.wrapped {
		direction: rtl;
		text-align: left;
	}
	div.wrapped > table {
		direction: ltr;
		display: inline-block;
		margin-right: -1px;
	}

	body.right .opcodewrapper {
		text-align: right !important;
	}
  </style>
</head>
<script>
function update(){
  if (document.getElementById("wrapcheckbox").checked){
  	document.body.classList.add('wrap');
  } else {
  	document.body.classList.remove('wrap');
  }
  if (document.getElementById("rightcheckbox").checked){
  	document.body.classList.add('right');
  } else {
  	document.body.classList.remove('right');
  }
}
</script>
<body class="wrap">

<p>
This document attempts to describe the Apple G13 GPU architecture, as used
in the M1 SoC. This is based on reverse engineering and is likely to have mistakes.
The Metal Shading Language is typically used to program these GPUs,
and this document uses Metal terminology. For example a CPU <em>SIMD-lane</em>
is a Metal <em>thread</em>, and a CPU <em>thread</em> is a Metal <em>SIMD-group</em>.
</p>

<p>
The G13 architecture has 32 threads per SIMD-group. Each SIMD-group has a
stack pointer (<code>sp</code>), a program counter (<code>pc</code>), a 32-bit
execution mask (<code>exec_mask</code>), and up to 128 general purpose registers.
</p>

<p>
General purpose registers each store one 32-bit value per thread. Each register
can be accessed as a 32-bit register, named <code>r0</code> to <code>r127</code>,
the low 16-bits of the register <code>r0l</code> to <code>r127l</code>, or
the high 16-bits of the register <code>r0h</code> to <code>r127h</code>.
Some instructions may also use pairs of contiguous 32-bit registers to operate
on 64-bit values, and memory operations may use up to four contiguous 16 or
32-bit registers (in both cases, encoded as the first register).
</p>

<p>
A number of physical registers are allocated to each SIMD-group, and registers
from <code>r0</code> to <code>r(N-1)</code> may be used, but accesses to higher
register numbers are not valid, and may read or corrupt data from other
SIMD-groups. Using fewer registers (e.g. by using 16-bit types instead of
32-bit types) allows more SIMD-groups to fit in the physical register file
(higher occupancy), which improves performance.
</p>

<p>
Certain instructions are hardcoded to use early registers. <code>r0l</code>
tracks the execution mask stack, and <code>r1</code> is used as the link register.
</p>

<p>
Shared state is less well understood, but includes 256 32-bit uniform registers
named <code>u0</code> to <code>u255</code> and similarly accessible as their
16-bit halves, <code>u0l</code> to <code>u255h</code>. These are used for values
that are the same across all threads, such as <code>threads_per_grid</code>, or
addresses of buffers.
</p>

<h2>Conditional Execution</h2>

<p>
Each thread within a SIMD-group may be deactivated, meaning the values in registers
for that thread will keep their current value. Whether or not each thread is active
is tracked in a 32-bit execution mask (in this document, by convention, a one bit
indicates the thread is active and a zero bit indicates it is not).
</p>

<p>
<code>r0l</code> tracks the execution mask stack. When used with flow-control
instructions, the value in <code>r0l</code> indicates how many 'pop' operations
will be needed to re-enable an inactive thread, or zero if the thread is active.
</p>

<p>
The execution mask stack instructions (<code>pop_exec</code>, <code>if_*cmp</code>,
<code>else_*cmp</code> and <code>while_*cmp</code>) are typically used to manage
<code>r0l</code>. They also update <code>exec_mask</code> based on the value in
<code>r0l</code>, and are the only known way to manipulate <code>exec_mask</code>.
However, <code>r0l</code> may be manipulated with other instructions. It should
be initialised to zero prior to using execution mask stack instructions, and
<code>break</code> statements may be implemented by conditionally moving
a value (corresponding to the number of stack-levels to break out of) to
<code>r0l</code> and using a <code>pop_exec 0</code> (which deactivates any
threads that now have non-zero values in <code>r0l</code>).
</p>

<p>
The <code>jmp_exec_none</code> instruction may be used to jump over an <code>if</code>
statement body if no threads are active, and the <code>jmp_exec_any</code> may be used
to jump back to the start of the loop only if one or more threads are still executing
the loop.
</p>

<p>
Execution masking generally prevents reading or writing values from inactive threads,
however SIMD shuffle instructions can read from inactive threads in some cases, which
can be valuable for research or debugging purposes (e.g. it allows observing non-zero
values in <code>r0l</code>).
</p>

<h2>Register Cache</h2>

<p>The GPU has a register cache, which keeps the contents of recently used general
purpose registers more quickly accessible. When instructions read or write GPRs,
they usually allow hints for the access to be encoded. The <code>cache</code> hint,
on a source or destination operand, indicates the value will be used again,
and should be cached (meaning other values, where this hint was not used, will
be preferred for eviction). The <code>discard</code> hint (on a source operand)
invalidates the value in the register cache after all operands have been read,
without writing it back to the register file.</p>

<p>While the <code>cache</code> hint should only change performance, the
<code>discard</code> hint will make future reads undefined, which could
lead to confusing issues. <code>discard</code> should probably not
be used within conditional execution, as inactive threads within the SIMD-group
may contain data that has not been written back to the register file that would
probably also also get discarded. The behaviour of this hint when execution is
partially or completely masked has not been tested.</p>

<p>
Either hint may be used multiple times even if the same operand appears twice,
and <code>discard</code> on a source register can be used with <code>cache</code>
on the same destination register.
</p>

''')


def html(s):
	return str(s).replace('&', '&amp;').replace('<', '&lt;')

for o in instruction_descriptors:
	if o.sizes[0] != o.sizes[1]:
		# if it has set opcode bits past the end of the short encoding,
		# it must be encoded with the long encoding
		if o.bits & ~((1 << (o.sizes[0] * 8)) - 1):
			o.add_constant(o.length_bit_pos, 1, 1)
		else:
			o.add_field(o.length_bit_pos, 1, 'L')

def operand_class_name(operand):
	n = operand.__class__.__name__
	if n.endswith('Desc'):
		n = n[:-len('Desc')]
	return n

def trim_indentation(s):
	s = s.replace('\t', '  ')
	lines = s.split('\n')
	indents = []
	for line in lines:
		if not line.strip():
			continue
		indents.append(len(line) - len(line.lstrip(' ')))
	m = min(indents) if indents else 0
	while lines and not lines[0].strip():
		lines.pop(0)
	while lines and not lines[-1].strip():
		lines.pop()
	return '\n'.join(i[m:] for i in lines)


print('''
<h1>Instructions</h1>

<p>
Instructions vary in length in multiples of two bytes up to twelve bytes (so far).
Some instructions have a long and short encoding. This is indicated by a bit
<code>L</code>, which, if zero, indicates the last two (or four) bytes of the
instruction are omitted, and any bits within those bytes should be read as zero.
So far only 12-byte instructions omit the last four bytes, and all others omit
the last two.
</p>

<p>
The encodings are described in little-endian, meaning bytes go right-to-left
(and top-to-bottom), but bits may be read in the usual numerical order.
</p>

<p>
Behaviour is mostly described in a Python-like pseudocode for now. In operand
descriptions, the <code>:</code> operator describes bit concatenation, with
the most-significant fields first. Elsewhere, values are considered to be
Python-style arbitrary-precision integers, and floating-point values are
considered to be arbitrary-precision floats (although double-precision with
round-to-odd may be an adequate approximation).
</p>

<p>
<label for="wrapcheckbox">
  <input type="checkbox" id="wrapcheckbox" onclick="update()" checked="checked">
  Wrap bit diagrams
</label>
<br>
<label for="rightcheckbox">
  <input type="checkbox" id="rightcheckbox" onclick="update()">
  Right-align bit diagrams
</label>
</p>
''')

for o in instruction_descriptors:
	if o.documentation_skip or o.name.startswith('TODO.'): # or not hasattr(o, 'pseudocode'):
		continue

	if hasattr(o, 'documentation_begin_group'):
		print('<h2>' + html(o.documentation_begin_group) + '</h2>')

	builder = Opcode(o.sizes[-1] * 8)

	for offset, size, value in o.constants:
		builder.add_constant(offset, size, value)

	for offset, size, name in o.fields:
		builder.add_field(offset, size, name)

	name = o.name
	if hasattr(o, 'documentation_name'):
		name += ' (%s)' % (o.documentation_name,)
	print('<h3>%s</h3>' % html(name))
	if hasattr(o, 'documentation_html'):
		print(o.documentation_html)
	print(builder.to_html())

	print('<pre>')
	any_operands = False
	for operand in o.ordered_operands:
		if getattr(operand, 'documentation_skip', False):
			continue
		arguments = []
		for name, subfields in operand.merged_fields:
			arguments.append(':'.join(name for _, _, name in subfields[::-1])) #' ' + html(name) + ': ' + html(subfields))
		for _, _, name in operand.fields:
			arguments.append(name)
		arguments += getattr(operand, 'documentation_extra_arguments', [])
		if getattr(operand, 'documentation_no_name', False):
			value = ', '.join(arguments)
		else:
			value = html(operand_class_name(operand)) + '(' + ', '.join(arguments) + ')'
		print(html(operand.name) + ' = ' + value)
		any_operands = True
	if any_operands:
		print()
	if hasattr(o, 'pseudocode'):
		print(html(trim_indentation(o.pseudocode)))
	else:
		print('TODO()')
	print('</pre>')

def operand_name(operand):
	n = operand.__name__
	if n.endswith('Desc'):
		n = n[:-len('Desc')]
	return n


print('<h1>Operands</h1>')

for o in applegpu.documentation_operands:

	print('<h2>' + html(operand_name(o)) + '</h2>')
	print('<pre>')
	print(html(trim_indentation(o.pseudocode.format(name=operand_name(o)))))
	print('</pre>')

print('<h1>Helper Pseudocode</h1>')
print('<pre>')
print(html(trim_indentation(applegpu.helper_pseudocode)))
print('</pre>')


print('''</body>
</html>''')
