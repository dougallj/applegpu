import subprocess
import os

class MetallibReplacer:
	_toolsDir = os.path.join(os.path.dirname(__file__), 'compiler_explorer_tools')
	_archiveExtractor = os.path.join(_toolsDir, 'metal-archive-extractor')
	validSections = ('__TEXT,__vertex', '__TEXT,__fragment', '__TEXT,__compute', '__TEXT,__text')

	@classmethod
	def _run_extractor(cls, arg, input):
		return subprocess.check_output([cls._archiveExtractor, arg, '-'], input=input).decode("utf-8").split("\n")

	@classmethod
	def _find_section(cls, metallib, name):
		sections = cls._run_extractor('--list-sections', metallib)
		for section in sections:
			(offset, _, rest) = section.partition(' ')
			(size, _, secname) = rest.partition(' ')
			if secname == name:
				return (int(offset), int(size))
		raise ValueError(f'No section {name}')

	@staticmethod
	def _parse_shader(line):
		(offset, _, shader) = line.partition(' ')
		return (int(offset, 0), shader)

	def __init__(self, metallib):
		self.metallib = metallib
		self.shaders = {}
		if not os.path.exists(self._archiveExtractor):
			subprocess.run(['make', '-C', self._toolsDir])
		for section_line in self._run_extractor('--list-sections', metallib):
			(section_off, _, rest) = section_line.partition(' ')
			(section_size, _, section_name) = rest.partition(' ')
			if section_name not in self.validSections:
				continue
			(section_off, section_size) = (int(section_off), int(section_size))
			section_data = metallib[section_off : section_off + section_size]
			(text_off, text_size) = self._find_section(section_data, '__TEXT,__text')
			shaders = sorted((self._parse_shader(x) for x in self._run_extractor('--list-shaders', section_data) if x), key=lambda x: x[0])
			nshaders = len(shaders)
			for i in range(nshaders):
				(shader_off, shader_name) = shaders[i]
				shader_end = shaders[i + 1][0] if i < nshaders - 1 else text_size
				if section_name not in self.shaders:
					self.shaders[section_name] = {}
				self.shaders[section_name][shader_name] = (section_off + text_off + shader_off, shader_end - shader_off)

	def _replace(self, metallib, section, name, replacement):
		(offset, size) = self.shaders[section][name]
		if len(replacement) > size:
			raise ValueError(f'Replacement shader for {name} in {section} is too big to fit in space used by old shader ({len(replacement)} > {size})')
		return metallib[:offset] + replacement + metallib[offset + len(replacement):]

	def replace(self, section, name, replacement):
		"""
		Replace the function with name `name` in section `section` of the metallib `metallib` with the shader code `replacement`
		Returns a new metallib with the shader replaced
		"""
		return self._replace(self.metallib, section, name, replacement)

	def replace_multiple(self, replacements):
		"""
		Replace multiple shaders in the metallib
		`replacements` should be an array of tuples of (section, name, replacement)
		Returns a new metallib with the shader replaced
		"""
		lib = self.metallib
		for (section, name, replacement) in replacements:
			lib = self._replace(lib, section, name, replacement)
	
	def guess_section(self, shader="_agc.main"):
		for section in self.validSections:
			if section in self.shaders:
				return section

if __name__ == '__main__':
	import argparse
	import assemble
	parser = argparse.ArgumentParser(description='replace compiled shaders in metallib binary archives')
	parser.add_argument('-i', '--input', required=True, help='the metallib to modify')
	parser.add_argument('-c', '--code', required=True, help='the replacement code')
	parser.add_argument('-o', '--output', required=True, help='the location to write the modified metallib')
	parser.add_argument('-s', '--section', default='auto', choices=('auto',) + MetallibReplacer.validSections, help='the code section of the shader to replace')
	parser.add_argument('-n', '--name', default='_agc.main', help='the name of the shader to replace')
	parser.add_argument('-b', '--binary', action='store_true', help='treat code as binary instead of assembly')
	args = parser.parse_args()
	code = assemble.assemble_file(args.code, args.binary)
	with open(args.input, 'rb') as ifile:
		metallib = ifile.read()
	replacer = MetallibReplacer(metallib)
	name = args.name
	section = args.section
	if section == 'auto':
		section = replacer.guess_section(name)
	out = replacer.replace(section, name, code)
	with open(args.output, 'wb') as ofile:
		ofile.write(out)

