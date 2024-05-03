# Apple GPU

This is a project working on reverse engineering the Apple G13 GPU architecture (as used by the M1), and hacking together documentation, a disassembler, an emulator and an assembler along the way. This is a somewhat messy work in progress, but it reflects the best of my understanding.

Documentation can be viewed at https://dougallj.github.io/applegpu/docs.html


## Disassembler

This is probably the most useful bit, still an early work in progress:

```
$ python3 disassemble.py code.bin
   0: 0501040d00c43200     device_load      2, 0, i32, pair, 4, r0_r1, u2_u3, 0, lsl 1
   8: 3800                 wait             0
   a: be890a042c00         convert          u32_to_f, $r2, r0.discard, 1
  10: be810a242c00         convert          u32_to_f, $r0, r1.discard, 1
  16: 9a85c4020200         fmul             $r1, r2.discard, 0.5
  1c: 0a05c282             rcp              r1, r1.discard
  20: 9a81c0020200         fmul             $r0, r0.discard, 0.5
  26: 0a01c082             rcp              r0, r0.discard
  2a: c508803d00803000     uniform_store    2, i16, pair, 0, r1l_r1h, 8
  32: c500a03d00803000     uniform_store    2, i16, pair, 0, r0l_r0h, 10
```

## Assembler

Useful for generating tests. Very hacky, and missing a lot of error checking, so check the disassembly output to see if it gave you what you asked for.

```
$ python3 assemble.py 'fmul $r1, r2.discard, 0.5 ; rcp r1, r1.discard'
9a85c4020200                     fmul             $r1, r2.discard, 0.5
0a05c282                         rcp              r1, r1.discard
```

## Emulator

Emulator is not a standalone tool, nor a final API, and so far has only been used for testing the logic of instructions against hardware. There's no flow control yet, but if you want to use it, it looks something like:

```
cs = applegpu.CoreState()

# use cs.set_reg32/cs.set_reg16 to initialise core state

remaining = instructions
while remaining:
	n = applegpu.opcode_to_number(remaining)
	desc = applegpu.get_instruction_descriptor(n)
	desc.exec(n, cs)
	size = desc.decode_size(n)
	remaining = remaining[size:]

# use cs.get_reg32/cs.get_reg16 to dump core state
```

However, it is used in the hardware tests.

## Hardware tests

Hardware tests run instructions on the actual GPU (so can only run on Apple Silicon), and emulate them, and compare state afterwards. This is achieved overwriting shader code in Metal binary archives with our own shaders.

To run the tests: `python3 hwtest.py`

Note: The python script will automatically compile its C helpers if they didn't yet exist, but not if they do exist but are out of date.  You can manually recompile with `make -C hwtestbed -j8`

Hardware testing is great because it's hard to mess up, quick to add new tests, and we have tests. Most other things about it are currently not great. The tests are a bit scattershot - some tests are thorough, but others only test what had to be work to implement other tests. It can be hard to see what's covered, so I've found it useful to experimentally break things to see if they're being covered by the tests or not.

If you just want to play around, try out `python3 hwtestbed.py <shader code>`.  Use `python3 hwtestbed.py --help` for its full set of options.  If you want to specify a lot of input registers, bash's brace expansion syntax can make this a lot easier e.g. `-r{"0,1","2,3","4,5","6,7"}{,,,,,,,}` expands to `-r0,1 -r2,3 -r4,5 -r6,7 -r0,1 -r2,3 -r4,5 -r6,7 ...` (32 total threads).

## Documentation

Documentation is generated, partially based on the instruction descriptions in `applegpu.py` using:

```
$ python3 genhtml.py > docs.html
```

Most of the text is in `genhtml.py`.
