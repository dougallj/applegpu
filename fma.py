# Based on golang's src/math/fma.go
#
# Copyright (c) 2009-2019 The Go Authors. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
#    * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#    * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import struct

F16_BIAS = 15
F16_FRACTION_BITS = 10
F16_FRACTION_MASK = (1 << F16_FRACTION_BITS) - 1
F16_EXPONENT_BITS = 5
F16_EXPONENT_MASK = (1 << F16_EXPONENT_BITS) - 1
F16_SIGN_SHIFT = F16_EXPONENT_BITS + F16_FRACTION_BITS

F32_BIAS = 127
F32_FRACTION_BITS = 23
F32_FRACTION_MASK = (1 << F32_FRACTION_BITS) - 1
F32_EXPONENT_BITS = 8
F32_EXPONENT_MASK = (1 << F32_EXPONENT_BITS) - 1
F32_SIGN_SHIFT = F32_EXPONENT_BITS + F32_FRACTION_BITS

F64_BIAS = 1023
F64_FRACTION_BITS = 52
F64_FRACTION_MASK = (1 << F64_FRACTION_BITS) - 1
F64_EXPONENT_BITS = 11
F64_EXPONENT_MASK = (1 << F64_EXPONENT_BITS) - 1
F64_SIGN_SHIFT = F64_EXPONENT_BITS + F64_FRACTION_BITS

F64_ONE = F64_BIAS << F64_FRACTION_BITS

F64_NAN_BITS = 0x7FF8000000000000
F64_INFINITY_BITS = 0x7FF0000000000000

ARM64_NANS = False
F64_QNAN_BIT = (1 << 51)

# Internal magic exponent values
ZERO_EXP = -(F64_BIAS * 8)  # zero: ensure it's smaller than the smallest product
INF_EXP = F64_BIAS * 8      # infinity: ensure it's larger than the largest product

def saturate64(bits):
	s = bits >> F64_SIGN_SHIFT
	e = (bits >> F64_FRACTION_BITS) & F64_EXPONENT_MASK
	f = bits & F64_FRACTION_MASK

	if s or (e == F64_EXPONENT_MASK and f):
		return 0
	if bits > F64_ONE:
		return F64_ONE
	return bits

# NOTE: this rounding trick doesn't account for the sign bit
# (e.g. for round to -inf modes)
# although could just use "round(m, ROUND_AWAY_FROM_ZERO) if sign else round(m, ROUND_TO_ZERO)"

# Rounding modes:
#
# Index bits are the least significant 3 bits when rounding, i.e. the first is
# the least-significant-bit if truncated, the second is the half bit, and the
# third is the dirty bit (set if any bit of lesser significance would be set).
# The result is added to the truncated fraction ("1" to round away from zero,
# or "0" to truncate.)
#
# For example, 010 and 110 both indicate an exact result, half way between
# representable values, but round differently in ROUND_NEAREST_EVEN and
# ROUND_TO_ODD modes, depending on whether the truncated value is odd or
# even.

ROUND_NEAREST_EVEN = [
	0, # 000
	0, # 001
	0, # 010
	1, # 011
	0, # 100
	0, # 101
	1, # 110
	1, # 111
]

ROUND_TO_ODD = [
	0, # 000
	1, # 001
	1, # 010
	1, # 011
	0, # 100
	0, # 101
	0, # 110
	0, # 111
]

ROUND_TO_ZERO = [
	0, # 000
	0, # 001
	0, # 010
	0, # 011
	0, # 100
	0, # 101
	0, # 110
	0, # 111
]

ROUND_AWAY_FROM_ZERO = [
	0, # 000
	1, # 001
	1, # 010
	1, # 011
	0, # 100
	1, # 101
	1, # 110
	1, # 111
]

def do_rounding(m, mode=ROUND_NEAREST_EVEN):
	return (m >> 2) + mode[m & 7]

def shr_compress(v, shift):
	flag = 1 if v & ((1 << shift) - 1) else 0
	return (v >> shift) | flag

def u32_to_f32(v):
	return struct.unpack('<f', struct.pack('<I', v))[0]

def f64_to_u64(f64):
	return struct.unpack('<Q', struct.pack('<d', f64))[0]

def u64_to_f64(u64):
	return struct.unpack('<d', struct.pack('<Q', u64))[0]

def u16_to_f16(u64):
	return struct.unpack('<e', struct.pack('<H', u64))[0]

def f16_to_u16(u64):
	return struct.unpack('<H', struct.pack('<e', u64))[0]

def leading_zeroes_64(v):
	return len(format(v, '064b').split('1')[0])

def leading_zeroes_128(v):
	return len(format(v, '0128b').split('1')[0])


def f16_to_f64(bits, ftz=False):
	s = bits >> F16_SIGN_SHIFT
	e = (bits >> F16_FRACTION_BITS) & F16_EXPONENT_MASK
	f = bits & F16_FRACTION_MASK

	if e == F16_EXPONENT_MASK:
		e = F64_EXPONENT_MASK
	elif e == 0:
		if f == 0 or ftz:
			e = 0
			f = 0
		else:
			while (f & (1 << F16_FRACTION_BITS)) == 0:
				f <<= 1
				e -= 1
			e += 1
			f &= F16_FRACTION_MASK
			e = (e - F16_BIAS + F64_BIAS)
	else:
		e = (e - F16_BIAS + F64_BIAS)

	return (s << F64_SIGN_SHIFT) | (e << F64_FRACTION_BITS) | (f << (F64_FRACTION_BITS - F16_FRACTION_BITS))

def f32_to_f64(bits, ftz=False):
	s = bits >> F32_SIGN_SHIFT
	e = (bits >> F32_FRACTION_BITS) & F32_EXPONENT_MASK
	f = bits & F32_FRACTION_MASK

	if e == F32_EXPONENT_MASK:
		e = F64_EXPONENT_MASK
	elif e == 0:
		if f == 0 or ftz:
			e = 0
			f = 0
		else:
			while (f & (1 << F32_FRACTION_BITS)) == 0:
				f <<= 1
				e -= 1
			e += 1
			f &= F32_FRACTION_MASK
			e = (e - F32_BIAS + F64_BIAS)
	else:
		e = (e - F32_BIAS + F64_BIAS)

	return (s << F64_SIGN_SHIFT) | (e << F64_FRACTION_BITS) | (f << (F64_FRACTION_BITS - F32_FRACTION_BITS))


def f64_to_f32(bits, ftz=False):
	s = bits >> F64_SIGN_SHIFT
	e = (bits >> F64_FRACTION_BITS) & F64_EXPONENT_MASK
	f = bits & F64_FRACTION_MASK

	if e == F64_EXPONENT_MASK:
		e = F32_EXPONENT_MASK
	elif e == 0 and f == 0:
		e = 0
	else:
		e = (e - F64_BIAS + F32_BIAS)
		if ftz:
			if e <= 0:
				f = 0
				e = 0
		else:
			if e < -F32_FRACTION_BITS:
				e = 0
				f = 1
			elif e <= 0:
				f |= 1 << F64_FRACTION_BITS
				f = shr_compress(f, -e + 1)
				e = 0
		if e >= F32_EXPONENT_MASK:
			e = F32_EXPONENT_MASK
			f = 0

	f = shr_compress(f, (F64_FRACTION_BITS - F32_FRACTION_BITS - 2))

	return (s << F32_SIGN_SHIFT) + (e << F32_FRACTION_BITS) + do_rounding(f)

def f64_to_f16(bits, ftz=False):
	s = bits >> F64_SIGN_SHIFT
	e = (bits >> F64_FRACTION_BITS) & F64_EXPONENT_MASK
	f = bits & F64_FRACTION_MASK

	if e == F64_EXPONENT_MASK:
		e = F16_EXPONENT_MASK
	elif e == 0 and f == 0:
		e = 0
	else:
		e = (e - F64_BIAS + F16_BIAS)
		if ftz:
			if e <= 0:
				f = 0
				e = 0
		else:
			if e < -F16_FRACTION_BITS:
				e = 0
				f = 1
			elif e <= 0:
				f |= 1 << F64_FRACTION_BITS
				f = shr_compress(f, -e + 1)
				e = 0
		if e >= F16_EXPONENT_MASK:
			e = F16_EXPONENT_MASK
			f = 0

	f = shr_compress(f, (F64_FRACTION_BITS - F16_FRACTION_BITS - 2))

	return (s << F16_SIGN_SHIFT) + (e << F16_FRACTION_BITS) + do_rounding(f)

def split(b):
	sign = b >> 63
	exp = (b >> 52) & F64_EXPONENT_MASK
	mantissa = b & F64_FRACTION_MASK

	if exp == F64_EXPONENT_MASK:
		exp = INF_EXP
		mantissa |= 1 << 52
	elif exp == 0 and mantissa == 0:
		exp = ZERO_EXP
	elif exp == 0:
		# Normalize value if subnormal.
		shift = leading_zeroes_64(mantissa) - 11
		mantissa <<= shift
		exp = 1 - shift
	else:
		# Add implicit 1 bit
		mantissa |= 1 << 52
	return sign, exp, mantissa


def is_snan(b):
	m = b & F64_FRACTION_MASK
	e = (b >> F64_FRACTION_BITS) & F64_EXPONENT_MASK
	return e == F64_EXPONENT_MASK and m != 0 and (m & F64_QNAN_BIT) == 0

def is_nan(b):
	m = b & F64_FRACTION_MASK
	e = (b >> F64_FRACTION_BITS) & F64_EXPONENT_MASK
	return e == F64_EXPONENT_MASK and m != 0

def is_inf(e, m):
	m &= ((1 << 52) - 1)
	return e == INF_EXP and m == 0


def bfma64(bx, by, bz, meta=None, rounding=ROUND_NEAREST_EVEN):
	if meta is None:
		meta = []

	xs, xe, xm = split(bx)
	ys, ye, ym = split(by)
	zs, ze, zm = split(bz)


	if ARM64_NANS:
		for v in (bz, bx, by):
			if is_snan(v):
				return v | F64_QNAN_BIT

		for v in (bz, bx, by):
			if is_nan(v):
				return v
	else:
		for v in (bz, bx, by):
			if is_nan(v):
				return F64_NAN_BITS

	if is_inf(xe, xm) and ym == 0:
		return F64_NAN_BITS

	if xm == 0 and is_inf(ye, ym):
		return F64_NAN_BITS

	# product
	ps = xs ^ ys
	pe = xe + ye - F64_BIAS + 1
	pm = (xm * ym) << 21

	if (is_inf(xe, xm) or is_inf(ye, ym)) and is_inf(ze, zm) and ps != zs:
		return F64_NAN_BITS

	zm <<= 74

	if (xm == 0 or ym == 0) and zm == 0 and zs == ps:
		return zs << 63

	# normalize to 126th bit
	if ((pm >> 126) & 1) == 0:
		pm <<= 1
		pe -= 1

	# Swap addition operands so |p| >= |z|
	if pe < ze or (pe == ze and (pm < zm)):
		ps, pe, pm, zs, ze, zm = zs, ze, zm, ps, pe, pm

	zm = shr_compress(zm, pe-ze)
	if ps == zs:
		# Adding
		pm += zm
		carry = (pm >> 127)
		if carry == 0:
			pe -= 1
		m = shr_compress(pm, (64 + carry))
	else:
		# Subtracting
		pm -= zm
		nz = leading_zeroes_128(pm)
		pe -= nz
		m = shr_compress(pm << (nz - 1), 64)

	if pe > 1022 + F64_BIAS or (pe == 1022 + F64_BIAS and (m+(1 << 9))>>63 == 1):
		# rounded value overflows exponent range
		return (ps << 63) | F64_INFINITY_BITS

	if pe < 0:
		m = shr_compress(m, -pe)
		pe = 0

	if m == 0:
		# exact, unrounded zero gets sign=0
		ps = 0

	m = shr_compress(m, 8)
	m = do_rounding(m, rounding)

	if m == 0:
		pe = 0

	return (ps << 63) + (pe << 52) + m

