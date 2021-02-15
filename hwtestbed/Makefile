all: compute.metallib main replacer.dylib

compute.metallib: Compute.metal
	xcrun -sdk macosx metal -c  -std=osx-metal2.0 -O3 -o compute.air $^
	xcrun -sdk macosx metallib -o $@ compute.air

main: main.mm
	xcrun -sdk macosx clang++ -framework Metal -framework MetalKit -fobjc-arc -O3 -o $@ main.mm

replacer.dylib: replacer.c
	clang replacer.c -framework IOKit -dynamiclib -o $@
