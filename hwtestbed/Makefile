all: compute.metallib hwtestbed

compute.metallib: compute.air
	xcrun -sdk macosx metallib -o $@ $^

hwtestbed: hwtestbed.o runner-mac.o
	xcrun -sdk macosx clang++ -framework Metal -fobjc-arc -O3 -o $@ $^

%.air: %.metal
	xcrun -sdk macosx metal -MMD -c -std=osx-metal2.0 -O3 -o $@ $<

%.o: %.cpp
	xcrun -sdk macosx clang++ -MMD -c -std=c++11 -O3 -o $@ $<

%.o: %.mm
	xcrun -sdk macosx clang++ -MMD -c -std=c++11 -O3 -fobjc-arc -o $@ $<

-include *.d

clean:
	rm *.metallib *.air *.o *.d hwtestbed

.PHONY: clean all
