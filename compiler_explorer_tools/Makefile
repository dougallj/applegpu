all: metal-archive-extractor metal-compile-tool

metal-archive-extractor: metal-archive-extractor.cpp
	clang -std=c++11 $^ -O2 -o $@

metal-compile-tool: metal-compile-tool.m
	clang $^ -O2 -framework Metal -framework Cocoa -fobjc-arc -o $@

clean:
	rm metal-compile-tool metal-archive-extractor

.phony: clean all
