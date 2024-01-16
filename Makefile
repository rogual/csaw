all:
	find /Users/me/Projects/secf/src -type f | grep -vF windows | grep -vF '.h' | xargs ./csaw
	clang++ -std=c++11 -shared -x objective-c++ \
		-include /Users/me/Projects/secf/src/headers.h \
		unit2.cc
