.SUFFIXES:

SRCDIR = src
TESTDIR = test

.PHONY: all libso test clean info

all: libso test

libso:
	@(cd $(SRCDIR) && make)

test:
	@(cd $(TESTDIR) && make)

clean:
	@(cd $(SRCDIR) && make clean);\
	(cd $(TESTDIR) && make clean)

info:
	@echo "Available make targets:"
	@echo "  all: make shared object and tests"
	@echo "  libso: make shared object"
	@echo "  test: make tests"
	@echo "  clean: clean all object files"
	@echo "  info: show this infomation"
