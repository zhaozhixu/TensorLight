CC ?= gcc
CXX ?= g++
CUCC ?= nvcc
AR = ar cr
ECHO = @echo
SHELL = /bin/sh

ifdef VERBOSE
AT =
else
AT = @
endif

UNAME_S := $(shell uname -s)

CFLAGS = -Wall -std=c99
CXXFLAGS = -std=c++11 -Wall
CUFLAGS = -m64 -arch=sm_30 -ccbin $(CC)
LDFLAGS = $(CFLAGS)
DEFINES =

ifeq ($(DEBUG), yes)
CFLAGS += -g -O0 -DTL_DEBUG
CXXFLAGS += -g -O0 -DTL_DEBUG
CUFLAGS += -lineinfo -O0
LDFLAGS += -g -O0
else
CFLAGS += -O3
CXXFLAGS += -O3
CUFLAGS += -O3
LDFLAGS += -O3
endif

INCPATHS = -I/usr/local/include
LDFLAGS += -L/usr/local/lib -lm

ifeq ($(WITH_CUDA), yes)
DEFINES += TL_CUDA
CFLAGS += -DTL_CUDA
CXXFLAGS += -DTL_CUDA
CUFLAGS += -DTL_CUDA
CUDA_INSTALL_DIR ?= /usr/local/cuda
INCPATHS += -I$(CUDA_INSTALL_DIR)/include
LDFLAGS += -L$(CUDA_INSTALL_DIR)/lib64 -lcudart -lcublas -lcurand -lstdc++
endif

CFLAGS += $(INCPATHS)
CXXFLAGS += $(INCPATHS)
CUFLAGS += $(INCPATHS)

define concat
  $1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend-c
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(CFLAGS) $1
endef

define make-depend-cxx
  $(AT)$(CXX) -MM -MF $3 -MP -MT $2 $(CXXFLAGS) $1
endef

define make-depend-cu
  $(AT)$(CUCC) -M $(CUFLAGS) $1 > $3.$$$$; \
  sed 's,.*\.o[ :]*,$2 : ,g' < $3.$$$$ > $3; \
  rm -f $3.$$$$
endef
