include ../config.mk
TARGET ?= tensorlight
TARGET_TEST = test_$(TARGET)

CC = gcc
CXX = g++
CUCC = nvcc
AR = ar cr
ECHO = @echo
SHELL = /bin/sh

ifdef VERBOSE
AT =
else
AT = @
endif

CFLAGS = -Wall
CXXFLAGS = -std=c++11 -Wall
CUFLAGS = -m64 -arch=sm_30 -ccbin $(CXX)
LDFLAGS = $(CFLAGS)

ifeq ($(DEBUG), 1)
CFLAGS += -g -O0 -DTL_DEBUG
CXXFLAGS += -g -O0 -DTL_DEBUG
CUFLAGS += -lineinfo
LDFLAGS += -g -O0
else
CFLAGS += -O3
CXXFLAGS += -O3
CUFLAGS +=
LDFLAGS += -O3
endif

INCPATHS = -I/usr/local/include `pkg-config --cflags check`
LDFLAGS += -L/usr/local/lib -lm `pkg-config --libs check` -lpthread

ifeq ($(CUDA_ENABLE), 1)
CUDA_INSTALL_DIR ?= /usr/local/cuda
CFLAGS += -DTL_CUDA
CXXFLAGS += -DTL_CUDA
CUFLAGS += -DTL_CUDA
INCPATHS += -I$(CUDA_INSTALL_DIR)/include
LDFLAGS += -L$(CUDA_INSTALL_DIR)/lib64 -lcuda -lcudart -lcublas -lcurand
endif

CFLAGS += $(INCPATHS)
CXXFLAGS += $(INCPATHS)
CUFLAGS += $(INCPATHS)

OBJDIR = obj
OBJS   = $(patsubst %.c,$(OBJDIR)/%.o,$(wildcard *.c))
OBJS  += $(patsubst %.cpp,$(OBJDIR)/%.o,$(wildcard *.cpp))
OBJS  += $(patsubst %.cc,$(OBJDIR)/%.o,$(wildcard *.cc))
CUOBJS =
ifeq ($(CUDA_ENABLE), 1)
CUOBJS += $(patsubst %.cu,$(OBJDIR)/%.o,$(wildcard *.cu))
endif

SRCOBJS = ../src/obj/lib$(TARGET).so

define concat
  $1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend-c
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(CFLAGS) $1
endef

define make-depend-cxx
  $(AT)$(CXX) -MM -MF $3 -MP -MT $2 $(CFLAGS) $1
endef

define make-depend-cu
  $(AT)$(CUCC) -M $(CUFLAGS) $1 > $3.$$$$; \
  sed 's,.*\.o[ :]*,$2 : ,g' < $3.$$$$ > $3; \
  rm -f $3.$$$$
endef

.PHONY: all clean
all: $(TARGET_TEST)
	$(ECHO) Running tests...
	$(AT)./$(TARGET_TEST)

$(TARGET_TEST): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $^ $(SRCOBJS)
	$(AT)$(CC) -o $@ $^ $(SRCOBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.c
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-c,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-cxx,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cc
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-cxx,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-cu,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR)
