.SUFFIXES:
TARGET = cnnsim
TARGET_TEST = test_$(TARGET)

CC = gcc
CUCC = nvcc

CFLAGS = -Wall
CUFLAGS = -m64 -arch=sm_35 -ccbin $(CC)
LDFLAGS = $(CFLAGS)

ifdef DEBUG
CFLAGS += -g -O0 -DDEBUG
CUFLAGS += -lineinfo
LDFLAGS += -g -O0
else
CFLAGS += -O3 -DNDEBUG
CUFLAGS +=
LDFLAGS += -O3
endif

ifdef VERBOSE
AT =
else
AT = @
endif

AR = ar cr
ECHO = @echo
SHELL = /bin/sh

define concat
  $1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(CFLAGS) $1
endef

define make-depend-cu
  $(AT)$(CUCC) -M $(CUFLAGS) $1 > $3.$$$$; \
  sed 's,.*\.o[ :]*,$2 : ,g' < $3.$$$$ > $3; \
  rm -f $3.$$$$
endef

TESTDIR = .
SRCDIR = ../src
SRCS_C_TMP = $(wildcard $(SRCDIR)/*.c)
SRCS_CXX_TMP = $(wildcard $(SRCDIR)/*.cpp)
SRCS_CU_TMP = $(wildcard $(SRCDIR)/*.cu)

SRCS_C = $(patsubst $(SRCDIR)/$(TARGET).c, , $(SRCS_C_TMP))
SRCS_CXX = $(patsubst $(SRCDIR)/$(TARGET).cpp, , $(SRCS_CXX_TMP))
SRCS_CU = $(patsubst $(SRCDIR)/$(TARGET).cu, , $(SRCS_CU_TMP))

SRCS_C_TEST = $(wildcard $(TESTDIR)/*.c)
SRCS_CXX_TEST = $(wildcard $(TESTDIR)/*.cpp)
SRCS_CU_TEST = $(wildcard $(TESTDIR)/*.cu)

OUTDIR = .
OBJDIR = $(call concat,$(OUTDIR),/obj)
OBJS   = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS_C))
OBJS  += $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS_CXX))
CUOBJS = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SRCS_CU))

OBJS  += $(patsubst $(TESTDIR)/%.c, $(OBJDIR)/%.o, $(SRCS_C_TEST))
OBJS  += $(patsubst $(TESTDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS_CXX_TEST))
CUOBJS += $(patsubst $(TESTDIR)/%.cu, $(OBJDIR)/%.o, $(SRCS_CU_TEST))

TRIPLE?=x86_64-linux
CUDA_INSTALL_DIR = /usr/local/cuda-8.0
CUDA_LIBDIR = lib
INCPATHS    =-I"$(CUDA_INSTALL_DIR)/include" -I"/usr/local/include"
LIBPATHS    =-L"$(CUDA_INSTALL_DIR)/targets/$(TRIPLE)/$(CUDA_LIBDIR)" -L"/usr/local/lib" -L"/usr/local/cuda/lib64" -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)"
LIBS = $(LIBPATHS) `pkg-config --libs opencv check` -lpthread
# LIBS += -lcudart -lcudart_static -lnvinfer
CFLAGS += $(INCPATHS) `pkg-config --cflags opencv check`
CUFLAGS += $(INCPATHS) `pkg-config --cflags opencv check`
# LIBS = $(LIBPATHS) -lcudart -lcudart_static -lnvinfer `pkg-config --libs opencv` -lcheck -pthread -lrt -lm -lsubunit
# CFLAGS += $(INCPATHS) `pkg-config --cflags opencv`
# CUFLAGS += $(INCPATHS) `pkg-config --cflags opencv`

LDFLAGS += $(LIBS)

.PHONY: all
all: $(OUTDIR)/$(TARGET_TEST)
	$(ECHO) Running tests...
	$(AT)$(OUTDIR)/$(TARGET_TEST)

$(OUTDIR)/$(TARGET_TEST): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $^
	$(AT)$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-cu,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(TESTDIR)/%.c
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(TESTDIR)/%.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(TESTDIR)/%.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-cu,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR)

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(OBJDIR)/*.d
endif
