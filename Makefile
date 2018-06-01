.SUFFIXES:
TARGET = tensorlight
MAJOR = 0
MINOR = 1
MICRO = 0
LIBTARGET_A = lib$(TARGET).a
LIBTARGET_SO = lib$(TARGET).so
LIBTARGET_SO_MAJOR_MINOR = $(LIBTARGET_SO).$(MAJOR).$(MINOR)
LIBTARGET_SO_MAJOR_MINOR_MICRO = $(LIBTARGET_SO).$(MAJOR).$(MINOR).$(MICRO)

SRC_DIR = src
OBJ_DIR = $(SRC_DIR)/obj
TEST_DIR = test
BUILD_DIR = build
BUILD_INCLUDE_DIR = $(BUILD_DIR)/include
BUILD_LIB_DIR = $(BUILD_DIR)/lib
INSTALL_DIR ?= /usr/local
INSTALL_INCLUDE_DIR = $(INSTALL_DIR)/include
INSTALL_LIB_DIR = $(INSTALL_DIR)/lib
PKGCONFIG_DIR ?= /usr/lib/pkgconfig

OBJ_A = $(OBJ_DIR)/$(LIBTARGET_A)
OBJ_SO = $(OBJ_DIR)/$(LIBTARGET_SO)
SRC_HEADERS = $(wildcard $(SRC_DIR)/tl_*.h)
HEADERS = $(patsubst $(SRC_DIR)/%.h,%.h,$(SRC_HEADERS))

BUILD_A = $(BUILD_LIB_DIR)/$(LIBTARGET_A)
BUILD_SO = $(BUILD_LIB_DIR)/$(LIBTARGET_SO)
BUILD_SO_MAJOR_MINOR = $(BUILD_LIB_DIR)/$(LIBTARGET_SO_MAJOR_MINOR)
BUILD_SO_MAJOR_MINOR_MICRO = $(BUILD_LIB_DIR)/$(LIBTARGET_SO_MAJOR_MINOR_MICRO)
BUILD_HEADERS = $(patsubst %.h,$(BUILD_INCLUDE_DIR)/%.h,$(HEADERS))

INSTALL_A = $(INSTALL_LIB_DIR)/$(LIBTARGET_A)
INSTALL_SO = $(INSTALL_LIB_DIR)/$(LIBTARGET_SO)
INSTALL_SO_MAJOR_MINOR = $(INSTALL_LIB_DIR)/$(LIBTARGET_SO_MAJOR_MINOR)
INSTALL_SO_MAJOR_MINOR_MICRO = $(INSTALL_LIB_DIR)/$(LIBTARGET_SO_MAJOR_MINOR_MICRO)
INSTALL_HEADERS = $(patsubst %.h,$(INSTALL_INCLUDE_DIR)/%.h,$(HEADERS))

ifdef VERBOSE
AT =
else
AT = @
endif

define make-build-dir
  $(AT)if [ ! -d $(BUILD_DIR) ]; then mkdir -p $(BUILD_DIR); fi
  $(AT)if [ ! -d $(BUILD_INCLUDE_DIR) ]; then mkdir -p $(BUILD_INCLUDE_DIR); fi
  $(AT)if [ ! -d $(BUILD_LIB_DIR) ]; then mkdir -p $(BUILD_LIB_DIR); fi
  cp $(SRC_HEADERS) $(BUILD_INCLUDE_DIR)
  cp $(OBJ_A) $(BUILD_A)
  cp $(OBJ_SO) $(BUILD_SO)
  cp $(OBJ_SO) $(BUILD_SO_MAJOR_MINOR)
  cp $(OBJ_SO) $(BUILD_SO_MAJOR_MINOR_MICRO)
endef

define make-install-dir
  $(AT)if [ ! -d $(INSTALL_DIR) ]; then mkdir -p $(INSTALL_DIR); fi
  $(AT)if [ ! -d $(INSTALL_INCLUDE_DIR) ]; then mkdir -p $(INSTALL_INCLUDE_DIR); fi
  $(AT)if [ ! -d $(INSTALL_LIB_DIR) ]; then mkdir -p $(INSTALL_LIB_DIR); fi
  cp $(BUILD_HEADERS) $(INSTALL_INCLUDE_DIR)
  cp $(BUILD_A) $(INSTALL_A)
  cp $(BUILD_SO) $(INSTALL_SO)
  cp $(BUILD_SO_MAJOR_MINOR) $(INSTALL_SO_MAJOR_MINOR)
  cp $(BUILD_SO_MAJOR_MINOR_MICRO) $(INSTALL_SO_MAJOR_MINOR_MICRO)
endef

.PHONY: all lib test clean info install uninstall

all: lib test

install:
	$(call make-install-dir)
	$(AT)perl scripts/gen_pkgconfig.pl $(INSTALL_DIR) $(MAJOR).$(MINOR).$(MICRO) $(PKGCONFIG_DIR)

test: lib
	$(AT)(cd $(TEST_DIR) && make)

lib:
	$(AT)(cd $(SRC_DIR) && make)
	$(call make-build-dir)

clean:
	$(AT)(cd $(SRC_DIR) && make clean);\
	(cd $(TEST_DIR) && make clean);\
	rm -rf $(BUILD_DIR)

uninstall:
	rm $(INSTALL_HEADERS)
	rm $(INSTALL_A)
	rm $(INSTALL_SO)
	rm $(INSTALL_SO_MAJOR_MINOR)
	rm $(INSTALL_SO_MAJOR_MINOR_MICRO)
	rm $(PKGCONFIG_DIR)/tensorlight.pc

info:
	@echo "Available make targets:"
	@echo "  all: compile library and run tests"
	@echo "  lib: compile library"
	@echo "  test: compile and run tests"
	@echo "  install: install headers and library files"
	@echo "  clean: clean up all object files"
	@echo "  uninstall: uninstall headers and library files"
	@echo "  info: show this infomation"
