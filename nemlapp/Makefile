MOOSE_SUBMODULE := $(CURDIR)/moose
ifneq ($(wildcard $(MOOSE_SUBMODULE)/framework/Makefile),)
  MOOSE_DIR ?= $(MOOSE_SUBMODULE)
else
  MOOSE_DIR ?= $(shell dirname `pwd`)/srlife/moose
endif

FRAMEWORK_DIR := $(MOOSE_DIR)/framework
include $(FRAMEWORK_DIR)/build.mk
include $(FRAMEWORK_DIR)/moose.mk

ALL_MODULES        := no
HEAT_TRANSFER      := yes
SOLID_MECHANICS    := yes
include $(MOOSE_DIR)/modules/modules.mk

NEML_DIR ?= neml
ADDITIONAL_INCLUDES := -I$(NEML_DIR)/include
ADDITIONAL_LIBS     := -L$(NEML_DIR)/lib -lneml -Wl,-rpath,$(abspath $(NEML_DIR)/lib)

APPLICATION_DIR  := $(CURDIR)
APPLICATION_NAME := nemlapp
BUILD_EXEC       := yes
DEP_APPS         := $(shell $(FRAMEWORK_DIR)/scripts/find_dep_apps.py $(APPLICATION_NAME))
GEN_REVISION     := no
include $(FRAMEWORK_DIR)/app.mk
