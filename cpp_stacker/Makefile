CXX      ?= g++
PREFIX   ?= /usr/local
BINDIR   ?= $(PREFIX)/bin
DESTDIR  ?=

OPTFLAGS := -O3
ifneq ($(PORTABLE),1)
OPTFLAGS += -march=native
endif

CXXFLAGS := -std=c++20 $(OPTFLAGS) -fopenmp -MMD -MP -Wall -Wextra -pedantic -Iinclude $(EXTRA_CXXFLAGS)
LDFLAGS  := -fopenmp -lraw_r $(EXTRA_LDFLAGS)

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/astro_stacker

SRCS := \
	src/raw_loader.cpp \
	src/calibration.cpp \
	src/demosaic.cpp \
	src/star_detector.cpp \
	src/registration.cpp \
	src/flat_field.cpp \
	src/integrator.cpp \
	src/io_export.cpp \
	src/main.cpp

OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean install uninstall

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

install: $(TARGET)
	install -d "$(DESTDIR)$(BINDIR)"
	install -m 755 $(TARGET) "$(DESTDIR)$(BINDIR)/astro_stacker"

uninstall:
	rm -f "$(DESTDIR)$(BINDIR)/astro_stacker"

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR)
