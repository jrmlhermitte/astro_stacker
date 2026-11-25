CXX      := g++
CXXFLAGS := -std=c++20 -O3 -march=native -fopenmp -MMD -MP -Wall -Wextra -pedantic -Iinclude
LDFLAGS  := -fopenmp -lraw_r

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

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR)
