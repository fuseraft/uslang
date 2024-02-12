CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic # -fsanitize=address
LDFLAGS := -lcurl

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
BIN_DIR := bin
TEST_DIR := tests

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
INCLUDE_FILES := $(wildcard $(INCLUDE_DIR)/*.h)
MAIN_FILE := $(SRC_DIR)/main.cpp

OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))

EXECUTABLE := $(BIN_DIR)/kiwi

LIB_TEST := ./test.kiwi
PLAY := $(TEST_DIR)/playground.kiwi

.PHONY: clean test play

format:
	find . -iname "*.cpp" -o -iname "*.h" | xargs clang-format -i --style=file

all: format $(EXECUTABLE)

test: $(EXECUTABLE)
	@echo "================================"
	$(EXECUTABLE) $(LIB_TEST)

play: $(EXECUTABLE)
	@echo "================================"
	$(EXECUTABLE) $(PLAY)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

$(EXECUTABLE): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(CXX) $^ -o $@ $(LDFLAGS)
	rm -f $(BUILD_DIR)/main.o

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(INCLUDE_FILES)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

