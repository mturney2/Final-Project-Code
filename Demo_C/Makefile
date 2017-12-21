# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= #-O3
ARCH   	:= -march=native

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := 
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : normXcorr2_FC



normXcorr2_FC: normXcorr_Host_Final.cu 
	module load cuda; nvcc -o normXcorr2_FC -O3 normXcorr_Host_Final.cu -ccbin $(BIN)
 

# TODO: add targets for building executables

.PHONY: clean
clean:
	rm -f normXcorr2_FC
