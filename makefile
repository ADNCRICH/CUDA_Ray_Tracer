CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
# NVCC_DBG       = -g -G
NVCC_DBG       = -g

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_89,code=sm_89

DIR ?= .
SRC ?= *.cu
BIN ?= ../bin
BIN_ ?= ./bin

cuda: $(DIR)/$(SRC)
	cd $(DIR) && $(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(BIN)/cuda $(SRC) -I /usr/local/include/libstb/

out: $(BIN_)/cuda
	$(BIN_)/cuda

cudart: $(DIR)/$(SRC)
	cd $(DIR) && $(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(BIN)/cuda $(SRC) -I /usr/local/include/libstb/ && $(BIN)/cuda

profile_basic: cudart
	nvprof ./cudart

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart

clean:
	rm -f cudart cudart.o