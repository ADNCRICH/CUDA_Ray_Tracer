CUDA_PATH     ?= /usr/local/cuda-12.3
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
# NVCC_DBG       = -g -G
NVCC_DBG       = -g 

# --ptxas-options=-v for register information
NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_89,code=sm_89

DIR ?= .
SRC ?= *.cu
BIN ?= ../bin
BIN_ ?= ./bin
TMP = ./_tmp

cuda: $(DIR)/$(SRC)
	cd $(DIR) && $(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(BIN)/cuda $(SRC) -I /usr/local/include/libstb/

out: $(BIN_)/cuda
	cd $(DIR) && $(BIN)/cuda

cudart: $(DIR)/$(SRC)
	cd $(DIR) && $(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(BIN)/cuda $(SRC) -I /usr/local/include/libstb/ && $(BIN)/cuda

benchmark: 
# nsys profile, don't output anything
	cd $(TMP) && nsys profile --stats=true $(BIN)/cuda && rm -rf ./*.nsys-rep ./*.sqlite

# profile_basic: cuda
# 	nvprof ./cuda

# # use nvprof --query-metrics
# profile_metrics: cuda
# 	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cuda

clean:
	rm -f cuda cuda.o