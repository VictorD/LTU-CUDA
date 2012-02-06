# Define installation location for MATLAB.
export MATLAB = /usr/local/matlab
MEX           = $(MATLAB)/bin/mex
MEXEXT        = .$(shell $(MATLAB)/bin/mexext)

# Define installation location for CUDA and compilation flags compatible
# with the CUDA include files.
CUDAHOME    = /usr/local/cuda
CUDANPPHOME = /usr/local/cuda
CUDAMATLABHOME = /usr/local/cuda/matlab
INCLUDEDIR  = -I$(CUDAHOME)/include -I$(CUDANPPHOME)/include -I$(MATLAB)/extern/include
INCLUDELIB  = -L$(CUDAHOME)/lib -lnpp -lcudart -lcufft -Wl,-rpath,$(CUDAHOME)/lib -Lcommon/lib
INCLUDELIB_CUDA  = -L$(CUDAHOME)/lib64 -L$(CUDANPPHOME)/lib -lnpp64 -lcudart -lcufft
CFLAGS      = -D_GNU_SOURCE -Xcudafe --variadic_macros"
COPTIMFLAGS = #-O3
BUILD_DIR = obj
BIN_DIR = bin

# nvmex is a modified mex script that knows how to handle CUDA .cu files.
NVMEX = $(CUDAHOME)/matlab/nvmex

# LTUCuda code
VPATH += src/lcuda
VPATH += src/lcuda/cu
INCLUDEDIR += -Isrc/lcuda -Icommon/inc
SOURCE_FILES = lcudamorph.c lcudatypes.c lcudafloat.c 

# MatLTUCuda code
VPATH += src/mlcuda
INCLUDEDIR += -Isrc/mlcuda
SOURCE_FILES += mlcuda.c mlcudafloat.c

OBJECTS = $(patsubst %,${BUILD_DIR}/%,${SOURCE_FILES:.c=.o})

           
# Mex files containing cuda code.
VPATH += src/lcuda/cu
SOURCE_FILES_CUDA = lcudamorphreconstruct.cu lcudamorphfloat.cu morphology.cu

OBJECTS_CUDA = $(patsubst %,${BUILD_DIR}/%,${SOURCE_FILES_CUDA:.cu=.o})

# List the mex files to be built.  The .mex extension will be replaced with the
# appropriate extension for this installation of MATLAB, e.g. .mexglx or
# .mexa64.

# Mex files not containing any cuda code.
VPATH += src/mex
MEXFILES = cudaimerode.c       \
           cudaimdilate.c		\
           cudaimalloc.c \
           cudaimfree.c \
           cudaimget.c \
           cudaimclone.c \
           reconstructp.c \
           reconstructcuda.c \
           vhgwCPU.c
           
MEX_BIN = $(MEXFILES:.c=$(MEXEXT))
NVMEX_BIN = $(NVMEXFILES:.cu=$(MEXEXT))

all: $(OBJECTS) $(OBJECTS_CUDA) $(NVMEX_BIN) $(MEX_BIN)

clean:
	rm -f $(BIN_DIR)/*.$(MEXEXT)
	rm -f $(BUILD_DIR)/*.o
	rm -r bin
	rm -r obj

.SUFFIXES: .c

%.mexglx:%.c
	$(MEX) -g -outdir $(BIN_DIR) -f /home/lightweight/cuda/old/Matlab_Cuda_1.1/nvopts.sh $< \
	$(INCLUDEDIR) $(INCLUDELIB) $(OBJECTS) $(OBJECTS_CUDA)

%.mexglx:%.cu
	$(NVMEX) -f /home/lightweight/cuda/old/Matlab_Cuda_1.1/nvopts.sh $< \
	$(INCLUDEDIR) $(INCLUDELIB)


%.mexa64:%.c $(OBJECTS) $(OBJECTS_CUDA)
	$(MEX) -g -outdir $(BIN_DIR) -f $(CUDAMATLABHOME)/nvopts.sh $< \
        $(INCLUDEDIR) $(INCLUDELIB) $(OBJECTS) $(OBJECTS_CUDA)
	
$(BUILD_DIR)/%.o:%.cu
	nvcc -g -G -arch sm_20 --compiler-options -fPIC -c -o $@ $< $(INCLUDEDIR) $(INCLUDELIB_CUDA)

$(BUILD_DIR)/%.o:%.c
	@mkdir -p $(dir $@)
	gcc -g -Wall -c -fPIC $< -o $@ $(INCLUDEDIR) $(INCLUDELIB)

docs:
	doxygen Makefile.doxygen
