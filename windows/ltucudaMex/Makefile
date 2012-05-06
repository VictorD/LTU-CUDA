
mexBinary    = H:\MATLAB\R2010a\bin\mex.bat
cudaLibPath  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64"
lcudaLibPath = "E:\LTUCUDA\LTU-CUDA\windows\ltucuda\lib"

OBJS= cudaimalloc.obj cudaimget.obj
LIBS= -lltucuda -lcudart

all: $(OBJS)

$(OBJS): 
    $(mexBinary) $@ -L$(cudaLibPath) -L$(lcudaLibPath) $(LIBS)