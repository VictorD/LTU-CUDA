CC=gcc
CUDAHOME    = /usr/local/cuda
INCLUDEDIR  = -I$(CUDAHOME)/include
INCLUDELIB  = -L$(CUDAHOME)/lib -lnpp -lcudart -lcufft -lglut -lglui -lGLEW
CFLAGS=-I. -I../src/lcuda $(INCLUDEDIR) -g
DEPS = morphology.cuh pinnedmem.cuh pgm.cuh

ODIR=obj
_OBJ = guiTest.o morphology.o pinnedmem.o pgm.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

#$(ODIR)/%.o: %.c $(DEPS)
#	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o:%.cu $(DEPS)
	nvcc -g -G -Xptxas -dlcm=cg --ptxas-options=-v -arch sm_13 --compiler-options -fPIC -c -o $@ $< $(INCLUDEDIR) $(INCLUDELIB)

bin/viewer: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(INCLUDELIB)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
