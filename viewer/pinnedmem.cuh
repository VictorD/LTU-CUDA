#ifndef CUDAHELP_H_
#define CUDAHELP_H_
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include "morphology.cuh"

enum memoryMode { PINNED, REGULAR };
cudaError mallocHost(void** h_mem ,uint memSize, memoryMode memMode, bool wc);
cudaError freeHost(void* h_mem, memoryMode memMode);
void exitOnError(const char *whereAt);

#endif
