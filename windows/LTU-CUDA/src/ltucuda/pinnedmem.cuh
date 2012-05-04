#ifndef PINNEDMEM_H_
#define PINNEDMEM_H_

#include <cuda_runtime_api.h>

enum memoryMode { PINNED, REGULAR };
cudaError_t mallocHost(void** h_mem ,unsigned int memSize, memoryMode memMode, bool wc);
cudaError_t freeHost(void* h_mem, memoryMode memMode);

#endif
