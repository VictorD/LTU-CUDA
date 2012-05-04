#include "pinnedmem.cuh"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

cudaError_t mallocHost(void** h_mem ,unsigned int memSize, memoryMode memMode, bool wc)
{
    if( PINNED == memMode ) {
#if CUDART_VERSION >= 2020
        return cudaHostAlloc( h_mem, memSize, (wc) ? cudaHostAllocWriteCombined : 0 );
#else
        if (wc) {printf("Write-Combined unavailable on CUDART_VERSION less than 2020, running is: %d", CUDART_VERSION);}
        return cudaMallocHost( h_mem, memSize );
#endif
    }
    else { // PAGEABLE memory mode
        *h_mem = malloc( memSize );
    }

    return cudaSuccess;
}

cudaError_t freeHost(void* h_mem, memoryMode memMode)
{
    if( PINNED == memMode ) {
        return cudaFreeHost(h_mem);
    }
    else {
        free(h_mem);
    }
    return cudaSuccess;
}
