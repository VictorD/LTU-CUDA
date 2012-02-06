#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "pinnedmem.cuh"

cudaError
mallocHost(void** h_mem ,uint memSize, memoryMode memMode, bool wc)
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

cudaError
freeHost(void* h_mem, memoryMode memMode)
{
    if( PINNED == memMode ) {
        return cudaFreeHost(h_mem);
    }
    else {
        free(h_mem);
    }
    return cudaSuccess;
}

/*
 * exitOnError: Show the error message and terminate the application.
 */ 
void exitOnError(const char *whereAt) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error at %s: %s\n", whereAt, cudaGetErrorString(error));
            exit(-1);
        }
}
