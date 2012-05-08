#include "transpose.cuh"

__global__ void transposeDiagonal(float *odata, float *idata, int width, int height, int nreps)
	{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	int blockIdx_x, blockIdx_y;

	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
	}

	int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;

	for (int r=0; r < nreps; r++) {
		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
		}

		__syncthreads();

		for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
			odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
		}
	}
}

void transposeImage(float *dev_in, float *dev_out, rect2d image) {
	dim3 grid(image.width/TILE_DIM, image.height/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
	transposeDiagonal<<<grid,threads>>>(dev_out, dev_in, image.width, image.height, 1);
}

cudaImage createTransposedImage(cudaImage input) {
	 cudaImage flipped	= createImage(input.height, input.width);
	 rect2d inputDim	= {input.width, input.height};
	 transposeImage(getData(input), getData(flipped), inputDim);
	 return flipped;
}