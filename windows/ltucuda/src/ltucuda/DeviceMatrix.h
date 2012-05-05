#ifndef _DEVICE_MATRIX_H_
#define _DEVICE_MATRIX_H_

#include "MatrixBase.h"
#include "kernels/fill.cuh"

template <typename T> class DeviceMatrix : public MatrixBase<T> { 
public:
	DeviceMatrix(int width, int height, T* data) : MatrixBase<T>(width, height, data) { 
		allocated = false;
		alloc();
	}

	void copyFromHostMatrix(MatrixBase<T> hostMatrix) {
		width = hostMatrix.getWidth();
		height = hostMatrix.getHeight();
		T* hostData = hostMatrix.getData();
		if (hostData != NULL) {
			cudaMemcpy2D(this->data, this->pitch, hostData, width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice);
			exitOnError("setImageDeviceData");
		}
	}

	T* copyToHost() {
		int w = this->width;
		int h = this->height;
		int bytesNeeded = width*height*sizeof(T);
		T *host;
		mallocHost((void**)&host,bytesNeeded, PINNED, false);
		cudaMemcpy2D(host, w * sizeof(T), this->devicePtr, pitch, w*sizeof(T), h, cudaMemcpyDeviceToHost);
		exitOnError("DeviceMatrix , copyToHost");
		return host;
	}

	void setPitch(int pitch) { this->pitch = pitch; }
	int getPitch() { return this->pitch; }

	bool applyFilter(FillFilter<T> f) {
		if (allocated) {
			f.filter(devicePtr, pitch, width, height);
			return true;
		} else {
			return false;
		}
	}

private:
	
	void alloc() {
		cudaMallocPitch((void **)&data, (size_t*)&this->pitch, this->width * sizeof(T), this->height);
		exitOnError("DeviceMatrix Alloc");
		allocated = true;
	}

	int pitch;
	bool allocated;
}; 


#endif