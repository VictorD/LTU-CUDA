#include "mlcuda.h"
#include "lcudatypes.h"
#include <string.h>

#define DEBUG2 1
#if DEBUG2
#define PRINTF2(...) mexPrintf(__VA_ARGS__)
#else
#define PRINTF2(...)
#endif

#define null 0

#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b

typedef struct queue_t{
	int* q;
	int head;
	int last;
	int size;
} queue;

#define QUEUE_HEADER 2
int* createQueues(int num, int size) {
	int* space = (int*) mxMalloc(num*(size+QUEUE_HEADER)*sizeof(int));
	memset(space, 0, num*(size+QUEUE_HEADER)*sizeof(int));
	return space;
}

int* moveQueuesToDevice(int *q, int num, int q_size) {
	int* device;
	cudaMalloc((void**)&device, num*(q_size+QUEUE_HEADER)*sizeof(int));
	cudaMemcpy(device, q, num*(q_size+QUEUE_HEADER)*sizeof(int), cudaMemcpyHostToDevice);
	return device;
}

void freeQueues(int* q) {
	mxFree(q);
}

void enqueue(int *q, int q_num, int q_size, int pos) {
	q_size = q_size+QUEUE_HEADER;
	int start = q_num*q_size;
	q[start+QUEUE_HEADER+q[start+1]++] = pos;
	if (q[start+1] == q_size) {
		q[start+1] = 0;
	}
}

int dequeue(int *q, int q_num, int q_size, int* pos) {
	q_size = q_size+2;
	int start = q_num*q_size;
	if (q[start] == q[start+1]) {
		return 0;
	}
	*pos = q[start+QUEUE_HEADER+q[start]++];
	if (q[start] == q_size)
		q[start] = 0;
	return 1;
}

__device__ void device_enqueue(int *q, int q_num, int q_size, int pos) {
	q_size = q_size+QUEUE_HEADER;
	int start = q_num*q_size;
	q[start+QUEUE_HEADER+q[start+1]++] = pos;
	if (q[start+1] == q_size) {
		q[start+1] = 0;
	}
}

__device__ int device_dequeue(int *q, int q_num, int q_size, int* pos) {
	q_size = q_size+2;
	int start = q_num*q_size;
	if (q[start] == q[start+1]) {
		return 0;
	}
	*pos = q[start+QUEUE_HEADER+q[start]++];
	if (q[start] == q_size)
		q[start] = 0;
	return 1;
}



lcuda8u maxNGPlus(lcuda8u* im, int w, int pos) {
	lcuda8u max;

	max = im[pos-w-1];
	max = MAX(max, im[pos-w]);
	max = MAX(max ,im[pos-w+1]);
	max = MAX(max, im[pos-1]);
	max = MAX(max, im[pos]);
	return max;
}

lcuda8u maxNGMinus(lcuda8u* im, int w, int pos) {
	lcuda8u max;

	max = im[pos+w+1];
	max = MAX(max, im[pos+w]);
	max = MAX(max, im[pos+w-1]);
	max = MAX(max, im[pos+1]);
	max = MAX(max, im[pos]);
	return max;
}

bool NGMinusEnqueue(lcuda8u* marker, lcuda8u* mask, int w, int p) {
	int q;

// ------- START DEFINE ------
#define CHECK(q_pos) \
	q = q_pos; \
	if (marker[q] < marker[p] && marker[q] < mask[q]) { return true; }
// ------- END DEFINE ------
	CHECK(p+w+1);
	CHECK(p+w);
	CHECK(p+w-1);
	CHECK(p+1);
	return false;
}

__global__ void swap(int* qs, lcuda8u* mask, int mask_pitch, lcuda8u* marker, int marker_pitch, int w, int h, int n, int ss, lcuda8u* more);
__global__ void process_q(int* qs, lcuda8u* mask, int mask_pitch, lcuda8u* marker, int marker_pitch, int w, int h, int n, int ss);


#define SPLITS 2

void reconcuda(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[]) {
	lcuda8u* in_mask;
	lcuda8u* in_marker;
	lcuda8u* out_marker;
	lcuda8u* mask;
	lcuda8u* marker;
	int* qs;
	int in_marker_w, in_marker_h;
	int in_sh;
	int w, h;
	int sh, ss;
	int i, j;
	int p;
	int qi;
	mxArray* mxp_marker;



//	if (nrhs != 2)
//		mxErrMsgTxt("The number of input arguments must be 2, "
//				"one marker and mask");
//
//	if (!mxIsUint8(prhs[0])) {
//		mxErrMsgTxt("The marker must be of type uint8");
//	}
//	if (!mxIsUint8(prhs[1])) {
//		mxErrMsgTxt("The mask must be of type uint8");
//	}

	in_marker = (lcuda8u*)mxGetData(prhs[0]);
	in_marker_w = mxGetM(prhs[0]);
	in_marker_h = mxGetN(prhs[0]);
	in_mask = (lcuda8u*)mxGetData(prhs[1]);
	in_sh = in_marker_w/SPLITS;

	h = in_marker_w+2;
	w = in_marker_h+2;
	sh = h/SPLITS;
	ss = w*sh;

	qs = createQueues(SPLITS, ss);

	mask = (lcuda8u *)mxMalloc(h*w*sizeof(lcuda8u));
	marker =  (lcuda8u *)mxMalloc(h*w*sizeof(lcuda8u));

	memset(marker, 0, w*h);
	memset(mask, 0, w*h);

	for (i = 1; i < h - 1; ++i) {
		for (j = 1; j < w - 1; ++j) {
			marker[w*i+j] = in_marker[(j-1)*in_marker_w + (i-1)];
			mask[w*i+j] = in_mask[(j-1)*in_marker_w + (i-1)];
		}
	}

	// RASTER order.
	for (i = 1; i < h - 1; ++i) {
		for (j = 1; j < w - 1; ++j) {
			p = i*w+j;
			marker[p] = MIN(maxNGPlus(marker, w, p), mask[p]);
		}
	}

	// ANTI-RASTER order.
	i = h - 2;
	for (qi = SPLITS-1; qi >= 0; --qi) {
		int next;
		if ( i == h - 2 || i == sh - 1) {
			next = i - in_sh;
		} else {
			next = i - sh;
		}
		for (; i > next; --i) {
			for (j = w - 2; j > 0; --j) {
				p = i*w+j;
				marker[p] = MIN(maxNGMinus(marker, w, p), mask[p]);
				if (NGMinusEnqueue(marker, mask, w, p)) {
					enqueue(qs, qi, ss, p);
				}
			}
		}
	}


	// Move computation to device

	int* device_q;
	device_q = moveQueuesToDevice(qs, SPLITS, ss);

	lcudaMatrix_8u device_mask = lcudaAllocMatrix_8u(w, h);
	lcudaMatrix_8u device_marker = lcudaAllocMatrix_8u(w, h);
	lcudaCpyToMatrix_8u(mask, device_mask);
	lcudaCpyToMatrix_8u(marker, device_marker);

	lcuda8u more[SPLITS];
	lcudaArray_8u device_more = lcudaAllocArray_8u(SPLITS);

	dim3 dimBlock(20, 20);
	dim3 dimGrid(16, 16);

	while (1) {
		process_q<<<dimGrid, dimBlock>>>(device_q,  device_mask.data, device_mask.pitch,
				device_marker.data, device_marker.pitch,  w,  h, SPLITS, ss);
		cudaMemset(device_more.data, 0 , device_more.length);
		cudaMemcpy(more, device_more.data, device_more.length, cudaMemcpyDeviceToHost);
		swap<<<dimGrid, dimBlock>>>(device_q, device_mask.data, device_mask.pitch,
				device_marker.data, device_marker.pitch, w, h, SPLITS, ss, device_more.data);
		cudaMemcpy(more, device_more.data, device_more.length, cudaMemcpyDeviceToHost);
		int con = 0;
		for (i = 0; i < SPLITS; ++i) {
			if (more[i]) {
				con = 1;
				break;
			}
		}
		if (!con) {
			break;
		}
	}

	//lcudaCpyFromMatrix_8u(device_mask, mask);
	//lcudaCpyFromMatrix_8u(device_marker, marker);
	lcudaFreeMatrix_8u(device_mask);
	lcudaFreeMatrix_8u(device_marker);
	lcudaFreeArray_8u(device_more);
	cudaFree(device_q);

	mxp_marker = mxCreateNumericMatrix(in_marker_w, in_marker_h, mxUINT8_CLASS, mxREAL);
	out_marker = (lcuda8u*)mxGetData(mxp_marker);

	for (i = 1; i < h - 1; ++i) {
		for (j = 1; j < w - 1; ++j) {
			out_marker[(j-1)*in_marker_w + (i-1)] = marker[w*i+j];
		}
	}

	mxFree(mask);
	mxFree(marker);
	plhs[0] = mxp_marker;
	freeQueues(qs);
}


__global__ void  process_q(int *qs, lcuda8u* mask, int mask_pitch, lcuda8u* marker, int marker_pitch, int w, int h, int n, int ss) {
	int index = threadIdx.x*threadIdx.y;
	int p;
	int my_h_b = (h/n)*index;
	int my_m_b = my_h_b*w;
	int my_h_t = (h/n)*(index+1) - 1;
	int my_m_t = my_h_t*w+w-1;
	while(device_dequeue(qs, index, ss, &p)) {
// ------- START DEFINE ------
#define CHECK_WHILE(point) \
		pq = point; \
		if (marker[pq] < marker[p] && marker[pq] != mask[pq]) { \
			marker[pq] = MIN(marker[p], mask[pq]); \
			device_enqueue(qs, index, ss, pq); \
		}
// ------- END DEFINE ------
		int pq;
		if (p-w >= my_m_b) {
			CHECK_WHILE(p-w-1);
			CHECK_WHILE(p-w);
			CHECK_WHILE(p-w+1);
		}
		CHECK_WHILE(p-1);
		CHECK_WHILE(p+1);
		if (p+w <= my_m_t) {
			CHECK_WHILE(p+w-1);
			CHECK_WHILE(p+w);
			CHECK_WHILE(p+w+1);
		}
	}
}

__global__ void  swap(int* qs,  lcuda8u* mask, int mask_pitch, lcuda8u* marker, int marker_pitch, int w, int h, int n, int ss, lcuda8u* more) {
	int index = threadIdx.x*threadIdx.y;
	int i;
	int my_b = (h/n)*(index+1) - 1;
	for (i = 0; i < w; ++i) {
		int q1, q2;
		int q1i, q2i;
		q1i = my_b*w+i;
		q2i = (my_b+1)*w+i;
		q1 = MAX(marker[q1i], marker[q1i-1]);
		q1 = MAX(q1, marker[q1i+1]);
		q2 = MAX(marker[q2i], marker[q2i-1]);
		q2 = MAX(q2, marker[q2i+1]);
		if (q1 < q2) {
			if (marker[q1i] != mask[q1i]) {
				marker[q1i] = MIN(q2, mask[q1i]);
				more[index] = 1;
				device_enqueue(qs, index, ss, q1i);
			}
		} else if (q2 < q1) {
			if (marker[q2i] != mask[q2i]) {
				marker[q2i] = MIN(q1, mask[q2i]);
				more[index] = 1;
				device_enqueue(qs, index+1, ss, q2i);
			}
		}
	}
}
