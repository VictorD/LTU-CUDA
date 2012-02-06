#include "mlcuda.h"
#include <string.h>

#define null 0

#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b

typedef struct queue_t{
	int* q;
	int head;
	int last;
	int size;
} queue;

queue* createQueue(int size) {
	queue* q = (queue*) mxMalloc(sizeof(queue));
	q->q = (int*) mxMalloc(size*sizeof(int));
	q->head = 0;
	q->last = 0;
	q->size = size;
	return q;
}

void freeQueue(queue* q) {
	mxFree(q->q);
	mxFree(q);
}

void enqueue(queue *q, int pos) {
	PRINTF("Enqueue %d\n", pos);
	q->q[q->last++] = pos;
	if (q->last == q->size) {
		q->last = 0;
	}
}

int dequeue(queue *q, int* pos) {
	if (q->head == q->last) {
		return 0;
	}
	*pos = q->q[q->head++];
	PRINTF("Dequeue %d\n", *pos);
	if (q->head == q->size)
		q->head = 0;
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

void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[]) {
	lcuda8u* in_mask;
	lcuda8u* in_marker;
	lcuda8u* out_marker;
	lcuda8u* mask;
	lcuda8u* marker;
	queue* q;
	int in_mask_w, in_mask_h;
	int in_marker_w, in_marker_h;
	int w, h;
	int i, j;
	int p;
	mxArray* mxp_marker;



	if (nrhs != 2)
		mxErrMsgTxt("The number of input arguments must be 2, "
				"one marker and mask");

	if (!mxIsUint8(prhs[0])) {
		mxErrMsgTxt("The marker must be of type uint8");
	}
	if (!mxIsUint8(prhs[1])) {
		mxErrMsgTxt("The mask must be of type uint8");
	}

	in_marker = (lcuda8u*)mxGetData(prhs[0]);
	in_marker_w = mxGetM(prhs[0]);
	in_marker_h = mxGetN(prhs[0]);
	in_mask = (lcuda8u*)mxGetData(prhs[1]);
	in_mask_w = mxGetM(prhs[1]);
	in_mask_h = mxGetN(prhs[1]);

	q = createQueue(in_marker_w*in_marker_h);

	h = in_marker_w+2;
	w = in_marker_h+2;

	mask = mxMalloc(h*w*sizeof(lcuda8u));
	marker =  mxMalloc(h*w*sizeof(lcuda8u));

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
	for (i = h - 2; i > 0; --i) {
		for (j = w - 2; j > 0; --j) {
			p = i*w+j;
			marker[i*w+j] = MIN(maxNGMinus(marker, w, p), mask[p]);
			if (NGMinusEnqueue(marker, mask, w, p)) {
				enqueue(q, p);
			}
		}
	}

	while(dequeue(q, &p)) {
// ------- START DEFINE ------
#define CHECK_WHILE(point) \
		pq = point; \
		if (marker[pq] < marker[p] && marker[pq] != mask[pq]) { \
			marker[pq] = MIN(marker[p], mask[pq]); \
			enqueue(q, pq); \
		}
// ------- END DEFINE ------
		int pq;
		CHECK_WHILE(p-w-1);
		CHECK_WHILE(p-w);
		CHECK_WHILE(p-w+1);
		CHECK_WHILE(p-1);
		CHECK_WHILE(p+1);
		CHECK_WHILE(p+w-1);
		CHECK_WHILE(p+w);
		CHECK_WHILE(p+w+1);
	}

	mxp_marker = mxCreateNumericMatrix(in_marker_w, in_marker_h, mxUINT8_CLASS, mxREAL);
	out_marker = mxGetData(mxp_marker);

	for (i = 1; i < h - 1; ++i) {
		for (j = 1; j < w - 1; ++j) {
			out_marker[(j-1)*in_marker_w + (i-1)] = marker[w*i+j];
		}
	}

	mxFree(mask);
	mxFree(marker);
	plhs[0] = mxp_marker;
	freeQueue(q);
}
