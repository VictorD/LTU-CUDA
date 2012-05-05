#ifndef _LCUDA_MATRIX_BASE_
#define _LCUDA_MATRIX_BASE_

template <typename T> class MatrixBase {
public:
	MatrixBase(int width, int height, T* data) : width(width), height(height), data(data) {}
	void setHeight(int height) { this->height = height; }
	void setWidth(int width) { this->width = width; }
	void setData(T *data) { this->data = data; }
	int getHeight() { return this->height; }
	int getWidth() { return this->width; }
	T* getData() { return this->data; }
protected:
	int height;
	int width;
	T* data;
};

#endif