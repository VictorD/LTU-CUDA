#ifndef MLCUDAFLOAT_H_
#define MLCUDAFLOAT_H_
/**
 * \defgroup mlcuda Matlab LTU Cuda
 * The Matlab LTU Cuda library contains functions to make it easier to interact
 * with the LTU Cuda library in Mex functions.
 */
/*@{*/
/**
 * \file mlcuda.h
 * \brief Main include file for the Matlab LTU Cuda library.
 *
 * \author Henrik Mäkitaavola
 */

#include <mex.h>
#include "lcudamorphfloat.cuh"
#include "lcuda.h"

/**
 * Allocates a matrix on the device and then copies a mxArray into this matrix.
 *
 * \param mx A pointer to the mxArray that should be copied to the device.
 * \return The matrix which contains the copied data.
 */
lcudaMatrix mlcudaMxToMatrix(const mxArray* mx);

/**
 * Allocates an mxArray on the host and then copies data from a matrix on the device to it.
 *
 * \param matrix The matrix on the device that should be copied to the host mxArray.
 * \return a pointer to a mxArray containing the data that was copied from the matrix on the device.
 */
mxArray* mlcudaMatrixToMx(lcudaMatrix matrix);

/**
 * Allocates a array on the device and then copies a mxArray into this array.
 * All the rows in the mxArray will be copied to the array in sequence.
 *
 * \param mx A pointer to the mxArray that should be copied to the device.
 * \return The array which contains the copied data.
 */
lcudaArray mlcudaMxToArray(const mxArray* mx);

/**
 * Allocates an mxArray on the host and then copies data from a array on the device to it.
 *
 * \param array The array on the device that should be copied to the host mxArray.
 * \return a pointer to a mxArray containing the data that was copied from the array on the device.
 */
mxArray* mlcudaArrayToMx(lcudaArray array);

/**
 * Creates a matrix from information contained in a Matlab struct.
 *
 * \param mx The matlab struct containing the information about the device matrix.
 */
lcudaMatrix mlcudaStructToMatrix(const mxArray* mx);

/**
 * Creates a Matlab struct containing information about a matrix on the device.
 *
 * \param matrix The matrix that the matlab struct should contain information about.
 */
mxArray *mlcudaMatrixToStruct(lcudaMatrix matrix);

/**
 * Creates a structuring element that can be used in a erosion or dilation from
 * a Matlab struct.
 *
 * \param mx The Matlab struct, preferably created with cudastrel().
 */
lcudaStrel_8u mlcudaStructToStrel(const mxArray* mx);

/**
 * Dilates or erodes an image on the device.
 *
 * \param im The image that should be dilated or eroded on the device.
 * \param lcudaSe The structuring element the image should be eroded/dilated with.
 * \param dilate True if the operation performed should be dilation, false for erode.
 */
void mlcudaErodeDilate(lcudaMatrix im, lcudaStrel_8u lcudaSe, bool dilate);

/*@}*/

#endif /* MLCUDAFLOAT_H_ */
