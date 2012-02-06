#ifndef CUDAFLOAT_H_
#define CUDAFLOAT_H_

/**
 * \defgroup lcudatypes LTU Cuda Type definitions
 * \ingroup lcuda
 * Basic types and allocation/manipulation functions in the LTU Cuda library.
 *
 */
/*@{*/


/**
 * \file lcudatypes.h
 * Header file for the LTU Cuda types and associated functions.
 * \author Henrik Mäkitaavola
 */

#include <cuda_runtime.h>
#include <npp.h>
#include "lcudaextern.h"

/**
 * \typedef lcudaFloat
 * Unsigned 8-bit integer
 */
typedef float lcudaFloat;

/**
 * \struct lcudaMatrix
 * Matrix allocated on the device with lcudaFloat entries.
 */
typedef struct {
	lcudaFloat *data; ///< Pointer to the matrix.
	int pitch; ///< Pitch of the data.
	int width; ///< Width.
	int height; ///< Height.
} lcudaMatrix;

/**
 * \struct lcudaArray
 * Array allocated on the device with lcudaFloat entries.
 */
typedef struct {
	lcudaFloat *data; ///< Pointer to the array.
	int length; ///< Length of the array.
	int width; ///< Optional member that specifies a width, if the array can be seen as a matrix.
	int height; ///< Optional member that specifies a height, if the array can be seen as a matrix.
} lcudaArray;

/**
 * Allocates a matrix on the device.
 *
 * \param width The width of the matrix.
 * \param height The height of the Matrix.
 * \return The matrix allocated on the device.
 */
EXTERN lcudaMatrix lcudaAllocMatrix(int width, int height);

/**
 * Frees the memory on the device allocated by a matrix.
 *
 * \param matrix The matrix allocated on the device that should be freed.
 */
EXTERN void lcudaFreeMatrix(lcudaMatrix matrix);

/**
 * Copies data from the host to a matrix on the device. The size of the data
 * that is being copied to the device must be atleast the same size as the matrix.
 *
 * \param data Data on the host.
 * \param matrix Matrix on the device that the data from the host should be stored in.
 */
EXTERN void lcudaCpyToMatrix(lcudaFloat *data, lcudaMatrix matrix);

/**
 * Copies data from a matrix on the device to the host.
 *
 * \param matrix Matrix on the device that is being copied to the host.
 * \param data Pointer to allocated memory on the host. The allocated space must be atleast the same size as the matrix.
 */
EXTERN void lcudaCpyFromMatrix(lcudaMatrix matrix, lcudaFloat *data);

/**
 * Clones a matrix.
 *
 * \param matrix The matrix that should be cloned.
 * \return A clone of the input matrix.
 */
EXTERN lcudaMatrix lcudaCloneMatrix(lcudaMatrix matrix);

/**
 * Allocates an array on the device.
 *
 * \param length The length of the array that should be allocated.
 * \return The array allocated on the device.
 */
EXTERN lcudaArray lcudaAllocArray(int length);

/**
 * Frees the memory on the device allocated by an array.
 *
 * \param array The array allocated on the device that should be freed.
 */
EXTERN void lcudaFreeArray(lcudaArray array);

/**
 * Copies data from the host to a array on the device. The size of the data
 * that is being copied to the device must be atleast the length of the array.
 *
 * \param data Data on the host.
 * \param array Array on the device that the data from the host should be stored in.
 */
EXTERN void lcudaCpyToArray(lcudaFloat *data, lcudaArray array);

/**
 * Copies data from a array on the device to the host.
 *
 * \param array Array on the device that is being copied to the host.
 * \param data Pointer to allocated memory on the host. The allocated space must be atleast the same length as the array.
 */
EXTERN void lcudaCpyFromArray(lcudaArray array, lcudaFloat *data);

/*}@*/

#endif /* CUDAFLOAT_H_ */
