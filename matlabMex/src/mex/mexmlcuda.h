/**
 * \defgroup mexmlcuda Mex Matlab LTU Cuda
 * The Mex Matlab LTU Cuda library contains functions that are callable directly
 * from Matlab.
 */


#ifndef MEXMLCUDA_H_
#define MEXMLCUDA_H_

#if DOXYGEN

/**
 * \struct mexMlcudaImStruct
 * \ingroup mexmlcuda
 * Struct containing the necessary information about a image on the device needed
 * to be passed between Matlab and mex functions.
 *
 * MARKER: Note that this struct is not actually available in the c code. It only
 * shows what the structure looks like and the names of its fields.
 * Is actually composed by a mxArray of class
 * struct.
 */
typedef struct {
	unsigned long address; ///< Address to the data of the image on the device.
	int width; ///< Width of the image.
	int height; ///< Height of the image.
	int pitch; ///< Pitch of the image data.
} mexMlcudaImStruct;

#endif

#endif /* MEXMLCUDA_H_ */
