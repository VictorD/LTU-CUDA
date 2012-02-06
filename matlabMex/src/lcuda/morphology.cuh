/**
 * \defgroup lcudamorph LTU Cuda Morphological operations
 * \ingroup lcuda
 *
 * Morphological operations in the LTU Cuda library.
 *
 */
/*@{*/

#ifndef LCUDAMORPHOLOGYHELPER_H_
#define LCUDAMORPHOLOGYHELPER_H_
#include "lcudatypes.h"
#include "lcudafloat.h"
#include "lcudamorph.h"
#include "lcudaextern.h"

void lcudaCopyBorder(lcudaMatrix src, lcudaMatrix dst, int color, int offsetX, int offsetY);
EXTERN void lcudaCopyBorder_8u(lcudaMatrix_8u src, lcudaMatrix_8u dst, int color, int offsetX, int offsetY);

void performErosion(const lcudaFloat * pSrc, Npp32s nSrcStep, lcudaFloat * pDst, Npp32s nDstStep, NppiSize srcROI, 
                        const Npp8u * pMask, const float* maskHeight, NppiSize maskSize, NppiPoint anchor, NppiSize borderSize, char isFlat, int seBinary);

EXTERN void performErosion_8u(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize srcROI, 
                        const Npp8u * pMask, const float * maskHeight, NppiSize maskSize, NppiPoint anchor, NppiSize borderSize, char isFlat, int seBinary);

void performDilation(const lcudaFloat * pSrc, Npp32s nSrcStep, lcudaFloat * pDst, Npp32s nDstStep, NppiSize srcROI, 
                        const Npp8u * pMask, const float* maskHeight, NppiSize maskSize, NppiPoint anchor, NppiSize borderSize, char isFlat, int seBinary);
EXTERN void performDilation_8u(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize srcROI, 
                        const Npp8u * pMask, const float * maskHeight, NppiSize maskSize, NppiPoint anchor, NppiSize borderSize, char isFlat, int seBinary);

#endif /* LCUDAMORPHOLOGYHELPER_H_ */
