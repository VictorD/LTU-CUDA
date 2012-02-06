#ifndef LCUDA_PGM_H_
#define LCUDA_PGM_H_

float* loadPGM(const char *filename, int *width, int *height);
void savePGM(const char *filename, float* imageData, int width, int height);

#endif
