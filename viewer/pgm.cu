#include "pgm.cuh"
#include <cstdio>
#include <cstdlib>
#include <cctype>
/* 
    Source for some of the parsing code: 
    http://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/
*/
void skipFileComments(FILE *fp);

float* loadPGM(const char *filename, int *width, int *height) {
    printf("Loading image %s\n", filename);

    FILE *pgmFile;
    char version[3];

    pgmFile = fopen(filename, "rb");
    if (pgmFile == NULL) {
        printf("pgmparse error: can't open file!\n");
        exit(-1);
    }

    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P5")) {
        fprintf(stderr, "Wrong filetype?\n");
        exit(-1);
    }

    int rows, cols;
    int maxGrey = 0;

    skipFileComments(pgmFile);
    fscanf(pgmFile, "%d", &cols);
    skipFileComments(pgmFile);
    fscanf(pgmFile, "%d", &rows);
    skipFileComments(pgmFile);
    fscanf(pgmFile, "%d", &maxGrey);
    fgetc(pgmFile); // Skip a newline(?)


    int bytesNeeded = sizeof(float)*rows*cols;
    printf("Rows: %d and Cols: %d\n", rows, cols);
    printf("Bytes needed: %d\n", bytesNeeded);
    printf("Max greyscale color: %d\n", maxGrey);
    float *imageData = (float*) malloc(bytesNeeded);

    int i,j, lo, hi;
    if (maxGrey > 255) {
        for(i = 0; i < rows; ++i)  {
            for(j = 0; j < cols; ++j) {
                hi = fgetc(pgmFile);
                lo = fgetc(pgmFile);
                imageData[i*rows+j] = (float)((hi<<8)+lo);
            }
            printf("\n");
        }
    } else {
        for(i = 0; i < rows; ++i) {
            for(j = 0; j < cols; ++j) {
                lo = fgetc(pgmFile);
                imageData[i*cols+j] = lo/255.0;
            }
        }
    }
    fclose(pgmFile);
    *width = cols;
    *height = rows;
    return imageData;
}


void savePGM(const char *filename, float* imageData, int width, int height) {
    printf("Saving image %s\n", filename);

    int i,j;
    FILE *file;
    file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error creating file\n");
        exit(-1);
    } else {
        fprintf(file,"P5\n%d %d\n255\n", width, height);
        for(i = 0; i < height; ++i) {
            for(j = 0; j < width; ++j)
              fputc(255.0*imageData[i*width + j],file);
        }
        fclose(file);
    }
}

void skipFileComments(FILE *fp)
{
	int ch;
	char line[100];

	while ((ch = fgetc(fp)) != EOF && isspace(ch))
		;
	if (ch == '#') {
		fgets(line, sizeof(line), fp);
		skipFileComments(fp);
	} else
		fseek(fp, -1, SEEK_CUR);
}
