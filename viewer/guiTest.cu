/****************************************************************************

  A simple GLUT program using the GLUI User Interface Library

  This program sets up a checkbox and a spinner, both with live variables.
  No callbacks are used.

  -----------------------------------------------------------------------
	   
  9/9/98 Paul Rademacher (rademach@cs.unc.edu)

  -----------------------------------------------------------------------

  CUDA application by Victor Danell (victor.danell@gmail.com)

****************************************************************************/
#include <cstdlib>
#include <string.h>
#include <GL/glew.h>
#include <GL/glui.h>

#include <iostream>
#include <memory>
#include <cassert>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "morphology.cuh"
#include "pgm.cuh"
#include <sys/time.h>
#include <cuda_gl_interop.h>

#define GL_TEXTURE_TYPE GL_TEXTURE_2D

GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint texid;
GLuint shader;

inline cudaError deviceReset()
{
#if CUDART_VERSION >= 4000
	return cudaDeviceReset();
#else
	return cudaThreadExit();
#endif
}

rect2d image;
float *host_in = NULL;
float *dev_in = NULL;
int    pitch_in;
timeval start, stop, result;
float *padded_dev_in;
float *padded_dev_in2;
float *data;
float *data2; 
int padded_pitch;
int padded_pitch2;

// Mask Settings
rect2d border;
int size = 43;
morphMask vhgwMaskHorizontal;
morphMask vhgwMaskVertical;
morphMask diagonalMask1;
morphMask crossMask;

/** These are the live variables passed into GLUI ***/
int hozFilter = 1;
int vertFilter = 1;
int crossFilter = 1;
int diagonalFilter = 1;
int main_window;


void loadImage(int argc, char **argv) {
    if (argc >= 1) {
        host_in = loadPGM(argv[1], &image.width, &image.height);
        dev_in = copyImageToDevice(host_in, image, &pitch_in);
    } else {
        printf("Please specify the path to a PGM image. (e.g. ../images/hmap.pgm)\n");
        exit(EXIT_FAILURE);
    }
}

void createSTREL() {
    vhgwMaskHorizontal = createFlatHorizontalLineMask(size);
    vhgwMaskVertical = createFlatVerticalLineMask(size);

    unsigned char crossMaskData[9] = {0,1,0,1,1,1,0,1,0};
    crossMask = createFlat3x3Mask(crossMaskData);


    unsigned char diagonal1Data[9] = {1,0,0,0,1,0,0,0,1};
    diagonalMask1 = createFlat3x3Mask(diagonal1Data);


    // Put large padding on image    
    border.width = 150;
    border.height = 150;

    // Create padded arrays on device, filled with 255.0f
    padded_dev_in = createPaddedArray(border, image, 255.0f, &padded_pitch);
    padded_dev_in2 = createPaddedArray(border, image, 255.0f, &padded_pitch2);


}

void initCUDA(int argc, char **argv) {
    loadImage(argc, argv);
    createSTREL();
}

void idleTime( void )
{
  /* According to the GLUT specification, the current window is 
     undefined during an idle callback.  So we need to explicitly change
     it if necessary */
  if ( glutGetWindow() != main_window ) 
    glutSetWindow(main_window);  

  glutPostRedisplay();
}

void computeFPS()
{
    char fps[256];
    float ifps = 1.0f / (result.tv_sec + result.tv_usec/ 1000000.0f);
    sprintf(fps, "Square filter size %dx%d @ %2.1f fps", 2*size+1, 2*size+1, ifps);  
    glutSetWindowTitle(fps);
}

// display results using OpenGL
void display()
{
    float* d_result;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void **)&d_result, &num_bytes, cuda_pbo_resource);

    // Erode
    gettimeofday(&start, NULL);
    vhgwMaskHorizontal.width = 2*size+1;
    vhgwMaskVertical.height = 2*size+1;

    float* nextSrc = padded_dev_in;
    int srcPitch = padded_pitch;

    float* nextDst = padded_dev_in2;
    int dstOffset = border.height * padded_pitch2/sizeof(float) + border.width;
    int dstPitch  = padded_pitch2;

    float *tmp = NULL;
    int tmpPitch = 0;

    // Reset image to original each frame
    data = padded_dev_in + border.height * padded_pitch/sizeof(float) + border.width;
    cudaMemcpy2D(data, padded_pitch, dev_in, pitch_in, image.width*sizeof(float), image.height, cudaMemcpyDeviceToDevice);
    exitOnError("Copy image data into padded array!");

    if (hozFilter) {
        performErosion(nextSrc, srcPitch, nextDst + dstOffset, dstPitch, image, vhgwMaskHorizontal, border);
        nextSrc = nextDst;
        dstOffset = border.height * srcPitch/sizeof(float) + border.width;
        dstPitch  = srcPitch;
    } 

    if (vertFilter) {
        performErosion(nextSrc, srcPitch, nextDst + dstOffset, dstPitch, image, vhgwMaskVertical, border);
        nextSrc = nextDst;
        dstOffset = border.height * srcPitch/sizeof(float) + border.width;
        dstPitch  = srcPitch;
    } 

    for(int i = 0; i < crossFilter; ++i) {
        performErosion(nextSrc, srcPitch, nextDst + dstOffset, dstPitch, image, crossMask, border);
        tmp = nextSrc;
        nextSrc = nextDst;
        nextDst = tmp;

        tmpPitch = dstPitch;
        dstPitch  = srcPitch; 
        srcPitch = tmpPitch;
    }

   for(int i = 0; i < diagonalFilter; ++i) {
        performErosion(nextSrc, srcPitch, nextDst + dstOffset, dstPitch, image, diagonalMask1, border);
        tmp = nextSrc;
        nextSrc = nextDst;
        nextDst = tmp;

        tmpPitch = dstPitch;
        dstPitch  = srcPitch; 
        srcPitch = tmpPitch;
    }

    // Update the CUDA/GL resource
    int srcOffset = border.height * srcPitch/sizeof(float) + border.width;
    cudaMemcpy2D(d_result, pitch_in, nextSrc + srcOffset, srcPitch, 
                    image.width*sizeof(float), image.height, cudaMemcpyDeviceToDevice);

    gettimeofday(&stop, NULL);
    timersub(&stop, &start, &result);

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    
    glClear(GL_COLOR_BUFFER_BIT);

    // Load PBO as texture 
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.width, image.height, GL_LUMINANCE, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Activate Fragment Shader
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    if (GL_TEXTURE_TYPE == GL_TEXTURE_2D) {
        glTexCoord2f(0.0f, 0.0f);          
        glVertex2f(0.0f, 1.0f);
        glTexCoord2f(1.0f, 0.0f);          
        glVertex2f(1.0f, 1.0f);
        glTexCoord2f(1.0f, 1.0f);          
        glVertex2f(1.0f, 0.0f);
        glTexCoord2f(0.0f, 1.0f);          
        glVertex2f(0.0f, 0.0f);
    }
    glEnd();
    glBindTexture(GL_TEXTURE_TYPE, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
	glutSwapBuffers();
    glutReportErrors();

    if (hozFilter || vertFilter || crossFilter || diagonalFilter) {
        computeFPS();
    }
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    float *host_out;
    switch(key) {
        case 27:
            // Exit on Esc
            exit(0);
            break;
        case 's':
            // Save a screenshot
            float* d_result;
            cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
            size_t num_bytes; 
            cudaGraphicsResourceGetMappedPointer((void **)&d_result, &num_bytes, cuda_pbo_resource);
            cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
        
            char filename[256];    
            sprintf(filename, "image_%s_%s_%dx%d_%d.pgm", (hozFilter ? "hoz":""), (vertFilter ? "vert":""), 2*size+1,2*size+1,(int)result.tv_usec);  
            host_out = copyImageToHost(d_result, image, pitch_in);
            savePGM(filename, host_out, image.width, image.height);
            break;
        case '=':
        case '+':
            if (size < 100) {
                size++;
            }
            break;
        case '-':
            if (size > 1) {
                size--;
            }
            break;
        default:
            break;
    }
}

void initGL(int *argc, char **argv)
{
    /****************************************/
    /*   Initialize GLUT and create window  */
    /****************************************/

    glutInit(argc, argv);
    glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
    glutInitWindowPosition( 50, 50 );
    glutInitWindowSize( 800, 600 );

    main_window = glutCreateWindow( "LTU CUDA Morphology 2011" );
    glutDisplayFunc( display );
    glutReshapeFunc( reshape );
    glutKeyboardFunc(keyboard);

    glewInit();

    if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
        PRINTF("Error: failed to get minimal extensions for demo\n");
        PRINTF("This sample requires:\n");
        PRINTF("  OpenGL version 2.0\n");
        PRINTF("  GL_ARB_fragment_program\n");
        PRINTF("  GL_EXT_framebuffer_object\n");
        exit(-1);
    }

    if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
        PRINTF("Error: failed to get minimal extensions for demo\n");
        PRINTF("This sample requires:\n");
        PRINTF("  OpenGL version 1.5\n");
        PRINTF("  GL_ARB_vertex_buffer_object\n");
        PRINTF("  GL_ARB_pixel_buffer_object\n");
        exit(-1);
    }

    /****************************************/
    /*         Here's the GLUI code         */
    /****************************************/

    GLUI *glui = GLUI_Master.create_glui( "LTU Morphology 2011" );
    new GLUI_Checkbox( glui, "Horizontal", &hozFilter );
    new GLUI_Checkbox( glui, "Vertical", &vertFilter );
    (new GLUI_Spinner( glui, "Mask Size:", &size ))->set_int_limits( 3, 100 ); 
    (new GLUI_Spinner( glui, "Cross 3x3:", &crossFilter ))->set_int_limits( 0, 30 ); 
    (new GLUI_Spinner( glui, "\\ 3x3:", &diagonalFilter ))->set_int_limits( 0, 30 ); 

    glui->set_main_gfx_window( main_window );

    /* We register the idle callback with GLUI, *not* with GLUT */
    GLUI_Master.set_glutIdleFunc( idleTime ); 
}

void setGLDevice() {
    PRINTF("Setting GL device for CUDA...\n");
    int num_devices, device;
    cudaGetDeviceCount(&num_devices);
    PRINTF("Devices found: %d\n", num_devices);
    if (num_devices >= 1) {
          int max_multiprocessors = 0, max_device = 0;
          for (device = 0; device < num_devices; device++) {
                  cudaDeviceProp properties;
                  cudaGetDeviceProperties(&properties, device);
                  if (max_multiprocessors < properties.multiProcessorCount) {
                          max_multiprocessors = properties.multiProcessorCount;
                          max_device = device;
                  }
          }

        if( max_device != -1 ) {
            PRINTF("Setting CUDA GL Device to: %d\n", max_device);
	        cudaGLSetGLDevice( max_device );
        } else {
            deviceReset();
	        PRINTF("Error initializing GL Device\n");
            exit(-1);
        }
    }
}

// shader for displaying floating-point texture
static const char *shader_code = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
//        const GLubyte *error_string;
//        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
//        PRINTF("Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

void initGLBuffers()
{
    // create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, image.width*image.height*sizeof(float), host_in, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    exitOnError("Cuda register GL Buffer");

    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, image.width, image.height, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

int main(int argc, char* argv[]) {
    initGL( &argc, argv ); // Init GLContext first so that CUDA knows about it.
	setGLDevice();

    initCUDA(argc, argv);
    initGLBuffers();

    glutMainLoop();

    // Free memory
    cudaFree(dev_in);
    cudaFree(vhgwMaskHorizontal.data);
    cudaFree(padded_dev_in);
    freeHost(host_in, REGULAR);

    return EXIT_SUCCESS;
}


