#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace std;

// This code creates a 2D array of integers that can be indexed in any range
// If M is declared
//      int **M;
// and M = imatrix(1,100,1,100) is executed, then
// the i,j entry in M can be referenced by M[i][j]

typedef int* ip;

int **imatrix(int nrl,int nrh,int ncl,int nch)

{
  int i;
  int **m;

  m = new ip[nrh-nrl+1];
  if (!m) {
     cout << "allocation failure in matrix";
     exit(1);
    }
  m -= nrl;
  for (i=nrl;i<=nrh;i++) {
    m[i] = new int[nch-ncl+1];
    if (!m[i]) {
       cout << "allocation failure in matrix";
       exit(1);
      }
    m[i] -= ncl;
   }
  return m;
}

/*
main(int argc,char *argv[])

{
    char line[80];
    unsigned char ch;
    int i,j,xres,yres,maxintensity,**image;
    ifstream fp;

    fp.open(argv[1]);
    fp.getline(line,80);
    if ((line[0]=='P')&&(line[1]=='5')) {
       while (fp.peek()=='#') fp.getline(line,80);
       fp >> xres >> yres >> maxintensity;
       fp.getline(line,80);  // read past eol
       image = imatrix(0,xres-1,0,yres-1); 
       for (i=0;i<xres;i++) 
         for (j=0;j<yres;j++) {
            fp.get(ch);
            image[i][j] = (int) ch;
          }
      }
     else cout << "Bad pgm format" << endl;
}*/