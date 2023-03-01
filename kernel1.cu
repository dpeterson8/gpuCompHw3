#include <stdio.h>
#include "kernel1.h"


extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
    extern __shared__ float s_data[];
    int s_rowwidth = blockDim.x + 2;
    //Write this kernel to achieve the same output as the provided k0, but you will have to use
    // shared memory.
    
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int x = ix + 1;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int y = iy + 1;
    int idx = y * floatpitch + x;

    if( y >= width - 1|| x >= width - 1 || y < 1 || x < 1 ) return;

    s_data[threadIdx.x + 1] = g_dataA[idx - floatpitch];
    s_data[threadIdx.x + 1 + s_rowwidth] = g_dataA[idx];
    s_data[threadIdx.x + 1 + (2 * s_rowwidth)] = g_dataA[idx + floatpitch];


    if(threadIdx.x == 0) {
        s_data[0] = g_dataA[idx - floatpitch - 1];
        s_data[s_rowwidth] = g_dataA[idx - 1];
        s_data[s_rowwidth * 2] = g_dataA[idx + floatpitch - 1];

    } else if (threadIdx.x == blockDim.x - 1 || x + 2 == width) {
        s_data[threadIdx.x + 2] = g_dataA[idx - floatpitch + 1];
        s_data[s_rowwidth + threadIdx.x + 2] = g_dataA[idx + 1];
        s_data[s_rowwidth * 2 + threadIdx.x + 2] = g_dataA[idx + floatpitch + 1];
    }

    // }

    __syncthreads();
    

    int threadId = threadIdx.x + 1;


    g_dataB[idx] = (
        0.2f * s_data[s_rowwidth + threadId] +               //itselfmake
        0.1f * s_data[s_rowwidth + threadId - 1] +       //W
        0.1f * s_data[s_rowwidth + threadId + 1] +      //E
        0.1f * s_data[threadId + 1] +       //NE
        0.1f * s_data[threadId] +       //N
        0.1f * s_data[s_rowwidth * 2 + threadId + 1] +       //SE
        0.1f * s_data[s_rowwidth * 2 + threadId ] +       //S
        0.1f * s_data[s_rowwidth * 2 + threadId - 1] +       //SW
        0.1f * s_data[threadId - 1]         //NW
    )  * 0.95f;

    // __syncthreads();

    // g_dataB[idx] = s_data[(s_rowwidth * 4) + threadId];

    //   g_dataB[idx] = s_data[s_rowwidth + threadId + 1];



    //g_dataB[idx] = s_data[blockDim.x * 3 + x];

}

