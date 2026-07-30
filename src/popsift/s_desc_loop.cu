/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/vec_macros.h"
#include "s_desc_loop.h"
#include "s_gradiant.h"
#include "sift_constants.h"

#include <cstdio>

using namespace popsift;

__device__ static inline
void ext_desc_loop_sub( const float         ang,
                        const Extremum*     ext,
                        float* __restrict__ features,
                        cudaTextureObject_t layer_tex,
                        const int           width,
                        const int           height )
{
#ifndef BLOCK_3_DIMS
    const int ix   = threadIdx.y;
    const int iy   = threadIdx.z;
    const int tile = ( ( ( iy << 2 ) + ix ) << 3 ); // base of the 8 floats written by this group of 16 threads
#else
    const int ix   = ( threadIdx.z &  0x3 );
    const int iy   = ( threadIdx.z >> 2 );
    const int tile = ( threadIdx.z << 3 );
#endif

    const float x    = ext->xpos;
    const float y    = ext->ypos;

    //> Samples the correct Gaussian-blur level from a 3D texture of the octave.
    const int   level = ext->lpos; // old_level;
    const float sig  = ext->sigma;

    //> window radius = 3.0 × sigma (DESC_MAGNIFY = 3.0)
    const float SBP  = fabsf(DESC_MAGNIFY * sig);

    if( SBP == 0 ) {
        return;
    }

    // const float cos_t = cosf(ang);
    // const float sin_t = sinf(ang);
    float cos_t;
    float sin_t;

    //> Convert angle to cosine and sine
    //> __sincosf is a device-side intrinsic that efficiently computes both the sine and cosine 
    //  of a single-precision floating-point angle in radians
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    //> The points that support the 4×4 spatial grid have the offsets -1.5, -0.5, 0.5, 1.5 from the keypoint.
    const float2 offsetpt = make_float2( ix - 1.5f,
                                         iy - 1.5f );

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;

    //> Rotate cell center into image space
    //> fmaf is a device-side intrinsic that efficiently computes the fused multiply-add single precision operation
    const float ptx = ::fmaf( csbp, offsetpt.x, ::fmaf( -ssbp, offsetpt.y, x ) );
    const float pty = ::fmaf( csbp, offsetpt.y, ::fmaf(  ssbp, offsetpt.x, y ) );

    const float bsz = fabsf(csbp) + fabsf(ssbp);
    const int   xmin = max(1,          (int)floorf(ptx - bsz));
    const int   ymin = max(1,          (int)floorf(pty - bsz));
    const int   xmax = min(width - 2,  (int)floorf(ptx + bsz));
    const int   ymax = min(height - 2, (int)floorf(pty + bsz));

    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    //> Loop over all pixels in a bounding box around this cell
    for( int i = threadIdx.x; popsift::any(i < loops); i+=blockDim.x )
    {
        if( i >= loops ) continue;

        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;     

        const float2 d = make_float2( jj - ptx, ii - pty );

        // const float nx = crsbp * dx + srsbp * dy;
        // const float ny = crsbp * dy - srsbp * dx;

        //> Map pixels inside the cell to the rotated descriptor coordinates
        //> and keep only those pixels inside the rotated cell (|nx|, |ny| < 1)
        const float2 n = make_float2( ::fmaf( crsbp, d.x,  srsbp * d.y ),
                                      ::fmaf( crsbp, d.y, -srsbp * d.x ) );
        const float2 nn = abs(n);
        if (nn.x < 1.0f && nn.y < 1.0f) {
            float grad_mag;
            float grad_theta;
            //> "layer_tex" is the Gaussian pyramid of the octave
            //> and specifically the corresponding scale level is "level=ext->lpos"
            //> This get_gradiant() function simply reads from the scale-blurred image
            //> get_gradiant lives in s_gradiant.h
            //> CH: I believe this is the bottleneck of computation
            get_gradiant( grad_mag, grad_theta, jj, ii, layer_tex, level );

            const float2 dn = n + offsetpt;

            //> Gaussian weight by distance from keypoint
            // const float  ww = __expf( -scalbnf(dn.x*dn.x + dn.y*dn.y, -3));
            const float ww  = __expf(-0.125f * (dn.x*dn.x + dn.y*dn.y)); // speedup !
            const float2 w  = make_float2( 1.0f - nn.x,
                                           1.0f - nn.y );
            
            //> Bilinear weight within the cell
            const float wgt = ww * w.x * w.y * grad_mag;

            //> Convert to descriptor orientation bin index
            //> (Orientation relative to keypoint -> 8 bins with linear interpolation)
            grad_theta -= ang;
            grad_theta += ( grad_theta <  0.0f  ? M_PI2 : 0.0f ); //  if (grad_theta <  0.0f ) grad_theta += M_PI2;
            grad_theta -= ( grad_theta >= M_PI2 ? M_PI2 : 0.0f ); //  if (grad_theta >= M_PI2) grad_theta -= M_PI2;

            const float tth  = __fmul_ru( grad_theta, M_4RPI ); // grad_theta * M_4RPI;
            const int   fo0  = (int)floorf(tth);
            const float do0  = tth - fo0;             
            const float wgt1 = 1.0f - do0;
            const float wgt2 = do0;

            int fo  = fo0 % DESC_BINS;
    
                // maf: multiply-add
                // _ru - round to positive infinity equiv to froundf since always >=0
            dpt[fo]   = __fmaf_ru( wgt1, wgt, dpt[fo] );   // dpt[fo]   += (wgt1*wgt);
            dpt[fo+1] = __fmaf_ru( wgt2, wgt, dpt[fo+1] ); // dpt[fo+1] += (wgt2*wgt);
        }
    }
    __syncthreads();

    dpt[0] += dpt[8];

    //> Warp-shuffle reduction: combine partial histograms from 16 cells
    //> Write 8 floats per cell to a 128-D descriptor
    /* reduction here */
    for (int i = 0; i < 8; i++) {
        dpt[i] += popsift::shuffle_down( dpt[i], 16 );
        dpt[i] += popsift::shuffle_down( dpt[i], 8 );
        dpt[i] += popsift::shuffle_down( dpt[i], 4 );
        dpt[i] += popsift::shuffle_down( dpt[i], 2 );
        dpt[i] += popsift::shuffle_down( dpt[i], 1 );
        dpt[i]  = popsift::shuffle     ( dpt[i], 0 );
    }

    if( threadIdx.x < 8 ) {
        features[tile+threadIdx.x] = dpt[threadIdx.x];
    }
}

struct DescriptorLoopCache
{
    float* features;
    float  ang;
    float  x;
    float  y;
    float  csbp;
    float  ssbp;
    float  crsbp;
    float  srsbp;
    float  bsz;
    int    level;
    int    valid;
};

__global__ void ext_desc_loop(int octave, cudaTextureObject_t layer_tex, int w, int h)
{
    const int   o_offset =  dct.ori_ps[octave] + blockIdx.x;
    Descriptor* desc     = &dbuf.desc           [o_offset];
    const int   ext_idx  =  dobuf.feat_to_ext_map[o_offset];
    Extremum*   ext      =  dobuf.extrema + ext_idx;

    const int   ext_base =  ext->idx_ori;
    const int   ori_num  =  o_offset - ext_base;
    const float ang      =  ext->orientation[ori_num];

    ext_desc_loop_sub( ang,
                       ext,
                       desc->features,
                       layer_tex,
                       w,
                       h );
}

/* One warp processes one of the 4x4 spatial cells of one descriptor.
 * Several warps share a block (default: 4 warps = 4 cells per block).
 * Dimensions: grid (num_of_keypoints, 16/warps_per_block) / block (32, warps_per_block, 1).
 * Cells of the same descriptor write disjoint 8-bin slices of the 128-D vector;
 * meaning that the global memory of the descriptor is partitioned into 16 disjoint 8-bin slices.
 */
__device__ static inline
void ext_desc_loop_per_cell_sub( const float         ang,
                                 const Extremum*     ext,
                                 float* __restrict__ features,
                                 cudaTextureObject_t layer_tex,
                                 const int           width,
                                 const int           height,
                                 const int           ix,
                                 const int           iy )
{
    //> get the tile index
    const int tile = ( ( ( iy << 2 ) + ix ) << 3 );

    //> get the keypoint position
    const float x     = ext->xpos;
    const float y     = ext->ypos;
    const int   level = ext->lpos;
    const float sig   = ext->sigma;
    const float SBP   = fabsf(DESC_MAGNIFY * sig);

    // if( SBP == 0 ) {
    //     return;
    // }

    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    const float2 offsetpt = make_float2( ix - 1.5f, iy - 1.5f );

    const float ptx = ::fmaf( csbp, offsetpt.x, ::fmaf( -ssbp, offsetpt.y, x ) );
    const float pty = ::fmaf( csbp, offsetpt.y, ::fmaf(  ssbp, offsetpt.x, y ) );

    const float bsz  = fabsf(csbp) + fabsf(ssbp);
    const int   xmin = max(1,         (int)floorf(ptx - bsz));
    const int   ymin = max(1,         (int)floorf(pty - bsz));
    const int   xmax = min(width - 2,  (int)floorf(ptx + bsz));
    const int   ymax = min(height - 2, (int)floorf(pty + bsz));

    const int wx    = xmax - xmin + 1;
    const int hy    = ymax - ymin + 1;
    const int loops = wx * hy;

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for( int i = threadIdx.x; popsift::any(i < loops); i += blockDim.x )
    {
        if( i >= loops ) continue;

        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;

        const float2 d  = make_float2( jj - ptx, ii - pty );
        const float2 n  = make_float2( ::fmaf( crsbp, d.x,  srsbp * d.y ),
                                       ::fmaf( crsbp, d.y, -srsbp * d.x ) );
        const float2 nn = abs(n);

        if( nn.x < 1.0f && nn.y < 1.0f ) {
            float grad_mag;
            float grad_theta;
            get_gradiant( grad_mag, grad_theta, jj, ii, layer_tex, level );

            const float2 dn = n + offsetpt;
            const float  ww = __expf( -0.125f * (dn.x*dn.x + dn.y*dn.y) );
            const float2 w  = make_float2( 1.0f - nn.x,
                                           1.0f - nn.y );
            const float wgt = ww * w.x * w.y * grad_mag;

            grad_theta -= ang;
            grad_theta += ( grad_theta <  0.0f  ? M_PI2 : 0.0f );
            grad_theta -= ( grad_theta >= M_PI2 ? M_PI2 : 0.0f );

            const float tth  = __fmul_ru( grad_theta, M_4RPI );
            const int   fo0  = (int)floorf(tth);
            const float do0  = tth - fo0;
            const float wgt1 = 1.0f - do0;
            const float wgt2 = do0;
            const int   fo   = fo0 % DESC_BINS;

            dpt[fo]   = __fmaf_ru( wgt1, wgt, dpt[fo] );
            dpt[fo+1] = __fmaf_ru( wgt2, wgt, dpt[fo+1] );
        }
    }

    dpt[0] += dpt[8];

    //> Reduce the 8-bin histogram via warp shuffling
    for( int i = 0; i < 8; i++ ) {
        dpt[i] += popsift::shuffle_down( dpt[i], 16 );
        dpt[i] += popsift::shuffle_down( dpt[i], 8 );
        dpt[i] += popsift::shuffle_down( dpt[i], 4 );
        dpt[i] += popsift::shuffle_down( dpt[i], 2 );
        dpt[i] += popsift::shuffle_down( dpt[i], 1 );
        dpt[i]  = popsift::shuffle( dpt[i], 0 );
    }

    //> Write only the corresponding "slice" of the 128-D vector to the output buffer
    if( threadIdx.x < 8 ) {
        features[tile + threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__ void ext_desc_loop_per_cell(int octave, cudaTextureObject_t layer_tex, int w, int h)
{
    //> One warp (threadIdx.y) owns one cell; blockIdx.y selects the cell group.
    const int cell = blockIdx.y * blockDim.y + threadIdx.y; // 0 .. 15

    //> get the keypoint index
    const int o_offset  = dct.ori_ps[octave] + blockIdx.x;
    Descriptor* desc    = &dbuf.desc[o_offset];
    const int   ext_idx = dobuf.feat_to_ext_map[o_offset];
    Extremum*   ext     = dobuf.extrema + ext_idx;

    const int   ori_num = o_offset - ext->idx_ori;
    const float ang     = ext->orientation[ori_num];

    //> get the row and column of the grid-cell
    const int ix = cell &  0x3;     //> column index
    const int iy = cell >> 2;       //> row index

    ext_desc_loop_per_cell_sub( ang,
                                ext,
                                desc->features,
                                layer_tex,
                                w,
                                h,
                                ix,
                                iy );
}

