/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "common/debug_macros.h"
#include "common/plane_2d.h"
#include "sift_extremum.h"
#include "sift_octave.h"
#include "sift_pyramid.h"

#undef BLOCK_3_DIMS

__global__ void ext_desc_loop(int octave, cudaTextureObject_t layer_tex, int width, int height);
__global__ void ext_desc_loop_cached(int octave, cudaTextureObject_t layer_tex, int width, int height);
__global__ void ext_desc_loop_per_cell(int octave, cudaTextureObject_t layer_tex, int width, int height);

namespace popsift
{

//> This is for each octave
inline static bool start_ext_desc_loop( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    //> one thread block per oriented feature (descriptor)
    //  i.e., one CUDA block = one 128-D SIFT descriptor
    grid.x = hct.ori_ct[octave];
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    //> Each thread block has 32 * 4 * 4 = 512 threads
    //> Inside the block, 16 thread groups (threadIdx.y × threadIdx.z).
    //  each handle one cell of the 4×4 spatial grid.
#ifndef BLOCK_3_DIMS
    block.x = 32;
    block.y = 4;
    block.z = 4;
#else
    block.x = 32;
    block.y = 1;
    block.z = 16;
#endif

    ext_desc_loop
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint( ),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    POP_SYNC_CHK;

    return true;
}

// Alternative descriptor kernel for benchmarking. One thread per block loads
// the keypoint data and computes the scale/rotation constants, then broadcasts
// them to the remaining threads through shared memory.
inline static bool start_ext_desc_loop_cached( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    grid.x = hct.ori_ct[octave];
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

#ifndef BLOCK_3_DIMS
    block.x = 32;
    block.y = 4;
    block.z = 4;
#else
    block.x = 32;
    block.y = 1;
    block.z = 16;
#endif

    ext_desc_loop_cached
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint(),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    POP_SYNC_CHK;

    return true;
}

// Alternative: one 32-thread block per 4x4 spatial cell (16 blocks per descriptor).
// Avoids cross-cell __syncthreads within a 512-thread block.
inline static bool start_ext_desc_loop_per_cell( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    grid.x = hct.ori_ct[octave];
    grid.y = 16;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = 1;
    block.z = 1;

    ext_desc_loop_per_cell
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint(),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    POP_SYNC_CHK;

    return true;
}

}; // namespace popsift

