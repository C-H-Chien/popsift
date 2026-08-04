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

//> Warps cooperating on one cell in ext_desc_loop_per_cell_multiwarp.
constexpr int DESC_LOOP_MULTIWARP_WARPS = 2;

//> Register-pipelined texture prefetch: M cells / M warps per block.
//> M must divide 16: {1,2,4,8,16}.
constexpr int DESC_LOOP_ASYNC_CELLS_PER_BLOCK = 4;

static_assert( DESC_LOOP_ASYNC_CELLS_PER_BLOCK >= 1 &&
               DESC_LOOP_ASYNC_CELLS_PER_BLOCK <= 16 &&
               ( 16 % DESC_LOOP_ASYNC_CELLS_PER_BLOCK ) == 0,
               "DESC_LOOP_ASYNC_CELLS_PER_BLOCK must be 1,2,4,8, or 16" );

__global__ void ext_desc_loop(int octave, cudaTextureObject_t layer_tex, int width, int height);
__global__ void ext_desc_loop_per_cell(int octave, cudaTextureObject_t layer_tex, int width, int height);
__global__ void ext_desc_loop_per_cell_multiwarp(int octave, cudaTextureObject_t layer_tex, int width, int height);
__global__ void ext_desc_loop_per_cell_prefetch(int octave, cudaTextureObject_t layer_tex, int width, int height);

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

// Alternative: one warp per 4x4 spatial cell, several warps per block
// (4 warps = 4 cells per block; 4 blocks cover all 16 cells of a descriptor).
// Avoids cross-cell __syncthreads within a 512-thread block.
inline static bool start_ext_desc_loop_per_cell( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    constexpr int warps_per_block = 1;
    constexpr int cells_per_desc  = 16;

    grid.x = hct.ori_ct[octave];
    grid.y = cells_per_desc / warps_per_block;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = warps_per_block;
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

// Alternative: one cell per block, several warps cooperate on that cell's
// pixel loop. Warp-shuffle reduces within each warp; shared memory combines
// warps before writing the cell's 8-bin slice.
inline static bool start_ext_desc_loop_per_cell_multiwarp( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    constexpr int warps_per_block = DESC_LOOP_MULTIWARP_WARPS;

    grid.x = hct.ori_ct[octave];
    grid.y = 16;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32 * warps_per_block;
    block.y = 1;
    block.z = 1;

    ext_desc_loop_per_cell_multiwarp
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint(),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    POP_SYNC_CHK;

    return true;
}

// Register-pipelined texture prefetch: M warps, M cells per block.
// Each warp owns one cell; texture samples stay in registers (curr/next)
// so the next fetch can be issued before histogram work on the current sample.
inline static bool start_ext_desc_loop_per_cell_prefetch( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    constexpr int cells_per_block = DESC_LOOP_ASYNC_CELLS_PER_BLOCK;

    grid.x = hct.ori_ct[octave];
    grid.y = 16 / cells_per_block;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = cells_per_block; // one warp per cell
    block.z = 1;

    ext_desc_loop_per_cell_prefetch
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint(),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    POP_SYNC_CHK;

    return true;
}

}; // namespace popsift

