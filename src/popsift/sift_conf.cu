/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/debug_macros.h"
#include "sift_conf.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

namespace popsift
{

Config::Config( )
    : _upscale_factor( 1.0f )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    , _gauss_mode( getGaussModeDefault() )
    , _sift_mode( Config::PopSift )
    , _log_mode( Config::None )
    , _scaling_mode( Config::ScaleDefault )
    // , _desc_mode( Config::VLFeat_Desc )
    , _desc_mode( Config::Loop )
    , _grid_filter_mode( Config::LargestScaleFirst )
    , verbose( false )
    // , _max_extrema( 20000 )
    , _max_extrema( 100000 )
    , _filter_max_extrema( -1 )
    , _filter_grid_size( 2 )
    , _assume_initial_blur( true )
    , _initial_blur( 0.5f )
    , _normalization_mode( getNormModeDefault() )
    , _normalization_multiplier( 0 )
    , _print_gauss_tables( false )
{
    int            currentDev;
    cudaDeviceProp currentProp;
    cudaError_t    err;

    err = cudaGetDevice( &currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device ID" );

    err = cudaGetDeviceProperties( &currentProp, currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device properties" );
}

void Config::setMode( Config::SiftMode m )
{
    _sift_mode = m;
}

void Config::setGaussMode( Config::GaussMode m )
{
    _gauss_mode = m;
}

void Config::setDescMode( const std::string& text )
{
    if( text == "loop" )
        setDescMode( Config::Loop );
    else if( text == "iloop" )
        setDescMode( Config::ILoop );
    else if( text == "grid" )
        setDescMode( Config::Grid );
    else if( text == "igrid" )
        setDescMode( Config::IGrid );
    else if( text == "notile" )
        setDescMode( Config::NoTile );
    else if( text == "vlfeat" )
        setDescMode( Config::VLFeat_Desc );
    else
        POP_FATAL( "specified descriptor extraction mode must be one of loop, grid or igrid" );
}

void Config::setDescMode( Config::DescMode m )
{
    _desc_mode = m;
}

const char* Config::getDescModeUsage( )
{
    return "Choice of descriptor extraction modes:\n"
           "loop, iloop, grid, igrid, notile, vlfeat\n"
	       "Default is loop\n"
           "loop is OpenCV-like horizontal scanning, sampling every pixel in a radius around the "
           "centers or the 16 tiles arond the keypoint. Each sampled point contributes to two "
           "histogram bins."
           "iloop is like loop but samples all constant 1-pixel distances from the keypoint, "
           "using the CUDA texture engine for interpolation. "
           "grid is like loop but works on rotated, normalized tiles, relying on CUDA 2D cache "
           "to replace the manual data aligment idea of loop. "
           "igrid iloop and grid. "
           "notile is like igrid but handles all 16 tiles at once.\n"
           "vlfeat is VLFeat-like horizontal scanning, sampling every pixel in a radius around "
           "keypoint itself, using the 16 tile centers only for weighting. Every sampled point "
           "contributes to up to eight histogram bins.";
}

void Config::setGaussMode( const std::string& m )
{
    if( m == "vlfeat" )
        setGaussMode( Config::VLFeat_Compute );
    else if( m == "vlfeat-hw-interpolated" )
        setGaussMode( Config::VLFeat_Relative );
    else if( m == "relative" )
        setGaussMode( Config::VLFeat_Relative );
    else if( m == "vlfeat-direct" )
        setGaussMode( Config::VLFeat_Relative_All );
    else if( m == "opencv" )
        setGaussMode( Config::OpenCV_Compute );
    else if( m == "fixed9" )
        setGaussMode( Config::Fixed9 );
    else if( m == "fixed15" )
        setGaussMode( Config::Fixed15 );
    else
        POP_FATAL( string("Bad Gauss mode.\n") + getGaussModeUsage() );
}

Config::GaussMode Config::getGaussModeDefault( )
{
    return Config::VLFeat_Compute;
}

const char* Config::getGaussModeUsage( )
{
    return
        "Choice of Gauss filter method. "
        "Options are: "
        "vlfeat (default), "
        "vlfeat-hw-interpolated, "
        "vlfeat-direct, "
        "opencv, "
        "fixed9, "
        "fixed15, "
        "relative (synonym for vlfeat-hw-interpolated)";
}

bool Config::getCanFilterExtrema() const
{
#if __CUDACC_VER_MAJOR__ >= 8
    return true;
#else
    return false;
#endif
}

void Config::setFilterSorting( const std::string& text )
{
    if( text == "up" )
        _grid_filter_mode = Config::SmallestScaleFirst;
    else if( text == "down" )
        _grid_filter_mode = Config::LargestScaleFirst;
    else if( text == "random" )
        _grid_filter_mode = Config::RandomScale;
    else
        POP_FATAL( "filter sorting mode must be one of up, down or random" );
}

void Config::setFilterSorting( Config::GridFilterMode m )
{
    _grid_filter_mode = m;
}

void Config::setVerbose( bool on )
{
    verbose = on;
}

void Config::setLogMode( LogMode mode )
{
    _log_mode = mode;
}

Config::LogMode Config::getLogMode( ) const
{
    return _log_mode;
}

void Config::setScalingMode( ScalingMode mode )
{
    _scaling_mode = mode;
}

/**
 * Normalization mode
 * Should the descriptor normalization use L2-like classic normalization
 * of the typically better L1-like RootSift normalization?
 */
void Config::setUseRootSift( bool on )
{
    if( on )
        _normalization_mode = RootSift;
    else
        _normalization_mode = Classic;
}

bool Config::getUseRootSift( ) const
{
    return ( _normalization_mode == RootSift );
}

Config::NormMode Config::getNormMode( NormMode m ) const 
{
    return _normalization_mode;
}

void Config::setNormMode( Config::NormMode m )
{
    _normalization_mode = m;
}

void Config::setNormMode( const std::string& m )
{
    if( m == "RootSift" ) setNormMode( Config::RootSift );
    else if( m == "classic" ) setNormMode( Config::Classic );
    else
        POP_FATAL( string("Bad Normalization mode.\n") + getGaussModeUsage() );
}

Config::NormMode Config::getNormModeDefault( )
{
    return Config::RootSift;
}

const char* Config::getNormModeUsage( )
{
    return
        "Choice of descriptor normalization modes. "
        "Options are: "
        "RootSift (L1-like, default), "
        "Classic (L2-like)";
}

/**
 * Normalization multiplier
 * A power of 2 multiplied with the normalized descriptor. Required
 * for the construction of 1-byte integer desciptors.
 * Usual choice is 2^8 or 2^9.
 */
void Config::setNormalizationMultiplier( int mul )
{
    _normalization_multiplier = mul;
}

int Config::getNormalizationMultiplier( ) const
{
    return _normalization_multiplier;
}

void Config::setDownsampling( float v ) { _upscale_factor = -v; }
void Config::setOctaves( int v ) { octaves = v; }
void Config::setLevels( int v ) { levels = v; }
void Config::setSigma( float v ) { sigma = v; }
void Config::setEdgeLimit( float v ) { _edge_limit = v; }
void Config::setThreshold( float v ) { _threshold = v; }
void Config::setPrintGaussTables() { _print_gauss_tables = true; }
void Config::setFilterMaxExtrema( int ext ) { _filter_max_extrema = ext; }
void Config::setFilterGridSize( int sz ) { _filter_grid_size = sz; }

void Config::setInitialBlur( float blur )
{
    if( blur == 0.0f ) {
        _assume_initial_blur = false;
        _initial_blur        = blur;
    } else {
        _assume_initial_blur = true;
        _initial_blur        = blur;
    }
}

Config::GaussMode Config::getGaussMode( ) const
{
    return _gauss_mode;
}

Config::SiftMode Config::getSiftMode() const
{
    return _sift_mode;
}

bool Config::hasInitialBlur( ) const
{
    return _assume_initial_blur;
}

float Config::getInitialBlur( ) const
{
    return _initial_blur;
}

float Config::getPeakThreshold() const
{
    return ( _threshold * 0.5f * 255.0f / levels );
}

bool Config::ifPrintGaussTables() const
{
    return _print_gauss_tables;
}

bool Config::equal( const Config& other ) const
{
    #define COMPARE(a) ( this->a != other.a )
    if( COMPARE( octaves ) ||
        COMPARE( levels ) ||
        COMPARE( sigma ) ||
        COMPARE( _edge_limit ) ||
        COMPARE( _threshold ) ||
        COMPARE( _upscale_factor ) ||
        COMPARE( _scaling_mode ) ||
        COMPARE( _max_extrema ) ||
        COMPARE( _gauss_mode ) ||
        COMPARE( _sift_mode ) ||
        COMPARE( _assume_initial_blur ) ||
        COMPARE( _initial_blur ) ||
        COMPARE( _normalization_mode ) ||
        COMPARE( _normalization_multiplier ) ) return false;
    return true;
}

void Config::print() const
{
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "PopSift Configuration Parameters" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Basic parameters
    std::cout << "Octaves:              ";
    if( octaves < 0 ) {
        std::cout << "auto (will be calculated as: log₂(min(w,h)) - 3 + scale_factor)" << std::endl;
    } else {
        std::cout << octaves << std::endl;
    }
    
    std::cout << "Levels per octave:    " << levels << std::endl;
    std::cout << "Sigma:                " << sigma << std::endl;
    
    // Calculate Gaussian kernel size based on initial sigma and mode
    int gauss_kernel_size = 0;
    float effective_sigma = sigma;
    if( _assume_initial_blur && _initial_blur > 0.0f ) {
        effective_sigma = sqrtf( sigma * sigma - _initial_blur * _initial_blur * powf( 2.0f, getUpscaleFactor() ) * powf( 2.0f, getUpscaleFactor() ) );
        if( effective_sigma < 0.0f ) effective_sigma = sigma;
    }
    
    switch( _gauss_mode ) {
        case VLFeat_Relative_All:
        case VLFeat_Compute: {
            int spn = std::min<int>( (int)ceilf( 4.0f * effective_sigma ) + 1, 127 );
            gauss_kernel_size = spn * 2 - 1; // Full width (half-width * 2 - 1 for center)
            break;
        }
        case VLFeat_Relative: {
            int spn = std::min<int>( (int)ceilf( 4.0f * effective_sigma ) + 1, 127 );
            if( ( spn & 1 ) == 0 ) spn += 1;
            gauss_kernel_size = spn * 2 - 1;
            break;
        }
        case OpenCV_Compute: {
            int span = int( roundf( 2.0f * 4.0f * effective_sigma + 1.0f ) ) | 1;
            span >>= 1;
            span += 1;
            span = std::min<int>( span, 127 );
            gauss_kernel_size = span * 2 - 1;
            break;
        }
        case Fixed9:
            gauss_kernel_size = 9;
            break;
        case Fixed15:
            gauss_kernel_size = 15;
            break;
        default:
            gauss_kernel_size = 0;
            break;
    }
    
    std::cout << "Gaussian kernel size: " << gauss_kernel_size << "x" << gauss_kernel_size;
    if( gauss_kernel_size > 0 && effective_sigma != sigma ) {
        std::cout << " (for effective sigma=" << std::fixed << std::setprecision(3) << effective_sigma 
                  << ", initial sigma=" << sigma << ")";
    } else if( gauss_kernel_size > 0 ) {
        std::cout << " (for initial sigma=" << std::fixed << std::setprecision(3) << effective_sigma << ")";
    }
    std::cout << std::endl;
    std::cout << "  (Note: kernel size varies by level and octave)" << std::endl;
    
    std::cout << "Threshold:            " << _threshold << std::endl;
    std::cout << "Edge limit:           " << _edge_limit << std::endl;
    
    // Upscale/downsampling
    std::cout << "Upscale factor:       " << getUpscaleFactor() << std::endl;
    if( getUpscaleFactor() < 0 ) {
        std::cout << "  (Downsampling by:  " << -getUpscaleFactor() << "x)" << std::endl;
    }
    
    // Initial blur
    std::cout << "Assume initial blur:  " << (_assume_initial_blur ? "yes" : "no") << std::endl;
    if( _assume_initial_blur ) {
        std::cout << "Initial blur:        " << _initial_blur << std::endl;
    }
    
    // Extrema filtering
    std::cout << "Max extrema:          " << getMaxExtrema() << std::endl;
    std::cout << "Filter max extrema:  ";
    if( _filter_max_extrema < 0 ) {
        std::cout << "unlimited" << std::endl;
    } else {
        std::cout << _filter_max_extrema << std::endl;
    }
    std::cout << "Filter grid size:     " << _filter_grid_size << "x" << _filter_grid_size << std::endl;
    
    // Modes
    const char* sift_mode_str = "Unknown";
    switch( _sift_mode ) {
        case PopSift: sift_mode_str = "PopSift"; break;
        case OpenCV: sift_mode_str = "OpenCV"; break;
        case VLFeat: sift_mode_str = "VLFeat"; break;
    }
    std::cout << "SIFT mode:            " << sift_mode_str << std::endl;
    
    const char* gauss_mode_str = "Unknown";
    switch( _gauss_mode ) {
        case VLFeat_Compute: gauss_mode_str = "VLFeat_Compute"; break;
        case VLFeat_Relative: gauss_mode_str = "VLFeat_Relative"; break;
        case VLFeat_Relative_All: gauss_mode_str = "VLFeat_Relative_All"; break;
        case OpenCV_Compute: gauss_mode_str = "OpenCV_Compute"; break;
        case Fixed9: gauss_mode_str = "Fixed9"; break;
        case Fixed15: gauss_mode_str = "Fixed15"; break;
    }
    std::cout << "Gauss mode:           " << gauss_mode_str << std::endl;
    
    const char* desc_mode_str = "Unknown";
    switch( _desc_mode ) {
        case Loop: desc_mode_str = "Loop"; break;
        case Grid: desc_mode_str = "Grid"; break;
        case VLFeat_Desc: desc_mode_str = "VLFeat_Desc"; break;
        case NoTile: desc_mode_str = "NoTile"; break;
        case ILoop: desc_mode_str = "ILoop"; break;
        case IGrid: desc_mode_str = "IGrid"; break;
    }
    std::cout << "Descriptor mode:      " << desc_mode_str << std::endl;
    
    const char* scaling_mode_str = (_scaling_mode == ScaleDirect) ? "ScaleDirect" : "ScaleDefault";
    std::cout << "Scaling mode:         " << scaling_mode_str << std::endl;
    
    const char* grid_filter_str = "Unknown";
    switch( _grid_filter_mode ) {
        case LargestScaleFirst: grid_filter_str = "LargestScaleFirst"; break;
        case RandomScale: grid_filter_str = "RandomScale"; break;
    }
    std::cout << "Grid filter mode:     " << grid_filter_str << std::endl;
    
    // Normalization
    const char* norm_mode_str = (_normalization_mode == RootSift) ? "RootSift (L1-like)" : "Classic (L2-like)";
    std::cout << "Normalization mode:   " << norm_mode_str << std::endl;
    std::cout << "Norm multiplier:      " << _normalization_multiplier;
    if( _normalization_multiplier > 0 ) {
        std::cout << " (2^" << _normalization_multiplier << ")";
    }
    std::cout << std::endl;
    
    // Logging
    const char* log_mode_str = (_log_mode == All) ? "All" : "None";
    std::cout << "Log mode:             " << log_mode_str << std::endl;
    std::cout << "Verbose:              " << (verbose ? "yes" : "no") << std::endl;
    std::cout << "Print Gauss tables:    " << (_print_gauss_tables ? "yes" : "no") << std::endl;
    
    // Peak threshold (computed)
    std::cout << "Peak threshold:       " << getPeakThreshold() << " (computed)" << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
}

}; // namespace popsift

