/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <popsift/common/device_prop.h>
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_config.h>
#include <popsift/version.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <chrono>

#ifdef USE_DEVIL
#include <devil_cpp_wrapper.hpp>
#endif
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include "pgmread.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "sift_config_python.h"

using namespace std;
namespace py = pybind11;

static bool print_dev_info  {false};
static bool print_time_info {false};
static bool write_as_uchar  {false};
static bool dont_write      {false};
static bool pgmread_loading {false};

static void parseargs(int argc, char** argv, popsift::Config& config, string& lFile, string& rFile) {
    using namespace boost::program_options;

    options_description options("Options");
    {
        options.add_options()
            ("help,h", "Print usage")
            ("verbose,v", bool_switch()->notifier([&](bool i) {if(i) config.setVerbose(); }), "")
            ("log", bool_switch()->notifier([&](bool i) {if(i) config.setLogMode(popsift::Config::All); }), "Write debugging files")

            ("left,l",  value<std::string>(&lFile)->required(), "\"Left\"  input file")
            ("right,r", value<std::string>(&rFile)->required(), "\"Right\" input file");
    
    }
    options_description parameters("Parameters");
    {
        parameters.add_options()
            ("octaves", value<int>(&config.octaves), "Number of octaves")
            ("levels", value<int>(&config.levels), "Number of levels per octave")
            ("sigma", value<float>()->notifier([&](float f) { config.setSigma(f); }), "Initial sigma value")

            ("threshold", value<float>()->notifier([&](float f) { config.setThreshold(f); }), "Contrast threshold")
            ("edge-threshold", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("edge-limit", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("downsampling", value<float>()->notifier([&](float f) { config.setDownsampling(f); }), "Downscale width and height of input by 2^N")
            ("initial-blur", value<float>()->notifier([&](float f) {config.setInitialBlur(f); }), "Assume initial blur, subtract when blurring first time");
    }
    options_description modes("Modes");
    {
    modes.add_options()
        ( "gauss-mode", value<std::string>()->notifier([&](const std::string& s) { config.setGaussMode(s); }),
          popsift::Config::getGaussModeUsage() )
        ("desc-mode", value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
         popsift::Config::getDescModeUsage() )
        ("popsift-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::PopSift); }),
        "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to 0.6, do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to original pos.")
        ("vlfeat-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::VLFeat); }),
        "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
        "In extrema refinement, steps up to 0.6, levels remain unchanged, "
        "do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh.")
        ("opencv-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::OpenCV); }),
        "During the initial upscale, shift pixels by 0.5. "
        "In extrema refinement, steps up to 0.5, "
        "reject points when reaching max iterations, "
        "first contrast threshold is floor(.5 * peak thresh). "
        "Computed filter width are lower than VLFeat/PopSift")
        ("direct-scaling", bool_switch()->notifier([&](bool b) { if(b) config.setScalingMode(popsift::Config::ScaleDirect); }),
         "Direct each octave from upscaled orig instead of blurred level.")
        ("norm-multi", value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }), "Multiply the descriptor by pow(2,<int>).")
        ( "norm-mode", value<std::string>()->notifier([&](const std::string& s) { config.setNormMode(s); }),
          popsift::Config::getNormModeUsage() )
        ( "root-sift", bool_switch()->notifier([&](bool b) { if(b) config.setNormMode(popsift::Config::RootSift); }),
          popsift::Config::getNormModeUsage() )
        ("filter-max-extrema", value<int>()->notifier([&](int f) {config.setFilterMaxExtrema(f); }), "Approximate max number of extrema.")
        ("filter-grid", value<int>()->notifier([&](int f) {config.setFilterGridSize(f); }), "Grid edge length for extrema filtering (ie. value 4 leads to a 4x4 grid)")
        ("filter-sort", value<std::string>()->notifier([&](const std::string& s) {config.setFilterSorting(s); }), "Sort extrema in each cell by scale, either random (default), up or down");

    }
    options_description informational("Informational");
    {
        informational.add_options()
        ("print-gauss-tables", bool_switch()->notifier([&](bool b) { if(b) config.setPrintGaussTables(); }), "A debug output printing Gauss filter size and tables")
        ("print-dev-info", bool_switch(&print_dev_info)->default_value(false), "A debug output printing CUDA device information")
        ("print-time-info", bool_switch(&print_time_info)->default_value(false), "A debug output printing image processing time after load()")
        ("write-as-uchar", bool_switch(&write_as_uchar)->default_value(false), "Output descriptors rounded to int Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or similar")
        ("dont-write", bool_switch(&dont_write)->default_value(false), "Suppress descriptor output")
        ("pgmread-loading", bool_switch(&pgmread_loading)->default_value(false), "Use the old image loader instead of LibDevIL")
        ;
        
        //("test-direct-scaling")
    }

    options_description all("Allowed options");
    all.add(options).add(parameters).add(modes).add(informational);
    variables_map vm;
    
    try
    {    
       store(parse_command_line(argc, argv, all), vm);

       if (vm.count("help")) {
           std::cout << all << '\n';
           exit(1);
       }

        notify(vm); // Notify does processing (e.g., raise exceptions if required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        std::cerr << "Usage:\n\n" << all << std::endl;
        exit(EXIT_FAILURE);
    }
}


static void collectFilenames( list<string>& inputFiles, const boost::filesystem::path& inputFile )
{
    vector<boost::filesystem::path> vec;
    std::copy( boost::filesystem::directory_iterator( inputFile ),
               boost::filesystem::directory_iterator(),
               std::back_inserter(vec) );
    for (const auto& currPath : vec)
    {
        if( boost::filesystem::is_regular_file(currPath) )
        {
            inputFiles.push_back( currPath.string() );

        }
        else if( boost::filesystem::is_directory(currPath) )
        {
            collectFilenames( inputFiles, currPath);
        }
    }
}

SiftJob* process_image( const string& inputFile, PopSift& PopSift )
{
    SiftJob* job;
    unsigned char* image_data;
    int w = 0, h = 0;
    bool image_loaded = false;

    // Try OpenCV first (for consistency with Python array-based loading)
    // OpenCV and DevIL convert to grayscale differently, causing feature differences!
#ifdef USE_OPENCV
    if( ! pgmread_loading )
    {
        cv::Mat img = cv::imread( inputFile, cv::IMREAD_GRAYSCALE );
        if( ! img.empty() ) {
            w = img.cols;
            h = img.rows;
            cout << "Loading " << w << " x " << h << " image " << inputFile << " (OpenCV)" << endl;
            
            // Allocate memory and copy data
            image_data = new unsigned char[w * h];
            memcpy( image_data, img.data, w * h );
            image_loaded = true;
        }
    }
#endif

    // Try DevIL as fallback (if OpenCV failed or is not available)
    if( ! image_loaded )
    {
#ifdef USE_DEVIL
        if( ! pgmread_loading )
        {
            ilImage img;
            if( img.Load( inputFile.c_str() ) == true ) {
                if( img.Convert( IL_LUMINANCE ) == true ) {
                    w = img.Width();
                    h = img.Height();
                    cout << "Loading " << w << " x " << h << " image " << inputFile << " (DevIL fallback)" << endl;
                    
                    // Allocate memory and copy data (don't use internal pointer)
                    image_data = new unsigned char[w * h];
                    memcpy( image_data, img.GetData(), w * h );
                    image_loaded = true;
                    // Now safe to clear the DevIL image
                    img.Clear();
                } else {
                    cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
                }
            }
        }
#endif
    }

    // Fall back to PGM reader if both DevIL and OpenCV failed
    if( ! image_loaded )
    {
        cout << "Loading " << inputFile << " (PGM fallback)" << endl;
        image_data = readPGMfile( inputFile, w, h );
        if( image_data == nullptr ) {
            cerr << "Could not load image " << inputFile << " with any available method" << endl;
            return nullptr;
        }
        image_loaded = true;
    }

    // Process the loaded image
    job = PopSift.enqueue( w, h, image_data );
    
    // Clean up memory (we always allocate our own copy now for DevIL and OpenCV)
#if defined(USE_DEVIL) || defined(USE_OPENCV)
    if( ! pgmread_loading && image_loaded ) {
        delete [] image_data;
    }
#endif

    return job;
}

// Python bindings for SIFT feature matching
struct MatchResult {
    std::vector<int> matches_left_idx;
    std::vector<int> matches_right_idx;
    std::vector<float> match_distances;
    float match_time_ms;
    float left_gpu_time_ms;
    float right_gpu_time_ms;
    int num_matches;
    int num_total_matches;
};


MatchResult match_sift_features_from_files_with_config(const std::string& left_file,
                                                      const std::string& right_file,
                                                      const SiftConfig& sift_config,
                                                      bool verbose = false,
                                                      bool print_time_info = false) {
    // Initialize CUDA
    popsift::cuda::reset();
    
    if (verbose) {
        std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;
        std::cout << "Matching features from files: " << left_file << " <-> " << right_file << std::endl;
    }
    
    // Create configuration
    popsift::Config config;
    if (verbose) {
        config.setVerbose();
    }
    
    // Apply custom SIFT configuration
    apply_sift_config(sift_config, config);
    
    // Initialize PopSift for matching
    PopSift popSift(config, popsift::Config::MatchingMode);
    
    // Process both images
    SiftJob* lJob = process_image(left_file, popSift);
    SiftJob* rJob = process_image(right_file, popSift);
    
    if (!lJob || !rJob) {
        throw std::runtime_error("Failed to process one or both images");
    }
    
    // Get device features for matching
    popsift::FeaturesDev* lFeatures = lJob->getDev();
    popsift::FeaturesDev* rFeatures = rJob->getDev();
    
    // Get GPU extraction times
    float left_gpu_time = lJob->getGpuTime();
    float right_gpu_time = rJob->getGpuTime();
    
    if (verbose) {
        std::cout << "Left image - Number of features: " << lFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << lFeatures->getDescriptorCount() << std::endl;
        if (print_time_info) {
            std::cout << "Left image - GPU extraction time: " << std::fixed << std::setprecision(2) 
                      << left_gpu_time << " ms" << std::endl;
        }
        std::cout << "Right image - Number of features: " << rFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << rFeatures->getDescriptorCount() << std::endl;
        if (print_time_info) {
            std::cout << "Right image - GPU extraction time: " << std::fixed << std::setprecision(2) 
                      << right_gpu_time << " ms" << std::endl;
        }
    }
    
    // Perform matching with CUDA timing and get results
    float match_time_ms = 0.0f;
    popsift::FeaturesDev::MatchInfo match_info = lFeatures->matchWithResults(rFeatures, &match_time_ms);
    
    // Get match results
    MatchResult result;
    result.match_time_ms = match_time_ms;
    result.left_gpu_time_ms = left_gpu_time;
    result.right_gpu_time_ms = right_gpu_time;
    result.num_matches = match_info.num_accepted_matches;
    result.num_total_matches = match_info.num_total_matches;
    
    // Copy match data
    result.matches_left_idx = match_info.left_feature_indices;
    result.matches_right_idx = match_info.right_feature_indices;
    result.match_distances = match_info.distances;
    
    if (print_time_info) {
        std::cout << "GPU SIFT matching time: " << std::fixed << std::setprecision(2) 
                  << match_time_ms << " ms" << std::endl;
    }
    
    // Cleanup
    delete lFeatures;
    delete rFeatures;
    popSift.uninit();
    
    return result;
}

MatchResult match_sift_features_from_files(const std::string& left_file,
    const std::string& right_file,
    bool verbose = false,
    bool print_time_info = false) 
{
    // Use default SiftConfig
    SiftConfig default_config;
    return match_sift_features_from_files_with_config(left_file, right_file, default_config, verbose, print_time_info);
}

MatchResult match_sift_features_from_arrays_with_config(py::array_t<unsigned char> left_image,
                                                       py::array_t<unsigned char> right_image,
                                                       const SiftConfig& sift_config,
                                                       bool verbose = false,
                                                       bool print_time_info = false) {
    // Initialize CUDA
    popsift::cuda::reset();
    
    if (verbose) {
        std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;
    }
    
    // Get image dimensions and data
    py::buffer_info left_buf = left_image.request();
    py::buffer_info right_buf = right_image.request();
    
    if (left_buf.ndim != 2 || right_buf.ndim != 2) {
        throw std::runtime_error("Both image arrays must be 2D (grayscale)");
    }
    
    int left_h = left_buf.shape[0];
    int left_w = left_buf.shape[1];
    int right_h = right_buf.shape[0];
    int right_w = right_buf.shape[1];
    
    unsigned char* left_data = static_cast<unsigned char*>(left_buf.ptr);
    unsigned char* right_data = static_cast<unsigned char*>(right_buf.ptr);
    
    if (verbose) {
        std::cout << "Left image: " << left_w << " x " << left_h << std::endl;
        std::cout << "Right image: " << right_w << " x " << right_h << std::endl;
    }
    
    // Create configuration
    popsift::Config config;
    if (verbose) {
        config.setVerbose();
    }
    
    // Apply custom SIFT configuration
    apply_sift_config(sift_config, config);
    
    // Initialize PopSift for matching
    PopSift popSift(config, popsift::Config::MatchingMode);
    
    // Process both images
    SiftJob* lJob = popSift.enqueue(left_w, left_h, left_data);
    SiftJob* rJob = popSift.enqueue(right_w, right_h, right_data);
    
    // Get device features for matching
    popsift::FeaturesDev* lFeatures = lJob->getDev();
    popsift::FeaturesDev* rFeatures = rJob->getDev();
    
    // Get GPU extraction times
    float left_gpu_time = lJob->getGpuTime();
    float right_gpu_time = rJob->getGpuTime();
    
    if (verbose) {
        std::cout << "Left image - Number of features: " << lFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << lFeatures->getDescriptorCount() << std::endl;
        if (print_time_info) {
            std::cout << "Left image - GPU extraction time: " << std::fixed << std::setprecision(2) 
                      << left_gpu_time << " ms" << std::endl;
        }
        std::cout << "Right image - Number of features: " << rFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << rFeatures->getDescriptorCount() << std::endl;
        if (print_time_info) {
            std::cout << "Right image - GPU extraction time: " << std::fixed << std::setprecision(2) 
                      << right_gpu_time << " ms" << std::endl;
        }
    }
    
    // Perform matching with CUDA timing and get results
    float match_time_ms = 0.0f;
    popsift::FeaturesDev::MatchInfo match_info = lFeatures->matchWithResults(rFeatures, &match_time_ms);
    
    // Get match results
    MatchResult result;
    result.match_time_ms = match_time_ms;
    result.left_gpu_time_ms = left_gpu_time;
    result.right_gpu_time_ms = right_gpu_time;
    result.num_matches = match_info.num_accepted_matches;
    result.num_total_matches = match_info.num_total_matches;
    
    // Copy match data
    result.matches_left_idx = match_info.left_feature_indices;
    result.matches_right_idx = match_info.right_feature_indices;
    result.match_distances = match_info.distances;
    
    if (print_time_info) {
        std::cout << "GPU SIFT matching time: " << std::fixed << std::setprecision(2) 
                  << match_time_ms << " ms" << std::endl;
    }
    
    // Cleanup
    delete lFeatures;
    delete rFeatures;
    popSift.uninit();
    
    return result;
}

MatchResult match_sift_features_from_arrays(py::array_t<unsigned char> left_image,
    py::array_t<unsigned char> right_image,
    bool verbose = false,
    bool print_time_info = false) 
{
    // Use default SiftConfig
    SiftConfig default_config;
    return match_sift_features_from_arrays_with_config(left_image, right_image, default_config, verbose, print_time_info);
}

// Batch processing for multiple image pairs
std::vector<MatchResult> match_multiple_pairs_from_arrays_with_config(
    const std::vector<py::array_t<unsigned char>>& left_images,
    const std::vector<py::array_t<unsigned char>>& right_images,
    const SiftConfig& sift_config,
    bool verbose = false,
    bool print_time_info = false) {
    
    if (left_images.size() != right_images.size()) {
        throw std::runtime_error("Number of left and right images must match");
    }
    
    if (left_images.empty()) {
        throw std::runtime_error("No image pairs provided");
    }
    
    size_t num_pairs = left_images.size();
    
    // Initialize CUDA
    popsift::cuda::reset();
    
    if (verbose) {
        std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;
        std::cout << "Batch processing " << num_pairs << " image pairs" << std::endl;
    }
    
    // Create configuration
    popsift::Config config;
    if (verbose) {
        config.setVerbose();
    }
    
    // Apply custom SIFT configuration
    apply_sift_config(sift_config, config);
    
    // Initialize PopSift for matching (reuse for all pairs)
    PopSift popSift(config, popsift::Config::MatchingMode);
    
    // Storage for jobs
    std::vector<SiftJob*> left_jobs;
    std::vector<SiftJob*> right_jobs;
    
    // Enqueue all images
    if (verbose) {
        std::cout << "Enqueuing all images..." << std::endl;
    }
    
    for (size_t i = 0; i < num_pairs; i++) {
        // Get image dimensions and data for left image
        py::buffer_info left_buf = left_images[i].request();
        if (left_buf.ndim != 2) {
            throw std::runtime_error("All image arrays must be 2D (grayscale)");
        }
        int left_h = left_buf.shape[0];
        int left_w = left_buf.shape[1];
        unsigned char* left_data = static_cast<unsigned char*>(left_buf.ptr);
        
        // Get image dimensions and data for right image
        py::buffer_info right_buf = right_images[i].request();
        if (right_buf.ndim != 2) {
            throw std::runtime_error("All image arrays must be 2D (grayscale)");
        }
        int right_h = right_buf.shape[0];
        int right_w = right_buf.shape[1];
        unsigned char* right_data = static_cast<unsigned char*>(right_buf.ptr);
        
        if (verbose) {
            std::cout << "Pair " << i << " - Left: " << left_w << "x" << left_h 
                      << ", Right: " << right_w << "x" << right_h << std::endl;
        }
        
        // Enqueue both images (pipeline will process them)
        SiftJob* lJob = popSift.enqueue(left_w, left_h, left_data);
        SiftJob* rJob = popSift.enqueue(right_w, right_h, right_data);
        
        left_jobs.push_back(lJob);
        right_jobs.push_back(rJob);
    }
    
    if (verbose) {
        std::cout << "All images enqueued. Processing and matching..." << std::endl;
    }
    
    // Process results for each pair
    std::vector<MatchResult> results;
    results.reserve(num_pairs);
    
    for (size_t i = 0; i < num_pairs; i++) {
        if (verbose) {
            std::cout << "Processing pair " << i << "..." << std::endl;
        }
        
        // Get device features (this blocks until processing is complete)
        popsift::FeaturesDev* lFeatures = left_jobs[i]->getDev();
        popsift::FeaturesDev* rFeatures = right_jobs[i]->getDev();
        
        // Get GPU extraction times
        float left_gpu_time = left_jobs[i]->getGpuTime();
        float right_gpu_time = right_jobs[i]->getGpuTime();
        
        if (verbose) {
            std::cout << "Pair " << i << " - Left features: " << lFeatures->getFeatureCount() 
                      << ", descriptors: " << lFeatures->getDescriptorCount() << std::endl;
            if (print_time_info) {
                std::cout << "Pair " << i << " - Left GPU time: " << std::fixed 
                          << std::setprecision(2) << left_gpu_time << " ms" << std::endl;
            }
            std::cout << "Pair " << i << " - Right features: " << rFeatures->getFeatureCount() 
                      << ", descriptors: " << rFeatures->getDescriptorCount() << std::endl;
            if (print_time_info) {
                std::cout << "Pair " << i << " - Right GPU time: " << std::fixed 
                          << std::setprecision(2) << right_gpu_time << " ms" << std::endl;
            }
        }
        
        // Perform matching
        float match_time_ms = 0.0f;
        popsift::FeaturesDev::MatchInfo match_info = lFeatures->matchWithResults(rFeatures, &match_time_ms);
        
        // Store results
        MatchResult result;
        result.match_time_ms = match_time_ms;
        result.left_gpu_time_ms = left_gpu_time;
        result.right_gpu_time_ms = right_gpu_time;
        result.num_matches = match_info.num_accepted_matches;
        result.num_total_matches = match_info.num_total_matches;
        result.matches_left_idx = match_info.left_feature_indices;
        result.matches_right_idx = match_info.right_feature_indices;
        result.match_distances = match_info.distances;
        
        if (print_time_info) {
            std::cout << "Pair " << i << " - GPU matching time: " << std::fixed 
                      << std::setprecision(2) << match_time_ms << " ms" << std::endl;
        }
        
        results.push_back(result);
        
        // Cleanup features for this pair
        delete lFeatures;
        delete rFeatures;
    }
    
    // Cleanup PopSift
    popSift.uninit();
    
    if (verbose) {
        std::cout << "Batch processing complete. Processed " << num_pairs << " pairs." << std::endl;
    }
    
    return results;
}

std::vector<MatchResult> match_multiple_pairs_from_arrays(
    const std::vector<py::array_t<unsigned char>>& left_images,
    const std::vector<py::array_t<unsigned char>>& right_images,
    bool verbose = false,
    bool print_time_info = false) {
    
    SiftConfig default_config;
    return match_multiple_pairs_from_arrays_with_config(left_images, right_images, default_config, verbose, print_time_info);
}

std::vector<MatchResult> match_multiple_pairs_from_files_with_config(
    const std::vector<std::string>& left_files,
    const std::vector<std::string>& right_files,
    const SiftConfig& sift_config,
    bool verbose = false,
    bool print_time_info = false) {
    
    if (left_files.size() != right_files.size()) {
        throw std::runtime_error("Number of left and right files must match");
    }
    
    if (left_files.empty()) {
        throw std::runtime_error("No file pairs provided");
    }
    
    size_t num_pairs = left_files.size();
    
    // Initialize CUDA
    popsift::cuda::reset();
    
    if (verbose) {
        std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;
        std::cout << "Batch processing " << num_pairs << " image pairs from files" << std::endl;
    }
    
    // Create configuration
    popsift::Config config;
    if (verbose) {
        config.setVerbose();
    }
    
    // Apply custom SIFT configuration
    apply_sift_config(sift_config, config);
    
    // Initialize PopSift for matching (reuse for all pairs)
    PopSift popSift(config, popsift::Config::MatchingMode);
    
    // Storage for jobs
    std::vector<SiftJob*> left_jobs;
    std::vector<SiftJob*> right_jobs;
    
    // Enqueue all images
    if (verbose) {
        std::cout << "Loading and enqueuing all images..." << std::endl;
    }
    
    for (size_t i = 0; i < num_pairs; i++) {
        if (verbose) {
            std::cout << "Pair " << i << ": " << left_files[i] << " <-> " << right_files[i] << std::endl;
        }
        
        // Process left and right images
        SiftJob* lJob = process_image(left_files[i], popSift);
        SiftJob* rJob = process_image(right_files[i], popSift);
        
        if (!lJob || !rJob) {
            throw std::runtime_error("Failed to process pair " + std::to_string(i));
        }
        
        left_jobs.push_back(lJob);
        right_jobs.push_back(rJob);
    }
    
    if (verbose) {
        std::cout << "All images enqueued. Processing and matching..." << std::endl;
    }
    
    // Process results for each pair
    std::vector<MatchResult> results;
    results.reserve(num_pairs);
    
    for (size_t i = 0; i < num_pairs; i++) {
        if (verbose) {
            std::cout << "Processing pair " << i << "..." << std::endl;
        }
        
        // Get device features (this blocks until processing is complete)
        popsift::FeaturesDev* lFeatures = left_jobs[i]->getDev();
        popsift::FeaturesDev* rFeatures = right_jobs[i]->getDev();
        
        // Get GPU extraction times
        float left_gpu_time = left_jobs[i]->getGpuTime();
        float right_gpu_time = right_jobs[i]->getGpuTime();
        
        if (verbose) {
            std::cout << "Pair " << i << " - Left features: " << lFeatures->getFeatureCount() 
                      << ", descriptors: " << lFeatures->getDescriptorCount() << std::endl;
            if (print_time_info) {
                std::cout << "Pair " << i << " - Left GPU time: " << std::fixed 
                          << std::setprecision(2) << left_gpu_time << " ms" << std::endl;
            }
            std::cout << "Pair " << i << " - Right features: " << rFeatures->getFeatureCount() 
                      << ", descriptors: " << rFeatures->getDescriptorCount() << std::endl;
            if (print_time_info) {
                std::cout << "Pair " << i << " - Right GPU time: " << std::fixed 
                          << std::setprecision(2) << right_gpu_time << " ms" << std::endl;
            }
        }
        
        // Perform matching
        float match_time_ms = 0.0f;
        popsift::FeaturesDev::MatchInfo match_info = lFeatures->matchWithResults(rFeatures, &match_time_ms);
        
        // Store results
        MatchResult result;
        result.match_time_ms = match_time_ms;
        result.left_gpu_time_ms = left_gpu_time;
        result.right_gpu_time_ms = right_gpu_time;
        result.num_matches = match_info.num_accepted_matches;
        result.num_total_matches = match_info.num_total_matches;
        result.matches_left_idx = match_info.left_feature_indices;
        result.matches_right_idx = match_info.right_feature_indices;
        result.match_distances = match_info.distances;
        
        if (print_time_info) {
            std::cout << "Pair " << i << " - GPU matching time: " << std::fixed 
                      << std::setprecision(2) << match_time_ms << " ms" << std::endl;
        }
        
        results.push_back(result);
        
        // Cleanup features for this pair
        delete lFeatures;
        delete rFeatures;
    }
    
    // Cleanup PopSift
    popSift.uninit();
    
    if (verbose) {
        std::cout << "Batch processing complete. Processed " << num_pairs << " pairs." << std::endl;
    }
    
    return results;
}

std::vector<MatchResult> match_multiple_pairs_from_files(
    const std::vector<std::string>& left_files,
    const std::vector<std::string>& right_files,
    bool verbose = false,
    bool print_time_info = false) {
    
    SiftConfig default_config;
    return match_multiple_pairs_from_files_with_config(left_files, right_files, default_config, verbose, print_time_info);
}

PYBIND11_MODULE(popsift_match, m) {
    m.doc() = "PopSift SIFT feature matching Python bindings";
    
    // Note: SiftConfig is registered in popsift_config module to avoid duplication
    
    // Define MatchResult structure
    py::class_<MatchResult>(m, "MatchResult")
        .def_readonly("matches_left_idx", &MatchResult::matches_left_idx)
        .def_readonly("matches_right_idx", &MatchResult::matches_right_idx)
        .def_readonly("match_distances", &MatchResult::match_distances)
        .def_readonly("match_time_ms", &MatchResult::match_time_ms)
        .def_readonly("left_gpu_time_ms", &MatchResult::left_gpu_time_ms)
        .def_readonly("right_gpu_time_ms", &MatchResult::right_gpu_time_ms)
        .def_readonly("num_matches", &MatchResult::num_matches)
        .def_readonly("num_total_matches", &MatchResult::num_total_matches);
    
    // Define functions
    m.def("match_features_from_files", &match_sift_features_from_files,
          "Match SIFT features from two image files",
          py::arg("left_file"),
          py::arg("right_file"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("match_features_from_files_with_config", &match_sift_features_from_files_with_config,
          "Match SIFT features from two image files with custom configuration",
          py::arg("left_file"),
          py::arg("right_file"),
          py::arg("sift_config"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("match_features_from_arrays", &match_sift_features_from_arrays,
          "Match SIFT features from two numpy arrays",
          py::arg("left_image"),
          py::arg("right_image"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("match_features_from_arrays_with_config", &match_sift_features_from_arrays_with_config,
          "Match SIFT features from two numpy arrays with custom configuration",
          py::arg("left_image"),
          py::arg("right_image"),
          py::arg("sift_config"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    // Batch processing functions
    m.def("match_multiple_pairs_from_arrays", &match_multiple_pairs_from_arrays,
          "Batch match SIFT features from multiple pairs of numpy arrays",
          py::arg("left_images"),
          py::arg("right_images"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("match_multiple_pairs_from_arrays_with_config", &match_multiple_pairs_from_arrays_with_config,
          "Batch match SIFT features from multiple pairs of numpy arrays with custom configuration",
          py::arg("left_images"),
          py::arg("right_images"),
          py::arg("sift_config"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("match_multiple_pairs_from_files", &match_multiple_pairs_from_files,
          "Batch match SIFT features from multiple pairs of image files",
          py::arg("left_files"),
          py::arg("right_files"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("match_multiple_pairs_from_files_with_config", &match_multiple_pairs_from_files_with_config,
          "Batch match SIFT features from multiple pairs of image files with custom configuration",
          py::arg("left_files"),
          py::arg("right_files"),
          py::arg("sift_config"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("get_version", []() { return std::string(POPSIFT_VERSION_STRING); },
          "Get PopSift version");
}

