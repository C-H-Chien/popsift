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

    // Try DevIL first (if available and not forced to use pgmread)
#ifdef USE_DEVIL
    if( ! pgmread_loading )
    {
        ilImage img;
        if( img.Load( inputFile.c_str() ) == true ) {
            if( img.Convert( IL_LUMINANCE ) == true ) {
                w = img.Width();
                h = img.Height();
                cout << "Loading " << w << " x " << h << " image " << inputFile << " (DevIL)" << endl;
                image_data = img.GetData();
                image_loaded = true;
                img.Clear();
            } else {
                cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
            }
        }
    }
#endif

    // Try OpenCV if DevIL failed or is not available
    if( ! image_loaded )
    {
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
    
    // Clean up memory (only if we allocated it ourselves)
#ifdef USE_OPENCV
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
    int num_matches;
};

MatchResult match_sift_features_from_files(const std::string& left_file,
                                         const std::string& right_file,
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
    
    if (verbose) {
        std::cout << "Left image - Number of features: " << lFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << lFeatures->getDescriptorCount() << std::endl;
        std::cout << "Right image - Number of features: " << rFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << rFeatures->getDescriptorCount() << std::endl;
    }
    
    // Perform matching with CUDA timing
    float match_time_ms = 0.0f;
    lFeatures->match(rFeatures, &match_time_ms);
    
    // Get match results
    MatchResult result;
    result.match_time_ms = match_time_ms;
    
    // Note: The current PopSift implementation doesn't provide direct access to match results
    // The matching is performed internally and results are printed to stdout
    // For now, we'll return empty match data with just the timing information
    // TODO: Implement proper match result extraction if needed
    
    result.num_matches = 0; // Placeholder - actual matches are printed to stdout by the CUDA kernel
    
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
    
    // Initialize PopSift for matching
    PopSift popSift(config, popsift::Config::MatchingMode);
    
    // Process both images
    SiftJob* lJob = popSift.enqueue(left_w, left_h, left_data);
    SiftJob* rJob = popSift.enqueue(right_w, right_h, right_data);
    
    // Get device features for matching
    popsift::FeaturesDev* lFeatures = lJob->getDev();
    popsift::FeaturesDev* rFeatures = rJob->getDev();
    
    if (verbose) {
        std::cout << "Left image - Number of features: " << lFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << lFeatures->getDescriptorCount() << std::endl;
        std::cout << "Right image - Number of features: " << rFeatures->getFeatureCount() 
                  << ", Number of descriptors: " << rFeatures->getDescriptorCount() << std::endl;
    }
    
    // Perform matching with CUDA timing
    float match_time_ms = 0.0f;
    lFeatures->match(rFeatures, &match_time_ms);
    
    // Get match results
    MatchResult result;
    result.match_time_ms = match_time_ms;
    
    // Note: The current PopSift implementation doesn't provide direct access to match results
    // The matching is performed internally and results are printed to stdout
    // For now, we'll return empty match data with just the timing information
    // TODO: Implement proper match result extraction if needed
    
    result.num_matches = 0; // Placeholder - actual matches are printed to stdout by the CUDA kernel
    
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

PYBIND11_MODULE(popsift_match, m) {
    m.doc() = "PopSift SIFT feature matching Python bindings";
    
    // Define MatchResult structure
    py::class_<MatchResult>(m, "MatchResult")
        .def_readonly("matches_left_idx", &MatchResult::matches_left_idx)
        .def_readonly("matches_right_idx", &MatchResult::matches_right_idx)
        .def_readonly("match_distances", &MatchResult::match_distances)
        .def_readonly("match_time_ms", &MatchResult::match_time_ms)
        .def_readonly("num_matches", &MatchResult::num_matches);
    
    // Define functions
    m.def("match_features_from_files", &match_sift_features_from_files,
          "Match SIFT features from two image files",
          py::arg("left_file"),
          py::arg("right_file"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("match_features_from_arrays", &match_sift_features_from_arrays,
          "Match SIFT features from two numpy arrays",
          py::arg("left_image"),
          py::arg("right_image"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("get_version", []() { return std::string(POPSIFT_VERSION_STRING); },
          "Get PopSift version");
}

