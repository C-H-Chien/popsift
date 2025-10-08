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

static bool print_dev_info  = false;
static bool print_time_info = false;
static bool write_as_uchar  = false;
static bool dont_write      = false;
static bool pgmread_loading = false;
static bool float_mode      = false;

static void parseargs(int argc, char** argv, popsift::Config& config, string& inputFile) {
    using namespace boost::program_options;

    options_description options("Options");
    {
        options.add_options()
            ("help,h", "Print usage")
            ("verbose,v", bool_switch()->notifier([&](bool i) {if(i) config.setVerbose(); }), "")
            ("log,l", bool_switch()->notifier([&](bool i) {if(i) config.setLogMode(popsift::Config::All); }), "Write debugging files")

            ("input-file,i", value<std::string>(&inputFile)->required(), "Input file");
    
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
        // "Choice of span (1-sided) for Gauss filters. Default is VLFeat-like computation depending on sigma. "
        // "Options are: vlfeat, relative, relative-all, opencv, fixed9, fixed15"
        ("desc-mode", value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
         popsift::Config::getDescModeUsage())
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
        ("write-as-uchar", bool_switch(&write_as_uchar)->default_value(false), "Output descriptors rounded to int.\n"
         "Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or similar")
        ("dont-write", bool_switch(&dont_write)->default_value(false), "Suppress descriptor output")
        ("pgmread-loading", bool_switch(&pgmread_loading)->default_value(false), "Use the old image loader instead of LibDevIL")
        ("float-mode", bool_switch(&float_mode)->default_value(false), "Upload image to GPU as float instead of byte")
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
           exit(EXIT_SUCCESS);
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
        if( float_mode )
        {
            cerr << "Cannot combine float-mode test with DevIL image reader" << endl;
            exit( -1 );
        }

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
    if( ! float_mode )
    {
        job = PopSift.enqueue( w, h, image_data );
        
        // Clean up memory (only if we allocated it ourselves)
#ifdef USE_OPENCV
        if( ! pgmread_loading && image_loaded ) {
            delete [] image_data;
        }
#endif
    }
    else
    {
        auto f_image_data = new float [w * h];
        for( int i=0; i<w*h; i++ )
        {
            f_image_data[i] = float( image_data[i] ) / 256.0f;
        }
        job = PopSift.enqueue( w, h, f_image_data );

        delete [] f_image_data;
        
        // Clean up memory (only if we allocated it ourselves)
#ifdef USE_OPENCV
        if( ! pgmread_loading && image_loaded ) {
            delete [] image_data;
        }
#endif
    }

    return job;
}

float read_job( SiftJob* job, bool really_write )
{
    popsift::Features* feature_list = job->get();
    float gpu_time = job->getGpuTime();
    
    std::cout << "Number of feature points: " << feature_list->getFeatureCount() << std::endl 
              << "Number of feature descriptors: " << feature_list->getDescriptorCount() << std::endl;

    if( really_write ) {
        std::ofstream of( "output-features.txt" );
        feature_list->print( of, write_as_uchar );
    }
    delete feature_list;
    
    return gpu_time;
}

// Python bindings for SIFT feature extraction
struct SiftResult {
    std::vector<float> keypoints_x;
    std::vector<float> keypoints_y;
    std::vector<float> keypoints_scale;
    std::vector<float> keypoints_orientation;
    std::vector<std::vector<float>> descriptors;
    float gpu_time_ms;
    int num_features;
    int num_descriptors;
};

SiftResult extract_sift_features_from_array(py::array_t<unsigned char> image_array, 
                                           bool verbose = false,
                                           bool print_time_info = false) {
    // Initialize CUDA
    popsift::cuda::reset();
    
    // Get image dimensions and data
    py::buffer_info buf_info = image_array.request();
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Image array must be 2D (grayscale)");
    }
    
    int h = buf_info.shape[0];
    int w = buf_info.shape[1];
    unsigned char* image_data = static_cast<unsigned char*>(buf_info.ptr);
    
    if (verbose) {
        std::cout << "Processing " << w << " x " << h << " image" << std::endl;
    }
    
    // Create configuration
    popsift::Config config;
    if (verbose) {
        config.setVerbose();
    }
    
    // Initialize PopSift
    PopSift popSift(config, popsift::Config::ExtractingMode, PopSift::ByteImages);
    
    // Process image
    SiftJob* job = popSift.enqueue(w, h, image_data);
    
    // Get results
    popsift::Features* features = job->get();
    float gpu_time = job->getGpuTime();
    
    // Extract keypoints and descriptors
    SiftResult result;
    result.gpu_time_ms = gpu_time;
    result.num_features = features->getFeatureCount();
    result.num_descriptors = features->getDescriptorCount();
    
    // Extract keypoint data
    const popsift::Feature* feature_data = features->getFeatures();
    const popsift::Descriptor* desc_data = features->getDescriptors();
    
    for (int i = 0; i < result.num_features; ++i) {
        result.keypoints_x.push_back(feature_data[i].xpos);
        result.keypoints_y.push_back(feature_data[i].ypos);
        result.keypoints_scale.push_back(feature_data[i].sigma);
        // Use the first orientation if available
        if (feature_data[i].num_ori > 0) {
            result.keypoints_orientation.push_back(feature_data[i].orientation[0]);
        } else {
            result.keypoints_orientation.push_back(0.0f);
        }
    }
    
    // Extract descriptors (128-dimensional)
    int desc_size = 128;
    for (int i = 0; i < result.num_descriptors; ++i) {
        std::vector<float> descriptor;
        for (int j = 0; j < desc_size; ++j) {
            descriptor.push_back(desc_data[i].features[j]);
        }
        result.descriptors.push_back(descriptor);
    }
    
    if (print_time_info) {
        std::cout << "GPU SIFT processing time: " << std::fixed << std::setprecision(2) 
                  << gpu_time << " ms" << std::endl;
    }
    
    // Cleanup
    delete features;
    delete job;
    popSift.uninit();
    
    return result;
}

SiftResult extract_sift_features_from_file(const std::string& filename,
                                          bool verbose = false,
                                          bool print_time_info = false) {
    // Initialize CUDA
    popsift::cuda::reset();
    
    if (verbose) {
        std::cout << "Loading image from file: " << filename << std::endl;
    }
    
    // Create configuration
    popsift::Config config;
    if (verbose) {
        config.setVerbose();
    }
    
    // Initialize PopSift
    PopSift popSift(config, popsift::Config::ExtractingMode, PopSift::ByteImages);
    
    // Process image using existing function
    SiftJob* job = process_image(filename, popSift);
    
    if (!job) {
        throw std::runtime_error("Failed to process image: " + filename);
    }
    
    // Get results
    popsift::Features* features = job->get();
    float gpu_time = job->getGpuTime();
    
    // Extract keypoints and descriptors
    SiftResult result;
    result.gpu_time_ms = gpu_time;
    result.num_features = features->getFeatureCount();
    result.num_descriptors = features->getDescriptorCount();
    
    // Extract keypoint data
    const popsift::Feature* feature_data = features->getFeatures();
    const popsift::Descriptor* desc_data = features->getDescriptors();
    
    for (int i = 0; i < result.num_features; ++i) {
        result.keypoints_x.push_back(feature_data[i].xpos);
        result.keypoints_y.push_back(feature_data[i].ypos);
        result.keypoints_scale.push_back(feature_data[i].sigma);
        // Use the first orientation if available
        if (feature_data[i].num_ori > 0) {
            result.keypoints_orientation.push_back(feature_data[i].orientation[0]);
        } else {
            result.keypoints_orientation.push_back(0.0f);
        }
    }
    
    // Extract descriptors (128-dimensional)
    int desc_size = 128;
    for (int i = 0; i < result.num_descriptors; ++i) {
        std::vector<float> descriptor;
        for (int j = 0; j < desc_size; ++j) {
            descriptor.push_back(desc_data[i].features[j]);
        }
        result.descriptors.push_back(descriptor);
    }
    
    if (print_time_info) {
        std::cout << "GPU SIFT processing time: " << std::fixed << std::setprecision(2) 
                  << gpu_time << " ms" << std::endl;
    }
    
    // Cleanup
    delete features;
    delete job;
    popSift.uninit();
    
    return result;
}

PYBIND11_MODULE(popsift_extract, m) {
    m.doc() = "PopSift SIFT feature extraction Python bindings";
    
    // Define SiftResult structure
    py::class_<SiftResult>(m, "SiftResult")
        .def_readonly("keypoints_x", &SiftResult::keypoints_x)
        .def_readonly("keypoints_y", &SiftResult::keypoints_y)
        .def_readonly("keypoints_scale", &SiftResult::keypoints_scale)
        .def_readonly("keypoints_orientation", &SiftResult::keypoints_orientation)
        .def_readonly("descriptors", &SiftResult::descriptors)
        .def_readonly("gpu_time_ms", &SiftResult::gpu_time_ms)
        .def_readonly("num_features", &SiftResult::num_features)
        .def_readonly("num_descriptors", &SiftResult::num_descriptors);
    
    // Define functions
    m.def("extract_features_from_array", &extract_sift_features_from_array,
          "Extract SIFT features from numpy array",
          py::arg("image_array"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("extract_features_from_file", &extract_sift_features_from_file,
          "Extract SIFT features from image file",
          py::arg("filename"),
          py::arg("verbose") = false,
          py::arg("print_time_info") = false);
    
    m.def("get_version", []() { return std::string(POPSIFT_VERSION_STRING); },
          "Get PopSift version");
}

