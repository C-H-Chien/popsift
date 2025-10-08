/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <string>
#include <vector>
#include <popsift/sift_conf.h>

// SIFT configuration parameters for Python interface
struct SiftConfig {
    int octaves = -1;  // -1 means use default
    int levels = -1;
    float sigma = -1.0f;
    float threshold = -1.0f;
    float edge_threshold = -1.0f;
    float downsampling = -1.0f;
    float initial_blur = -1.0f;
    std::string gauss_mode = "";
    std::string desc_mode = "";
    std::string popsift_mode = "";
    std::string vlfeat_mode = "";
    std::string opencv_mode = "";
    std::string direct_scaling = "";
    int norm_multi = -1;
    std::string norm_mode = "";
    std::string root_sift = "";
    int filter_max_extrema = -1;
    int filter_grid = -1;
    std::string filter_sort = "";
    bool print_gauss_tables = false;
};

// Helper function to apply SiftConfig to PopSift Config
void apply_sift_config(const SiftConfig& sift_config, popsift::Config& config);
