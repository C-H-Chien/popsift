/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_config_python.h"
#include <popsift/sift_conf.h>

// Helper function to apply SiftConfig to PopSift Config
void apply_sift_config(const SiftConfig& sift_config, popsift::Config& config) {
    if (sift_config.octaves != -1) config.setOctaves(sift_config.octaves);
    if (sift_config.levels != -1) config.setLevels(sift_config.levels);
    if (sift_config.sigma != -1.0f) config.setSigma(sift_config.sigma);
    if (sift_config.threshold != -1.0f) config.setThreshold(sift_config.threshold);
    if (sift_config.edge_threshold != -1.0f) config.setEdgeLimit(sift_config.edge_threshold);
    if (sift_config.downsampling != -1.0f) config.setDownsampling(sift_config.downsampling);
    if (sift_config.initial_blur != -1.0f) config.setInitialBlur(sift_config.initial_blur);
    
    if (!sift_config.gauss_mode.empty()) config.setGaussMode(sift_config.gauss_mode);
    if (!sift_config.desc_mode.empty()) config.setDescMode(sift_config.desc_mode);
    
    if (sift_config.popsift_mode == "true") config.setMode(popsift::Config::PopSift);
    if (sift_config.vlfeat_mode == "true") config.setMode(popsift::Config::VLFeat);
    if (sift_config.opencv_mode == "true") config.setMode(popsift::Config::OpenCV);
    if (sift_config.direct_scaling == "true") config.setScalingMode(popsift::Config::ScaleDirect);
    
    if (sift_config.norm_multi != -1) config.setNormalizationMultiplier(sift_config.norm_multi);
    if (!sift_config.norm_mode.empty()) config.setNormMode(sift_config.norm_mode);
    if (sift_config.root_sift == "true") config.setNormMode(popsift::Config::RootSift);
    
    if (sift_config.filter_max_extrema != -1) config.setFilterMaxExtrema(sift_config.filter_max_extrema);
    if (sift_config.filter_grid != -1) config.setFilterGridSize(sift_config.filter_grid);
    if (!sift_config.filter_sort.empty()) config.setFilterSorting(sift_config.filter_sort);
    
    if (sift_config.print_gauss_tables) config.setPrintGaussTables();
}
