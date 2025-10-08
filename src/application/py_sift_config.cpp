/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sift_config_python.h"

namespace py = pybind11;

PYBIND11_MODULE(popsift_config, m) {
    m.doc() = "PopSift SIFT configuration Python bindings";
    
    // Define SiftConfig structure
    py::class_<SiftConfig>(m, "SiftConfig")
        .def(py::init<>())
        .def_readwrite("octaves", &SiftConfig::octaves)
        .def_readwrite("levels", &SiftConfig::levels)
        .def_readwrite("sigma", &SiftConfig::sigma)
        .def_readwrite("threshold", &SiftConfig::threshold)
        .def_readwrite("edge_threshold", &SiftConfig::edge_threshold)
        .def_readwrite("downsampling", &SiftConfig::downsampling)
        .def_readwrite("initial_blur", &SiftConfig::initial_blur)
        .def_readwrite("gauss_mode", &SiftConfig::gauss_mode)
        .def_readwrite("desc_mode", &SiftConfig::desc_mode)
        .def_readwrite("popsift_mode", &SiftConfig::popsift_mode)
        .def_readwrite("vlfeat_mode", &SiftConfig::vlfeat_mode)
        .def_readwrite("opencv_mode", &SiftConfig::opencv_mode)
        .def_readwrite("direct_scaling", &SiftConfig::direct_scaling)
        .def_readwrite("norm_multi", &SiftConfig::norm_multi)
        .def_readwrite("norm_mode", &SiftConfig::norm_mode)
        .def_readwrite("root_sift", &SiftConfig::root_sift)
        .def_readwrite("filter_max_extrema", &SiftConfig::filter_max_extrema)
        .def_readwrite("filter_grid", &SiftConfig::filter_grid)
        .def_readwrite("filter_sort", &SiftConfig::filter_sort)
        .def_readwrite("print_gauss_tables", &SiftConfig::print_gauss_tables);
}
