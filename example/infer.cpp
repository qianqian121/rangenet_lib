/* Copyright (c) 2019 Xieyuanli Chen, Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */

#include <cmath>

// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

// c++ stuff
#include <chrono>
#include <iomanip>  // for setfill
#include <iostream>
#include <string>

// net stuff
#include <selector.hpp>
namespace cl = rangenet::segmentation;

// standalone lib h
#include "infer.hpp"

// boost
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include "rslidar_point.h"

typedef std::tuple< u_char, u_char, u_char> color;

int main(int argc, const char *argv[]) {
  // define options
  std::string scan;
  std::string path;
  std::string backend = "tensorrt";
  bool verbose = false;

  // Parse options
  try {
    po::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")(
        "scan,s", po::value<std::string>(&scan),
        "LiDAR scan to infer. No Default")(
        "path,p", po::value<std::string>(),
        "Directory to get the inference model from. No default")(
        "verbose,v", po::bool_switch(),
        "Verbose mode. Calculates profile (time to run)");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    if (!vm["scan"].empty()) {
      std::cout << "scan: " << scan << std::endl;
    } else {
      std::cerr << "No scan! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }

    // make defaults count, parameter check, and print
    if (vm.count("path")) {
      path = vm["path"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "path: " << path << std::endl;
    } else {
      std::cerr << "No path! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }
    if (vm.count("verbose")) {
      verbose = vm["verbose"].as<bool>();
//      verbose = true;
      std::cout << "verbose: " << verbose << std::endl;
    } else {
      std::cout << "verbose: " << verbose << ". Using default!" << std::endl;
    }

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  } catch (const po::error &ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  // create a network
  std::unique_ptr<cl::Net> net = cl::make_net(path, backend);

  // set verbosity
  net->verbosity(verbose);

  // predict each image
  std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  std::cout << "Predicting image: " << scan << std::endl;

  uint32_t num_points;
  std::vector<float> values;
  // Open a scan
  if (scan.substr(scan.size() - 3) == "pcd") {
    pcl::PointCloud<RslidarPoint> pcl_cloud;
    pcl::io::loadPCDFile(scan, pcl_cloud);
    num_points = pcl_cloud.size();
    values.resize(num_points * 4);
    int idx = 0;
    for (const auto& pcl_point : pcl_cloud) {
      values[idx++] = pcl_point.x;
      values[idx++] = pcl_point.y;
      values[idx++] = pcl_point.z;
      values[idx++] = std::min(1.0, static_cast<float>(pcl_point.intensity) / 127.0);
    }
  } else {
    std::ifstream in(scan.c_str(), std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "Could not open the scan!" << std::endl;
      return 1;
    }

    in.seekg(0, std::ios::end);
    num_points = in.tellg() / (4 * sizeof(float));
    in.seekg(0, std::ios::beg);

    values.resize(4 * num_points);
    in.read((char*)&values[0], 4 * num_points * sizeof(float));
  }

  // predict
  std::vector<std::vector<float>> semantic_scan = net->infer(values, num_points);

  // get point cloud
  std::vector<cv::Vec3f> points = net->getPoints(values, num_points);

  // get color mask
  std::vector<cv::Vec3b> color_mask = net->getLabels(semantic_scan, num_points);

  // print the output
  if (verbose) {
    cv::viz::Viz3d window("semantic scan");
    cv::viz::WCloud cloudWidget(points, color_mask);
    while (!window.wasStopped()) {
      window.showWidget("cloud", cloudWidget);
      window.spinOnce(30, true);
    }
  }
  std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;

  std::cout << "Example finished! "<< std::endl;

  return 0;
}
