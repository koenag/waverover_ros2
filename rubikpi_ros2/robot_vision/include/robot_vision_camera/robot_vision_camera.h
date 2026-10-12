#ifndef ROBOT_VISION_CAMERA_H
#define ROBOT_VISION_CAMERA_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <iostream>
#include <string>
#include <thread>
#include <sys/time.h>

using namespace cv;
using namespace std;

// GStreamer data structure for qtiqmmfsrc pipeline
typedef struct _CustomData {
  GstElement *pipeline;
  GstElement *source;        // qtiqmmfsrc
  GstElement *capsfiltersrc;
  GstElement *convert1;      // First videoconvert (before videobalance)
  GstElement *videobalance;  // Video adjustments
  GstElement *convert2;      // Second videoconvert (after videobalance)
  GstElement *capsfilterapp;
  GstElement *appsink;
} CustomData;

// Enhanced camera parameter structure
struct CameraParam {
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  cv::Mat rectification_matrix;
  cv::Mat projection_matrix;
  int image_width;
  int image_height;
  std::string distortion_model;
  
  void read(const FileStorage& fs) {
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_coefficients;
    fs["rectification_matrix"] >> rectification_matrix;
    fs["projection_matrix"] >> projection_matrix;
    fs["image_width"] >> image_width;
    fs["image_height"] >> image_height;
    
    // Try to read distortion model, default to plumb_bob if not found
    std::string model;
    fs["distortion_model"] >> model;
    distortion_model = model.empty() ? "plumb_bob" : model;
  }
};

class RobotVisionCamera : public rclcpp::Node {
private:
  // ROS2 publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  
  // GStreamer elements
  CustomData data;
  GstBus *bus = nullptr;
  GstMessage *msg = nullptr;
  GstStateChangeReturn ret;
  
  // Camera parameters
  int frame_rate;
  int width;
  int height;
  int camera_id;             // Camera ID for qtiqmmfsrc
  std::string input_format;  // Input format (NV12, etc.)
  std::string output_format; // Output format (RGB, BGR, etc.)
  std::string topic_name;
  bool image_compress;
  bool image_rectify;
  std::string camera_parameter_path;
  int jpeg_quality;
  
  // Video adjustment parameters
  double brightness;
  double contrast;
  double saturation;
  double hue;
  
  // Camera calibration and info
  CameraParam cam_param;
  Mat map1, map2;
  sensor_msgs::msg::CameraInfo camera_info_msg_;
  bool camera_info_loaded_;

public:
  RobotVisionCamera(const std::string & name);
  ~RobotVisionCamera();
  
  void getROSParams();
  void prepareRemap();
  void loadCameraInfo();
  void fillCameraInfoMessage();
  void buildGStreamerPipeline();
  void startGStreamerPipeline();
  
  // Static callback function for GStreamer
  static GstFlowReturn processData(GstElement * sink, RobotVisionCamera* node);
};

#endif // ROBOT_VISION_CAMERA_H