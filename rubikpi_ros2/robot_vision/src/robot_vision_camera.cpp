#include "robot_vision_camera/robot_vision_camera.h"

RobotVisionCamera::RobotVisionCamera(const std::string & name)
  : Node(name, rclcpp::NodeOptions().use_intra_process_comms(true)), camera_info_loaded_(false)
{
  getROSParams();

  // Use Sensor Data QoS (Best Effort) for camera topics
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

  // Create publishers for raw, compressed images, and camera info
  pub_ = this->create_publisher<sensor_msgs::msg::Image>(topic_name, qos);
  compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(topic_name + "/compressed", qos);
  camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(topic_name.substr(0, topic_name.find_last_of('/')) + "/camera_info", qos);
  

  // Load camera info first
  loadCameraInfo();

  if (image_rectify) {
    prepareRemap();
  }

  buildGStreamerPipeline();
}

void RobotVisionCamera::getROSParams(){
  // Handle parameters
  this->declare_parameter<int>("frame_rate", 10);
  this->declare_parameter<int>("width", 1280);
  this->declare_parameter<int>("height", 720);
  this->declare_parameter<int>("camera_id", 0);  // Camera ID for qtiqmmfsrc
  this->declare_parameter<std::string>("input_format", "NV12");  // NV12 input format
  this->declare_parameter<std::string>("output_format", "BGR");  // BGR output format
  this->declare_parameter<std::string>("topic_name", "camera/image_raw");
  this->declare_parameter<bool>("image_compress", true);
  this->declare_parameter<bool>("image_rectify", false);
  this->declare_parameter<std::string>("camera_parameter_path", "/home/ubuntu/ros2_ws/rubikpi_ros2/robot_vision/config/camera_parameter.yaml");
  this->declare_parameter<int>("jpeg_quality", 90);
  
  // Video adjustment parameters
  this->declare_parameter<double>("brightness", 0.05);
  this->declare_parameter<double>("contrast", 1.0);
  this->declare_parameter<double>("saturation", 0.2);
  this->declare_parameter<double>("hue", 0.1);

  // Get parameter values
  frame_rate = this->get_parameter("frame_rate").as_int();
  width = this->get_parameter("width").as_int();
  height = this->get_parameter("height").as_int();
  camera_id = this->get_parameter("camera_id").as_int();
  input_format = this->get_parameter("input_format").as_string();
  output_format = this->get_parameter("output_format").as_string();
  topic_name = this->get_parameter("topic_name").as_string();
  image_compress = this->get_parameter("image_compress").as_bool();
  image_rectify = this->get_parameter("image_rectify").as_bool();
  camera_parameter_path = this->get_parameter("camera_parameter_path").as_string();
  jpeg_quality = this->get_parameter("jpeg_quality").as_int();
  
  brightness = this->get_parameter("brightness").as_double();
  contrast = this->get_parameter("contrast").as_double();
  saturation = this->get_parameter("saturation").as_double();
  hue = this->get_parameter("hue").as_double();

  // Log parameters
  RCLCPP_INFO(this->get_logger(), "frame_rate: %d", frame_rate);
  RCLCPP_INFO(this->get_logger(), "width: %d", width);
  RCLCPP_INFO(this->get_logger(), "height: %d", height);
  RCLCPP_INFO(this->get_logger(), "camera_id: %d", camera_id);
  RCLCPP_INFO(this->get_logger(), "input_format: %s", input_format.c_str());
  RCLCPP_INFO(this->get_logger(), "output_format: %s", output_format.c_str());
  RCLCPP_INFO(this->get_logger(), "topic_name: %s", topic_name.c_str());
  RCLCPP_INFO(this->get_logger(), "image_compress: %s", image_compress ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "image_rectify: %s", image_rectify ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "jpeg_quality: %d", jpeg_quality);
  RCLCPP_INFO(this->get_logger(), "brightness: %.2f", brightness);
  RCLCPP_INFO(this->get_logger(), "contrast: %.2f", contrast);
  RCLCPP_INFO(this->get_logger(), "saturation: %.2f", saturation);
  RCLCPP_INFO(this->get_logger(), "hue: %.2f", hue);
}

// Load camera info from YAML file
void RobotVisionCamera::loadCameraInfo() {
  cv::FileStorage fs(camera_parameter_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_WARN(this->get_logger(), "Failed to open camera YAML: %s. Camera info will not be published.", camera_parameter_path.c_str());
    camera_info_loaded_ = false;
    return;
  }

  try {
    // Read camera parameters using enhanced structure
    cam_param.read(fs);
    fs.release();

    // Fill the ROS CameraInfo message
    fillCameraInfoMessage();
    camera_info_loaded_ = true;
    
    RCLCPP_INFO(this->get_logger(), "Camera info loaded successfully from: %s", camera_parameter_path.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Error loading camera info: %s", e.what());
    camera_info_loaded_ = false;
    fs.release();
  }
}

// Fill CameraInfo message from loaded parameters
void RobotVisionCamera::fillCameraInfoMessage() {
  camera_info_msg_.header.frame_id = "camera_frame";
  
  // Use runtime dimensions from node parameters
  camera_info_msg_.width = width;
  camera_info_msg_.height = height;
  
  // Distortion model
  camera_info_msg_.distortion_model = cam_param.distortion_model;
  
  // Camera matrix (K)
  if (!cam_param.camera_matrix.empty() && cam_param.camera_matrix.rows == 3 && cam_param.camera_matrix.cols == 3) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        camera_info_msg_.k[i * 3 + j] = cam_param.camera_matrix.at<double>(i, j);
      }
    }
  } else {
    RCLCPP_WARN(this->get_logger(), "Invalid camera matrix dimensions");
    // Fill with identity matrix as fallback
    std::fill(camera_info_msg_.k.begin(), camera_info_msg_.k.end(), 0.0);
    camera_info_msg_.k[0] = camera_info_msg_.k[4] = camera_info_msg_.k[8] = 1.0;
  }
  
  // Distortion coefficients (D)
  if (!cam_param.distortion_coefficients.empty()) {
    camera_info_msg_.d.clear();
    
    // Handle both row and column vectors
    cv::Mat d_flat = cam_param.distortion_coefficients;
    if (d_flat.cols == 1 && d_flat.rows > 1) {
      d_flat = d_flat.t(); // Convert to row vector
    }
    
    for (int i = 0; i < d_flat.cols; i++) {
      camera_info_msg_.d.push_back(d_flat.at<double>(0, i));
    }
  }
  
  // Rectification matrix (R)
  if (!cam_param.rectification_matrix.empty() && 
      cam_param.rectification_matrix.rows == 3 && 
      cam_param.rectification_matrix.cols == 3) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        camera_info_msg_.r[i * 3 + j] = cam_param.rectification_matrix.at<double>(i, j);
      }
    }
  } else {
    // Fill with identity matrix as default
    std::fill(camera_info_msg_.r.begin(), camera_info_msg_.r.end(), 0.0);
    camera_info_msg_.r[0] = camera_info_msg_.r[4] = camera_info_msg_.r[8] = 1.0;
  }
  
  // Projection matrix (P)
  if (!cam_param.projection_matrix.empty() && 
      cam_param.projection_matrix.rows == 3 && 
      cam_param.projection_matrix.cols == 4) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        camera_info_msg_.p[i * 4 + j] = cam_param.projection_matrix.at<double>(i, j);
      }
    }
  } else {
    // Create projection matrix from camera matrix if not available
    std::fill(camera_info_msg_.p.begin(), camera_info_msg_.p.end(), 0.0);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        camera_info_msg_.p[i * 4 + j] = camera_info_msg_.k[i * 3 + j];
      }
    }
  }
  
  // Binning and ROI (set to defaults)
  camera_info_msg_.binning_x = 0;
  camera_info_msg_.binning_y = 0;
  camera_info_msg_.roi.x_offset = 0;
  camera_info_msg_.roi.y_offset = 0;
  camera_info_msg_.roi.height = 0;
  camera_info_msg_.roi.width = 0;
  camera_info_msg_.roi.do_rectify = false;
}

// Enhanced prepareRemap function that uses the loaded camera info
void RobotVisionCamera::prepareRemap() {
  if (!camera_info_loaded_) {
    RCLCPP_ERROR(this->get_logger(), "Cannot prepare remap: camera info not loaded");
    image_rectify = false;
    return;
  }

  // Use the already loaded camera parameters
  cv::Mat K = cam_param.camera_matrix;
  cv::Mat D = cam_param.distortion_coefficients;

  // Validate parameters
  if (K.empty() || D.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Missing camera_matrix or distortion_coefficients");
    image_rectify = false;
    return;
  }

  if (K.rows != 3 || K.cols != 3) {
    RCLCPP_ERROR(this->get_logger(), "camera_matrix must be 3x3, got %dx%d", K.rows, K.cols);
    image_rectify = false;
    return;
  }

  // Ensure proper data types
  K.convertTo(K, CV_64F);
  D.convertTo(D, CV_64F);

  // Flatten D to 1xN if it's Nx1
  if (D.cols == 1 && D.rows > 1) D = D.t();

  if (D.rows == 1 && (D.cols==4 || D.cols==5 || D.cols==8 || D.cols==12 || D.cols==14)) {
    // Valid distortion coefficient count
  } else {
    RCLCPP_ERROR(this->get_logger(), "distortion_coefficients must be 1x{4,5,8,12,14}, got %dx%d", D.rows, D.cols);
    image_rectify = false;
    return;
  }

  // Choose runtime image size
  cv::Size imageSize(width, height);

  // Build undistortion maps
  cv::Mat newK = getOptimalNewCameraMatrix(K, D, imageSize, 0.0, imageSize, 0);
  cv::initUndistortRectifyMap(K, D, cv::Mat(), newK, imageSize, CV_16SC2, map1, map2);

  RCLCPP_INFO(this->get_logger(),
    "Rectify ready. size=%dx%d, D=%dx%d, K=(%g,%g,%g | %g,%g,%g | %g,%g,%g)",
    imageSize.width, imageSize.height, D.rows, D.cols,
    K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2),
    K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2),
    K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2));
}

RobotVisionCamera::~RobotVisionCamera(){
  if (bus) {
    gst_object_unref(bus);
  }
  if (data.pipeline) {
    gst_element_set_state(data.pipeline, GST_STATE_NULL);
    gst_object_unref(data.pipeline);
  }
}

// Gstreamer pipeline definition and setup for qtiqmmfsrc
void RobotVisionCamera::buildGStreamerPipeline(){
  gst_init(0, nullptr);

  // Input caps for qtiqmmfsrc
  std::string input_caps = "video/x-raw,format=" + input_format + 
                           ",width=" + std::to_string(width) + 
                           ",height=" + std::to_string(height) +
                           ",framerate=" + std::to_string(frame_rate) + "/1";
  
  std::string output_caps = "video/x-raw,format=" + output_format;

  RCLCPP_INFO(this->get_logger(), "Input caps: %s", input_caps.c_str());
  RCLCPP_INFO(this->get_logger(), "Output caps: %s", output_caps.c_str());
  RCLCPP_INFO(this->get_logger(), "Camera ID: %d", camera_id);

  // Create GStreamer elements for qtiqmmfsrc pipeline
  // Pipeline: qtiqmmfsrc ! capsfilter ! videoconvert ! videobalance ! videoconvert ! capsfilter ! appsink
  data.source = gst_element_factory_make("qtiqmmfsrc", "source");
  data.capsfiltersrc = gst_element_factory_make("capsfilter", "capsfiltersrc");
  data.convert1 = gst_element_factory_make("videoconvert", "convert1");
  data.videobalance = gst_element_factory_make("videobalance", "videobalance");
  data.convert2 = gst_element_factory_make("videoconvert", "convert2");
  data.capsfilterapp = gst_element_factory_make("capsfilter", "capsfilterapp");
  data.appsink = gst_element_factory_make("appsink", "sink");
  data.pipeline = gst_pipeline_new("robot-vision-camera");

  if (!data.pipeline || !data.source || !data.capsfiltersrc || !data.convert1 ||
      !data.videobalance || !data.convert2 || !data.capsfilterapp || !data.appsink) {
    RCLCPP_ERROR(this->get_logger(), "Not all GStreamer elements could be created.");
    return;
  }

  // Build the pipeline
  gst_bin_add_many(GST_BIN(data.pipeline), data.source, data.capsfiltersrc, data.convert1,
                   data.videobalance, data.convert2, data.capsfilterapp, data.appsink, nullptr);
  
  if (gst_element_link_many(data.source, data.capsfiltersrc, data.convert1, data.videobalance,
                           data.convert2, data.capsfilterapp, data.appsink, nullptr) != TRUE) {
    RCLCPP_ERROR(this->get_logger(), "Pipeline elements could not be linked.");
    gst_object_unref(data.pipeline);
    return;
  }

  // Configure qtiqmmfsrc
  g_object_set(G_OBJECT(data.source), 
               "camera", camera_id,
               nullptr);
  
  // Configure input caps filter
  g_object_set(G_OBJECT(data.capsfiltersrc), "caps",
               gst_caps_from_string(input_caps.c_str()), nullptr);
  
  // Configure videobalance with color adjustments
  g_object_set(G_OBJECT(data.videobalance),
               "brightness", brightness,
               "contrast", contrast,
               "saturation", saturation,
               "hue", hue,
               nullptr);

  // Configure output caps and appsink
  g_object_set(G_OBJECT(data.capsfilterapp), "caps",
               gst_caps_from_string(output_caps.c_str()), nullptr);
  
  g_object_set(G_OBJECT(data.appsink), 
               "emit-signals", TRUE,
               "sync", FALSE,
               "max-buffers", 10,
               "drop", TRUE,
               nullptr);
  
  g_signal_connect(data.appsink, "new-sample", G_CALLBACK(this->processData), this);
  
  RCLCPP_INFO(this->get_logger(), "GStreamer pipeline built successfully");
}

void RobotVisionCamera::startGStreamerPipeline(){
  // Start pipeline
  ret = gst_element_set_state(data.pipeline, GST_STATE_PLAYING);
  if(ret == GST_STATE_CHANGE_FAILURE){
    RCLCPP_ERROR(this->get_logger(), "Unable to set the pipeline to the playing state.");
    gst_object_unref(data.pipeline);
    return;
  }
  
  RCLCPP_INFO(this->get_logger(), "GStreamer pipeline started");
  
  bus = gst_element_get_bus(data.pipeline);

  // Wait until error or EOS
  msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
      (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  // Parse message
  if (msg != NULL) {
    GError *err;
    gchar *debug_info;

    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR:
        gst_message_parse_error(msg, &err, &debug_info);
        RCLCPP_ERROR(this->get_logger(), "Error received from element %s: %s",
            GST_OBJECT_NAME(msg->src), err->message);
        RCLCPP_ERROR(this->get_logger(), "Debugging information: %s",
            debug_info ? debug_info : "none");
        g_clear_error(&err);
        g_free(debug_info);
        break;
      case GST_MESSAGE_EOS:
        RCLCPP_INFO(this->get_logger(), "End-Of-Stream reached.");
        break;
      default:
        RCLCPP_WARN(this->get_logger(), "Unexpected message received.");
        break;
    }
    gst_message_unref(msg);
  }
}

/* Callback for appsink to parse the video stream and publish images. 
   Static function as it will be cast as a C CALLBACK function.*/
GstFlowReturn RobotVisionCamera::processData(GstElement * sink, RobotVisionCamera* node){
  GstSample *sample;
  GstBuffer *buffer;
  GstMapInfo map_info;

  // Retrieve buffer
  g_signal_emit_by_name(sink, "pull-sample", &sample);

  if(sample){
    buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *caps_structure = gst_caps_get_structure(caps, 0);

    gboolean res;
    int width, height;
    res = gst_structure_get_int(caps_structure, "width", &width);
    res |= gst_structure_get_int(caps_structure, "height", &height);
    if (!res) {
      RCLCPP_WARN(node->get_logger(), "Could not get image dimensions");
    }

    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
      gst_buffer_unmap(buffer, &map_info);
      gst_sample_unref(sample);
      return GST_FLOW_ERROR;
    }

    // Create timestamp
    auto timestamp = node->now();
    // Parse data from buffer
    cv::Mat frame_rgb(cv::Size(width, height), CV_8UC3, (char*)map_info.data, cv::Mat::AUTO_STEP);

    if (node->image_rectify){ 
      Mat frame_rgb_rect;
      remap(frame_rgb, frame_rgb_rect, node->map1, node->map2, INTER_LINEAR);
      frame_rgb = frame_rgb_rect;
    }

    sensor_msgs::msg::Image::UniquePtr cam_msg(new sensor_msgs::msg::Image());
    cam_msg->header.stamp = timestamp;
    cam_msg->header.frame_id = "camera_frame";
    cam_msg->height = frame_rgb.rows;
    cam_msg->width = frame_rgb.cols;
    cam_msg->encoding = "rgb8";
    cam_msg->is_bigendian = false;
    cam_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(frame_rgb.step);
    cam_msg->data.assign(frame_rgb.datastart, frame_rgb.dataend);
    node->pub_->publish(std::move(cam_msg));  // Publish.

    // Publish camera info with same timestamp
    if (node->camera_info_loaded_) {
      auto camera_info_msg = std::make_unique<sensor_msgs::msg::CameraInfo>(node->camera_info_msg_);
      camera_info_msg->header.stamp = timestamp;
      node->camera_info_pub_->publish(std::move(camera_info_msg));
    }

    // Publish compressed image if enabled
    if (node->image_compress) {
      sensor_msgs::msg::CompressedImage::UniquePtr compressed_msg(new sensor_msgs::msg::CompressedImage());
      compressed_msg->header.stamp = timestamp;
      compressed_msg->header.frame_id = "camera_frame";
      compressed_msg->format = "jpeg";

      // Set JPEG compression parameters
      std::vector<int> compression_params;
      compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
      compression_params.push_back(node->jpeg_quality);
      
      std::vector<uchar> buffer_compressed;
      if (cv::imencode(".jpg", frame_rgb, buffer_compressed, compression_params)) {
        compressed_msg->data = buffer_compressed;
        node->compressed_pub_->publish(std::move(compressed_msg));
      } else {
        RCLCPP_WARN(node->get_logger(), "Failed to compress image");
      }
    }

    // Free resources
    gst_buffer_unmap(buffer, &map_info);
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }

  return GST_FLOW_ERROR;
}

int main(int argc, char *argv[]){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RobotVisionCamera>("robot_vision_camera");
  
  // Use executor for ROS2 Humble compatibility
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  
  // Start GStreamer pipeline in a separate thread
  std::thread pipeline_thread([&node]() {
    node->startGStreamerPipeline();
  });
  
  // Spin the executor
  executor.spin();
  
  // Cleanup
  pipeline_thread.join();
  rclcpp::shutdown();
  
  return 0;
}