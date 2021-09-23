#include "ros/ros.h"
#include "std_msgs/String.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
using namespace message_filters;

#include "../include/base_realsense_node.h"
using namespace realsense2_camera;

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
using namespace InferenceEngine;

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/rgbd.hpp>
#include <boost/thread.hpp>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <visualization_msgs/Marker.h>
#include <cmath>
#define pi 3.14159265

// initialize
cv::Mat frame;  
size_t width;
size_t height;
cv::Mat depth_frame;
cv::Mat bbox_depth;

// parameters for RANSAC
int iter_maxNum = 50000;
int inPlaneNum_max = 0;


// functions
void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
	/* Resize and copy data from the image to the input blob */
	Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
	matU8ToBlob<uint8_t>(frame, frameBlob);
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<T> & mask) {
        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors = {11.0, 15.0, 33.0, 19.0, 33.0, 91.0, 122.0, 32.0, 106.0, 104.0, 340.0, 42.0, 222.0, 99.0, 899.0, 117.0, 951.0, 251.0};
    //{10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};

    YoloParams() {}

    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        anchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(mask);
    }
};

void ParseYOLOV3Output(const YoloParams &params, const std::string & output_name,
                       const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + output_name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));

    auto side = out_blob_h;
    auto side_square = side * side;
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < params.num; ++n) {
            int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
            int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
            for (int j = 0; j < params.classes; ++j) {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

// ROS funciton----------------------------------------------------------------------------------------------------------------
void DepthCallback(const sensor_msgs::ImageConstPtr& depth_img) {
	ROS_INFO("Depth Seq: [%d]", depth_img->header.seq);
    //ROS_WARN("%s", depth_img->encoding.c_str());
    depth_frame = cv_bridge::toCvCopy(depth_img, sensor_msgs::image_encodings::TYPE_32FC1)->image;

	/*
	double minVal; 
	double maxVal; 
	cv::minMaxLoc(depth_frame, &minVal, &maxVal);
	printf("before min max: %f %f\n", minVal, maxVal);
	cv::threshold(depth_frame, depth_frame, 2000, 0, CV_THRESH_TOZERO_INV);
	cv::minMaxLoc(depth_frame, &minVal, &maxVal);
	printf("after min max: %f %f\n", minVal, maxVal);
	*/

	//normalize(depth_frame, depth_frame, 0, 1, CV_MINMAX);

	//std::cout << "image data: " << depth_frame.at<float>(296, 303) << std::endl;//獲取座標為240,320的深度值, 單位是毫米
	//cv::imshow("ori depth image", depth_frame);
	//cv::waitKey(1);
}

void pointCloud2ToZ(const sensor_msgs::PointCloud2 &msg)
{
	sensor_msgs::PointCloud out_pointcloud;
	sensor_msgs::convertPointCloud2ToPointCloud(msg, out_pointcloud);
	for (int i=0; i<out_pointcloud.points.size(); i++) {
		std::cout << out_pointcloud.points[i].x << ", " << out_pointcloud.points[i].y << ", " << out_pointcloud.points[i].z << std::endl;
	}
	std::cout << "------" << std::endl;
}

void ColorCallback(const sensor_msgs::ImageConstPtr& color_img) {
    ROS_WARN("%s", color_img->encoding.c_str());
    cv::Mat data;
    data = cv_bridge::toCvCopy(color_img, sensor_msgs::image_encodings::BGR8)->image;
    //cv::imshow("test", data);
    cv::waitKey(1);
}

void ImuCallback(const sensor_msgs::ImuConstPtr& imu_msg) {
	ROS_INFO("Imu Seq: [%d]", imu_msg->header.seq);
    ROS_INFO("Imu angular_velocity x: [%f], y: [%f], z: [%f], w: [%f]", 
	imu_msg->angular_velocity.x,imu_msg->angular_velocity.y,imu_msg->angular_velocity.z);
}

void AccCallback(const sensor_msgs::ImuConstPtr& acc_msg) {
	ROS_INFO("Acc Seq: [%d]", acc_msg->header.seq);
    ROS_INFO("Acc angular_velocity x: [%f], y: [%f], z: [%f], w: [%f]", 
	acc_msg->linear_acceleration.x,acc_msg->linear_acceleration.y,acc_msg->linear_acceleration.z);
}

void callback(const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img){
	ROS_WARN("%s", color_img->encoding.c_str());
    cv::Mat data;
    data = cv_bridge::toCvCopy(color_img, sensor_msgs::image_encodings::BGR8)->image;
    //cv::imshow("test", data);
    cv::waitKey(1);

	ROS_WARN("%s", depth_img->encoding.c_str());
    data = cv_bridge::toCvCopy(depth_img, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    std::cout << "image data: " << data.at<float>(296, 303) << std::endl;//表示获取图像坐标为240,320的深度值,单位是毫米
}

void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg){
	ROS_INFO("Image Seq: [%d]", image_msg->header.seq);
    cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
    //cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
	color_mat.copyTo(frame);
	
    //cv::imshow("ori color image", frame);

    width  = (size_t)color_mat.size().width;
    height = (size_t)color_mat.size().height;
}

int main(int argc, char** argv) {
    // use grid concept, assume the grid size is 20 * 10 * 25
    int resolution_x = 640;
    int resolution_y = 480;
    int resolution_z = 3000;
    int x_range = 20; // 20
    int y_range = 10; // 10
    int z_range = 20; // 25
    float exist_grid[int(resolution_x / x_range)][int(resolution_y / y_range)][int(resolution_z / z_range)] {};
    for (int i = 0; i < int(resolution_x / x_range); i++){
        for (int j = 0; j < int(resolution_y / y_range); j++){
            for (int k = 0; k < int(resolution_z / z_range); k++){
                exist_grid[i][j][k] = 0.5f;
            }
        }
    }
    cv::Mat a_b_c_d = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);

    ros::init(argc, argv, "listener");
    ros::NodeHandle node;   
    //ros::Subscriber color_sub, depth_sub, imu_sub, acc_sub;
	
    ros::Publisher marker_pub = node.advertise<visualization_msgs::Marker>("visualization_marker", 10);
    ros::Rate r(1);
    srand(1);

	ros::Subscriber image_sub = node.subscribe("/camera/color/image_raw", 1, imageCallback);
	ros::Subscriber depth_sub = node.subscribe("/camera/aligned_depth_to_color/image_raw", 1, DepthCallback); // /camera/aligned_depth_to_color/image_raw /camera/depth/image_rect_raw

	std::string FLAGS_d = "CPU"; //"MYRIAD"; // CPU
    // ***** attention! the input image should be BGR format, RGB is for opencv display!
	std::string FLAGS_m = "/opt/intel/openvino_2020.4.287/deployment_tools/model_optimizer/pallet_0818.xml"; // pallet_0818 frozen_darknet_yolov3_model
	std::string FLAGS_l = "";
	std::string FLAGS_c = "";
	std::string FLAGS_labels = "";
	std::string FLAGS_u = "";
	bool FLAGS_pc = false;
	double FLAGS_t = 0.5; // 0.5
	double FLAGS_iou_t = 0.4; // 0.4
	bool FLAGS_r = false;
	bool FLAGS_no_show = false;
	bool FLAGS_auto_resize = false;

	try{
		// --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /**Loading extensions to the devices **/

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        /** Reading network model **/
        auto cnnNetwork = ie.ReadNetwork(FLAGS_m);
        /** Reading labels (if specified) **/
        std::vector<std::string> labels;
        if (!FLAGS_labels.empty()) {
            std::ifstream inputFile(FLAGS_labels);
            std::string label; 
            while (std::getline(inputFile, label)) {
                labels.push_back(label);
            }
            if (labels.empty())
                throw std::logic_error("File empty or not found: " + FLAGS_labels);
        }
        // -----------------------------------------------------------------------------------------------------

        /** YOLOV3-based network should have one input and three output **/
        // --------------------------- 3. Configuring input and output -----------------------------------------
        // --------------------------------- Preparing input blobs ---------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks that have only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }

        ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
        SizeVector& inSizeVector = inputShapes.begin()->second;
        inSizeVector[0] = 1;  // set batch to 1
        cnnNetwork.reshape(inputShapes);
        // --------------------------------- Preparing output blobs -------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }

        std::map<std::string, YoloParams> yoloParams;
        if (auto ngraphFunction = cnnNetwork.getFunction()) {
            for (const auto op : ngraphFunction->get_ops()) {
                auto outputLayer = outputInfo.find(op->get_friendly_name());
                if (outputLayer != outputInfo.end()) {
                    auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                    if (!regionYolo) {
                        throw std::runtime_error("Invalid output type: " +
                            std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                    }
                    yoloParams[outputLayer->first] = YoloParams(regionYolo);
                }
            }
        }
        else {
            throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
        }

        if (!labels.empty() && static_cast<int>(labels.size()) != yoloParams.begin()->second.classes) {
            throw std::runtime_error("The number of labels is different from numbers of model classes");
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Creating infer request -----------------------------------------------
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        bool isLastFrame = false;
        bool isAsyncMode = false;  // execution is always started using SYNC mode
        bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_render_time = 0;

        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;
        std::cout << "To switch between sync/async modes, press TAB key in the output window" << std::endl;
        cv::Size graphSize{static_cast<int>(1920 / 4), 60};
        Presenter presenter(FLAGS_u, 1080 - graphSize.height - 10, graphSize);
        while (ros::ok()) {

			ros::spinOnce();

            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the Async mode, we capture frame to populate the NEXT infer request
            // in the regular mode, we capture frame to the CURRENT infer request

            if (isAsyncMode) {
                if (isModeChanged) {
                    FrameToBlob(frame, async_infer_request_curr, inputName);
                }
                //if (!isLastFrame) {
                //    FrameToBlob(next_frame, async_infer_request_next, inputName);
                //}
            } else if (!isModeChanged) {
                FrameToBlob(frame, async_infer_request_curr, inputName);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the true Async mode, we start the NEXT infer request while waiting for the CURRENT to complete
            // in the regular mode, we start the CURRENT request and wait for its completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    async_infer_request_curr->StartAsync();
                }
                /*
                if (!isLastFrame) {
                    async_infer_request_next->StartAsync();
                }
                */
            } else if (!isModeChanged) {
                async_infer_request_curr->StartAsync();
            }

            if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);

                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();
                presenter.drawGraphs(frame);
                std::ostringstream out;
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (ocv_decode_time + ocv_render_time) << " ms";
                cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                out.str("");
                out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
                cv::putText(frame, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
                if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                    out.str("");
                    out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                                cv::Scalar(255, 0, 0));
                }

                // ---------------------------Processing output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
                unsigned long resized_im_h = getTensorHeight(inputDesc);
                unsigned long resized_im_w = getTensorWidth(inputDesc);
                std::vector<DetectionObject> objects;
                // Parsing outputs
                for (auto &output : outputInfo) {
                    auto output_name = output.first;
                    Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
                    ParseYOLOV3Output(yoloParams[output_name], output_name, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
                }
                // Filtering overlapping boxes
                std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
                for (size_t i = 0; i < objects.size(); ++i) {
                    if (objects[i].confidence == 0)
                        continue;
                    for (size_t j = i + 1; j < objects.size(); ++j)
                        if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                            objects[j].confidence = 0;
                }
                // Drawing boxes
                for (auto &object : objects) {
                    auto label = object.class_id;
                    float confidence = object.confidence;

                    if (object.confidence < FLAGS_t){
                        continue;
                    }
                    
                    if (FLAGS_r) {
                        std::cout << "[" << label << "] element, prob = " << confidence <<
                                  "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }

                    if (confidence > FLAGS_t) {
                        /** Drawing only objects when >confidence_threshold probability **/
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        cv::putText(frame,
                                    (!labels.empty() ? labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                                    cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    cv::Scalar(0, 0, 255));
                        cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                                      cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));

                        // try to output the depth information in the person bbox
                        if (label == 0)
                        {   // if the label is pallet
                            if (object.xmin < 0)
                                object.xmin = 0;
                            
                            if (object.ymin < 0)
                                object.ymin = 0;

                            if (object.xmax >= depth_frame.cols)
                                object.xmax = depth_frame.cols;

                            if (object.ymax >= depth_frame.rows)
                                object.ymax = depth_frame.rows;

                            //normalize(depth_frame, depth_frame, 0, 1, CV_MINMAX);
                            bbox_depth = depth_frame(cv::Range(object.ymin, object.ymax), cv::Range(object.xmin, object.xmax));

                            // 將x, y, z pubilsh在rviz上顯示
                            visualization_msgs::Marker points, line_strip, line_list, red_points; 

                            //初始化
                            points.header.frame_id = line_strip.header.frame_id = line_list.header.frame_id = "/camera_link";
                            points.header.stamp = line_strip.header.stamp = line_list.header.stamp = ros::Time::now();
                            points.ns = line_strip.ns = line_list.ns = "points_and_lines";
                            points.action = line_strip.action = line_list.action = visualization_msgs::Marker::ADD;
                            points.pose.orientation.w = line_strip.pose.orientation.w = line_list.pose.orientation.w = 1.0;

                            red_points.header.frame_id = line_strip.header.frame_id = line_list.header.frame_id = "/camera_link";
                            red_points.header.stamp = line_strip.header.stamp = line_list.header.stamp = ros::Time::now();
                            red_points.ns = line_strip.ns = line_list.ns = "points_and_lines";
                            red_points.action = line_strip.action = line_list.action = visualization_msgs::Marker::ADD;
                            red_points.pose.orientation.w = line_strip.pose.orientation.w = line_list.pose.orientation.w = 1.0;

                            //分配3个id
                            points.id = 0;
                            line_strip.id = 1;
                            line_list.id = 2;
                            red_points.id = 3;

                            //初始化形状
                            points.type = visualization_msgs::Marker::POINTS;
                            red_points.type = visualization_msgs::Marker::POINTS;
                            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
                            line_list.type = visualization_msgs::Marker::LINE_LIST;

                            //初始化大小
                            // POINTS markers use x and y scale for width/height respectively
                            points.scale.x = 1;
                            points.scale.y = 1;
                            red_points.scale.x = 0.5;
                            red_points.scale.y = 0.5;

                            // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
                            line_strip.scale.x = 0.1;
                            line_list.scale.x = 10;

                            //初始化颜色
                            // Points are green
                            points.color.g = 1.0f;
                            points.color.a = 1.0;
                            red_points.color.r = 1.0f;
                            red_points.color.a = 1.0;

                            // Line strip is blue
                            line_strip.color.b = 1.0;
                            line_strip.color.a = 1.0;

                            // Line list is red
                            line_list.color.r = 1.0;
                            line_list.color.a = 1.0;

                            // get the maximum range value
                            //double min, max;
                            //cv::minMaxLoc(bbox_depth, &min, &max);
                            // create a index for each grid
                            int grid[int(resolution_x / x_range)][int(resolution_y / y_range)][int(resolution_z / z_range)] {};
                            
                            // put the points in the grid
                            int xxx, yyy, zzz;
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){
                                    if (int(bbox_depth.at<float>(i, j) < 1)) continue; // if the distance is zero, discard
                                    if (int(bbox_depth.at<float>(i, j) / z_range) > int(resolution_z / z_range)) continue; // if the distance bigger than 3m, discard

                                    xxx = int((j + object.xmin) / x_range);
                                    yyy = int((i + object.ymin) / y_range);
                                    zzz = int(bbox_depth.at<float>(i, j) / z_range);
                                    grid[xxx][yyy][zzz]++;
                                    //std::cout << xxx << " " << yyy << " " << zzz << " " << grid[xxx][yyy][zzz] << std::endl;
                                }
                            }

                            // show grid value
                            /*
                            for (int k = 0; k < int(resolution_z / z_range); k++){
                                for (int j = 0; j < int(resolution_y / y_range); j++){
                                    for (int i = 0; i < int(resolution_x / x_range); i++){
                                        if (k < 80 && k > 50){
                                            std::cout << grid[i][j][k] << " ";
                                        }
                                    }
                                    if (k < 80 && k > 50){
                                        std::cout << std::endl;
                                    }
                                }
                                if (k < 80 && k > 50){
                                    std::cout << "++++++++++++++++++++++++++++++" << std::endl;
                                }
                            }
                            */

                            // create a index to search the grid in z axis
                            int grid_z[int(resolution_x / x_range)][int(resolution_y / y_range)] {0};
                            int grid_z_index[int(resolution_x / x_range)][int(resolution_y / y_range)] {0};
                            // find the maximum z grid
                            for (int j = 0; j < int(resolution_y / y_range); j++){
                                for (int i = 0; i < int(resolution_x / x_range); i++){
                                    for (int k = 0; k < int(resolution_z / z_range); k++){
                                        if (grid[i][j][k] > grid_z[i][j]){
                                            grid_z[i][j] = grid[i][j][k];
                                            grid_z_index[i][j] = k;
                                        }
                                    }
                                }
                            }
                            
                            // plot the whole pallet points at rviz
                            int max_depth_sample = 1;
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){
                                    if (int(bbox_depth.at<float>(i, j) < 1)) continue;
                                    geometry_msgs::Point p1;
                                    p1.x = j + object.xmin;
                                    p1.y = i + object.ymin;
                                    p1.z = bbox_depth.at<float>(i, j);
                                    red_points.points.push_back(p1);
                                    /*
                                    if (int(bbox_depth.at<float>(i, j) / z_range) == grid_z_index[int((j + object.xmin) / x_range)][int((i + object.ymin) / y_range)]){
                                        if (grid_z[int((j + object.xmin) / x_range)][int((i + object.ymin) / y_range)] > 50){
                                            if (int(bbox_depth.at<float>(i, j) / z_range) < 50 && int(bbox_depth.at<float>(i, j) / z_range) > 10){
                                                
                                                geometry_msgs::Point p;
                                                p.x = j;
                                                p.y = i;
                                                p.z = bbox_depth.at<float>(i, j);
                                                points.points.push_back(p);
                                                
                                                max_depth_sample++;
                                            }
                                        }
                                    }
                                    */
                                }
                            }
                            marker_pub.publish(red_points);
                            //marker_pub.publish(points);
                            /*
                            // prepare the points for pallet plane fitting
                            int total_point_number = 0; // total point number
                            int total_x = 0;
                            int total_y = 0;
                            int total_z = 0;
                            cv::Mat x_y_1 = cv::Mat::zeros(cv::Size(3, max_depth_sample), CV_32FC1); // x, y = col row
                            cv::Mat zzzz = cv::Mat::zeros(cv::Size(1, max_depth_sample), CV_32FC1);

                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){
                                    if (int(bbox_depth.at<float>(i, j) < 1)) continue;

                                    if (int(bbox_depth.at<float>(i, j) / z_range) == grid_z_index[int((j + object.xmin) / x_range)][int((i + object.ymin) / y_range)]){
                                        if (grid_z[int((j + object.xmin) / x_range)][int((i + object.ymin) / y_range)] > 50){
                                            if (int(bbox_depth.at<float>(i, j) / z_range) < 50 && int(bbox_depth.at<float>(i, j) / z_range) > 10){
                                                total_x += j;
                                                total_y += i;
                                                total_z += bbox_depth.at<float>(i, j);
                                                
                                                x_y_1.at<float>(total_point_number, 0) = j;
                                                x_y_1.at<float>(total_point_number, 1) = i;
                                                x_y_1.at<float>(total_point_number, 2) = 1;
                                                zzzz.at<float>(total_point_number, 0) = -(bbox_depth.at<float>(i, j));
                                                
                                                total_point_number++;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // calculate the pallet plane equation
                            cv::Mat a_b_c_d = ((x_y_1.t()*x_y_1).inv()*x_y_1.t()*zzzz);
                            //std::cout << "a_b_c_d: " << a_b_c_d << " " << std::endl;

                            // display the normal vector for the pallet plane at rviz
                            geometry_msgs::Point p1;
                            p1.x = int(total_x / total_point_number);
                            p1.y = int(total_y / total_point_number);
                            p1.z = int(total_z / total_point_number);
                            line_list.points.push_back(p1);

                            geometry_msgs::Point p2;
                            p2.x = int(a_b_c_d.at<float>(0, 0)*10000);
                            p2.y = int(a_b_c_d.at<float>(0, 1)*10000);
                            p2.z = 1*10000;
                            line_list.points.push_back(p2);
                            marker_pub.publish(line_list);
                            */
                            // if this grid pass the condition, * 1.2 else * 0.9
                            for (int k = 0; k < int(resolution_z / z_range); k++){                        
                                for (int j = 0; j < int(resolution_y / y_range); j++){
                                    for (int i = 0; i < int(resolution_x / x_range); i++){
                                        if (grid[i][j][k] > 75 && k < 70 && k > 40){
                                            exist_grid[i][j][k] = exist_grid[i][j][k] * 1.2;
                                        }
                                        else{
                                            exist_grid[i][j][k] = exist_grid[i][j][k] * 0.9;
                                        }
                                    }
                                }
                            }

                            // if the possibility > 0.75, view as a existed grid
                            /*
                            for (int k = 0; k < int(resolution_z / z_range); k++){                        
                                for (int j = 0; j < int(resolution_y / y_range); j++){
                                    for (int i = 0; i < int(resolution_x / x_range); i++){
                                        if (exist_grid[i][j][k] > 0.75){
                                            max_depth_sample++;
                                        }
                                    }
                                }
                            }
                            */
                            
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){
                                    xxx = int((j + object.xmin) / x_range);
                                    yyy = int((i + object.ymin) / y_range);
                                    zzz = int(bbox_depth.at<float>(i, j) / z_range);
                                    if (exist_grid[xxx][yyy][zzz] > 0.99){
                                        max_depth_sample++;
                                    }
                                }
                            }
                            
                            // prepare the points for pallet plane fitting
                            int total_point_number = 0; // total point number
                            float total_x = 0;
                            float total_y = 0;
                            float total_z = 0;
                            cv::Mat x_y_1 = cv::Mat::zeros(cv::Size(3, max_depth_sample), CV_32FC1); // x, y = col row
                            cv::Mat zzzz = cv::Mat::zeros(cv::Size(1, max_depth_sample), CV_32FC1);

                            // display the grid selection result at rviz and prepare plane fitting
                            
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){
                                    xxx = int((j + object.xmin) / x_range);
                                    yyy = int((i + object.ymin) / y_range);
                                    zzz = int(bbox_depth.at<float>(i, j) / z_range);
                                    if (exist_grid[xxx][yyy][zzz] > 0.99){
                                        geometry_msgs::Point p;
                                        p.x = j + object.xmin;
                                        p.y = i + object.ymin;
                                        p.z = bbox_depth.at<float>(i, j);
                                        points.points.push_back(p);

                                        total_x += j + object.xmin;
                                        total_y += i + object.ymin;
                                        total_z += bbox_depth.at<float>(i, j);
                                        
                                        x_y_1.at<float>(total_point_number, 0) = j + object.xmin;
                                        x_y_1.at<float>(total_point_number, 1) = i + object.ymin;
                                        x_y_1.at<float>(total_point_number, 2) = 1;
                                        zzzz.at<float>(total_point_number, 0) = -(bbox_depth.at<float>(i, j));

                                        total_point_number++; 
                                    }
                                }
                            }
                            
                            /*
                            for (int k = 0; k < int(resolution_z / z_range); k++){                        
                                for (int j = 0; j < int(resolution_y / y_range); j++){
                                    for (int i = 0; i < int(resolution_x / x_range); i++){
                                        
                                        geometry_msgs::Point p;
                                        p.x = i * x_range;
                                        p.y = j * y_range;
                                        p.z = k * z_range;
                                        points.points.push_back(p);
                                        
                                        if (exist_grid[i][j][k] > 0.75){
                                            
                                            geometry_msgs::Point p;
                                            p.x = i * x_range;
                                            p.y = j * y_range;
                                            p.z = k * z_range;
                                            points.points.push_back(p);
                                            
                                            total_x += i * x_range;
                                            total_y += j * y_range;
                                            total_z += k * z_range;
                                            
                                            x_y_1.at<float>(total_point_number, 0) = i * x_range;
                                            x_y_1.at<float>(total_point_number, 1) = j * y_range;
                                            x_y_1.at<float>(total_point_number, 2) = 1;
                                            zzzz.at<float>(total_point_number, 0) = -(k * z_range);
                                            
                                            total_point_number++;
                                        }
                                    }
                                }
                            }
                            */
                            marker_pub.publish(points);
                            
                            // calculate the pallet plane equation
                            //cv::Mat abcd = ((x_y_1.t() * x_y_1).inv() * x_y_1.t() * zzzz);
                            //a_b_c_d = a_b_c_d*0.99 + abcd*0.01;
                            a_b_c_d = ((x_y_1.t() * x_y_1).inv() * x_y_1.t() * zzzz);
                            std::cout << "a_b_c_d: " << a_b_c_d << " " << std::endl;

                            // display the normal vector for the pallet plane at rviz
                            geometry_msgs::Point p1;
                            p1.x = int(total_x / total_point_number);
                            p1.y = int(total_y / total_point_number);
                            p1.z = int(total_z / total_point_number);
                            line_list.points.push_back(p1);

                            geometry_msgs::Point p2;
                            p2.x = int(total_x / total_point_number) + int(a_b_c_d.at<float>(0, 0) * 1000);
                            p2.y = int(total_y / total_point_number);//int(a_b_c_d.at<float>(0, 1) * 10000);
                            p2.z = int(total_z / total_point_number) + 1 * 1000;
                            line_list.points.push_back(p2);
                            marker_pub.publish(line_list);
                            
                            //****************************************************************************************************************
                            // get the maximum range value
                            //double min, max;
                            //cv::minMaxLoc(bbox_depth, &min, &max);
                            /*
                            int depth_sample_count[500] {0};                            
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){


                                    if (int(bbox_depth.at<float>(i, j) < 1)) continue;
                                    depth_sample_count[int(bbox_depth.at<float>(i, j)/50)]++;
                                    
                                    geometry_msgs::Point p;
                                    p.x = j;
                                    p.y = i;
                                    p.z = bbox_depth.at<float>(i, j);
                                    red_points.points.push_back(p);
                                }
                            }

                            marker_pub.publish(red_points);
                            marker_pub.publish(points);
                            */
                            /*
                            // find the maximum slice
                            int max_depth_sample_index = 0;
                            int max_depth_sample = 0;
                            for (int i = 0; i < 500; i++) {
                                if (depth_sample_count[i] > max_depth_sample){
                                    max_depth_sample = depth_sample_count[i];
                                    max_depth_sample_index = i;
                                }
                            }
                            
                            // copy the points
                            int total_point_number = 0; // total point number
                            int total_x = 0;
                            int total_y = 0;
                            int total_z = 0;
                            */
                            /*
                            cv::Mat test = cv::Mat::zeros(cv::Size(2, 3), CV_32FC1); // x, y = col row
                            test.at<float>(0, 0) = 6.0f;
                            test.at<float>(0, 1) = 1.0f;
                            test.at<float>(1, 0) = 2.0f;
                            test.at<float>(1, 1) = 3.0f;
                            test.at<float>(2, 0) = 4.0f;
                            test.at<float>(2, 1) = 5.0f;
                            // result be like 
                            M = [6, 1;
                                 2, 3;
                                 4, 5]
                            */    
                            /*
                            cv::Mat x_y_z = cv::Mat::zeros(cv::Size(3, max_depth_sample), CV_32FC1); // x, y = col row
                            cv::Mat x_y_1 = cv::Mat::zeros(cv::Size(3, max_depth_sample), CV_32FC1);
                            cv::Mat zzz = cv::Mat::zeros(cv::Size(1, max_depth_sample), CV_32FC1);
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){
                                    if (int(bbox_depth.at<float>(i, j)/50) == max_depth_sample_index){
                                        x_y_1.at<float>(total_point_number, 0) = j;
                                        x_y_1.at<float>(total_point_number, 1) = i;
                                        x_y_1.at<float>(total_point_number, 2) = 1;
                                        zzz.at<float>(total_point_number, 0) = -(bbox_depth.at<float>(i, j));
                                        
                                        x_y_z.at<float>(0, total_point_number) = j; // x 
                                        x_y_z.at<float>(1, total_point_number) = i; // y
                                        x_y_z.at<float>(2, total_point_number) = bbox_depth.at<float>(i, j); // z
                                        
                                        total_point_number++;
                                        total_x += j;
                                        total_y += i;
                                        total_z += bbox_depth.at<float>(i, j);
                                    }
                                }
                            }
                            
                            cv::Mat a_b_c_d = -((x_y_1.t()*x_y_1).inv()*x_y_1.t()*zzz);
                            std::cout << "a_b_c_d: " << a_b_c_d << " " << std::endl;

                            geometry_msgs::Point p1;
                            p1.x = int(total_x / total_point_number);
                            p1.y = int(total_y / total_point_number);
                            p1.z = int(total_z / total_point_number);
                            line_list.points.push_back(p1);

                            geometry_msgs::Point p2;
                            p2.x = int(a_b_c_d.at<float>(0, 0)*10000);
                            p2.y = int(a_b_c_d.at<float>(0, 1)*10000);
                            p2.z = int(a_b_c_d.at<float>(0, 2)*10000);
                            line_list.points.push_back(p2);
                            //marker_pub.publish(line_list);

                            // check even or odd at x axis, loop a counter for miuns or add for every x position
                            // set the circle center at x / 2 + 1
                            
                            int slice_long = 50;
                            int low_z = max_depth_sample_index * 50;
                            int high_z = (max_depth_sample_index + 1) * 50;
                            int even_or_odd = bbox_depth.cols % 2;
                            int circle_R, circle_L, circle_C;

                            if (even_or_odd == 0){
                                circle_C = int(bbox_depth.cols / 2 + 1);
                                circle_L = int(bbox_depth.cols / 2);
                                circle_R = int(bbox_depth.cols / 2 - 1);
                            }else{
                                circle_C = int(bbox_depth.cols / 2 + 1);
                                circle_L = int(bbox_depth.cols / 2);
                                circle_R = int(bbox_depth.cols / 2);
                            }
                            
                            int circle_sample_count[180] {0};
                            for (int degree = 0; degree < 180; degree++){
                                for (int i = 0; i < bbox_depth.rows; i++){
                                    for (int j = 0; j < circle_C; j++){
                                        int circle_j = circle_C - j - 1;
                                        // 重寫轉角方式
                                        int now_low_z = low_z - int(tan(degree*pi/180) * circle_j / 2);
                                        int now_high_z = high_z - int(tan(degree*pi/180) * circle_j / 2);
                                        
                                        if (now_low_z < bbox_depth.at<float>(i, j) && bbox_depth.at<float>(i, j) < now_high_z){
                                            circle_sample_count[degree]++;
                                        }
                                    }
                                    // remember circle_C part
                                    for (int j = circle_C + 1; j < bbox_depth.cols; j++){
                                        int circle_j = j - circle_C;
                                        int now_low_z = low_z - int(tan(degree*pi/180) * circle_j / 2);
                                        int now_high_z = high_z - int(tan(degree*pi/180) * circle_j / 2);

                                        if (now_low_z < bbox_depth.at<float>(i, j) && bbox_depth.at<float>(i, j) < now_high_z){
                                            circle_sample_count[degree]++;
                                        }
                                    }
                                }
                            }

                            int max_degree_index = 0;
                            int max_degree_count = 0;
                            for (int degree = 0; degree < 180; degree++){
                                std ::cout << "circle_sample_count " << circle_sample_count[degree] << std::endl;
                                if (max_degree_count < circle_sample_count[degree]){
                                    max_depth_sample_index = degree;
                                    max_degree_count = circle_sample_count[degree];
                                }
                            }
                            std::cout << "max_depth_sample_index: " << max_depth_sample_index << " max_degree_count: " << max_degree_count << std::endl;
                            
                            // output plane points
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < circle_C; j++){
                                    int circle_j = circle_C - j - 1;
                                    int now_low_z = low_z - int(tan(max_depth_sample_index*pi/180) * circle_j / 2);
                                    int now_high_z = high_z - int(tan(max_depth_sample_index*pi/180) * circle_j / 2);
                                    
                                    if (now_low_z < bbox_depth.at<float>(i, j) && bbox_depth.at<float>(i, j) < now_high_z){
                                        geometry_msgs::Point p;
                                        p.x = j;
                                        p.y = i;
                                        p.z = bbox_depth.at<float>(i, j);
                                        points.points.push_back(p);
                                    }
                                }
                                // remember circle_C part
                                for (int j = circle_C + 1; j < bbox_depth.cols; j++){
                                    int circle_j = j - circle_C;
                                    int now_low_z = low_z + int(tan(max_depth_sample_index*pi/180) * circle_j / 2);
                                    int now_high_z = high_z + int(tan(max_depth_sample_index*pi/180) * circle_j / 2);

                                    if (now_low_z < bbox_depth.at<float>(i, j) && bbox_depth.at<float>(i, j) < now_high_z){
                                        geometry_msgs::Point p;
                                        p.x = j;
                                        p.y = i;
                                        p.z = bbox_depth.at<float>(i, j);
                                        points.points.push_back(p);
                                    }
                                }
                            }
                            marker_pub.publish(points);                   
                            //r.sleep();
                            */         
                            /*
                            float average_x, average_y, average_z;
                            average_x = total_x / total_point_number;
                            average_y = total_y / total_point_number;
                            average_z = total_z / total_point_number;
                            cv::Mat avg_x_y_z = cv::Mat::zeros(cv::Size(3, 1), CV_32FC1);
                            avg_x_y_z.at<float>(0, 0) = average_x;
                            avg_x_y_z.at<float>(0, 1) = average_y;
                            avg_x_y_z.at<float>(0, 2) = average_z;
                            std::cout << total_x << " " << total_y << " " << total_z << " " << total_point_number << std::endl;
                            std::cout << avg_x_y_z << std::endl;
                            for (int r = 0; r < x_y_z.rows; ++r) {
                                x_y_z.row(r) = x_y_z.row(r) - avg_x_y_z;
                            }

                            // show in rviz
                            for (int i = 0; i < bbox_depth.rows; i++){
                                for (int j = 0; j < bbox_depth.cols; j++){
                                    if (int(bbox_depth.at<float>(i, j)/50) == max_depth_sample_index){
                                        geometry_msgs::Point p;
                                        p.x = j;
                                        p.y = i;
                                        p.z = bbox_depth.at<float>(i, j);
                                        points.points.push_back(p);
                                    }
                                }
                            }
                                    
                            
                            marker_pub.publish(points);
                            //marker_pub.publish(line_strip);
                            //marker_pub.publish(line_list);
                            //r.sleep();
                            
                            cv::Mat U, W, V;
                            cv::SVD::compute(x_y_z, W, U, V);
                            std::cout << U.rows << " " << U.cols << " " << std::endl;
                            std::cout << W.rows << " " << W.cols << " " << std::endl;
                            std::cout << V.rows << " " << V.cols << " " << std::endl;
                            //std::cout << U.at<float>(2, 0) << " " << U.at<float>(2, 1) << " " << U.at<float>(2, 2) << std::endl;
                            //std::cout << V.at<float>(2, 0) << " " << V.at<float>(2, 1) << " " << V.at<float>(2, 2) << std::endl;
                            float AAA = V.at<float>(2, 0);
                            float BBB = V.at<float>(2, 1);
                            float CCC = V.at<float>(2, 2);
                            float DDD = -(average_x * AAA + average_y * BBB + average_z * CCC); // ax + by + cz + d = 0
                            printf("----------- %fx + %fy + %fz + %f = 0\n", AAA, BBB, CCC , DDD);

                            geometry_msgs::Point p1;
                            p1.x = average_x;
                            p1.y = average_y;
                            p1.z = average_z;
                            line_list.points.push_back(p1);
                            geometry_msgs::Point p2;
                            p2.x = AAA*1000;
                            p2.y = BBB*1000;
                            p2.z = CCC*1000;
                            line_list.points.push_back(p2);
                            marker_pub.publish(line_list);
                            */
                            cv::imshow("Detection results", frame);
                            cv::imshow("cut depth", bbox_depth);
                        }
                    }
                }
            }

            if (!FLAGS_no_show) {
                
                //cv::imshow("ori depth", depth_frame);
                //cv::waitKey(0);
            }

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode, we swap the NEXT and CURRENT requests for the next iteration
            //frame = next_frame;
            //next_frame = cv::Mat();
            if (isAsyncMode) {
                async_infer_request_curr.swap(async_infer_request_next);
            }

            const int key = cv::waitKey(1);
            if (27 == key)  // Esc
                break;
            if (9 == key) {  // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            } 
			else {
                presenter.handleKey(key);
            }
        }
        // end while loop

        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;

        /** Showing performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(*async_infer_request_curr, std::cout, getFullDeviceName(ie, FLAGS_d));
        }

        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;

    return 0;
}
