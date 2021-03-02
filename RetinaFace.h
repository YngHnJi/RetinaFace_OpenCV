#ifndef RETINAFACE_TRT_RETINAFACE_H
#define RETINAFACE_TRT_RETINAFACE_H

#include "assets.h"
#include <opencv2/opencv.hpp>

class RetinaFace {
    struct FaceBox {
        float x;
        float y;
        float w;
        float h;
    };

    struct FaceRes {
        float confidence;
        FaceBox face_box;
        std::vector<cv::Point2f> keypoints;
        bool has_mask = false;
    };

public:
    explicit RetinaFace();
    ~RetinaFace();
    void LoadModel(std::string onnx_file); // load model
    std::vector<Bbox> RunModel(cv::Mat& img);
    bool empty() {return model.empty();}

private:
    void GenerateAnchors();
    cv::Mat prepareImage(cv::Mat& input_img);
    std::vector<FaceRes> postProcess(cv::Mat &vec_Mat, cv::Mat &result_matrix);
    //std::vector<Bbox> RetinaFace::postProcess_debug(cv::Mat& src_img, cv::Mat& result_matrix)
    void NmsDetect(std::vector<FaceRes>& detections);
    static float IOUCalculate(const FaceBox& det_a, const FaceBox& det_b);
    std::string onnx_file;
    //std::string engine_file;
    //nvinfer1::ICudaEngine* engine = nullptr;
    //nvinfer1::IExecutionContext* context = nullptr;
    int BATCH_SIZE; //default 1
    int INPUT_CHANNEL; //default 3
    int IMAGE_WIDTH; //default 640
    int IMAGE_HEIGHT; //default 640
    float obj_threshold; // default 0.5
    float nms_threshold; // default 0.45
    bool detect_mask; // default False
    float mask_thresh; // default 0.5
    float landmark_std; // default 1

    cv::Mat refer_matrix;
    int anchor_num = 2;
    int bbox_head = 4;
    int landmark_head = 10;
    std::vector<int> feature_sizes;
    std::vector<int> feature_steps; //default [32, 16, 8]
    std::vector<int> feature_maps;
    std::vector<std::vector<int>> anchor_sizes; //default [[512, 256], [128, 64], [32, 16]]
    int sum_of_feature;
    
    cv::dnn::Net model;
};

#endif //RETINAFACE_TRT_RETINAFACE_H