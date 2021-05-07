#include "RetinaFace.h"
#include <numeric>

RetinaFace::RetinaFace() {
    BATCH_SIZE = 1;
    INPUT_CHANNEL = 3;
    IMAGE_WIDTH = 640; // default 640
    IMAGE_HEIGHT = 640; // default 640
    obj_threshold = 0.5;  // default 0.5
    nms_threshold = 0.45; // default 0.45
    detect_mask = false;
    mask_thresh = 0.5;
    landmark_std = 1;
    feature_steps = { 32, 16, 8 };
    for (const int step : feature_steps) {
        assert(step != 0);
        int feature_map = IMAGE_HEIGHT / step;
        feature_maps.push_back(feature_map);
        int feature_size = feature_map * feature_map;
        feature_sizes.push_back(feature_size);

        //std::cout << "Feature Size: " << feature_size << std::endl;
    }
    anchor_sizes = {{512, 256}, {128, 64}, {32, 16}};
    sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0) * anchor_num;
    //std::cout << "Sum of feature: " << sum_of_feature << std::endl;

    GenerateAnchors();
}

RetinaFace::~RetinaFace() = default;

void RetinaFace::LoadModel(std::string onnx_path) { // Load Model from onnx file path
    std::cout << "Loading Detection Model" << std::endl;
    model = cv::dnn::readNetFromONNX(onnx_path);
    std::cout << "Detection Model Loaded" << std::endl;
}

std::vector<Bbox> RetinaFace::RunModel(cv::Mat& img) {
    // input cv::Mat output from cv::dnn::blobFromImage(face_cv, alpha_, cv::Size(), mean_);
    // output std::vector<Bbox> form

    std::vector<Bbox> result;
    //result.clear();

    cv::Mat img_prepared = prepareImage(img);

    cv::Scalar mean_mxnet_(0.0, 0.0, 0.0);
    float alpha_mxnet_ = 1.0;
    cv::Size size_detection(640, 640); //default (640,640)
    bool swap_BGR = true; // default: cv::imread BGR order, but model trained in RGB order

    cv::Mat blob = cv::dnn::blobFromImage(img_prepared, alpha_mxnet_, size_detection, mean_mxnet_, swap_BGR);

    // Run model here, return out
    model.setInput(blob);
    cv::Mat out = model.forward(); // store model output shape (1, 16800, 15)

    for (int item = 0; item < sum_of_feature; ++item) {
        float* current_row = (float*)out.data + item + 15;
    }

    auto faces = postProcess(img, out);

    auto rects = faces;
    if (rects.size() != 0) {
        for (const auto& rect : rects) {
            Bbox box;

            box.score = rect.confidence;
            box.x1 = rect.face_box.x - rect.face_box.w / 2;
            box.y1 = rect.face_box.y - rect.face_box.h / 2;
            box.x2 = rect.face_box.x + rect.face_box.w / 2;
            box.y2 = rect.face_box.y + rect.face_box.h / 2;
            box.ppoint[0] = rect.keypoints[0].x;
            box.ppoint[1] = rect.keypoints[0].y;
            box.ppoint[2] = rect.keypoints[1].x;
            box.ppoint[3] = rect.keypoints[1].y;
            box.ppoint[4] = rect.keypoints[2].x;
            box.ppoint[5] = rect.keypoints[2].y;
            box.ppoint[6] = rect.keypoints[3].x;
            box.ppoint[7] = rect.keypoints[3].y;
            box.ppoint[8] = rect.keypoints[4].x;
            box.ppoint[9] = rect.keypoints[4].y;

            result.push_back(box);
        }
    }
    else {
        Bbox box;

        box.score = 0;
        box.x1 = 0;
        box.y1 = 0;
        box.x2 = 0;
        box.y2 = 0;
        box.ppoint[0] = 0;
        box.ppoint[1] = 0;
        box.ppoint[2] = 0;
        box.ppoint[3] = 0;
        box.ppoint[4] = 0;
        box.ppoint[5] = 0;
        box.ppoint[6] = 0;
        box.ppoint[7] = 0;
        box.ppoint[8] = 0;
        box.ppoint[9] = 0;

        result.push_back(box);
    }

    return result;
}


void RetinaFace::GenerateAnchors() {
    // use this gererated anchors in postprocess

    float base_cx = 7.5;
    float base_cy = 7.5;

    refer_matrix = cv::Mat(sum_of_feature, bbox_head, CV_32FC1);
    int line = 0;
    for (size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) {
        for (int height = 0; height < feature_maps[feature_map]; ++height) {
            for (int width = 0; width < feature_maps[feature_map]; ++width) {
                for (int anchor = 0; anchor < anchor_sizes[feature_map].size(); ++anchor) {
                    auto* row = refer_matrix.ptr<float>(line);
                    row[0] = base_cx + (float)width * feature_steps[feature_map];
                    row[1] = base_cy + (float)height * feature_steps[feature_map];
                    row[2] = anchor_sizes[feature_map][anchor];
                    row[3] = anchor_sizes[feature_map][anchor];
                    line++;
                }
            }
        }
    }
}

cv::Mat RetinaFace::prepareImage(cv::Mat &input_img) {
    // input cv::Mat read from "ConvertFRVTImage2CVImg_RGB(faces[f], face_cv);"
    // output resized cv::Mat img, return value will goes to "cv::dnn::blobFromImage(face_cv, alpha_, cv::Size(), mean_);"
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    //float* data = result.data();

    float ratio = float(IMAGE_WIDTH) / float(input_img.cols) < float(IMAGE_HEIGHT) / float(input_img.rows) ? float(IMAGE_WIDTH) / float(input_img.cols) : float(IMAGE_HEIGHT) / float(input_img.rows);
    cv::Mat prepared_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
    cv::Mat rsz_img;
    cv::resize(input_img, rsz_img, cv::Size(), ratio, ratio);
    rsz_img.copyTo(prepared_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));

    return prepared_img;
}

std::vector<RetinaFace::FaceRes> RetinaFace::postProcess(cv::Mat &src_img, cv::Mat &result_matrix){

    std::vector<FaceRes> result;
    //float* out = output + outSize;
    float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT) ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);

    int result_cols = (detect_mask ? 2 : 1) + bbox_head + landmark_head; //result_cols 15
    //cv::Mat result_matrix = cv::Mat(sum_of_feature, result_cols, CV_32FC1, out);

    for (int item = 0; item < sum_of_feature; ++item) {
        float* current_row = (float*)result_matrix.data + item * result_cols;

        if (current_row[0] > obj_threshold) {
            //std::cout << current_row[0] << std::endl;
            FaceRes headbox;
            headbox.confidence = current_row[0];
            auto* anchor = refer_matrix.ptr<float>(item);
            auto* bbox = current_row + 1;
            auto* keyp = current_row + 1 + bbox_head;
            auto* mask = current_row + 1 + bbox_head + landmark_head;

            headbox.face_box.x = (anchor[0] + bbox[0] * anchor[2]) * ratio;
            headbox.face_box.y = (anchor[1] + bbox[1] * anchor[3]) * ratio;
            headbox.face_box.w = anchor[2] * exp(bbox[2]) * ratio;
            headbox.face_box.h = anchor[3] * exp(bbox[3]) * ratio;

            headbox.keypoints = {
                cv::Point2f((anchor[0] + keyp[0] * anchor[2] * landmark_std) * ratio,
                            (anchor[1] + keyp[1] * anchor[3] * landmark_std) * ratio),
                cv::Point2f((anchor[0] + keyp[2] * anchor[2] * landmark_std) * ratio,
                            (anchor[1] + keyp[3] * anchor[3] * landmark_std) * ratio),
                cv::Point2f((anchor[0] + keyp[4] * anchor[2] * landmark_std) * ratio,
                            (anchor[1] + keyp[5] * anchor[3] * landmark_std) * ratio),
                cv::Point2f((anchor[0] + keyp[6] * anchor[2] * landmark_std) * ratio,
                            (anchor[1] + keyp[7] * anchor[3] * landmark_std) * ratio),
                cv::Point2f((anchor[0] + keyp[8] * anchor[2] * landmark_std) * ratio,
                           (anchor[1] + keyp[9] * anchor[3] * landmark_std) * ratio)
            };

            if (detect_mask && mask[0] > mask_thresh){
                headbox.has_mask = true;
            }
            result.push_back(headbox);
        }
    }
    NmsDetect(result);

    return result;
}

void RetinaFace::NmsDetect(std::vector<FaceRes> & detections) {
    sort(detections.begin(), detections.end(), [=](const FaceRes& left, const FaceRes& right) {
        return left.confidence > right.confidence;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
            if (iou > nms_threshold)
                detections[j].confidence = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceRes& det)
    { return det.confidence == 0; }), detections.end());
}

float RetinaFace::IOUCalculate(const RetinaFace::FaceBox & det_a, const RetinaFace::FaceBox & det_b) {
    cv::Point2f center_a(det_a.x + det_a.w / 2, det_a.y + det_a.h / 2);
    cv::Point2f center_b(det_b.x + det_b.w / 2, det_b.y + det_b.h / 2);
    cv::Point2f left_up(std::min(det_a.x, det_b.x), std::min(det_a.y, det_b.y));
    cv::Point2f right_down(std::max(det_a.x + det_a.w, det_b.x + det_b.w), std::max(det_a.y + det_a.h, det_b.y + det_b.h));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x > det_b.x ? det_a.x : det_b.x;
    float inter_t = det_a.y > det_b.y ? det_a.y : det_b.y;
    float inter_r = det_a.x + det_a.w < det_b.x + det_b.w ? det_a.x + det_a.w : det_b.x + det_b.w;
    float inter_b = det_a.y + det_a.h < det_b.y + det_b.h ? det_a.y + det_a.h : det_b.y + det_b.h;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}   