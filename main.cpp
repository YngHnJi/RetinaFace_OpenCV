// Author Young-hoon Ji, younghoon_ji@kookmin.ac.kr

#include <iostream>
#include <string>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "RetinaFace.h"
#include <io.h>
#include <time.h>

#define FEATURE_VECTOR_SIZE 15

std::vector<std::string> get_files_inDirectory(const std::string& _path, const std::string& _filter){
	
	std::string searching_dir = _path + "*." + _filter;
	
	std::vector<std::string> return_;
	int cnt = 0;

	struct _finddata_t fd;
	intptr_t handle;
	if ((handle = _findfirst(searching_dir.c_str(), &fd)) == -1L)

		std::cout << "No file in directory!" << std::endl;

	do {
		//std::cout << fd.name << std::endl;
		return_.push_back(fd.name);
		cnt += 1;
	} while (_findnext(handle, &fd) == 0);

	_findclose(handle);

	return return_;
}


int main(void){
	std::string model_path(".\\model\\R50.onnx");;
	std::string base_path(".\\data\\");
	std::string output_dir(".\\result");

	std::string file_filter("png");

	std::vector<std::string> file_list = get_files_inDirectory(base_path, file_filter);
	//file_list.erase(file_list.begin(), file_list.begin() + 2);

	RetinaFace RetinaFace;
	RetinaFace.LoadModel(model_path);

	std::vector<Bbox> result;

	for (std::string file : file_list)
	{
		std::string image_path = base_path + file;
		cv::Mat img = cv::imread(image_path);

		if (img.empty())
		{
			std::cout << "Could not read the image: " << file << std::endl;
			continue;
		}
		
		result = RetinaFace.RunModel(img);


		//cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		char name[256];
		cv::Scalar color(255, 0, 0);
		sprintf_s(name, "%.2f", result[0].score);

		cv::putText(img, name, cv::Point(result[0].x1, result[0].y1), cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2);
		////cv::Rect box(rect.face_box.x - rect.face_box.w / 2, rect.face_box.y - rect.face_box.h / 2, rect.face_box.w, rect.face_box.h);
		cv::Rect box(result[0].x1, result[0].y1, result[0].x2 - result[0].x1, result[0].y2 - result[0].y1);
		cv::rectangle(img, box, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
		
		for (int k = 0; k < 5; k++) {
			cv::Point2f key_point = cv::Point2f(result[0].ppoint[2 * k], result[0].ppoint[2 * k + 1]);
			if (k % 3 == 0)
				cv::circle(img, key_point, 3, cv::Scalar(0, 255, 0), -1);
			else if (k % 3 == 1)
				cv::circle(img, key_point, 3, cv::Scalar(255, 0, 255), -1);
			else
				cv::circle(img, key_point, 3, cv::Scalar(0, 255, 255), -1);
		}
		
		cv::imshow("result", img);
		cv::waitKey(0);
		
		//std::string
		//cv::imwrite()

	}

	return 0;
}