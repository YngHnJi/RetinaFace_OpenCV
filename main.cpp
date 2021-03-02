#include <iostream>
#include <string>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "RetinaFace.h"
#include <io.h>
#include <time.h>

#define FEATURE_VECTOR_SIZE 15

std::vector<std::string> get_files_inDirectory(const std::string& _path, const std::string& _filter){
	
	std::string searching_dir = _path + "*.*";
	
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

void CreateRotatedDualImg(cv::Mat& orig_img, cv::Mat& dbl_img)
{
	cv::Mat rot_orig;
	cv::rotate(orig_img, rot_orig, cv::ROTATE_180);
	if (orig_img.cols > orig_img.rows) {
		dbl_img.create(orig_img.rows * 2, orig_img.cols, orig_img.type());
		orig_img.copyTo(dbl_img(cv::Rect(0, 0, orig_img.cols, orig_img.rows)));
		rot_orig.copyTo(dbl_img(cv::Rect(0, orig_img.rows, orig_img.cols, orig_img.rows)));
	}
	else {
		dbl_img.create(orig_img.rows, orig_img.cols * 2, orig_img.type());
		orig_img.copyTo(dbl_img(cv::Rect(0, 0, orig_img.cols, orig_img.rows)));
		rot_orig.copyTo(dbl_img(cv::Rect(orig_img.cols, 0, orig_img.cols, orig_img.rows)));
	}
}

void CreateResizeImg(cv::Mat& orig_img, cv::Mat& dbl_img)
{
	//cv::Mat rot_orig;
	//cv::rotate(orig_img, rot_orig, cv::ROTATE_180);
	if (orig_img.cols > orig_img.rows) {
		dbl_img.create(orig_img.rows * 2, orig_img.cols, orig_img.type());
		orig_img.copyTo(dbl_img(cv::Rect(0, 0, orig_img.cols, orig_img.rows)));
		//rot_orig.copyTo(dbl_img(cv::Rect(0, orig_img.rows, orig_img.cols, orig_img.rows)));
	}
	else {
		dbl_img.create(orig_img.rows, orig_img.cols * 2, orig_img.type());
		orig_img.copyTo(dbl_img(cv::Rect(0, 0, orig_img.cols, orig_img.rows)));
		//rot_orig.copyTo(dbl_img(cv::Rect(orig_img.cols, 0, orig_img.cols, orig_img.rows)));
	}
}


int main(void) // NO Dual Mode
{
	int cnt_true = 0;
	int cnt_false = 0;
	int cnt_zero = 0;

	std::cout << "OpenCV version : " << CV_VERSION << std::endl;
	
	std::string base_path{"./temp/"};
	std::string output_dir("./result");
	//std::string file_filter(".ppm");

	
	// std::ofstream file2write;
	// file2write.open("./result/result.txt");
	// if (!file2write.is_open()) {
	// 	std::cout << "file not opened" << std::endl;

	// 	return 0;
	// }
	

	std::vector<std::string> file_list = get_files_inDirectory(base_path, file_filter);
	file_list.erase(file_list.begin(), file_list.begin() + 2);
	
	//std::cout << "Debugging" << std::endl;

	RetinaFace RetinaFace;
	RetinaFace.LoadModel();

	std::vector<Bbox> result;

	//for (const std::string file : file_list)
	for (std::string file : file_list)
	{
		std::string image_path = base_path + file;
		cv::Mat img = cv::imread(image_path);

		if (img.empty())
		{
			std::cout << "Could not read the image: " << file << std::endl;
			continue;
		}
		//std::cout << "===> Data loaded" << std::endl;
		std::cout << "Processing image: " << file << " -->";
		//std::cout << "  rows: " << img.rows << "  cols: " << img.cols << std::endl;


		//std::cout << "Model Loaded" << std::endl;

		result = RetinaFace.RunModel(img);

		//file2write << file << " " << std::to_string(result[0].score) << "\n";

		std::cout << " " << result[0].score << ": ";

		std::cout << "\n"; // debug purpose YH JI
		//std::cout << "Debug" << std::endl;
		

		///////////////////// color the dots in 5 landmark coordinates
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		char name[256];
		cv::Scalar color(255, 0, 0);
		sprintf_s(name, "%.2f", result[0].score);

		cv::putText(img, name, cv::Point(result[0].x1, result[0].y1), cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2);
		////cv::Rect box(rect.face_box.x - rect.face_box.w / 2, rect.face_box.y - rect.face_box.h / 2, rect.face_box.w, rect.face_box.h);
		cv::Rect box(result[0].x1, result[0].y1, result[0].x2 - result[0].x1, result[0].y2 - result[0].y1);
		/cv::rectangle(img, box, color, 2, cv::LINE_8, 0);
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
		/////////////////// color the dots in 5 landmark coordinates
		cv::imshow("result", img);
		cv::waitKey(0);
	

		std::string output_file = output_dir + "result_" + file;
		cv::imwrite(output_file, img);
		//file2write << file << " " << result[0].score << "\n";
	}
	//file2write.close();


	return 0;
}