#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <ctime>

inline void Draw_corners(cv::Mat* src, std::vector<cv::Point2i>* corners, char source_window[]);

struct coordinates {
	float x, y, z;
	coordinates(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
};

/*!���������, �������� ����� ������� � ���������� ��� �������� �����.
@uint len - ����� �������;
@cv::Point2i coords - ��������� �������� ���������� ����� {x,y};
$
#
len_coords() : len(0), coords(cv::Point2i()) {}
len_coords(uint lenght, cv::Point2i cordinates) : len(lenght), coords(cordinates) {}
len_coords(uint lenght) : len(lenght), coords(cv::Point2i()) {}
bool operator<(const len_coords& a) const{return (len < a.len);}*/
struct len_coords {
	uint len;
	coordinates coords;
	len_coords(uint lenght = 0, coordinates cords = coordinates()) : len(lenght), coords(cords) {}
	bool operator<(const len_coords& a) const
	{
		return (len < a.len);
	}

};

cv::Mat disparityMap(cv::Mat imgleft, cv::Mat imgright){
	int ndisparities = 80;
	int SADWindowSize = 7;
	cv::Mat imgDisparity16S = cv::Mat(imgleft.rows, imgleft.cols, CV_16S);
	cv::Mat imgDisparity8U = cv::Mat(imgleft.rows, imgleft.cols, CV_8UC1);
	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(ndisparities, SADWindowSize);
	sbm->compute(imgleft, imgright, imgDisparity16S);
	double minVal; double maxVal;

	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
	return imgDisparity8U;
}

std::vector<coordinates> reconstruction3D(cv::Mat dispMap, std::vector<cv::Point2i> ground_corners)
{
	double focus = 3.6;
	double base = 26;
	double pixelSize = 0.0041;
	std::vector<coordinates> coords;
	uint size = ground_corners.size();
	coords.resize(size);
	for (uint i = 0; i < size; ++i){
		if (dispMap.at<uchar>(ground_corners[i]) != 0) {
			float z = (focus*base / (pixelSize*dispMap.at<uchar>(ground_corners[i])));
			float y = (ground_corners[i].y - (double)dispMap.rows / 2) / (dispMap.at<uchar>(ground_corners[i]))*base;
			float x = (ground_corners[i].x - (double)dispMap.cols / 2) / (dispMap.at<uchar>(ground_corners[i]))*base;
			coords[i] = coordinates(x, y, z);
		}
		else coords[i] = coordinates(INFINITY, INFINITY, INFINITY);
	}
	return coords;
}

void startSubProcces(){
	system("python /home/pi/tmp5united/disp-1.py");
}


int main() {
	///startSubProcces();
	char sample_filename[] = "sample/13.jpg";
	char sample_window[] = "Image sample";
	char ground_filename[] = "ground/14.jpg";
	char ground_window[] = "Image ground";
	///��������� ��������� �����
	///--------------------------------------------------
	/*!������� ��� ��������� ���� � ����������.
	@
	$
	#*/
	int vector_len_eps = 0;
	/*!����� �������� ����.
	@
	$���� ���� ������ �� ���������������.
	#*/
	double qualityLevel = 0.01;
	/*!���������� ����� ������������� ������.
	@
	$���� ���������� ����� ������ ������ ����� ��������, ��������� ��� ����, ����� ������ ��������.
	#*/
	double minDistance = 10;
	/*!������� ����� ��� ���������� ����������� �������.
	@
	$
	#*/
	int blockSize = 3;
	/*!��������, �����������, ������� �� ������������ �������� �������.
	@
	$
	#*/
	bool useHarrisDetector = false;
	/*!�������� ��� ��������� �������.
	@
	$
	#���� ���� �������� ������� �� ������������, ������� ���������� ��� �������� �������-��������*/
	double k_harris = 0.04;
	///--------------------------------------------------

	/// ���������� ��� ������������� ����������
	///--------------------------------------------------
	/*!���������� ��� �������� �������� ��������� �� X ��� ���������� �����  �������.
	@
	$
	#���������� ��� ��������� ��������� ���������� � �����.*/
	float tmp_coords_x;
	/*!���������� ��� �������� �������� ��������� �� Y ��� ���������� �����  �������.
	@
	$
	#���������� ��� ��������� ��������� ���������� � �����.*/
	float tmp_coords_y;
	float tmp_coords_z;
	/*!���������� ��� �������� ���������� X ����� ������ �������.
	@
	$
	#���������� ��� ��������� ��������� ��������� � ������� ������� �������� � �����.*/
	float tmp_corner_x;
	/*!���������� ��� �������� ���������� Y ����� ������ �������.
	@
	$
	#���������� ��� ��������� ��������� ��������� � ������� ������� �������� � �����.*/
	float tmp_corner_y;
	float tmp_corner_z;
	/*!���������� ��� �������� ������ ������� �����,������� ����������� ������ � ��������.
	@
	$
	#*/
	uint number_of_max_ground_matches;
	/*!���������� ��� �������� ������ ������� �������, � ������� ������ ����� ������� �����.
	@
	$
	#*/
	uint number_of_max_sample_matches;
	/*!���-�� ���������� � ��������.
	@
	$
	#*/
	uint count_of_good_match = 0;
	/*!������������ ���-�� ���������� � ��������
	@
	$
	#*/
	uint max_count_of_good_match = 0;
	///--------------------------------------------------

	///��������� �������, ��������� ��������� �����  � ���������� ����
	///--------------------------------------------------
	/*!������� �������� ����������� ������� � ��������� ������.
	@
	$
	#������ ������ ������� �������� ���� char (0-255), 0 - ������ ����, 255-�����.*/
	cv::Mat sample_gray;
	sample_gray = cv::imread(sample_filename, 0);
	cv::Mat imgleft_sample = sample_gray(cv::Rect(0, 0, sample_gray.cols / 2, sample_gray.rows)),
		imgright_sample = sample_gray(cv::Rect(sample_gray.cols / 2, 0, sample_gray.cols / 2, sample_gray.rows));
	cv::Mat disparity_sample = disparityMap(imgleft_sample, imgright_sample);
	cv::imwrite("disp_sample.jpg", disparity_sample);
	/*!���-�� ����� ������� � �������.
	@
	$
	#cv::goodFeaturesToTrack ������� � ������ ������ ������� ������������ ���������� �����. 
	#����� � ����������, ���������� ���������� ������ ������� ��������� �������.*/
	uint count_of_sample_corners = 10000;
	/*!������, �������� ��������� cv::Point2i, � ������� ���������� ���������� x � y ����� �������.
	@
	$
	#������ �������� �� ����� ������� ���������, ��� ��� ������� cv::goodFeaturesToTrack ������� ������ ������.*/
	std::vector<cv::Point2i> sample_corners;
	/*!������ ��������, �������� ����� �������� �� ������ ������ ����� �� ��������� ������ �����.
	@
	$��� ������ ����������� ������ ����� �� ����������� ����� �� ���������� �����. 
	$��� ����� ������������ ��� ������ ������� �� �����, � ���� ���������� ����� �� ���� ������� �������� ����������, 
	$������ ��� ���������� �� ����������� � ��� ������ ������� ����� �� ���. 
	#*/
	uint **len_vectors_sample;
	cv::goodFeaturesToTrack(imgleft_sample, sample_corners, count_of_sample_corners, qualityLevel, minDistance, 
		cv::Mat(), blockSize, useHarrisDetector, k_harris);
	count_of_sample_corners = static_cast<uint>(sample_corners.size());
	std::vector<coordinates> Coordinates_sample = reconstruction3D(disparity_sample, sample_corners);
	len_vectors_sample = new uint *[count_of_sample_corners - 1];

	clock_t start = clock();
	for (uint j = 0; j < count_of_sample_corners - 1; ++j) {
		len_vectors_sample[j] = new uint[count_of_sample_corners - 1 - j];
		uint pos = 0;
		tmp_corner_x = Coordinates_sample[j].x;
		tmp_corner_y = Coordinates_sample[j].y;
		tmp_corner_z = Coordinates_sample[j].z;
		for (uint i = j + 1; i < count_of_sample_corners; ++i) {
			tmp_coords_x = Coordinates_sample[i].x - tmp_corner_x;
			tmp_coords_y = Coordinates_sample[i].y - tmp_corner_y;
			tmp_coords_z = Coordinates_sample[i].z - tmp_corner_z;
			len_vectors_sample[j][pos++] = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y + tmp_coords_z*tmp_coords_z;
		}
	}
	std::cout << "sample lenght: " << (clock() - start) << "ms" << std::endl;
	std::cout << "count of sample corners detected: " << sample_corners.size() << std::endl;
	///--------------------------------------------------

	///��������� �����, ��������� ��������� ����� � ���������� ����
	///--------------------------------------------------
	/*!������� �������� ����������� ����� � ��������� ������.
	@
	$
	#������ ������ ������� �������� ���� char (0-255), 0 - ������ ����, 255-�����.*/
	cv::Mat ground_gray;
	ground_gray = cv::imread(ground_filename, 0);
	cv::Mat imgleft_ground = ground_gray(cv::Rect(0, 0, ground_gray.cols / 2, ground_gray.rows)),
		imgright_ground = ground_gray(cv::Rect( ground_gray.cols / 2, 0, ground_gray.cols / 2, ground_gray.rows));
	cv::Mat disparity_ground = disparityMap(imgleft_ground, imgright_ground);
	cv::imwrite("disp_ground.jpg", disparity_ground);
	/*!���-�� ����� ����� � �������.
	@
	$
	#cv::goodFeaturesToTrack ������� � ������ ������ ������� ������������ ���������� �����. 
	#����� � ����������, ���������� ���������� ������ ������� ��������� �������.*/
	uint count_of_ground_corners = 10000;
	/*!������, �������� ��������� cv::Point2i, � ������� ���������� ���������� x � y ����� �������.
	@
	$
	#������ �������� �� ����� ������� ���������, ��� ��� ������� cv::goodFeaturesToTrack ������� ������ ������.*/
	std::vector<cv::Point2i> ground_corners;
	/*!������ �������� �������� , �������� ����� �������� �� ������ ������ ����� �� ��������� ������ �����,
	� ��� �� ���������� ����� ����� �������.
	@
	$
	#*/
	len_coords **len_vectors_ground;
	cv::goodFeaturesToTrack(imgleft_ground, ground_corners, count_of_ground_corners, qualityLevel, minDistance, 
		cv::Mat(), blockSize, useHarrisDetector, k_harris);
	count_of_ground_corners = static_cast<uint>(ground_corners.size());
	std::vector<coordinates> Coordinates_ground = reconstruction3D(disparity_ground, ground_corners);
	len_vectors_ground = new len_coords *[count_of_ground_corners];

	start = clock();
	for (uint j = 0; j < count_of_ground_corners; ++j) {
		len_vectors_ground[j] = new len_coords[count_of_ground_corners];
		tmp_corner_x = Coordinates_ground[j].x;
		tmp_corner_y = Coordinates_ground[j].y;
		tmp_corner_z = Coordinates_ground[j].z;
		for (uint i = 0; i < count_of_ground_corners; ++i) {
			tmp_coords_x = Coordinates_ground[i].x;
			tmp_coords_y = Coordinates_ground[i].y;
			tmp_coords_z = Coordinates_ground[i].z;
			len_vectors_ground[j][i].coords = coordinates(tmp_coords_x,tmp_coords_y,tmp_coords_z);
			tmp_coords_x -= tmp_corner_x;
			tmp_coords_y -= tmp_corner_y;
			tmp_coords_z -= tmp_corner_z;
			len_vectors_ground[j][i].len = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y + tmp_coords_z*tmp_coords_z;
		}
		std::sort(len_vectors_ground[j], len_vectors_ground[j] + count_of_ground_corners);
	}
	std::cout << "ground lenght: " << (clock() - start) << "ms" << std::endl;
	std::cout << "count of ground corners detected: " << ground_corners.size() << std::endl;
	///--------------------------------------------------

	///�������� ������
	///--------------------------------------------------
	start = clock();
	for (uint k = 0; k < count_of_sample_corners - 1; ++k) {
		for (uint j = 0; j < count_of_ground_corners; ++j) {
			for (uint i = 0; i < count_of_sample_corners - 1 - k; ++i) {
				if (std::binary_search(len_vectors_ground[j], len_vectors_ground[j] + count_of_ground_corners, 
					len_coords(len_vectors_sample[k][i]))) {
					++count_of_good_match;
				}
			}
			if (max_count_of_good_match < count_of_good_match) {
				max_count_of_good_match = count_of_good_match;
				number_of_max_ground_matches = j; number_of_max_sample_matches = k;
			}
			count_of_good_match = 0;
			if (max_count_of_good_match > (count_of_sample_corners - 1)*0.6) break;
		}
	}
	len_coords *low;
	for (uint i = 0; i < count_of_sample_corners - 1 - number_of_max_sample_matches; ++i) {
		low = std::lower_bound(len_vectors_ground[number_of_max_ground_matches], 
			len_vectors_ground[number_of_max_ground_matches] + count_of_ground_corners, 
			len_coords(len_vectors_sample[number_of_max_sample_matches][i]));
		if (low->len == len_vectors_sample[number_of_max_sample_matches][i]) {
			cv::circle(imgleft_ground, cv::Point((low->coords.x),(low->coords.y)), 10, cv::Scalar(255));
		}
	}
	std::cout << "find: " << (clock() - start) << "ms" << std::endl;
	std::cout << max_count_of_good_match << std::endl;
	///--------------------------------------------------

	Draw_corners(&imgleft_sample, &sample_corners, sample_window);
	Draw_corners(&imgleft_ground, &ground_corners, ground_window);
	cv::imshow(ground_window, imgleft_ground);
	cv::waitKey(0);

	for (uint i = 0; i < count_of_sample_corners - 1; ++i) delete[] len_vectors_sample[i];
	delete[] len_vectors_sample;
	for (uint i = 0; i < count_of_ground_corners; ++i) delete[] len_vectors_ground[i];
	delete[] len_vectors_ground;
	return(0);
}


void Draw_corners(cv::Mat* src, std::vector<cv::Point2i>* corners, char source_window[]) {

	uint size = static_cast<uint>(corners->size());

	std::cout << "count of corners detected: " << corners->size() << std::endl;
	int r = 4;
	for (uint i = 0; i < size; ++i)
	{
		cv::circle(*src, (*corners)[i], r, cv::Scalar(0, 255, 0), -1, 8, 0);
	}

	cv::namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	cv::imshow(source_window, *src);
}