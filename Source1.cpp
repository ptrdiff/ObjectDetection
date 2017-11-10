#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <ctime>

inline void Draw_corners(cv::Mat* src, std::vector<cv::Point2i>* corners, char source_window[]);

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
	cv::Point2i coords;
	len_coords(uint lenght = 0, cv::Point2i cords = cv::Point()) : len(lenght), coords(cords) {}
	bool operator<(const len_coords& a) const
	{
		return (len < a.len);
	}
};


int main_1() {
	char sample_filename[] = "sample/redbull.jpg";
	char sample_window[] = "Image sample";
	char ground_filename[] = "ground/find_redbull.jpg";
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
	int tmp_coords_x;
	/*!���������� ��� �������� �������� ��������� �� Y ��� ���������� �����  �������.
	@
	$
	#���������� ��� ��������� ��������� ���������� � �����.*/
	int tmp_coords_y;
	/*!���������� ��� �������� ���������� X ����� ������ �������.
	@
	$
	#���������� ��� ��������� ��������� ��������� � ������� ������� �������� � �����.*/
	int tmp_corner_x;
	/*!���������� ��� �������� ���������� Y ����� ������ �������.
	@
	$
	#���������� ��� ��������� ��������� ��������� � ������� ������� �������� � �����.*/
	int tmp_corner_y;
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
	if (sample_gray.empty()) return -1;
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
	cv::goodFeaturesToTrack(sample_gray, sample_corners, count_of_sample_corners, qualityLevel, minDistance,
		cv::Mat(), blockSize, useHarrisDetector, k_harris);
	count_of_sample_corners = static_cast<uint>(sample_corners.size());
	len_vectors_sample = new uint *[count_of_sample_corners - 1];

	clock_t start = clock();
	for (uint j = 0; j < count_of_sample_corners - 1; ++j) {
		len_vectors_sample[j] = new uint[count_of_sample_corners - 1 - j];
		uint pos = 0;
		tmp_corner_x = sample_corners[j].x;
		tmp_corner_y = sample_corners[j].y;
		for (uint i = j + 1; i < count_of_sample_corners; ++i) {
			tmp_coords_x = sample_corners[i].x - tmp_corner_x;
			tmp_coords_y = sample_corners[i].y - tmp_corner_y;
			len_vectors_sample[j][pos++] = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y;
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
	if (ground_gray.empty()) return -1;
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
	cv::goodFeaturesToTrack(ground_gray, ground_corners, count_of_ground_corners, qualityLevel, minDistance,
		cv::Mat(), blockSize, useHarrisDetector, k_harris);
	count_of_ground_corners = static_cast<uint>(ground_corners.size());
	len_vectors_ground = new len_coords *[count_of_ground_corners];

	start = clock();
	for (uint j = 0; j < count_of_ground_corners; ++j) {
		len_vectors_ground[j] = new len_coords[count_of_ground_corners];
		tmp_corner_x = ground_corners[j].x;
		tmp_corner_y = ground_corners[j].y;
		for (uint i = 0; i < count_of_ground_corners; ++i) {
			tmp_coords_x = ground_corners[i].x;
			tmp_coords_y = ground_corners[i].y;
			len_vectors_ground[j][i].coords.x = tmp_coords_x;
			len_vectors_ground[j][i].coords.y = tmp_coords_y;
			tmp_coords_x -= tmp_corner_x;
			tmp_coords_y -= tmp_corner_y;
			len_vectors_ground[j][i].len = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y;
		}
		std::sort(len_vectors_ground[j], len_vectors_ground[j] + count_of_ground_corners);
	}
	std::cout << "ground lenght: " << (clock() - start) << "ms" << std::endl;
	std::cout << "count of ground corners detected: " << ground_corners.size() << std::endl;
	///--------------------------------------------------

	///�������� ������
	///--------------------------------------------------
	start = clock();
	for (uint d = 0; d < 283; ++d) {
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
				///if (max_count_of_good_match > (count_of_sample_corners - 1)*0.6) break;
			}
		}
		len_coords *low;
		for (uint i = 0; i < count_of_sample_corners - 1 - number_of_max_sample_matches; ++i) {
			low = std::lower_bound(len_vectors_ground[number_of_max_ground_matches],
				len_vectors_ground[number_of_max_ground_matches] + count_of_ground_corners,
				len_coords(len_vectors_sample[number_of_max_sample_matches][i]));
			if (low->len == len_vectors_sample[number_of_max_sample_matches][i]) {
				cv::circle(ground_gray, low->coords, 10, cv::Scalar(255));
			}
		}
	}
	std::cout << "find: " << (clock() - start) << "ms" << std::endl;
	std::cout << max_count_of_good_match << std::endl;
	///--------------------------------------------------

	Draw_corners(&sample_gray, &sample_corners, sample_window);
	Draw_corners(&ground_gray, &ground_corners, ground_window);
	cv::imshow(ground_window, ground_gray);
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