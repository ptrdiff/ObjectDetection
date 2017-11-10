#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\calib3d.hpp>
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <ctime>

/*!структура хранящая 3D - координаты точек в пространстве
@x,y,z  - координаты точек
$
#
coordinates(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
*/
struct coordinates {
	float x, y, z;
	int x_2D, y_2D;
	coordinates(float a = 0, float b = 0, float c = 0, int d = 0, int e = 0) : x(a), y(b), z(c), x_2D(d), y_2D(e) {}
};

/*!функция построения карты глубины, возвращает её
@ndisparities - максимальное смещение
@SADWindowSize - размер сканирующего окна
@cv::Mat imgleft, cv::Mat imgright -левое и правое изображения
$
#
*/
cv::Mat disparityMap(cv::Mat imgleft, cv::Mat imgright) {
	int ndisparities = 80;
	int SADWindowSize = 11;
	cv::Mat imgDisparity16S = cv::Mat(imgleft.rows, imgleft.cols, CV_16S);
	cv::Mat imgDisparity8U = cv::Mat(imgleft.rows, imgleft.cols, CV_8UC1);
	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(ndisparities, SADWindowSize);

	///-----------------------------------------------------------------------------------------------------------------------
	///создание карты глубины
	sbm->compute(imgleft, imgright, imgDisparity16S);
	double minVal; double maxVal;

	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
	///-----------------------------------------------------------------------------------------------------------------------

	///------------------------------------------------------------------------------------------------------------------------
	///удаление максимумов на disparityMap то есть точек чьи координаты ближе 15 сантиметров
	for (int i = 1; i<imgDisparity8U.rows - 1; i++)
		for (int j = 1; j < imgDisparity8U.cols - 1; j++)
		{
			imgDisparity8U.at<uchar>(i, j) = (imgDisparity8U.at<uchar>(i, j) > 150) ? 0 : imgDisparity8U.at<uchar>(i, j);
		}
	///------------------------------------------------------------------------------------------------------------------------
	return imgDisparity8U;
}

/*!функция востановления координат в пространстве
@std::vector<cv::Point2i> ground_corners - массив пиксельных координат относительно вернего левого угла
@cv::Mat dispMap - карты глубины
$
#
*/
std::vector<coordinates> reconstruction3D(cv::Mat &dispMap, std::vector<cv::Point2f> &ground_corners)
{
	double focus = 3.6;
	double base = 25;
	double pixelSize = 1 / 375.2;
	std::vector<coordinates> coords;
	uint size = ground_corners.size();
	for (uint i = 0; i < size; ++i) {
		if (dispMap.at<uchar>(ground_corners[i]) != 0) {
			float z = (focus*base / (pixelSize*dispMap.at<uchar>(ground_corners[i])));
			float y = (ground_corners[i].y - (double)dispMap.rows / 2) / (dispMap.at<uchar>(ground_corners[i]))*base;
			float x = (ground_corners[i].x - (double)dispMap.cols / 2) / (dispMap.at<uchar>(ground_corners[i]))*base;
			coords.push_back(coordinates(x, y, z, ground_corners[i].x, ground_corners[i].y));
		}
	}
	return coords;
}

class Scene {
	class Example {
		std::vector<cv::Point2f> example_keypoints;
		uint example_count_of_keypoints;
		double **example_len_vectors;
	public:
		cv::Mat example_image_left, example_image_right, example_image, example_image_disparity;
		Example(cv::Mat image = cv::Mat()) : example_image(image), example_count_of_keypoints(0), example_len_vectors(nullptr) {
			if (!example_image.empty()) {
				example_image_left = example_image(cv::Rect(0, 0, example_image.cols / 2, example_image.rows));
				example_image_right = example_image(cv::Rect(example_image.cols / 2, 0, example_image.cols / 2, example_image.rows));
				example_image_disparity = disparityMap(example_image_left, example_image_right);
			}
		}
		~Example() {
			if (example_len_vectors) {
				for (uint i = 0; i < example_count_of_keypoints - 1; ++i) {
					delete[] example_len_vectors[i];
					example_len_vectors[i] = nullptr;
				}
				delete[] example_len_vectors;
				example_len_vectors = nullptr;
				example_count_of_keypoints = 0;
			}
		}
		void Compute(int maxCorners, double qualityLevel, double minDistance) {
			cv::goodFeaturesToTrack(example_image_left, example_keypoints, maxCorners, qualityLevel, minDistance);
			std::vector<coordinates> Coordinates_sample = reconstruction3D(example_image_disparity, example_keypoints);
			example_count_of_keypoints = static_cast<uint>(Coordinates_sample.size());
			if (example_count_of_keypoints == 0) return;
			float tmp_coords_x;
			float tmp_coords_y;
			float tmp_coords_z;
			float tmp_corner_x;
			float tmp_corner_y;
			float tmp_corner_z;
			example_len_vectors = new double *[example_count_of_keypoints - 1];
			for (uint j = 0; j < example_count_of_keypoints - 1; ++j) {
				example_len_vectors[j] = new double [example_count_of_keypoints - 1 - j];
				uint pos = 0;
				tmp_corner_x = Coordinates_sample[j].x;
				tmp_corner_y = Coordinates_sample[j].y;
				tmp_corner_z = Coordinates_sample[j].z;
				for (uint i = j + 1; i < example_count_of_keypoints; ++i) {
					tmp_coords_x = Coordinates_sample[i].x - tmp_corner_x;
					tmp_coords_y = Coordinates_sample[i].y - tmp_corner_y;
					tmp_coords_z = Coordinates_sample[i].z - tmp_corner_z;
					example_len_vectors[j][pos++] = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y + tmp_coords_z*tmp_coords_z;
				}
			}
		}
		uint KeypointsCount() { return example_count_of_keypoints; }
		double **LenVectors() { return example_len_vectors; }
	};
	struct coords_3D_2D {
		cv::Point3f coords_3D;
		cv::Point2i coords_2D;
		coords_3D_2D(cv::Point3f cords_3D = cv::Point3f(), cv::Point2i cords_2D = cv::Point2i()) :
			coords_3D(cords_3D), coords_2D(cords_2D) {}
	};
	struct len_coords {
		double len;
		coords_3D_2D coords;
		bool find_check_flag;
		bool check_flag;
		len_coords(double lenght = 0, coords_3D_2D cords = coords_3D_2D()) : len(lenght), coords(cords), find_check_flag(0), check_flag(0) {}
		bool operator<(const len_coords& a) const
		{
			return (len < a.len);
		}
	};
	class Scope {
		std::vector<cv::Point2f> scope_keypoints;
		uint scope_count_of_keypoints;
		len_coords **scope_len_vectors;
	public:
		cv::Mat scope_image_left, scope_image_right, scope_image, scope_image_disparity;
		Scope(cv::Mat image = cv::Mat()) : scope_image(image), scope_count_of_keypoints(0), scope_len_vectors(nullptr) {
			if (!scope_image.empty()) {
				scope_image_left = scope_image(cv::Rect(0, 0, scope_image.cols / 2, scope_image.rows));
				scope_image_right = scope_image(cv::Rect(scope_image.cols / 2, 0, scope_image.cols / 2, scope_image.rows));
				scope_image_disparity = disparityMap(scope_image_left, scope_image_right);
			}
		}
		~Scope() {
			if (scope_len_vectors) {
				for (uint i = 0; i < scope_count_of_keypoints; ++i) {
					delete[] scope_len_vectors[i];
					scope_len_vectors[i] = nullptr;
				}
				delete[] scope_len_vectors;
				scope_len_vectors = nullptr;
				scope_count_of_keypoints = 0;
			}
		}
		void Compute(int maxCorners, double qualityLevel, double minDistance) {
			cv::goodFeaturesToTrack(scope_image_left, scope_keypoints, maxCorners, qualityLevel, minDistance);
			std::vector<coordinates> Coordinates_groind = reconstruction3D(scope_image_disparity, scope_keypoints);
			scope_count_of_keypoints = static_cast<uint>(Coordinates_groind.size());
			if (scope_count_of_keypoints == 0) return;
			float tmp_coords_x;
			float tmp_coords_y;
			float tmp_coords_z;
			float tmp_corner_x;
			float tmp_corner_y;
			float tmp_corner_z;
			scope_len_vectors = new len_coords *[scope_count_of_keypoints];
			clock_t start = clock();
			for (uint j = 0; j < scope_count_of_keypoints; ++j) {
				scope_len_vectors[j] = new len_coords[scope_count_of_keypoints];
				tmp_corner_x = Coordinates_groind[j].x;
				tmp_corner_y = Coordinates_groind[j].y;
				tmp_corner_z = Coordinates_groind[j].z;
				for (uint i = 0; i < scope_count_of_keypoints; ++i) {
					tmp_coords_x = Coordinates_groind[i].x;
					tmp_coords_y = Coordinates_groind[i].y;
					tmp_coords_z = Coordinates_groind[i].z;
					scope_len_vectors[j][i].coords.coords_3D.x = tmp_coords_x;
					scope_len_vectors[j][i].coords.coords_3D.y = tmp_coords_y;
					scope_len_vectors[j][i].coords.coords_3D.z = tmp_coords_z;
					tmp_coords_x -= tmp_corner_x;
					tmp_coords_y -= tmp_corner_y;
					tmp_coords_z -= tmp_corner_z;
					scope_len_vectors[j][i].len = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y + tmp_coords_z*tmp_coords_z;
					scope_len_vectors[j][i].coords.coords_2D.x = Coordinates_groind[i].x_2D;
					scope_len_vectors[j][i].coords.coords_2D.y = Coordinates_groind[i].y_2D;
				}
				std::stable_sort(scope_len_vectors[j], scope_len_vectors[j] + scope_count_of_keypoints);
			}
			std::cout << "ground processing: " << clock() - start << "ms" << std::endl;
			std::cout << "count of ground corners detected: " << scope_count_of_keypoints << std::endl;
		}
		uint KeypointsCount() { return scope_count_of_keypoints; }
		len_coords **LenVectors() { return scope_len_vectors; }
	};
	const uint N;
	Example *examples;
	Scope ground;
	std::list<coords_3D_2D> best_object_coords;
	double procent_of_coincidence;
public:
	Scene(const uint n, const char names[][128], int maxCorners = 10000, double qualityLevel = 0.01, double minDistance = 10)
		: N(n), examples(nullptr), ground(Scope()), procent_of_coincidence(0) {
		examples = new Example[N];
		for (uint i = 0; i < N; ++i) {
			examples[i] = Example(cv::imread(names[i], 0));
			examples[i].Compute(maxCorners, qualityLevel, minDistance);
		}
	}
	~Scene() {
		if (examples) {
			delete[] examples;
			examples = nullptr;
		}
	}
	void SetScope(cv::Mat scope_names, int maxCorners = 10000, double qualityLevel = 0.01, double minDistance = 10) {
		ground.~Scope();
		best_object_coords.clear();
		procent_of_coincidence = 0.0;
		ground = Scope(scope_names);
		ground.Compute(maxCorners, qualityLevel, minDistance);
	}
	void Find(Example &sample, double quality, uint count_for_binsearch, double eps) {
		uint number_of_max_ground_matches = 0;
		uint number_of_max_sample_matches = 0;
		uint count_of_good_match = 0;
		uint max_count_of_good_match = 0;
		uint count_of_image_sample_corners = sample.KeypointsCount();
		uint count_of_image_ground_corners = ground.KeypointsCount();
		if (count_of_image_sample_corners == 0 || count_of_image_ground_corners == 0) return;
		uint count_of_image_sample_corners_for_binsearch = cv::min(count_of_image_sample_corners - 1, count_for_binsearch);
		double **len_vectors_image_sample = sample.LenVectors();
		len_coords **len_vectors_image_ground = ground.LenVectors();
		len_coords tmp_len_low, tmp_len_up, *lower, *upper;

		for (uint k = 0; k < count_of_image_sample_corners_for_binsearch; ++k) {
			for (uint j = 0; j < count_of_image_ground_corners; ++j) {
				for (uint i = 0; i < count_of_image_sample_corners_for_binsearch - k; ++i) {
					tmp_len_low = len_coords(len_vectors_image_sample[k][i] - eps);
					tmp_len_up = len_coords(len_vectors_image_sample[k][i] + eps);
					lower = std::lower_bound(len_vectors_image_ground[j], len_vectors_image_ground[j] + count_of_image_ground_corners,
						tmp_len_low);
					upper = std::upper_bound(len_vectors_image_ground[j], len_vectors_image_ground[j] + count_of_image_ground_corners,
						tmp_len_up);
					for (auto it = lower; it != upper; ++it) {
						if (it->find_check_flag == 0) {
							++count_of_good_match;
							it->find_check_flag = 1;
						}
					}
				}
				if (max_count_of_good_match < count_of_good_match) {
					max_count_of_good_match = count_of_good_match;
					number_of_max_ground_matches = j; number_of_max_sample_matches = k;
				}
				count_of_good_match = 0;
				if (max_count_of_good_match > (count_of_image_sample_corners - 1)*quality) break;
			}
		}

		if (max_count_of_good_match == 0) return;

		max_count_of_good_match = 0;
		std::list<coords_3D_2D> object_coords;

		for (uint i = 0; i < count_of_image_sample_corners - 1 - number_of_max_sample_matches; ++i) {
			tmp_len_low = len_coords(len_vectors_image_sample[number_of_max_sample_matches][i] - eps);
			tmp_len_up = len_coords(len_vectors_image_sample[number_of_max_sample_matches][i] + eps);
			lower = std::lower_bound(len_vectors_image_ground[number_of_max_ground_matches],
				len_vectors_image_ground[number_of_max_ground_matches] + count_of_image_ground_corners, tmp_len_low);
			upper = std::upper_bound(len_vectors_image_ground[number_of_max_ground_matches],
				len_vectors_image_ground[number_of_max_ground_matches] + count_of_image_ground_corners, tmp_len_up);
			for (auto it = lower; it != upper; ++it) {
				if (it->check_flag == 0) {
					object_coords.push_back(it->coords);
					it->check_flag = 1;
					++max_count_of_good_match;
				}
			}
		}
		double tmp_procent_of_coincidence = static_cast<double>(max_count_of_good_match) / (count_of_image_sample_corners - 1);
		if (procent_of_coincidence < tmp_procent_of_coincidence) {
			procent_of_coincidence = tmp_procent_of_coincidence;
			best_object_coords = object_coords;
		}

	}
	void ShowBestSearch() {
		if (!best_object_coords.empty()) {
			cv::Mat last_search = ground.scope_image_left.clone();
			std::cout << best_object_coords.size() << " count" << std::endl;
			for (std::list<coords_3D_2D>::iterator it = best_object_coords.begin(); it != best_object_coords.end(); ++it) {
				cv::circle(last_search, cv::Point((it->coords_2D.x),(it->coords_2D.y)), 10, cv::Scalar(255));
				cv::putText(last_search, std::to_string(static_cast<int>(it->coords_3D.z)), cv::Point((it->coords_2D.x), (it->coords_2D.y)), cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255));
			}
			cv::imshow("match", last_search);
			std::cout << procent_of_coincidence * 100 << '%' << std::endl;
			std::cout << "----------------------------------------" << std::endl;
			cv::waitKey(0);
		}
		else {
			std::cout << "Not found!" << std::endl;
		}
	}
	void Detect(double quality = 0.5, uint count_for_binsearch = 4, double eps = 0.0) {
		clock_t start = clock();
		for (uint i = 0; i < N; ++i) {
			Find(examples[i],quality,count_for_binsearch, eps );
		}
		std::cout << "Finding: " << clock() - start << "ms" << std::endl;
	}

};

int main() {
	const char templates_names[][128] = {
		{"sample/im0.jpg" },
		{ "sample/im0 (1).jpg" },
		{ "sample/im0 (2).jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/im0 (5).jpg" },
		{ "sample/im0 (6).jpg" },
		{ "sample/im0 (7).jpg" },
		{ "sample/im0 (8).jpg" },
		{ "sample/im0 (9).jpg" },
		{ "sample/im0 (10).jpg" },
		{ "sample/im0 (11).jpg" },
		{ "sample/im0 (12).jpg" },
		{ "sample/im0 (13).jpg" },
		{ "sample/im0 (14).jpg" },
		{ "sample/im0 (15).jpg" },
		{ "sample/im0 (16).jpg" },
		{ "sample/im0 (17).jpg" },
		{ "sample/im0 (18).jpg" },
		{ "sample/im0 (19).jpg" },
		{ "sample/im0 (20).jpg" },
		{ "sample/im0 (21).jpg" },
		{ "sample/im0 (22).jpg" },
		{ "sample/im0 (23).jpg" },
		{ "sample/im0 (24).jpg" },
		{ "sample/im0 (25).jpg" },
		{ "sample/im0 (26).jpg" },
		{ "sample/im0 (27).jpg" },
		{ "sample/im0 (28).jpg" },
		{ "sample/im0 (29).jpg" },
		{ "sample/im0 (30).jpg" },
		{ "sample/im0 (31).jpg" },
		{ "sample/im0 (32).jpg" }
	};

	Scene my_scene(33, templates_names);

	for (uint i = 0; i < 1; ++i) {
		my_scene.SetScope(cv::imread(templates_names[i],0));
		my_scene.Detect();
		my_scene.ShowBestSearch();
	}

	system("pause");
	return 0;
}