#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <ctime>

class Scene {
	class Example {
		std::vector<cv::Point2i> example_keypoints;
		uint example_count_of_keypoints;
		int **example_len_vectors;
	public:
		cv::Mat example_image;
		Example(cv::Mat image = cv::Mat()) : example_image(image), example_count_of_keypoints(0), example_len_vectors(nullptr) {}
		~Example() {
			if (example_len_vectors) {
				for (size_t i = 0; i < static_cast<size_t>(example_count_of_keypoints - 1); ++i) {
					delete[] example_len_vectors[i];
					example_len_vectors[i] = nullptr;
				}
				delete[] example_len_vectors;
				example_len_vectors = nullptr;
				example_count_of_keypoints = 0;
			}
		}
		void Compute(int maxCorners, double qualityLevel, double minDistance) {
			cv::goodFeaturesToTrack(example_image, example_keypoints, maxCorners, qualityLevel, minDistance);
			example_count_of_keypoints = static_cast<uint>(example_keypoints.size());
			if (example_count_of_keypoints == 0) return;
			int tmp_coords_x;
			int tmp_coords_y;
			int tmp_corner_x;
			int tmp_corner_y;
			example_len_vectors = new int *[example_count_of_keypoints - 1];
			for (uint j = 0; j < example_count_of_keypoints - 1; ++j) {
				example_len_vectors[j] = new int [example_count_of_keypoints - 1 - j];
				uint pos = 0;
				tmp_corner_x = example_keypoints[j].x;
				tmp_corner_y = example_keypoints[j].y;
				for (uint i = j + 1; i < example_count_of_keypoints; ++i) {
					tmp_coords_x = example_keypoints[i].x - tmp_corner_x;
					tmp_coords_y = example_keypoints[i].y - tmp_corner_y;
					example_len_vectors[j][pos++] = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y;
				}
			}
		}
		uint KeypointsCount() { return example_count_of_keypoints; }
		int **LenVectors() { return example_len_vectors; }
	};
	struct len_coords {
		int len;
		cv::Point2i coords;
		bool find_check_flag;
		bool check_flag;
		len_coords(int lenght = 0, cv::Point2i cords = cv::Point2i()) : len(lenght), coords(cords), find_check_flag(0), check_flag(0) {}
		bool operator<(const len_coords& a) const
		{
			return (len < a.len);
		}
	};
	class Scope {
		std::vector<cv::Point2i> scope_keypoints;
		uint scope_count_of_keypoints;
		len_coords **scope_len_vectors;
	public:
		cv::Mat scope_image;
		Scope(cv::Mat image = cv::Mat()) : scope_image(image), scope_count_of_keypoints(0), scope_len_vectors(nullptr) {}
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
			cv::goodFeaturesToTrack(scope_image, scope_keypoints, maxCorners, qualityLevel, minDistance);
			scope_count_of_keypoints = static_cast<uint>(scope_keypoints.size());
			if (scope_count_of_keypoints == 0) return;
			int tmp_coords_x;
			int tmp_coords_y;
			int tmp_corner_x;
			int tmp_corner_y;
			scope_len_vectors = new len_coords *[scope_count_of_keypoints];
			clock_t start = clock();
			for (uint j = 0; j < scope_count_of_keypoints; ++j) {
				scope_len_vectors[j] = new len_coords[scope_count_of_keypoints];
				tmp_corner_x = scope_keypoints[j].x;
				tmp_corner_y = scope_keypoints[j].y;
				for (uint i = 0; i < scope_count_of_keypoints; ++i) {
					tmp_coords_x = scope_keypoints[i].x;
					tmp_coords_y = scope_keypoints[i].y;
					scope_len_vectors[j][i].coords.x = tmp_coords_x;
					scope_len_vectors[j][i].coords.y = tmp_coords_y;
					tmp_coords_x -= tmp_corner_x;
					tmp_coords_y -= tmp_corner_y;
					scope_len_vectors[j][i].len = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y;
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
	std::list<cv::Point2i> best_object_coords;
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
	void Find(Example &sample, double quality, uint count_for_binsearch, int eps) {
		uint number_of_max_ground_matches = 0;
		uint number_of_max_sample_matches = 0;
		uint count_of_good_match = 0;
		uint max_count_of_good_match = 0;
		uint count_of_image_sample_corners = sample.KeypointsCount();
		uint count_of_image_ground_corners = ground.KeypointsCount();
		if (count_of_image_sample_corners == 0 || count_of_image_ground_corners == 0) return;
		uint count_of_image_sample_corners_for_binsearch = cv::min(count_of_image_sample_corners - 1, count_for_binsearch);
		int **len_vectors_image_sample = sample.LenVectors();
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
		std::list<cv::Point2i> object_coords;

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
		double tmp_procent_of_coincidence = static_cast<double>(max_count_of_good_match) / static_cast<double>(count_of_image_sample_corners - 1);
		if (procent_of_coincidence < tmp_procent_of_coincidence) {
			procent_of_coincidence = tmp_procent_of_coincidence;
			best_object_coords = object_coords;
		}

	}
	void ShowBestSearch() {
		if (!best_object_coords.empty()) {
			cv::Mat last_search = ground.scope_image.clone();
			std::cout << best_object_coords.size() << " count" << std::endl;
			for (std::list<cv::Point2i>::iterator it = best_object_coords.begin(); it != best_object_coords.end(); ++it) {
				cv::circle(last_search, *it, 20, cv::Scalar(255));
				cv::putText(last_search, std::to_string(it->x) + ' ' + std::to_string(it->y), *it, cv::FONT_HERSHEY_PLAIN,0.5,cv::Scalar(255));
			}
			cv::resize(last_search, last_search, cv::Size(), 0.5, 0.5);
			cv::imshow("match", last_search);
			std::cout << procent_of_coincidence * 100 << '%' << std::endl;
			std::cout << "----------------------------------------" << std::endl;
			cv::waitKey(0);
		}
		else {
			std::cout << "Not found!" << std::endl;
		}
	}
	void Detect(double quality = 0.5, uint count_for_binsearch = 4, int eps = 0.0) {
		clock_t start = clock();
		for (uint i = 0; i < N; ++i) {
			Find(examples[i],quality,count_for_binsearch, eps );
		}
		std::cout << "Finding: " << clock() - start << "ms" << std::endl;
	}

};

int main() {
	cv::VideoCapture cap,cap1;
	cv::Mat frame;
	cap.open(0);
	cap1.open(-1);

	for (int i = 0; i <= 39; ++i) { 
		std::cout << cap.get(i) << ' ';
		std::cout << cap1.get(i) << std::endl;
	}


	while (1) {
		if (!cap.read(frame)) return -2;
		imshow("W", frame);
		cv::waitKey(33);
	}
	return 0;
}