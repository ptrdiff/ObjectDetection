#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <ctime>

class Scene {
	class Example {
		std::vector<cv::Point2i> example_keypoints;
		std::vector<std::vector<int>> example_len_vectors;
	public:
		cv::Mat example_image;
		Example(cv::Mat image = cv::Mat()) : example_image(image) {}
		~Example() {}
		void Compute(int maxCorners, double qualityLevel, double minDistance) {
			cv::goodFeaturesToTrack(example_image, example_keypoints, maxCorners, qualityLevel, minDistance);
			if (example_keypoints.size() == 0) return;
			example_len_vectors.resize(example_keypoints.size() - 1);
			for (uint j = 0; j < example_keypoints.size() - 1 ; ++j) {
				example_len_vectors[j].resize(example_keypoints.size() - 1 - j);
				uint pos = 0;
				int tmp_corner_x = example_keypoints[j].x;
				int tmp_corner_y = example_keypoints[j].y;
				for (uint i = j + 1; i < example_keypoints.size(); ++i) {
					int tmp_coords_x = example_keypoints[i].x - tmp_corner_x;
					int tmp_coords_y = example_keypoints[i].y - tmp_corner_y;
					example_len_vectors[j][pos++] = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y;
				}
			}
		}
		std::vector<std::vector<int>>& LenVectors() { return example_len_vectors; }
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
		std::vector<std::vector<len_coords>> scope_len_vectors;
	public:
		cv::Mat scope_image;
		Scope(cv::Mat image = cv::Mat()) : scope_image(image){}
		~Scope() {}
		void Compute(int maxCorners, double qualityLevel, double minDistance) {
			cv::goodFeaturesToTrack(scope_image, scope_keypoints, maxCorners, qualityLevel, minDistance);
			if (scope_keypoints.size() == 0) return;
			scope_len_vectors.resize(scope_keypoints.size());
			clock_t start = clock();
			for (uint j = 0; j < scope_keypoints.size(); ++j) {
				scope_len_vectors[j].resize(scope_keypoints.size());
				int tmp_corner_x = scope_keypoints[j].x;
				int tmp_corner_y = scope_keypoints[j].y;
				for (uint i = 0; i < scope_keypoints.size(); ++i) {
					int tmp_coords_x = scope_keypoints[i].x;
					int tmp_coords_y = scope_keypoints[i].y;
					scope_len_vectors[j][i].coords.x = tmp_coords_x;
					scope_len_vectors[j][i].coords.y = tmp_coords_y;
					tmp_coords_x -= tmp_corner_x;
					tmp_coords_y -= tmp_corner_y;
					scope_len_vectors[j][i].len = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y;
				}
				std::stable_sort(scope_len_vectors[j].begin(), scope_len_vectors[j].end());
			}
			std::cout << "ground processing: " << clock() - start << "ms" << std::endl;
			std::cout << "count of ground corners detected: " << scope_keypoints.size() << std::endl;
		}
		std::vector<std::vector<len_coords>>& LenVectors() { return scope_len_vectors; }
	};
	std::vector<Example> examples;
	Scope ground;
	std::list<cv::Point2i> best_object_coords;
	double procent_of_coincidence;
public:
	Scene(const uint n, const char names[][128], int maxCorners = 10000, double qualityLevel = 0.01, double minDistance = 10)
		: ground(Scope()), procent_of_coincidence(0.0) {
		examples.resize(n);
		for (uint i = 0; i < n; ++i) {
			examples[i] = Example(cv::imread(names[i], 0));
			examples[i].Compute(maxCorners, qualityLevel, minDistance);
		}
	}
	~Scene() {}
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
		std::vector<std::vector<int>> len_vectors_image_sample = sample.LenVectors();
		std::vector<std::vector<len_coords>> len_vectors_image_ground = ground.LenVectors();
		if (len_vectors_image_sample.size() == 0 || len_vectors_image_ground.size() == 0) return;
		uint count_of_image_sample_corners_for_binsearch = static_cast<uint>(cv::min(len_vectors_image_sample.size(), static_cast<size_t>(count_for_binsearch)));

		len_coords tmp_len_low, tmp_len_up;

		for (uint k = 0; k < count_of_image_sample_corners_for_binsearch; ++k) {
			for (uint j = 0; j < len_vectors_image_ground.size(); ++j) {
				for (uint i = 0; i < count_of_image_sample_corners_for_binsearch - k; ++i) {
					tmp_len_low = len_coords(len_vectors_image_sample[k][i] - eps);
					tmp_len_up = len_coords(len_vectors_image_sample[k][i] + eps);
					auto lower = std::lower_bound(len_vectors_image_ground[j].begin(), len_vectors_image_ground[j].end(), tmp_len_low);
					auto upper = std::upper_bound(len_vectors_image_ground[j].begin(), len_vectors_image_ground[j].end(), tmp_len_up);
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
				if (max_count_of_good_match >(len_vectors_image_sample.size())*quality) break;
			}
		}

		if (max_count_of_good_match == 0) return;

		max_count_of_good_match = 0;
		std::list<cv::Point2i> object_coords;

		for (uint i = 0; i < len_vectors_image_sample.size() - number_of_max_sample_matches; ++i) {
			tmp_len_low = len_coords(len_vectors_image_sample[number_of_max_sample_matches][i] - eps);
			tmp_len_up = len_coords(len_vectors_image_sample[number_of_max_sample_matches][i] + eps);
			auto lower = std::lower_bound(len_vectors_image_ground[number_of_max_ground_matches].begin(), len_vectors_image_ground[number_of_max_ground_matches].end(), tmp_len_low);
			auto upper = std::upper_bound(len_vectors_image_ground[number_of_max_ground_matches].begin(), len_vectors_image_ground[number_of_max_ground_matches].end(), tmp_len_up);
			for (auto it = lower; it != upper; ++it) {
				if (it->check_flag == 0) {
					object_coords.push_back(it->coords);
					it->check_flag = 1;
					++max_count_of_good_match;
				}
			}
		}
		double tmp_procent_of_coincidence = static_cast<double>(max_count_of_good_match) / static_cast<double>(len_vectors_image_sample.size());
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
				cv::circle(last_search, *it, 10, cv::Scalar(255));
				cv::putText(last_search, std::to_string(it->x) + ' ' + std::to_string(it->y), *it, cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(255));
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
		for (uint i = 0; i < examples.size(); ++i) {
			Find(examples[i], quality, count_for_binsearch, eps);
		}
		std::cout << "Finding: " << clock() - start << "ms" << std::endl;
	}

};

int main_1() {
	const char templates_names[][128] = {
		{ "temp/0.jpg" },
		{ "temp/1.jpg" },
		{ "temp/2.jpg" },
		{ "temp/3.jpg" },
		{ "temp/4.jpg" },
		{ "temp/5.jpg" },
		{ "temp/6.jpg" },
		{ "temp/7.jpg" },
		{ "temp/8.jpg" },
		{ "temp/9.jpg" },
		{ "temp/10.jpg" },
		{ "temp/11.jpg" },
		{ "temp/12.jpg" },
		{ "temp/13.jpg" },
		{ "temp/14.jpg" },
		{ "temp/15.jpg" }
	};
	const char ground_names[][128] = {
		{ "scope/0.jpg" },
		{ "scope/1.jpg" },
		{ "scope/2.jpg" },
		{ "scope/3.jpg" },
		{ "scope/4.jpg" },
		{ "scope/5.jpg" },
		{ "scope/6.jpg" },
		{ "scope/7.jpg" },
		{ "scope/8.jpg" },
		{ "scope/9.jpg" }
	};

	Scene my_scene(16, templates_names, 200, 0.001, 10);

	for (uint i = 0; i < 10; ++i) {
		my_scene.SetScope(cv::imread(ground_names[i], 0), 4000, 0.001, 10);
		my_scene.Detect(1, 4, 1);
		my_scene.ShowBestSearch();
	}

	system("pause");
	return 0;
}