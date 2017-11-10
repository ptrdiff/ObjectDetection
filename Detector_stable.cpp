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
		uint example_count_of_keypoints;
		uint **example_len_vectors;
	public:
		cv::Mat example_image;
		Example(cv::Mat image = cv::Mat()) : example_image(image), example_count_of_keypoints(0),example_len_vectors(nullptr) {}
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
			cv::goodFeaturesToTrack(example_image, example_keypoints, maxCorners, qualityLevel, minDistance);
			example_count_of_keypoints = static_cast<uint>(example_keypoints.size());
			if (example_count_of_keypoints == 0) return;
			int tmp_coords_x;
			int tmp_coords_y;
			int tmp_corner_x;
			int tmp_corner_y;
			example_len_vectors = new uint *[example_count_of_keypoints - 1];
			for (uint j = 0; j < example_count_of_keypoints - 1; ++j) {
				example_len_vectors[j] = new uint[example_count_of_keypoints - 1 - j];
				uint pos = 0;
				tmp_corner_x = example_keypoints[j].x;
				tmp_corner_y = example_keypoints[j].y;
				for (uint i = j + 1; i < example_count_of_keypoints; ++i) {
					tmp_coords_x = example_keypoints[i].x - tmp_corner_x;
					tmp_coords_y = example_keypoints[i].y - tmp_corner_y;
					example_len_vectors[j][pos++] = tmp_coords_x*tmp_coords_x + tmp_coords_y*tmp_coords_y;
				}
			}
			///std::cout << "count of sample corners detected: " << example_count_of_keypoints << std::endl;
		}
		uint KeypointsCount() { return example_count_of_keypoints; }
		uint **LenVectors() { return example_len_vectors; }
	};
	struct len_coords {
		uint len;
		cv::Point2i coords;
		len_coords(uint lenght = 0, cv::Point2i cords = cv::Point2i()) : len(lenght), coords(cords) {}
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
		Scope(cv::Mat image = cv::Mat()) : scope_image(image), scope_count_of_keypoints(0),scope_len_vectors(nullptr) {}
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
	Scene(const uint n, const char names[][128], int maxCorners = 10000, double qualityLevel = 0.01,double minDistance = 10 ) 
		: N(n), examples(nullptr), ground(Scope()), procent_of_coincidence(0){ 
		examples = new Example[N]; 
		for (uint i = 0; i < N; ++i) { 
			examples[i].example_image = cv::imread(names[i], 0);
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
		ground = scope_names;
		ground.Compute(maxCorners, qualityLevel,minDistance);
	}
	void Find(Example &sample, double quality = 0.5) {
		uint number_of_max_ground_matches;
		uint number_of_max_sample_matches;
		uint count_of_good_match = 0;
		uint max_count_of_good_match = 0;
		uint count_of_image_sample_corners = sample.KeypointsCount();
		uint count_of_image_sample_corners_for_binsearch = cv::min(count_of_image_sample_corners - 1, static_cast<uint>(4));
		uint count_of_image_ground_corners = ground.KeypointsCount();
		if (count_of_image_sample_corners == 0 && count_of_image_ground_corners == 0) return;
		uint **len_vectors_image_sample = sample.LenVectors();
		len_coords **len_vectors_image_ground = ground.LenVectors();

		for (uint k = 0; k < count_of_image_sample_corners_for_binsearch; ++k) {
			for (uint j = 0; j < count_of_image_ground_corners; ++j) {
				for (uint i = 0; i < count_of_image_sample_corners_for_binsearch - k; ++i) {
					if (std::binary_search(len_vectors_image_ground[j], len_vectors_image_ground[j] + count_of_image_ground_corners,
						len_coords(len_vectors_image_sample[k][i]))) {
						++count_of_good_match;
					}
				}
				if (max_count_of_good_match < count_of_good_match) {
					max_count_of_good_match = count_of_good_match;
					number_of_max_ground_matches = j; number_of_max_sample_matches = k;
				}
				count_of_good_match = 0;
				if (max_count_of_good_match >(count_of_image_sample_corners - 1)*quality) break;
			}
		}

		max_count_of_good_match = 0;
		std::list<cv::Point2i> object_coords;
		len_coords *low;
		for (uint i = 0; i < count_of_image_sample_corners - 1 - number_of_max_sample_matches; ++i) {
			low = std::lower_bound(len_vectors_image_ground[number_of_max_ground_matches],
				len_vectors_image_ground[number_of_max_ground_matches] + count_of_image_ground_corners,
				len_coords(len_vectors_image_sample[number_of_max_sample_matches][i]));
			if (low->len == len_vectors_image_sample[number_of_max_sample_matches][i]) {
				object_coords.push_back(low->coords);
				++max_count_of_good_match;
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
			cv::Mat last_search = ground.scope_image.clone();
			for (std::list<cv::Point2i>::iterator it = best_object_coords.begin(); it != best_object_coords.end(); ++it)
				cv::circle(last_search, *it, 10, cv::Scalar(255));
			cv::imshow("match", last_search);
			std::cout << procent_of_coincidence << '%' <<std::endl;
			cv::waitKey(0);
		}
	}
	void Detect() {
		clock_t start = clock();
		for (uint i = 0; i < N; ++i) {
			Find(examples[i]);
		}
		std::cout << "--------------------Finding: " << clock() - start << "ms" << std::endl;
	}

};

int main() {
	const char templates_names[][128] = {
		{"sample/im0.jpg"},
		{"sample/im1.jpg"},
		{"sample/im2.jpg"},
		{"sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{"sample/cube.jpg"},
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" },
		{ "sample/im0.jpg" },
		{ "sample/im1.jpg" },
		{ "sample/im2.jpg" },
		{ "sample/redbull.jpg" },
		{ "sample/1.jpg" },
		{ "sample/2.jpg" },
		{ "sample/3.jpg" },
		{ "sample/4.jpg" },
		{ "sample/5.jpg" },
		{ "sample/6.jpg" },
		{ "sample/im0 (3).jpg" },
		{ "sample/im0 (4).jpg" },
		{ "sample/cube.jpg" }
	};
	Scene my_scene(286, templates_names);

	my_scene.SetScope(cv::imread("ground/im0.jpg", 0));
	my_scene.Detect();
	my_scene.ShowBestSearch();

	my_scene.SetScope(cv::imread("ground/im0 (1).jpg", 0));
	my_scene.Detect();
	my_scene.ShowBestSearch();

	my_scene.SetScope(cv::imread("ground/im0 (2).jpg", 0));
	my_scene.Detect();
	my_scene.ShowBestSearch();

	return 0;
}