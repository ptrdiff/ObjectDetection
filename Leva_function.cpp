#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include<algorithm>
#include<cmath>
/*!структура хранящая 3D - координаты точек в пространстве
@x,y,z  - координаты точек
$
#
coordinates(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
*/
struct coordinates {
	float x, y, z;
	coordinates(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
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
	///------------------------------------------------------------------------------------------------------------------------
	///усреденение карты для инфемумов, работает криво, удалить если не нужно
	for (int i = 1; i<imgDisparity8U.rows-1; i++)
		for (int j = 1; j < imgDisparity8U.cols-1; j++)
		{
			if (imgDisparity8U.at<uchar>(i, j) == 0)
			{
				int sum = 0;
				int count = 0;
				for (int k = i - 1; k < i + 2; k++)
				{
					for (int l = j - 1; l < j + 2; l++)
					{
						if (imgDisparity8U.at<uchar>(k, l) !=0 && imgDisparity8U.at<uchar>(k, l) < 150)
						{
							sum += imgDisparity8U.at<uchar>(k, l);
							count++;
						}
					}
				}
				if(count!=0)
					imgDisparity8U.at<uchar>(i, j) = (uchar)(sum / count);

			}
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
std::vector<coordinates> reconstruction3D(cv::Mat dispMap, std::vector<cv::Point2i> ground_corners)
{
	double focus = 3.6;
	double base = 25;
	double pixelSize = 1/375.2;
	std::vector<coordinates> coords;
	uint size = ground_corners.size();
	coords.resize(size);
	for (uint i = 0; i < size; ++i) {
		if (dispMap.at<uchar>(ground_corners[i]) != 0) {
			float z = (focus*base / (pixelSize*dispMap.at<uchar>(ground_corners[i])));
			float y = (ground_corners[i].y - (double)dispMap.rows / 2) / (dispMap.at<uchar>(ground_corners[i]))*base;
			float x = (ground_corners[i].x - (double)dispMap.cols / 2) / (dispMap.at<uchar>(ground_corners[i]))*base;
			coords[i] = coordinates(x, y, z);
		}
		else coords[i] = coordinates(INFINITY, INFINITY, INFINITY);
	}
	for (uint i = 0; i < size; ++i) std::cout << coords[i].x << ' ' << coords[i].y << ' ' << coords[i].z << std::endl;
	system("pause");
	return coords;
}

