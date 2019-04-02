/*
author: lochappy<ttanloc@gmail.com>
date: 04Apr2019
*/
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

namespace CoherentLine
{
	namespace EdgeTangentFlow
	{
		CV_EXPORTS_W void compute_edge_tangent_flow(cv::InputArray in_img, cv::OutputArray out_img);
		CV_EXPORTS_W void compute_refined_edge_tangent_flow(cv::InputArray in_img, int kernelSize, cv::OutputArray out_img);


		void computeNewVector(int x, int y,
								const int kernel,
								const cv::Mat &flowField, 
								cv::Mat &refinedETF);

		float computePhi(cv::Vec3f x, cv::Vec3f y);
		float computeWs(cv::Point2f x, cv::Point2f y, int r);
		float computeWm(float gradmag_x, float gradmag_y);
		float computeWd(cv::Vec3f x, cv::Vec3f y);

	}
}