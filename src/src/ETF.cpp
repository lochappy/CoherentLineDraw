/*
author: lochappy<ttanloc@gmail.com>
date: 04Apr2019
*/
#include "ETF.hpp"

using namespace cv;

namespace CoherentLine
{
	namespace EdgeTangentFlow
	{
		Mat gradientMag;

		void compute_edge_tangent_flow(InputArray in_img, OutputArray out_img){
			cv::Mat  src  = in_img.getMat();
			cv::Mat& flowField = out_img.getMatRef();

			const int width = src.cols;
			const int height = src.rows;
			const int channel = src.channels();

			CV_Assert(channel == 1 && "Input image must have 1 channel") ;

			Mat src_n;
			normalize(src, src_n, 0.0, 1.0, NORM_MINMAX, CV_32FC1);
			//GaussianBlur(src_n, src_n, Size(51, 51), 0, 0);

			// Generate grad_x and grad_y
			Mat grad_x, grad_y;
			Sobel(src_n, grad_x, CV_32FC1, 1, 0, 5);
			Sobel(src_n, grad_y, CV_32FC1, 0, 1, 5);

			//Compute gradient
			magnitude(grad_x, grad_y, gradientMag);
			normalize(gradientMag, gradientMag, 0.0, 1.0, NORM_MINMAX);

			flowField = Mat::zeros(src.size(), CV_32FC3);

			#pragma omp parallel for
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					const float u = grad_x.at<float>(i, j);
					const float v = grad_y.at<float>(i, j);
					flowField.at<Vec3f>(i, j) = normalize(Vec3f(-v, u, 0)); //rotate 90 degree
				}
			}

		}
		void compute_refined_edge_tangent_flow(cv::InputArray in_img, int kernelSize, cv::OutputArray out_img){
			cv::Mat flowField;
			compute_edge_tangent_flow(in_img, flowField);

			cv::Mat& refinedETF = out_img.getMatRef();
			refinedETF = Mat::zeros(flowField.size(), CV_32FC3);

			#pragma omp parallel for
			for (int r = 0; r < flowField.rows; r++) {
				for (int c = 0; c < flowField.cols; c++) {
					computeNewVector(c, r, kernelSize, flowField, refinedETF);
				}
			}
		}


		/*
		* Paper's Eq(1)
		*/
		void computeNewVector(int x, int y, const int kernel,
							const cv::Mat &flowField, 
							cv::Mat &refinedETF) 
			{
			const Vec3f t_cur_x = flowField.at<Vec3f>(y, x);
			Vec3f t_new = Vec3f(0, 0, 0);

			for (int r = y - kernel; r <= y + kernel; r++) {
				for (int c = x - kernel; c <= x + kernel; c++) {
					if (r < 0 || r >= refinedETF.rows || c < 0 || c >= refinedETF.cols) continue;

					const Vec3f t_cur_y = flowField.at<Vec3f>(r, c);
					float phi = computePhi(t_cur_x, t_cur_y);
					float w_s = computeWs(Point2f(x, y), Point2f(c, r), kernel);
					float w_m = computeWm(gradientMag.at<float>(y, x), gradientMag.at<float>(r, c));
					float w_d = computeWd(t_cur_x, t_cur_y);
					t_new += phi*t_cur_y*w_s*w_m*w_d;
				}
			}
			refinedETF.at<Vec3f>(y, x) = normalize(t_new);
		}

		/*
		* Paper's Eq(5)
		*/
		float computePhi(cv::Vec3f x, cv::Vec3f y) {
			return x.dot(y) > 0 ? 1 : -1;
		}

		/*
		* Paper's Eq(2)
		*/
		float computeWs(cv::Point2f x, cv::Point2f y, int r) {
			return norm(x - y) < r ? 1 : 0;
		}

		/*
		* Paper's Eq(3)
		*/
		float computeWm(float gradmag_x, float gradmag_y) {
			float wm = (1 + tanh(gradmag_y - gradmag_x)) / 2;
			return wm;
		}

		/*
		* Paper's Eq(4)
		*/
		float computeWd(cv::Vec3f x, cv::Vec3f y) {
			return abs(x.dot(y));
		}

	}
}
