/*
author: lochappy<ttanloc@gmail.com>
date: 04Apr2019
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>

#define SIGMA_RATIO 1.6 
#define STEPSIZE 1.0

using namespace std;

namespace CoherentLine
{
	namespace FDoG
	{
		CV_EXPORTS_W void getCoherentLineImage( cv::InputArray ori_img, 
									  			cv::InputArray etf_img,
									  			const float rho, 
									  			const float sigma_c,
												const float sigma_m,
												const float tau,
									  			cv::OutputArray out_img);

		// Perform eq.(6) on each pixel
		CV_EXPORTS_W void gradientDoG(cv::InputArray ori_img, 
									  cv::InputArray etf_img,
									  const float rho, 
									  const float sigma_c,
									  cv::OutputArray out_img);

		// Perform eq.(9) on each pixel
		CV_EXPORTS_W void flowDoG(cv::InputArray gradDoG_img,
								  cv::InputArray etf_img,
								  const float sigma_m, 
								  cv::OutputArray fFoG_img);

	}
}