/*
author: lochappy<ttanloc@gmail.com>
date: 04Apr2019
*/
#include "FDoG.hpp"

using namespace cv;
using namespace std;

namespace CoherentLine
{
	namespace FDoG
	{
		// Eq.(8)
		inline float gauss(float x, float mean, float sigma) {
			return (exp((-(x - mean)*(x - mean)) / (2 * sigma*sigma)) / sqrt(M_PI * 2.0 * sigma * sigma));
		}

		// eq.(10)
		inline void binaryThresholding(cv::Mat& src, cv::Mat& dst, const float tau){
			#pragma omp parallel for
			for (int y = 0; y < dst.rows; y++) {
				for (int x = 0; x < dst.cols; x++) {
					float H = src.at<float>(y, x);
					int v = H < tau ? 0 : 255;
					dst.at<uchar>(y, x) = v;
				}
			}
		}
		inline void MakeGaussianVector(float sigma, vector<float>& GAU) {
			const float threshold = 0.001;

			int i = 0;
			while (1) {
				i++;
				if (gauss((float)i, 0.0, sigma) < threshold)
					break;
			}
			GAU.clear();
			GAU.resize(i + 1);

			GAU[0] = gauss(0.0, 0.0, sigma);
			for (int j = 1; j < GAU.size(); j++) {
				GAU[j] = gauss((float)j, 0.0, sigma);
			}
		}

		void gradientDoG(cv::InputArray ori_img, 
							cv::InputArray etf_img,
							const float rho, 
							const float sigma_c,
							cv::OutputArray out_img){

			cv::Mat  src  = ori_img.getMat();
			cv::Mat flowField = etf_img.getMat();

			CV_Assert(src.channels() == 1 && "Input original image must have 1 channel") ;
			CV_Assert(src.depth() == CV_8U && "Input original image must be type of CV_U8");
			CV_Assert(flowField.channels() == 3 && "Input edge tangent flow image must have 3 channels") ;
			CV_Assert(flowField.depth() == CV_32F && "Input original image must be type of CV_32F");
			CV_Assert(src.rows == flowField.rows && src.cols == flowField.cols && 
						"Input original image and its edge tangent flow must be in the same size") ;

			src.convertTo(src,CV_32FC1,1.0 / 255.0);
			flowField.convertTo(flowField,CV_32F);

			cv::Mat& dst = out_img.getMatRef();
			dst = Mat::zeros(src.size(), CV_32FC1);

			const float sigma_s = SIGMA_RATIO*sigma_c;
			vector<float> gau_c, gau_s;
			MakeGaussianVector(sigma_c, gau_c);
			MakeGaussianVector(sigma_s, gau_s);

			const int kernel = gau_s.size() - 1;

			#pragma omp parallel for
			for (int y = 0; y < dst.rows; y++) {
				for (int x = 0; x < dst.cols; x++) {
					float gau_c_acc = 0;
					float gau_s_acc = 0;
					float gau_c_weight_acc = 0;
					float gau_s_weight_acc = 0;
					Vec3f tmp = flowField.at<Vec3f>(y, x);
					Point2f gradient = Point2f(-tmp[0], tmp[1]);

					if (gradient.x == 0 && gradient.y == 0) continue;
					
					for (int step = -kernel; step <= kernel; step++) {
						float row = y + gradient.y * step;
						float col = x + gradient.x * step;

						if (col > (float)dst.cols - 1 || col < 0.0 || row > (float)dst.rows - 1 || row < 0.0) continue;

						float value = src.at<float>((int)round(row), (int)round(col));

						int gau_idx = abs(step);
						float gau_c_weight = (gau_idx >= gau_c.size()) ? 0.0 : gau_c[gau_idx];
						float gau_s_weight = gau_s[gau_idx];

						gau_c_acc += value * gau_c_weight;
						gau_s_acc += value * gau_s_weight;
						gau_c_weight_acc += gau_c_weight;
						gau_s_weight_acc += gau_s_weight;
					}

					float v_c = gau_c_acc / gau_c_weight_acc;
					float v_s = gau_s_acc / gau_s_weight_acc;
					dst.at<float>(y, x) = v_c - rho*v_s;
				}
			}

		}// end gradientDoG

		//////////////////////////////////////

		void flowDoG(cv::InputArray gradDoG_img, cv::InputArray etf_img, const float sigma_m, cv::OutputArray fFoG_img){

			cv::Mat  src  = gradDoG_img.getMat();
			cv::Mat flowField = etf_img.getMat();

			CV_Assert(src.channels() == 1 && "Input original image must have 1 channel") ;
			CV_Assert(src.depth() == CV_32F && "Input original image must be type of CV_32F");
			CV_Assert(flowField.channels() == 3 && "Input edge tangent flow image must have 3 channels") ;
			CV_Assert(flowField.depth() == CV_32F && "Input original image must be type of CV_32F");
			CV_Assert(src.rows == flowField.rows && src.cols == flowField.cols && 
						"Input original image and its edge tangent flow must be in the same size") ;
			
			cv::Mat& dst = fFoG_img.getMatRef();
			dst = Mat::zeros(src.size(), CV_32FC1);

			vector<float> gau_m;
			MakeGaussianVector(sigma_m, gau_m);

			const int img_w = src.cols;
			const int img_h = src.rows;
			const int kernel_half = gau_m.size() - 1;

			#pragma omp parallel for
			for (int y = 0; y < img_h; y++) {
				for (int x = 0; x < img_w; x++) {
					float gau_m_acc = -gau_m[0] * src.at<float>(y, x);
					float gau_m_weight_acc = -gau_m[0];
						
					// Intergral alone ETF
					Point2f pos(x, y);
					for (int step = 0; step < kernel_half; step++) {
						Vec3f tmp = flowField.at<Vec3f>((int)round(pos.y), (int)round(pos.x));
						Point2f direction = Point2f(tmp[1], tmp[0]);

						if (direction.x == 0 && direction.y == 0) break;
						if (pos.x > (float)img_w - 1 || pos.x < 0.0 || pos.y >(float)img_h - 1 || pos.y < 0.0) break;

						float value = src.at<float>((int)round(pos.y), (int)round(pos.x));
						float weight = gau_m[step];

						gau_m_acc += value*weight;
						gau_m_weight_acc += weight;

						// move alone ETF direction 
						pos += direction;
						
						if ((int)round(pos.x) < 0 || (int)round(pos.x) > img_w - 1 || (int)round(pos.y) < 0 || (int)round(pos.y) > img_h - 1) break;
					}

					// Intergral alone inverse ETF
					pos = Point2f(x, y);
					for (int step = 0; step < kernel_half; step++) {
						Vec3f tmp = -flowField.at<Vec3f>((int)round(pos.y), (int)round(pos.x));
						Point2f direction = Point2f(tmp[1], tmp[0]);

						if (direction.x == 0 && direction.y == 0) break;
						if (pos.x >(float)img_w - 1 || pos.x < 0.0 || pos.y >(float)img_h - 1 || pos.y < 0.0) break;

						float value = src.at<float>((int)round(pos.y), (int)round(pos.x));
						float weight = gau_m[step];

						gau_m_acc += value*weight;
						gau_m_weight_acc += weight;

						// move alone ETF direction 
						pos += direction;

						if ((int)round(pos.x) < 0 || (int)round(pos.x) > img_w - 1 || (int)round(pos.y) < 0 || (int)round(pos.y) > img_h - 1) break;
					}

					
					dst.at<float>(y, x) = (gau_m_acc / gau_m_weight_acc) > 0 ? 1.0 : 1 + tanh(gau_m_acc / gau_m_weight_acc);
				}
			}

			normalize(dst, dst, 0, 1, NORM_MINMAX);
		} // end flowDoG

		void getCoherentLineImage( cv::InputArray ori_img, 
								   cv::InputArray etf_img,
								   const float rho, 
								   const float sigma_c,
								   const float sigma_m,
								   const float tau,
								   cv::OutputArray out_img){

			Mat mDoG, mFDoG;						   
			gradientDoG(ori_img, etf_img, rho, sigma_c, mDoG);
			flowDoG(mDoG, etf_img, sigma_m,mFDoG);

			cv::Mat& result = out_img.getMatRef();
			result = Mat::zeros(mDoG.size(), CV_8UC1);
			binaryThresholding(mFDoG, result, tau);						   
			
		}
		
	}// end FDoG
}