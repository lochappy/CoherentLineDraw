"""
author: lochappy<ttanloc@gmail.com>
date: 04Apr2019
"""
import cv2
import sys
import cld

kernelSize = 5 # Edge tangent flow kernel size
sigma_m = 4.0 # Degree of coherence
sigma_c = 5.0 # Line width
rho = 0.997 # Noise
tau = 0.8 # Binary threshold


im = cv2.imread('./data/test0_small.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("0-Original image", im)

etfImg = cld.EdgeTangentFlow.compute_edge_tangent_flow(im)
cv2.imshow("1a-etfImg", etfImg)

refinedEtfImg = cld.EdgeTangentFlow.compute_refined_edge_tangent_flow(im,kernelSize)
cv2.imshow("1b-refinedEtfImg", refinedEtfImg)

gradientDoGImg = cld.FDoG.gradientDoG(im, etfImg,rho,sigma_c)
cv2.imshow("2-gradientDoGImg", gradientDoGImg)

fDoGImg = cld.FDoG.flowDoG(gradientDoGImg,etfImg,sigma_m)
cv2.imshow("3-gradientDoGImg", gradientDoGImg)

coherentLineImage = cld.FDoG.getCoherentLineImage(im,refinedEtfImg,rho,sigma_c,sigma_m,tau)
cv2.imshow("4-coherentLineImage", coherentLineImage)

cv2.imwrite('./data/test0_small_cld.jpg',coherentLineImage)

cv2.waitKey(0)



