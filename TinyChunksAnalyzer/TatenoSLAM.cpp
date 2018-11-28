#include "stdafx.h"

//implementing the SLAM-CNN hybrid as per "CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction", Tateno et al. (2017)

void Tateno_Monocular_SLAM()
{
#pragma region Definitions

	//Definitions:
	//ki:set of keyframes k1, ..., kn ∈ K

	//Each ki keyframe has:
	//Tki: a frame pose
	//Dk: a depth map
	//Uk: a depth uncertainty map

	//u: generic depth map element. u = (x,y), which ranges in the image domain, i.e. u ∈ Ω ⊂ R^2. udot = homogeneous representation

	//t: frame

	//T^kit: current camera pose. Composed of [Rt, tt] where:
	//Rt: 3x3 rotation cv::Matrix, ∈ SO(3)
	//tt: 3d translation vector, ∈ R^3

	//It: intensity image of frame t
	//Iki: intensity image of keyframe ki

#pragma endregion

	//For each new frame t:
	//First calc Tkit
	//The transforcv::Mation between the current camera pose and the last keyframe is given by the diference in intensity maps
	//via a weighted Gauss-Newton optomization
}

#pragma region E

//Huber Norm
//As defined in "LSD-SLAM: Large-Scale Direct Monocular SLAM", Engel et al. (2014)
//paramDelta
//5 * sqrt(6000*loopclosureStrictness);
//paramE = error;
cv::Mat p(cv::Mat a, double paramDelta = 1, double paramE = 1)
{
	double dsqr = paramDelta * paramDelta;
	if (paramE <= dsqr) { // inlier
		a.at<double>(0, 0) = paramE;
		a.at<double>(0, 1) = 1.;
		a.at<double>(0, 2) = 0.;
	}
	else { // outlier
		double sqrte = sqrt(paramE); // absolute value of the error
		a.at<double>(0, 0) = 2 * sqrte*paramDelta - dsqr; // rho(e)   = 2 * delta * e^(1/2) - delta^2
		a.at<double>(0, 1) = paramDelta / sqrte;        // rho'(e)  = delta / sqrt(e)
		a.at<double>(0, 2) = -0.5 * a.at<double>(0, 1) / paramE;    // rho''(e) = -1 / (2*e^(3/2)) = -1/2 * (delta/e) / e
	}
	return a;
}

cv::Mat sigma(cv::Mat a)
{
	//measure residual incertainty
	//how
	return a;
}

cv::Mat r(double pixel, cv::Mat a)
{

	return a;
}

cv::Mat E(cv::Mat Tkit, cv::Mat currentFrame)
{
	//E(Tkit) = sum(each element in omega, uBar)(p(r(uBar, Tkit)/sigma(r(uBar, Tkit))))
	//(Should Tkit actually be Tki? Or is it implying it's computed in place?)
	cv::Mat currentPose;
	
	for (int x = 0; x < currentFrame.rows; x++)
		for (int y = 0; y < currentFrame.cols; y++)
			currentPose += p(r(currentFrame.at<double>(x,y), Tkit) / sigma(r(currentFrame.at<double>(x, y), Tkit)));

	return currentPose;
}
#pragma endregion