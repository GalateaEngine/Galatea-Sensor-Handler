#include "stdafx.h"

int numLandmarks;
int numDimensions;
double oldRot;
cv::Mat F;
cv::Mat FNorm;
bool generateGuide = true;
cv::Mat uControls;
cv::Mat sigmaUncertainty;
cv::Mat zObservation;
cv::Mat xState;
cv::Mat rtMovementUncertainty;
cv::Mat qtSensorUncertainty;


//Contains the equations to predict robot motion between the last step and the current step given the controls input
cv::Mat g(cv::Mat meanPos, cv::Mat controlCommand)
{
	//generate "guide" matrix so we only affect the columns we want to, but only once
	if (generateGuide)
	{
		F = cv::Mat(2 * (numDimensions + 1) + 3, numDimensions, CV_64FC1, double(0));//PRE TRANSPOSED
		F.at<double>(0, 0) = 1;
		F.at<double>(1, 1) = 1;
		F.at<double>(2, 2) = 1;
		cv::transpose(F, FNorm);
		generateGuide = false;
	}

	//generate velocity model vector
	cv::Mat vM(numDimensions + 1, 1, CV_64FC1, double(0));
	//calc v
	double v = controlCommand.at<double>(0, 0);
	double w = controlCommand.at<double>(1, 0);
	double theta = controlCommand.at<double>(2, 0);
	double vOw = v / w;
	double wChangeOverTime = oldRot - w;
	oldRot = w;
	vM.at<double>(0, 0) = (-1 * vOw*sin(theta)) + (vOw * sin(theta + wChangeOverTime));
	vM.at<double>(1, 0) = (vOw * cos(theta)) - (vOw*cos(theta + wChangeOverTime));
	vM.at<double>(2, 0) = wChangeOverTime;

	//so the final equation is
	return meanPos + (F * vM);
}

//Generates and applies the Jacobian matrix to the input function to make it linear for the given time frame
cv::Mat CalcSigmaBar(cv::Mat controlCommand)
{
	//Generate Jacobian matrix G(t)^x
	cv::Mat Gtx(numDimensions + 1, numDimensions + 1, CV_64FC1, double(0));
	cv::Mat GtxTranspose(numDimensions + 1, numDimensions + 1, CV_64FC1, double(0));
	
	double v = controlCommand.at<double>(0, 0);
	double w = controlCommand.at<double>(1, 0);
	double theta = controlCommand.at<double>(2, 0);
	double vOw = v / w;
	double wChangeOverTime = oldRot - w;
	oldRot = w;

	Gtx.at<double>(0, 0) = 1;
	Gtx.at<double>(1, 1) = 1;
	Gtx.at<double>(2, 2) = 1;
	Gtx.at<double>(0, 2) = (-1 * vOw*cos(theta)) + (vOw * cos(theta + wChangeOverTime));
	Gtx.at<double>(1, 2) = (-1 * vOw*sin(theta)) + (vOw * sin(theta + wChangeOverTime));

	cv::Mat Gt((numDimensions + 1) + 2 * numLandmarks, (numDimensions + 1) + 2 * numLandmarks, CV_64FC1, double(0));
	cv::Mat GtTranspose((numDimensions + 1) + 2 * numLandmarks, (numDimensions + 1) + 2 * numLandmarks, CV_64FC1, double(0));
	for (int i = 0; i < (numDimensions + 1) + 2 * numLandmarks; i++)
	{
		Gt.at<double>(i, i) = 1;
	}

	Gtx.copyTo(Gt(cv::Rect(0, 0, Gtx.cols, Gtx.rows)));
	cv::transpose(Gtx, GtxTranspose);
	cv::transpose(Gt, GtTranspose);

	cv::Mat sigmaBar = (Gt * sigmaUncertainty * GtTranspose) + (F * rtMovementUncertainty * FNorm); //this needs to be updated for speed, see lecture 5 37mins

	return sigmaBar;
}

cv::Mat calcHit(cv::Mat sigmaBar, cv::Mat uBar, cv::Mat landmarks)
{
	cv::Mat Hit((2 * numLandmarks) + 3, 2, CV_64FC1, double(0));
	//for each landmark
	for (int j = 0; j < landmarks.rows; j++)
	{
		//calc h first
		cv::Mat delta(2, 1, CV_64FC1);
		int landmarkID = landmarks.at<double>(j, 0);
		double landmarkX = landmarks.at<double>(j, 1);
		double landmarkY = landmarks.at<double>(j, 2);
		delta.at<double>(0, 0) = landmarkX - uBar.at<double>(0, 0);
		delta.at<double>(1, 0) = landmarkY - uBar.at<double>(0, 1);
		cv::Mat deltaT;
		cv::transpose(delta, deltaT);
		cv::Mat qmat = deltaT * delta;
		double q = qmat.at<double>(0, 0);
		double h1 = sqrt(q);
		double h2 = atan2(delta.at<double>(1, 0), delta.at<double>(0, 0)) - uBar.at<double>(0, 2);

		//put values directly into Hit matrix to save on generating F2
		//not that since we have multiple landmarks we are generating the AVERAGE determined robot position as determined by the landmarks
		Hit.at<double>(0, 0) += (-1 * sqrt(q) * delta.at<double>(0, 0)) / landmarks.rows;
		Hit.at<double>(1, 0) += delta.at<double>(0, 1) / landmarks.rows;

		Hit.at<double>(0, 1) += (-1 * sqrt(q) * delta.at<double>(0, 1)) / landmarks.rows;
		Hit.at<double>(1, 1) += (-1 * delta.at<double>(0, 0)) / landmarks.rows;

		//Hit.at<double>(0, 2) += 0; // we can just leave this at zero
		Hit.at<double>(1, 2) += -q / landmarks.rows;

		Hit.at<double>(0, landmarkID * 2) = sqrt(q) * delta.at<double>(0, 0);
		Hit.at<double>(1, landmarkID * 2) = -delta.at<double>(0, 1);

		Hit.at<double>(0, (landmarkID * 2) + 1) = sqrt(q) * delta.at<double>(0, 1);
		Hit.at<double>(1, (landmarkID * 2) + 1) = delta.at<double>(0, 0);
	}

	return Hit;
}

//implements the bare bones kalman filter
//Landmarks structure: each row has landmark ID, landmark x, landmark y
void Extended_Kalman_Filter(cv::Mat meanPos, cv::Mat Landmarks)
{
	//Prediction Step
	
	//uBar = g(ut, ut-1)
	cv::Mat uBar = g(meanPos, uControls);

	//sigmaBar = G(t)*Sigma(t-1)*G(t)^T + R(t)
	cv::Mat sigmaBar = CalcSigmaBar(uControls);


	//Update Step
	
	//Calculate Kalman Gain
	cv::Mat Hit = calcHit(sigmaBar, uBar, Landmarks);

}