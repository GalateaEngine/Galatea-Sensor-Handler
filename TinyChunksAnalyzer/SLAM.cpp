#include "stdafx.h"

int numLandmarks;
int numDimensions;
double sensorDistUncertainty = 0.1;
double sensorRotUncertainty = 0.1;
std::map<int, cv::Point2d> landmarkDB;
double oldRot;
cv::Mat F;
cv::Mat FNorm;
bool generateGuide = true;
cv::Mat meanPos;
cv::Mat sigmaUncertainty;
cv::Mat rtMovementUncertainty;
cv::Mat qtSensorUncertainty;
cv::Mat hPreCalc;


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

//Generates and applies the Jacobian matrix to the input function to make it linear for the given time frame
cv::Mat CalcOmegaBar(cv::Mat controlCommand, cv::Mat omegaInv)
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

	cv::Mat omegaBarInv = (Gt * omegaInv * GtTranspose) + (F * rtMovementUncertainty * FNorm); //this needs to be updated for speed, see lecture 5 37mins

	return omegaBarInv.inv();
}

cv::Mat calcHit(cv::Mat sigmaBar, cv::Mat uBar, cv::Mat landmarks, int landmarkIndex)
{
	cv::Mat Hit((2 * numLandmarks) + 3, 2, CV_64FC1, double(0));
	hPreCalc = cv::Mat(2, 1, CV_64FC1);
	//for each landmark
	int j = landmarkIndex;
	int landmarkID = landmarks.at<double>(j, 0);
	double landmarkDist = landmarks.at<double>(j, 1);
	double landmarkRot = landmarks.at<double>(j, 2);

	//if new landmark
	if (!landmarkDB.count(landmarkID))
	{
		double tpointx, tpointy;
		tpointx = uBar.at<double>(0, 0) + (landmarkDist * cos(landmarkRot + uBar.at<double>(0, 2)));
		tpointy = uBar.at<double>(0, 0) + (landmarkDist * cos(landmarkRot + uBar.at<double>(0, 2)));
		cv::Point2d lmPoint(tpointx, tpointy);
		landmarkDB.emplace(landmarkID, lmPoint);
	}

	//set x and y for point
	cv::Point lmp = landmarkDB[landmarkID];
	double landmarkX = lmp.x;
	double landmarkY = lmp.y;

	//calc h first
	cv::Mat delta(2, 1, CV_64FC1);
	delta.at<double>(0, 0) = landmarkX - uBar.at<double>(0, 0);
	delta.at<double>(1, 0) = landmarkY - uBar.at<double>(0, 1);
	cv::Mat deltaT;
	cv::transpose(delta, deltaT);
	cv::Mat qmat = deltaT * delta;
	double q = qmat.at<double>(0, 0);
	double h1 = sqrt(q);
	double h2 = atan2(delta.at<double>(1, 0), delta.at<double>(0, 0)) - uBar.at<double>(0, 2);
	//store this for when we need it later
	hPreCalc.at<double>(0) = h1;
	hPreCalc.at<double>(1) = h2;

	//put values directly into Hit matrix to save on generating F2
	Hit.at<double>(0, 0) += (-1 * sqrt(q) * delta.at<double>(0, 0));
	Hit.at<double>(1, 0) += delta.at<double>(0, 1);

	Hit.at<double>(0, 1) += (-1 * sqrt(q) * delta.at<double>(0, 1));
	Hit.at<double>(1, 1) += (-1 * delta.at<double>(0, 0));

	//Hit.at<double>(0, 2) += 0; // we can just leave this at zero
	Hit.at<double>(1, 2) += -q;

	Hit.at<double>(0, landmarkID * 2) = sqrt(q) * delta.at<double>(0, 0);
	Hit.at<double>(1, landmarkID * 2) = -delta.at<double>(0, 1);

	Hit.at<double>(0, (landmarkID * 2) + 1) = sqrt(q) * delta.at<double>(0, 1);
	Hit.at<double>(1, (landmarkID * 2) + 1) = delta.at<double>(0, 0);

	return Hit;
}

//Initializes our EKF values for the first state

//implements the bare bones kalman filter
//Landmarks structure: each row has ID, distance reading, sensor rotation
void Extended_Kalman_Filter(cv::Mat Landmarks, cv::Mat controls)
{
	//set up qt, rt here
	qtSensorUncertainty = cv::Mat(2, 2, CV_64FC1, double(0));
	qtSensorUncertainty.at<double>(0, 0) = pow(sensorDistUncertainty, 2);
	qtSensorUncertainty.at<double>(1, 1) = pow(sensorRotUncertainty, 2);

	//Prediction Step

	//predict position
	//uBar = g(ut, ut-1)
	cv::Mat uBar = g(meanPos, controls);

	//predict 
	//sigmaBar = G(t)*Sigma(t-1)*G(t)^T + R(t)
	cv::Mat sigmaBar = CalcSigmaBar(controls);


	//Update Step

	//Calculate Kalman Gain
	cv::Mat ident = cv::Mat::eye(sigmaBar.rows, sigmaBar.cols, CV_64FC1);
	for (int j = 0; j < Landmarks.rows; j++)
	{
		cv::Mat Hit = calcHit(sigmaBar, uBar, Landmarks, j);
		cv::Mat HitT;
		cv::transpose(Hit, HitT);
		cv::Mat inverseSection;
		cv::invert((Hit * sigmaBar * HitT) + qtSensorUncertainty, inverseSection);
		cv::Mat KalmanGain = sigmaBar * HitT*inverseSection;

		//get the landmark we are comparing
		cv::Mat tlandmark(2, 1, CV_64FC1);
		tlandmark.at<double>(0, 0) = Landmarks.at<double>(j, 1);
		tlandmark.at<double>(1, 0) = Landmarks.at<double>(j, 2);

		//refine our predictions
		uBar = uBar + KalmanGain * (tlandmark - hPreCalc);
		sigmaBar = (ident - KalmanGain * Hit) * sigmaBar;
	}

	//refine our uncertainty measure
	cv::Mat sigma = sigmaBar;
	cv::Mat u = uBar;
	sigmaUncertainty = sigma;
	meanPos = u;
}

cv::Mat sigmaX;
cv::Mat sigmaXBar;
cv::Mat sigmaWm;
cv::Mat sigmaWc;

int doubleDim = numDimensions * 2;

//unscented parameters
double kappa = 3;
double alpha = 0.5;
double lambda;
double beta = 2;

void setup_UKF()
{
	lambda = alpha * alpha;
	lambda *= (numDimensions + kappa);
	lambda -= numDimensions;
}

//uses Cholesky decomposition ot find a matrix such that for input a the return value l is such that a = l*l^T
cv::Mat MatrixSqrt(cv::Mat a)
{
	int n = a.cols;

	cv::Mat ret(n, n, CV_64FC1);
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c <= r; c++)
		{
			if (c == r)
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					sum += ret.at<double>(c, j) * ret.at<double>(c, j);
				}
				ret.at<double>(c, c) = sqrt(a.at<double>(c, c) - sum);
			}
			else
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
					sum += ret.at<double>(r, j) * ret.at<double>(c, j);
				ret.at<double>(r, c) = (1.0 / ret.at<double>(c, c)) * (a.at<double>(r, c) - sum);
			}
		}
	}
	return ret;
}

//calculates the sigma points
void CalcSigmaPoints(cv::Mat controlCommand)
{
	sigmaX = cv::Mat(2, 2 * numDimensions, CV_64FC1);
	sigmaWm = cv::Mat(1, 2 * numDimensions, CV_64FC1);
	sigmaWc = cv::Mat(1, 2 * numDimensions, CV_64FC1);
	//first calc sigma points in X
	sigmaX.col(0) = meanPos;
	for (int i = 1; i < numDimensions; i++)
	{
		sigmaX.col(i) = meanPos + (lambda * MatrixSqrt(sigmaUncertainty));
	}
	for (int i = numDimensions; i < doubleDim; i++)
	{
		sigmaX.col(i) = meanPos - (lambda * MatrixSqrt(sigmaUncertainty));
	}

	//next calc sigma points in w(m and c)
	sigmaWm.at<double>(0, 0) = lambda / (numDimensions + lambda);
	sigmaWc.at<double>(0, 0) = sigmaWm.at<double>(0, 0) + (1 - (alpha*alpha) + beta);

	sigmaXBar = g(sigmaX, controlCommand);
	for (int i = 1; i < doubleDim; i++)
	{
		sigmaWm.at<double>(0, i) = 1 / (2 * (numDimensions + lambda));
		sigmaWc.at<double>(0, i) = sigmaWm.at<double>(0, i);
	}
}


void Unscented_Kalman_Filter(cv::Mat Landmarks, cv::Mat controls)
{
	//set up qt, rt here
	qtSensorUncertainty = cv::Mat(2, 2, CV_64FC1, double(0));
	qtSensorUncertainty.at<double>(0, 0) = pow(sensorDistUncertainty, 2);
	qtSensorUncertainty.at<double>(1, 1) = pow(sensorRotUncertainty, 2);

	//Prediction Step

	//calc sigma points
	CalcSigmaPoints(controls);

	//predict position
	cv::Mat uBar(numDimensions, 1, CV_64FC1);
	for (int i = 0; i < doubleDim; i++)
	{
		uBar += sigmaWm.col(i) * sigmaXBar.col(i);
	}

	//predict sigma bar
	cv::Mat sigmaBar;
	for (int i = 0; i < doubleDim; i++)
	{
		cv::Mat transposeCalc;
		cv::transpose((sigmaXBar.col(i) - meanPos), transposeCalc);
		sigmaBar += sigmaWc.col(i) * (sigmaXBar.col(i) - meanPos) * transposeCalc + rtMovementUncertainty;
	}


	//Update Step

	//Calculate Kalman Gain
	cv::Mat ident = cv::Mat::eye(sigmaBar.rows, sigmaBar.cols, CV_64FC1);
	for (int j = 0; j < Landmarks.rows; j++)
	{
		cv::Mat Z = calcHit(sigmaX, uBar, Landmarks, j);
		cv::Mat z(1, numDimensions, CV_64FC1, double(0));
		for (int i = 0; i < doubleDim; i++)
		{
			z += sigmaWm.col(i) * Z.col(i);
		}
		
		//calc new S param
		cv::Mat S(1, numDimensions, CV_64FC1, double(0));
		for (int i = 0; i < doubleDim; i++)
		{
			cv::Mat zCombine = Z.col(i) - z;
			cv::Mat zCombineT;
			cv::transpose(zCombine, zCombineT);
			S += sigmaWc.col(i) * zCombine.col(i) * zCombineT + qtSensorUncertainty;
		}

		//calc neosigmabar
		cv::Mat neoSigmaBar(1, numDimensions, CV_64FC1, double(0));
		for (int i = 0; i < doubleDim; i++)
		{
			cv::Mat zCombine = Z.col(i) - z;
			cv::Mat zCombineT;
			cv::transpose(zCombine, zCombineT);
			S += sigmaWc.col(i) * (sigmaX.col(i) - uBar) * zCombineT;
		}

		cv::Mat inverseS;
		cv::invert(S, inverseS);
		cv::Mat KalmanGain = neoSigmaBar * inverseS;

		//get the landmark we are comparing
		cv::Mat tlandmark(2, 1, CV_64FC1);
		tlandmark.at<double>(0, 0) = Landmarks.at<double>(j, 1);
		tlandmark.at<double>(1, 0) = Landmarks.at<double>(j, 2);

		//refine our predictions
		uBar += KalmanGain * (tlandmark - z);
		sigmaBar -= (KalmanGain * S * KalmanGain.t());
	}

	//refine our uncertainty measure
	cv::Mat sigma = sigmaBar;
	cv::Mat u = uBar;
	sigmaUncertainty = sigma;
	meanPos = u;
}

//implements the slightly more efficient information filter
//Landmarks structure: each row has ID, distance reading, sensor rotation
cv::Mat omega;
cv::Mat psi;
void Extended_Information_Filter(cv::Mat Landmarks, cv::Mat controls)
{
	//set up qt, rt here
	qtSensorUncertainty = cv::Mat(2, 2, CV_64FC1, double(0));
	qtSensorUncertainty.at<double>(0, 0) = pow(sensorDistUncertainty, 2);
	qtSensorUncertainty.at<double>(1, 1) = pow(sensorRotUncertainty, 2);

	//Prediction Step

	cv::Mat omegaInv = omega.inv();

	//predict omega
	cv::Mat omegaBar = CalcOmegaBar(controls, omegaInv).inv();

	//predict psi
	cv::Mat movEst = g(omegaInv * psi, controls);
	cv::Mat psiBar = omegaBar * movEst;

	//Update Step

	//Calculate Kalman Gain
	cv::Mat ident = cv::Mat::eye(omegaBar.rows, omegaBar.cols, CV_64FC1);
	for (int j = 0; j < Landmarks.rows; j++)
	{
		cv::Mat Hit = calcHit(omegaBar, psiBar, Landmarks, j);
		cv::Mat HitT;
		cv::transpose(Hit, HitT);
		cv::Mat qInv;
		cv::invert(qtSensorUncertainty, qInv);

		//get the landmark we are comparing
		cv::Mat tlandmark(2, 1, CV_64FC1);
		tlandmark.at<double>(0, 0) = Landmarks.at<double>(j, 1);
		tlandmark.at<double>(1, 0) = Landmarks.at<double>(j, 2);

		//refine our predictions
		cv::Mat HitTTimesqInv = HitT * qInv;
		omegaBar = omegaBar + HitTTimesqInv * Hit;
		psiBar = psiBar + HitTTimesqInv*(tlandmark - hPreCalc + (Hit*movEst));
	}

	//refine our uncertainty measure
	omega = omegaBar;
	psi = psiBar;
}


