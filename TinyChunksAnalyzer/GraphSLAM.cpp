#pragma once
#include "stdafx.h"

//turns the given point into a homogeneouse point
cv::Mat GraphSLAMer::makeHomo(cv::Mat x)
{
	int size = x.rows;
	cv::Mat xbar(size + 1, 1, CV_64FC1);
	int i = 0;
	if (x.type() == 4)
	{
		for (; i < size; i++)
		{
			xbar.at<double>(i, 0) = x.at<int>(i, 0);
		}
	}
	else
	{
		for (; i < size; i++)
		{
			xbar.at<double>(i, 0) = x.at<double>(i, 0);
		}
	}
	xbar.at<double>(i, 0) = 1.0;
	return xbar;
}

cv::Mat GraphSLAMer::deHomo(cv::Mat xbar)
{
	int size = xbar.rows;
	int cols = xbar.cols;
	cv::Mat x(size - 1, cols, CV_64FC1);
	for (int j = 0; j < cols; j++)
	{
		int i = 0;
		double scale = xbar.at<double>(size - 1, j);
		for (; i < size - 1; i++)
		{
			x.at<double>(i, j) = xbar.at<double>(i, j) / scale;
		}
	}
	return x;
}

void printMat(Mat &a)
{
	std::cout << a;
	std::cout << std::endl;
}

void printMat(const Mat &a)
{
	std::cout << a;
	std::cout << std::endl;
}

//simple cholesky decomposition
cv::Mat GraphSLAMer::MatrixSqrt(cv::Mat a)
{
	int n = a.cols;
	//printMat(a);
	cv::Mat ret = cv::Mat::eye(n, n, CV_64FC1);
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c <= r; c++)
		{
			//printMat(ret);
			if (c == r)
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					double tval = ret.at<double>(c, j);
					sum += tval * tval;
				}
				double tval = a.at<double>(c, c) - sum;
				ret.at<double>(c, c) = sqrt(tval);
			}
			else
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					double v1 = ret.at<double>(r, j);
					double v2 = ret.at<double>(c, j);
					sum += v1 * v2;
				}
				double v1 = (1.0 / ret.at<double>(c, c));
				double v2 = a.at<double>(r, c);
				ret.at<double>(r, c) = v1 * (v2 - sum);
			}
		}
	}
	return ret;
}

//code roughly from "my math lib"
void GraphSLAMer::InvertLowerTriangluar(cv::Mat &l)
{
	int i, j, k;
	double sum;
	int n = l.rows;

	//Invert the diagonal elements of the lower triangular matrix L.

	for (k = 0; k < n; k++) {
		l.at<double>(k, k) = 1.0 / l.at<double>(k, k);
	}

	//Invert the remaining lower triangular matrix L row by row.

	for (i = 1; i < n; i++)
	{
		for (j = 0; j < i; j++)
		{
			sum = 0.0;
			for (k = j; k < i; k++)
				sum += l.at<double>(i, k) * l.at<double>(k, j);
			l.at<double>(i, j) = -l.at<double>(i, i) * sum;
		}
	}
}

//solve for delta x in (H * dX = -b)
cv::Mat GraphSLAMer::SolveSparseMatrix(cv::Mat H, cv::Mat b)
{
	cv::Mat l = MatrixSqrt(H);
	InvertLowerTriangluar(l);
	//16xi * ixi
	return -b * (l.t() * l);
}

double GraphSLAMer::CalcErrorVal(cv::Mat residuals)
{
	double sum = 0.0;
	for (int i = 0; i < residuals.cols; i++)
	{
		sum += residuals.at<double>(0, i);
	}
	return sum;
}


cv::Mat GraphSLAMer::derivative(SIM3 & cameraPose, KeyFrame & keyframe, cv::Mat & image, double rmean, int bIndexX) //b is our guessed position, x a given pixel
{
	Mat y1(keyframe.quadTreeNodeCount, 1, CV_64FC1);
	Mat y2(keyframe.quadTreeNodeCount, 1, CV_64FC1);
	double ty1, ty2;
	int tries = 0;
	Mat nCamPose;
	Mat nParams = cameraPose.getParameters();
	double * paraIndex = &nParams.ptr<double>(0)[bIndexX];
	paraIndex[0] += alpha[bIndexX];
	cameraPose.setParameters(nParams);
	nCamPose = cameraPose.getlieMatrix();
	y1 = ComputeResiduals(nCamPose, keyframe, image, true, rmean);
	ty1 = ComputeResidualError(nCamPose, keyframe, image, true, rmean);
	paraIndex[0] -= (2 * alpha[bIndexX]);
	cameraPose.setParameters(nParams);
	nCamPose = cameraPose.getlieMatrix();
	y2 = ComputeResiduals(nCamPose, keyframe, image, true, rmean);
	ty2 = ComputeResidualError(nCamPose, keyframe, image, true, rmean);
	paraIndex[0] += alpha[bIndexX];
	cameraPose.setParameters(nParams);
	return (y1 - y2) / (alpha[bIndexX]);
}

cv::Mat computeHessian()
{
	return Mat();
}


cv::Mat GraphSLAMer::ComputeJacobian(SIM3 & cameraPose, KeyFrame keyframe, cv::Mat & image, double rmean, int numRes)//b is our guessed position, x is our pixel info, y is residual
{
	cv::Mat result(keyframe.quadTreeNodeCount, 0, CV_64FC1);
	bool badJacob = false;
	//for the sim(3) vars

	for (int i = 0; i < 6; i++)
	{
		//compute this section of the jacobian and store for jacobian compilation
		cv::Mat tmat = derivative(cameraPose, keyframe, image, rmean, i);
		if (cv::countNonZero(tmat) < 1)
		{
			alpha[i] *= 2;
		}
		result.push_back(tmat);
	}

	return result;
}


cv::Mat GraphSLAMer::ComputeResiduals(Mat & cameraPose, KeyFrame keyframe, cv::Mat & image, bool dcheck, double rmean)//b is our guessed position, x is our pixel info, y is residual
{
	cv::Mat keyTrans;
	keyTrans = keyframe.cameraTransformationAndScaleS.getlieMatrix();

	Mat residuals(1, keyframe.quadTreeNodeCount, CV_64FC1);
	double * resRowIndex = residuals.ptr<double>(0);

	double * cPosePoint0 = cameraPose.ptr<double>(0);
	double * cPosePoint1 = cameraPose.ptr<double>(1);
	double * cPosePoint2 = cameraPose.ptr<double>(2);
	double cameraScale = cameraPose.at<double>(3, 3);

	double * kfPosePoint0 = keyTrans.ptr<double>(0);
	double * kfPosePoint1 = keyTrans.ptr<double>(1);
	double * kfPosePoint2 = keyTrans.ptr<double>(2);
	double kfScale = keyTrans.at<double>(3, 3);

	Mat * processedImage = &keyframe.scaledImageI;


	//for our image
	for (int x = 0; x < keyframe.quadTreeNodeCount; x++)
	{
		QuadTreeNode * leaf = &keyframe.quadTreeLeaves[x];
		//get pixel location with respect to our new frame
		double keyDepthAtU = 1.0 / leaf->mean;
		if (keyDepthAtU == 0 || isinf(keyDepthAtU))
		{
			resRowIndex[x] = 0;
			continue;
		}

		uchar *kfImage = (*processedImage).ptr<uchar>(leaf->position.x);
		//project into world space
		cv::Point * pref = &(leaf->position);
		double xv, yv, zv;
		int xIndex, yIndex;
		projectCameraPointToWorldPointP(kfPosePoint0, kfPosePoint1, kfPosePoint2, kfScale, pref->x, pref->y, keyDepthAtU, xv, yv, zv);
		projectWorldPointToCameraPointU(cPosePoint0, cPosePoint1, cPosePoint2, cameraScale, xv, yv, zv, xIndex, yIndex);
		//do a bounds check, continue if we are out of range
		if (xIndex < 0 || yIndex < 0 || xIndex >= image.rows || yIndex >= image.cols)
		{
			resRowIndex[x] = 0;
			continue;
		}

		//calc photometric residue
		double r = (kfImage[leaf->position.y] - image.ptr<uchar>(xIndex)[yIndex]) / 255.0;
		resRowIndex[x] = HuberNorm((r * r) * (1.0 / leaf->depthDeviation), 3); //I guess 3 is just a maagggiiicc number (2 sources doing visual odometry have used it so who am I to question it?)
	}
	return residuals;
}


double GraphSLAMer::ComputeResidualError(Mat & cameraPose, KeyFrame keyframe, cv::Mat & image, bool dcheck, double rmean)//b is our guessed position, x is our pixel info, y is residual
{
	cv::Mat keyTrans;
	keyTrans = keyframe.cameraTransformationAndScaleS.getlieMatrix();

	double residual = 0;
	int residualCount = 0;

	double * cPosePoint0 = cameraPose.ptr<double>(0);
	double * cPosePoint1 = cameraPose.ptr<double>(1);
	double * cPosePoint2 = cameraPose.ptr<double>(2);
	double cameraScale = cameraPose.at<double>(3, 3);

	double * kfPosePoint0 = keyTrans.ptr<double>(0);
	double * kfPosePoint1 = keyTrans.ptr<double>(1);
	double * kfPosePoint2 = keyTrans.ptr<double>(2);
	double kfScale = keyTrans.at<double>(3, 3);

	Mat * processedImage = &keyframe.scaledImageI;


	//for our image
	for (int x = 0; x < keyframe.quadTreeNodeCount; x++)
	{
		QuadTreeNode * leaf = &keyframe.quadTreeLeaves[x];
		//get pixel location with respect to our new frame
		double keyDepthAtU = 1.0 / leaf->mean;
		if (keyDepthAtU == 0 || isinf(keyDepthAtU)) continue;

		uchar *kfImage = (*processedImage).ptr<uchar>(leaf->position.x);
		//project into world space
		cv::Point * pref = &(leaf->position);
		double xv, yv, zv;
		int xIndex, yIndex;
		projectCameraPointToWorldPointP(kfPosePoint0, kfPosePoint1, kfPosePoint2, kfScale, pref->x, pref->y, keyDepthAtU, xv, yv, zv);
		projectWorldPointToCameraPointU(cPosePoint0, cPosePoint1, cPosePoint2, cameraScale, xv, yv, zv, xIndex, yIndex);
		//do a bounds check, continue if we are out of range
		if (xIndex < 0 || yIndex < 0 || xIndex >= image.rows || yIndex >= image.cols) continue;

		//calc photometric residue
		double r = (kfImage[leaf->position.y] - image.ptr<uchar>(xIndex)[yIndex]) / 255.0;
		residual += r * r;
		residualCount++;
	}
	if (residualCount <= 10) return 1e+300;
	return residual;
}

//predicts new position based on dampened approximate velocity
void GraphSLAMer::forwardPredictPosition(cv::Mat &lastPos, cv::Mat &Velocity)
{
	double dt = 0.5;
	double a = 0.85, b = 0.005;

	double xk, vk, rk;
	double xm;

	for (int i = 0; i < 6; i++)
	{
		xm = lastPos.at<double>(0, i);
		vk = Velocity.at<double>(0, i);
		xk = xm + (vk * dt);

		rk = xm - xk;

		xk += a * rk;
		vk += (b * rk) / dt;

		lastPos.at<double>(0, i) = xk;
		Velocity.at<double>(0, i) = vk;
	}
}



//puts the projected point from pi into camera space
void GraphSLAMer::projectWorldPointToCameraPointU(double * poseRow1, double * poseRow2, double * poseRow3, double poseScale, double inX, double inY, double inZ, int & returnX, int & returnY)
{
	//apply reverse rotation
	inX = (inX * poseRow1[0] + inX * poseRow1[1] + inX * poseRow1[2]) * poseScale;
	inY = (inY * poseRow2[0] + inY * poseRow2[1] + inY * poseRow2[2]) * poseScale;
	inZ = (inZ * poseRow3[0] + inZ * poseRow3[1] + inZ * poseRow3[2]) * poseScale;


	//apply reverse translation
	inX -= poseRow1[3] * poseScale;
	inY -= poseRow2[3] * poseScale;
	inZ -= poseRow3[3] * poseScale;

	//3dx * fx_d / depth + cx_d = x
	returnX = inX * fx_d / inZ + cx_d;
	returnY = inY * fy_d / inZ + cy_d;
}

//puts the projected point from pi into camera space
void GraphSLAMer::projectCameraPointToWorldPointP(double * poseRow1, double * poseRow2, double * poseRow3, double poseScale, int inX, int inY, double depth, double & returnX, double & returnY, double & returnZ)
{
	returnX = (inX - cx_d) * depth / fx_d;
	returnY = (inY - cy_d) * depth / fy_d;
	returnZ = depth;

	//apply rotation
	returnX = (returnX * poseRow1[0] + returnX * poseRow2[0] + returnX * poseRow3[0]) * poseScale;
	returnY = (returnY * poseRow1[1] + returnY * poseRow2[1] + returnY * poseRow3[1]) * poseScale;
	returnZ = (returnZ * poseRow1[2] + returnZ * poseRow2[2] + returnZ * poseRow3[2]) * poseScale;

	//apply translation
	returnX += poseRow1[3] * poseScale;
	returnY += poseRow2[3] * poseScale;
	returnZ += poseRow3[3] * poseScale;
}


double GraphSLAMer::HuberNorm(double x, double epsilon)
{
	if (abs(x) <= epsilon) return (x * x) / (2 * epsilon);
	return abs(x) - (epsilon / 2.0);
}


cv::Mat GraphSLAMer::applyVarianceWeights(cv::Mat & jacobianTranspose, KeyFrame kf)
{
	int residualSize = jacobianTranspose.cols;
	int rowSize = jacobianTranspose.rows;
	cv::Mat results(rowSize, residualSize, CV_64FC1);
	for (int r = 0; r < rowSize; r++)
	{
		double * rowPtr = jacobianTranspose.ptr<double>(r);
		double * resPtr = results.ptr<double>(r);
		for (int c = 0; c < residualSize; c++)
		{
			resPtr[c] = rowPtr[c] * (1.0 / kf.quadTreeLeaves[c].depthDeviation);
			if (isinf(resPtr[c]) || isnan(resPtr[c]))
			{
				resPtr[c] = resPtr[c];
			}
		}
	}
	return results;
}

//computes the update
cv::Mat GraphSLAMer::TransformJacobian(cv::Mat & jacobian, KeyFrame kf, cv::Mat residuals)
{
	//double max = 0, min = 0;
	//cv::minMaxIdx(jacobian, &min, &max);
	//jacobian = (jacobian - min) / max;
	//findDependantVals(jacobian);
	//printMat(JT);
	cv::Mat ojtj = jacobian * jacobian.t();
	cv::Mat JTJ = applyVarianceWeights(jacobian, kf) *  jacobian.t(); // because our javobian is already the wrong way lol
	//printMat(JTJ);
	//invert using decomposition then lower matrix inverse
	//cv::Mat l = MatrixSqrt(JTJ);
	//std::cout << l << std::endl;
	//InvertLowerTriangluar(l);

	//we use the transpose because the hessian of of a vector jacobian.t * a vector jacobian = H.t
	cv::Mat JTJi = -JTJ.inv();//l*l.t(); // (JT * J)^-1

	std::cout << cv::determinant(JTJ) << std::endl;
	//printMat(JTJi);

	cv::Mat JTJiJT = JTJi * jacobian; // (JT * J)^-1 * JT
	return applyVarianceWeights(JTJiJT, kf) * residuals.t(); // (JT * J)^-1 * JT * r
}


GraphSLAMer::SIM3 GraphSLAMer::CalcGNPosOptimization(cv::Mat & image, KeyFrame keyframe)
{
	//set initial camera pose
	SIM3 cameraPose(keyframe.cameraTransformationAndScaleS.getRotationMat(), keyframe.cameraTransformationAndScaleS.getTranslation(), keyframe.cameraTransformationAndScaleS.getParameters());

	//run gauss-newton optimization
	Mat camPose = cameraPose.getlieMatrix();
	Mat oldCamParas;
	double rmean = 0;
	//ComputeMedianResidualAndCorrectedPhotometricResiduals(cameraPose, image, keyframe, rmean);
	double residualSum = 1;//ComputeResidualError(camPose, keyframe, image, false, rmean);
	//if (residualSum < 0.010) return keyframe.cameraTransformationAndScaleS;
	double oldResidual = 0;
	double lambda = 1;
	while (fabs(oldResidual - residualSum) / oldResidual > 0.01)//while we have not converged
	{
		oldResidual = residualSum;

		camPose = cameraPose.getlieMatrix();
		//calculate all residuals and the sum

		//calculate error with current residuals
		//cv::Mat errorVec = CalcErrorVec(residuals);
		cv::Mat residuals = ComputeResiduals(camPose, keyframe, image, false, rmean);
		double error = CalcErrorVal(residuals);
		residualSum = error;

		if (error > 1.0e+200)
		{
			std::cout << "hwat";
		}

		//update pose estimate
		cv::Mat jacobianRes = ComputeJacobian(cameraPose, keyframe, image, rmean, image.cols * image.rows);

		//calculate deltax from derivatives
		cv::Mat deltaX = TransformJacobian(jacobianRes, keyframe, residuals).t();
		//SolveSparsecv::Matrix(H + lambda, b);
		double lnerror = 0;

		cameraPose = SIM3(deltaX) * cameraPose;

		/*cv::Mat deltaX = (rmean - error) / jacobianMat;
		SIM3 oldCam(cameraPose.getRotationMat(), cameraPose.getTranslation(), cameraPose.getParameters());
		for (int i = 5; i >= 0; i--)
		{
			Mat selectedvalue = cv::Mat::zeros(1, 6, CV_64FC1);
			int fcount = 0;
			lambda = 0.01;
			selectedvalue.at<double>(0,i) = deltaX.at<double>(0, i);
			while (fcount < 10)
			{
				cameraPose.addParameters(selectedvalue * lambda);
				camPose = cameraPose.getlieMatrix();

				double nerror = ComputeResidualError(camPose, keyframe, image, false, rmean);

				fcount++;

				lnerror = nerror;

				if (error <= nerror)
				{
					cameraPose = oldCam;
					lambda /= 2;
				}
				else
				{
					error = nerror;
					oldCam = cameraPose;
				}
			}
		}*/
	}
	return cameraPose;
}

double getPercent(double x1, double x2)
{
	if (x1 > x2)
	{
		return (x1 - x2) / x1;
	}
	else
	{
		return (x2 - x1) / x2;
	}
}


void GraphSLAMer::ComputeQuadtreeForKeyframe(KeyFrame &kf)
{
	double thresholdSquared = 0.01;// 0.01;//10% post square
	cv::Mat image = kf.scaledImageI;
	int numPixles = image.rows * image.cols;
	int treeSize = numPixles;
	int tpixels = numPixles;
	for (int i = 0; i < quadTreeDepth; i++)
	{
		tpixels /= 4;
		treeSize += tpixels;
	}
	int index = treeSize - numPixles;
	std::vector<QuadTreeNode> nodes(treeSize);

	//first set up our image pyramid
	int prows = image.rows;
	int pcols = image.cols;
	cv::Mat lastImage = image;

	//push back original image
	cv::Mat pimage(prows, pcols, CV_8UC1);
	for (int x = 0; x < prows; x++)
	{
		for (int y = 0; y < pcols; y++)
		{
			pimage.at<uchar>(x, y) = lastImage.at<uchar>(x, y);
		}
	}
	lastImage = pimage.clone();
	kf.pyramid.push_back(pimage.clone());

	//create power pyramid
	for (int i = 0; i < quadTreeDepth; i++)
	{
		cv::Mat pimage(prows / 2, pcols / 2, CV_8UC1);
		for (int x = 0; x < prows; x += 2)
		{
			for (int y = 0; y < pcols; y += 2)
			{
				double avg = 0;
				for (int j = 0; j < 2; j++)
				{
					for (int k = 0; k < 2; k++)
					{
						avg += lastImage.at<uchar>(x + j, y + k);
					}
				}
				pimage.at<uchar>(x / 2, y / 2) = avg / 4.0;
			}
		}
		lastImage = pimage.clone();
		double val = lastImage.at<uchar>(0, 0);
		double val2 = pimage.at<uchar>(0, 0);
		kf.pyramid.push_back(pimage.clone());
		prows /= 2;
		pcols /= 2;
	}

	//place image into quadtree
	for (int x = 0; x < image.rows; x += 2)
	{
		for (int y = 0; y < image.cols; y += 2)
		{
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					QuadTreeNode leaf;
					leaf.numChildren = 0;
					leaf.strikes = 0;
					leaf.avgIntensity = image.at<uchar>(x + i, y + j);

					//calculate and store gradient
					double x1, x2, y1, y2;
					//bounds check gradiant, use current value if tripped
					if (x + i - 1 > 0) x1 = image.at<uchar>(x + i - 1, y + j);
					else x1 = leaf.avgIntensity;

					if (x + i + 1 < image.rows) x2 = image.at<uchar>(x + i + 1, y + j);
					else x2 = leaf.avgIntensity;

					if (y + j - 1 > 0) y1 = image.at<uchar>(x + i, y + j - 1);
					else y1 = leaf.avgIntensity;

					if (y + j + 1 < image.cols) y2 = image.at<uchar>(x + i, y + j + 1);
					else y2 = leaf.avgIntensity;

					leaf.xGradient = x1 - x2;
					leaf.yGradient = y1 - y2;

					leaf.layer = 0;
					leaf.position = Point(x + i, y + j);
					leaf.width = 1;
					leaf.length = 1;

					double savg = (x1 + x2 + y1 + y2) / 4;

					double percent = getPercent(savg, leaf.avgIntensity);
					percent *= percent;
					if (percent > thresholdSquared)
					{
						leaf.fLeaf = true;
					}
					else leaf.fLeaf = false;
					leaf.valid = true;

					nodes[index] = leaf;
					index++;
				}
			}
		}
	}

	//construct higher levels of the quad tree
	int groupSizeX = image.rows;
	int groupSizeY = image.cols;
	index = treeSize - numPixles;
	std::vector<QuadTreeNode> finalNodes;
	for (int l = 0; l < quadTreeDepth; l++)
	{
		groupSizeX /= 2;
		groupSizeY /= 2;
		int indexOff = 0;
		int curGroupSize = groupSizeX * groupSizeY;
		for (int x = 0; x < groupSizeX; x += 2)
		{
			for (int y = 0; y < groupSizeY; y += 2)
			{
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						QuadTreeNode branch;
						branch.numChildren = 4;
						double avgIntensity = 0.0;
						double xGradient = 0;
						double yGradient = 0;
						branch.fLeaf = false;
						bool fleafSkip = false; //check if we can skip rollup 

						//index is the start of the lower layer
						//xyOffset is the offset from our x and y values, every X must skip double group size of y, and y is also doubled
						//this is to account fort he lower level being twice the size of the current
						//the i and j offsets are to vary our insertion order, they ca be directly applied to x and y
						int xyOffset = ((x + i) * 4 * groupSizeY) + 4 * (y + j);

						//set children from the lower group
						for (int k = 0; k < 4; k++)
						{

							//our sub offset k must read the first 4 values in order, the branches are ordered in the proper for mat on insertion
							branch.children[k] = &nodes[index + xyOffset + k];
							branch.children[k]->parent = &branch;
							avgIntensity += branch.children[k]->avgIntensity;
							xGradient += branch.children[k]->xGradient;
							yGradient += branch.children[k]->yGradient;
							//if any of the children are final leaves, we cannot roll up any further
							if (branch.children[k]->fLeaf)
							{
								fleafSkip = true;
							}

						}
						branch.avgIntensity = avgIntensity / 4.0;
						branch.layer = l + 1;
						branch.strikes = 0;
						//this is an approximation and WRONG, in fact we should be using max found instead of avg
						//but this might be close enough, we can check later
						branch.xGradient = xGradient / 4;
						branch.yGradient = yGradient / 4;

						if (fleafSkip)
						{
							//since we are skipping, add all non-fleaf children to the final vector and set this node's fleaf value to true
							for (int k = 0; k < 4; k++)
							{
								if (branch.children[k]->fLeaf && branch.children[k]->valid)
								{
									finalNodes.push_back(*branch.children[k]);
								}
								else if (!branch.children[k]->fLeaf)
								{
									branch.children[k]->fLeaf = true;
									branch.children[k]->valid = true;
									finalNodes.push_back(*branch.children[k]);
								}
							}
							branch.fLeaf = true;
							branch.valid = false;
						}
						else
						{
							//set branch position and structure info
							branch.position = branch.children[0]->position;
							branch.length = branch.children[0]->length * 2;
							branch.width = branch.children[0]->width * 2;

							//do the threshold check for trim otherwise
							//check if we need to trim branch
							bool trim = true;
							for (int i = 0; i < 4; i++)
							{
								double percent = getPercent(branch.children[i]->avgIntensity, branch.avgIntensity);
								percent *= percent;
								if (percent > thresholdSquared)
								{
									trim = false;
									break;
								}
							}

							//trim if nesseccary
							if (trim)
							{
								branch.numChildren = 0;
							}
							else
							{
								//check if final leaf, and add to final vector if true
								//a branch is a final leaf if it is the first branch from the end not to be trimmed
								//wait, to get here we must have not encountered any fleaf children (see above) so we don;t need a check lol
								branch.fLeaf = true;
								branch.valid = true;
							}

						}

						//store branch in proper group pattern (00,01,10,11)
						//2 in a row, skip y size, do 2
						nodes[(index - curGroupSize) + (x * groupSizeY) + (2 * y) + (i * 2) + j] = branch;
						indexOff++;
					}
				}

			}
		}
		index -= curGroupSize;
	}

	//default case for when we are looking at basicly a solid colour (or testing why things are breaking ;) )
	//changing this in order to  search the final top level and add any nodes that are not yet fleafed
	//if (finalNodes.size() == 0)
	{
		int l = quadTreeDepth - 1;
		for (int x = 0; x < groupSizeX * groupSizeY; x++)
		{
			if (!nodes[x].fLeaf)
			{
				nodes[x].valid = true;
				nodes[x].fLeaf = true;
				finalNodes.push_back(nodes[x]);
			}
		}
	}

	//now the finalNodes vector contains all our leaves that we are using in the keyframe
	kf.quadTreeNodeCount = finalNodes.size();
	kf.quadTreeLeaves = new QuadTreeNode[kf.quadTreeNodeCount];
	std::copy(finalNodes.begin(), finalNodes.end(), kf.quadTreeLeaves);
}

double maxDepth = 0;;
//calculates the depths by comparing the image, after plcement into a power of 2 pyramid, against the keyframe quadtree leaves
void GraphSLAMer::computeDepthsFromStereoPair(KeyFrame & kf, cv::Mat & image, cv::Mat & cameraParams, SIM3 cameraPos, bool initialize)
{
	int prows = image.rows;
	int pcols = image.cols;
	std::vector<cv::Mat> pyramid;
	cv::Mat lastImage = image;

	//push back original image
	cv::Mat pimage(prows, pcols, CV_8UC1);
	for (int x = 0; x < prows; x++)
	{
		for (int y = 0; y < pcols; y++)
		{
			pimage.at<uchar>(x, y) = lastImage.at<uchar>(x, y);
		}
	}
	lastImage = pimage.clone();
	pyramid.push_back(pimage.clone());

	//create power pyramid
	for (int i = 0; i < quadTreeDepth; i++)
	{
		cv::Mat pimage(prows / 2, pcols / 2, CV_8UC1);
		for (int x = 0; x < prows; x += 2)
		{
			for (int y = 0; y < pcols; y += 2)
			{
				double avg = 0;
				for (int j = 0; j < 2; j++)
				{
					for (int k = 0; k < 2; k++)
					{
						avg += lastImage.at<uchar>(x + j, y + k);
					}
				}
				pimage.at<uchar>(x / 2, y / 2) = avg / 4.0;
			}
		}
		lastImage = pimage.clone();
		double val = lastImage.at<uchar>(0, 0);
		double val2 = pimage.at<uchar>(0, 0);
		pyramid.push_back(pimage.clone());
		prows /= 2;
		pcols /= 2;
	}

	//first generate the fundamental matrix
	//get offset from keyframe to image
	cv::Mat translation = kf.cameraTransformationAndScaleS.getTranslation() - cameraPos.getTranslation();
	cv::Mat rotation = kf.cameraTransformationAndScaleS.getRotationMat() - cameraPos.getRotationMat();
	double* transPtr = translation.ptr<double>(0);


	double sum = 0;
	double direction = (transPtr[0] > 0) - (transPtr[0] < 0);

	if (!initialize)
	{
		sum = transPtr[0] * transPtr[0];
		sum += transPtr[1] * transPtr[1];
		sum += transPtr[2] * transPtr[2];
	}
	else sum = 0.001; //for initialization purposes

	double baseline = sqrt(sum);

	//see baseline disparity depth calculation
	//we may need to recitfy our image sections, but the paper says the difference is small enough not to matter
	double focalXTimesBase = fx_d * baseline;


	//extract s = promote translate from vector to mat in cross multiply format
	cv::Mat S = cv::Mat::zeros(3, 3, CV_64FC1);

	if (!initialize)
	{
		S.at<double>(0, 0) = 0;
		S.at<double>(1, 0) = transPtr[2];
		S.at<double>(2, 0) = -transPtr[1];

		S.at<double>(0, 1) = -transPtr[2];
		S.at<double>(1, 1) = 0;
		S.at<double>(2, 1) = transPtr[0];

		S.at<double>(0, 2) = transPtr[1];
		S.at<double>(1, 2) = -transPtr[0];
		S.at<double>(2, 2) = 0;
	}
	else
	{
		//S = cv::Mat::zeros(3, 3, CV_64FC1);
		S.at<double>(2, 1) = 0.01; //we can say it moved 1 x unit
		S.at<double>(1, 2) = -0.01; //we can say it moved 1 x unit
		//S.at<double>(0, 2) = 1; //we can say it moved 1 y unit
		//S.at<double>(2, 0) = -1; //we can say it moved 1 y unit
	}



	//no initial rotation
	cv::Mat R;

	if (initialize)
	{
		R = cv::Mat::eye(3, 3, CV_64FC1);
	}
	else
	{
		R = cv::Mat(3, 3, CV_64FC1);
		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				R.at<double>(x, y) = rotation.at<double>(x, y);
	}

	//calculate Mi inverse
	//3x3 * 3x4 = 3x4
	//cv::Mat imageParamsTimesPose = cameraParams * cameraPos.getExtrinsicMatrix();
	//cv::Mat imageParamsTimesPoseInv = imageParamsTimesPose.inv();

	//Construct F = Mk^(-T)EMi^(-1), E = RS, M = intrinsic camera params
	//3x3 * (3x3 * 3x3) * 3x3
	cv::Mat E = (S * R);
	//cv::Mat test = cameraParams * cameraParamsInv;
	cv::Mat F = cameraParams.t() * E * cameraParams;

	double totalInvDepth = 0.0;

	//for each fleaf in the keyframe we search for a match
	for (int i = 0; i < kf.quadTreeNodeCount; i++)
	{

		//extract the leaf
		QuadTreeNode leaf = kf.quadTreeLeaves[i];

		double pixelMod = imageScale * pow(2, leaf.layer);


		//skip if the node is invalid
		if (!leaf.valid) continue;

		//store the value we will be comparing against
		double kValue = leaf.avgIntensity;

		//extract the image pyramid layer we will be comparing against
		cv::Mat pimage = pyramid[leaf.layer];

		//get the run of pixels we need for our leaf
		double kfPixels[5];
		int kfRightShift = 5 - leaf.position.y / leaf.length;
		int kfLeftShift = pimage.cols - (leaf.position.y / leaf.length + 5);
		int shift = 0.0;
		if (kfLeftShift < 0) shift = kfLeftShift;
		else if (kfRightShift > 0) shift = kfRightShift;
		for (int p = 0; p < 5; p++)
		{
			kfPixels[p] = kf.pyramid[leaf.layer].ptr<uchar>(leaf.position.x / leaf.length)[leaf.position.y / leaf.length + p + shift];
		}


		//calculate line equation
		cv::Mat position(1, 3, CV_64FC1);
		position.at<double>(0, 0) = (leaf.position.y * imageScale);// +(320 / imageScale);
		position.at<double>(0, 1) = (leaf.position.x * imageScale);// +(240 / imageScale);
		position.at<double>(0, 2) = 1;
		cv::Mat lineParams = position * F;

		//store constant values for epipolar line
		double a = lineParams.at<double>(0, 0);
		double b = lineParams.at<double>(0, 1);
		double c = lineParams.at<double>(0, 2);

		//values for storing our max and pixel position
		int bestPosX;
		int bestPosY;
		double minSSD = std::numeric_limits<double>::infinity();

		//to store locations of points along the epipolar line
		int goodCount = 0;
		double * eLineVals = new double[pimage.cols];
		int * eLineLocsX = new int[pimage.cols];
		int * eLineLocsY = new int[pimage.cols];
		//for the entire epipolar line try and find our best match
		//changing this to search the proper region
		/*for (int x = 0; x < pimage.cols; x++)
		{
			int y = (((-x * pixelMod * a) - c) / b) / pixelMod;*/
		double lineConst = -c / b;
		double yIncrement = -a / b;
		double infDepthPoint = (lineConst +  (leaf.position.x * pixelMod) * yIncrement)/pixelMod;
		int maxOffset = pimage.cols;
		if (leaf.depth != 0)
		{
			maxOffset = leaf.depth + 2 * leaf.depthDeviation;
			yIncrement *= direction;
		}
		int y = infDepthPoint;
		int x = leaf.position.x;
		int xIncrement = direction;
		for (int lineInd = 0; lineInd < maxOffset; lineInd++)
		{
			maxOffset += yIncrement;
			y = maxOffset;
			x += xIncrement;

			/*Matx31d np(x, fullY, 1);

			Mat hz = position.t() * F * Mat(np);

			if (hz.at<double>(0) > 0.0001)
			{
				hz = hz;
			}*/
			//YES THANK YOU HZ


			if (y < 0 || y >= pimage.rows)
			{
				break;
			}


			//grab and store entire row
			eLineVals[goodCount] = pimage.ptr<uchar>(y)[x];
			eLineLocsX[goodCount] = y;
			eLineLocsY[goodCount] = x;
			goodCount++;
		}

		int numValues = goodCount - 5;
		for (int x = 0; x < numValues; x++)
		{
			double curSSD = 0;
			for (int y = 0; y < 5; y++)
			{
				double diff = eLineVals[x + y] - kfPixels[y];
				curSSD += sqrt(diff * diff);
			}

			//update our min if we need too
			if (curSSD < minSSD)
			{
				bestPosX = eLineLocsX[x] - shift;
				bestPosY = eLineLocsY[x];
				minSSD = curSSD;
			}
		}

		if (isinf(minSSD))
		{
			minSSD = minSSD;
		}

		delete[] eLineVals;
		delete[] eLineLocsX;
		delete[] eLineLocsY;

		//kf.quadTreeLeaves[i].depth = minSSD;
		//continue;

		//we now have our best ssd value and the most likley location
		//thus we can kalman update our depth map and variances,
		//or if the ssd value is too large put a strike against the current leaf
		//Finally, if a leaf has too many strikes we rule it invalid
		if (minSSD < 70) //arbitrary threshold, currently
		{

			//calculate pixel diff
			//this is the vector created from the camera centers towards the pixels selected, subtracted from one another
			double xDiff = abs(leaf.position.y * imageScale - bestPosY * pixelMod);


			//see baseline disparity depth calculation
			//we may need to recitfy our image sections, but the paper says the difference is small enough not to matter
			double depth = 1 / ((focalXTimesBase) / xDiff);

			totalInvDepth += depth;

			if (depth > maxDepth) maxDepth = depth;

			if (depth == 0 || isnan(depth))
			{
				depth = depth;
			}

			//if depth is uninitialized just set it
			if (initialize)
			{
				if (isnan(depth))
				{
					kf.quadTreeLeaves[i].depth = 0; // undefined, no paralax
					kf.quadTreeLeaves[i].depthDeviation = rand() + 1;//we can attenuate or strengthen sensor dependancy later
					kf.quadTreeLeaves[i].mean = 0;
				}
				else
				{
					kf.quadTreeLeaves[i].depth = depth;
					kf.quadTreeLeaves[i].depthDeviation = rand() + 1;//we can attenuate or strengthen sensor dependancy later
					kf.quadTreeLeaves[i].mean = depth;
				}
			}
			else
			{
				//>hurf durf use a kalman filter for a single value
				//no
				//update depth based on variance, and update variance too
				if (!isnan(depth))
				{
					//calc mean
					kf.quadTreeLeaves[i].updateCount++;
					double curMean = kf.quadTreeLeaves[i].mean;
					double newMean = +(1 / kf.quadTreeLeaves[i].updateCount) * (depth - curMean);
					kf.quadTreeLeaves[i].depthDeviation = ((kf.quadTreeLeaves[i].depthDeviation * (kf.quadTreeLeaves[i].updateCount - 1)) + (depth - newMean)) / kf.quadTreeLeaves[i].updateCount;
					if (kf.quadTreeLeaves[i].depthDeviation == 0 || isnan(kf.quadTreeLeaves[i].depthDeviation))
					{
						kf.quadTreeLeaves[i].depthDeviation = kf.quadTreeLeaves[i].depthDeviation;
					}
					kf.quadTreeLeaves[i].mean = newMean;
					kf.quadTreeLeaves[i].depth = depth;
					//may need to change this to a more advanced filter later, but we'll try this for now
				}
			}
		}
		else
		{
			//mark this node with a strike
			kf.quadTreeLeaves[i].strikes++;
			if (kf.quadTreeLeaves[i].strikes > 10)//another arbitrary threshold...
			{
				kf.quadTreeLeaves[i].valid = false;
			}
		}
	}

	//normalize depths
	for (int i = 0; i < kf.quadTreeNodeCount; i++)
	{
		//kf.quadTreeLeaves[i].depth /= 
	}
}

void GraphSLAMer::projectDepthNodesToDepthMap(KeyFrame & kf)
{
	std::list<int> invalidChunks;
	std::list<int> validChunks;
	std::list<int> retryList;
	for (int i = 0; i < kf.quadTreeNodeCount; i++)
	{
		//paper uses interpolation for assignment, lets try skipping it for now, and we can use our own fast poly algo later
		QuadTreeNode * qtn = &kf.quadTreeLeaves[i];
		int x = qtn->position.x;
		int y = qtn->position.y;
		int xSize = qtn->position.x + qtn->length;
		int ySize = qtn->position.y + qtn->width;

		//only set if valid
		if (qtn->valid)
		{
			for (; x < xSize; x++)
			{
				for (y = qtn->position.y; y < ySize; y++)
				{
					kf.inverseDepthD.at<double>(x, y) = qtn->mean;
				}
			}
			validChunks.push_back(i);
		}
		else
		{
			invalidChunks.push_back(i);
		}
	}

	//keeping trying to interpolate nodes until all have a value for everything
	int lastSize = 0;
	while (invalidChunks.size() != lastSize)
	{
		lastSize = invalidChunks.size();
		auto it = std::begin(invalidChunks);
		while (it != std::end(invalidChunks))
		{
			QuadTreeNode * qtn = &kf.quadTreeLeaves[*it];

			//find surrounding nodes
			double avgDepth = 0.0;
			int invalidCount = 0;
			int invalidLimit = qtn->length * 2;
			int numValues = qtn->length * 4;

			//west side
			int tlength = qtn->position.y + qtn->length;
			cv::Point2f pt(qtn->position.x, qtn->position.y);
			if (pt.x == 0)
			{
				pt.x = 1;
				pt.y = tlength;
				numValues -= qtn->length;
			}
			else
			{
				pt.x -= 1;

				for (; pt.y < tlength; pt.y++)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}
				pt.x++;
			}


			//south side
			tlength = qtn->position.x + qtn->length;
			if (pt.y == kf.inverseDepthD.cols)
			{
				pt.x = tlength;
				pt.y = kf.inverseDepthD.rows - 1;
				numValues -= qtn->length;
			}
			else
			{
				for (; pt.x < tlength; pt.x++)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}

				if (invalidCount > invalidLimit) { it++;  continue; };
				pt.y--;
			}


			//east side
			tlength = qtn->position.y;
			if (pt.x == kf.inverseDepthD.rows)
			{
				pt.y = tlength;
				pt.x = qtn->position.x;
				numValues -= qtn->length;
			}
			else
			{
				for (; pt.y > tlength; pt.y--)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}

				if (invalidCount > invalidLimit) { it++;  continue; }
				pt.x--;
			}

			//north side
			if (pt.y != 0)
			{
				tlength = qtn->position.x;
				for (; pt.x > tlength; pt.x--)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}

				if (invalidCount > invalidLimit) { it++;  continue; };
			}
			else
			{
				numValues -= qtn->length;
			}


			//set to interpolated value(gradiant?)
			qtn->depth = avgDepth / numValues;

			//remove from invalid list and add to retry list and valid list
			qtn->valid = true;
			qtn->strikes = 0;
			retryList.push_back(*it);
			it = invalidChunks.erase(it);
		}
	}

	//cycle through the retry list and project all the depths back onto the depthmap 
	for (int i : retryList)
	{
		QuadTreeNode * qtn = &kf.quadTreeLeaves[i];
		int x = qtn->position.x;
		int y = qtn->position.y;
		int xSize = qtn->position.x + qtn->length;
		int ySize = qtn->position.y + qtn->width;
		//set up covariance matrix for this node
		for (int j = 0; j < kf.quadTreeNodeCount; j++)
		{
			QuadTreeNode * subNode = &kf.quadTreeLeaves[j];
		}
		for (; x < xSize; x++)
		{
			for (; y < ySize; y++)
			{
				kf.inverseDepthD.at<double>(x, y) = 1 / qtn->mean;
			}
		}
	}


}


//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
//K: is a 3x3 real mat with the camera parameters
//pi: perspective projection function
double increment = 0;
GraphSLAMer::SIM3 GraphSLAMer::LS_Graph_SLAM(cv::Mat cameraFrame)
{
	bool makeNewKeyframe;

	Mat magFrame;
	cvtColor(cameraFrame, magFrame, CV_BGR2GRAY);


	//find most likley position of camera based on last keyframe
	SIM3 position = CalcGNPosOptimization(magFrame, lastKey);

	if (!(cv::countNonZero(position.getParameters() != lastKey.cameraTransformationAndScaleS.getParameters()) == 0))
	{
		//construct depth quadtrees based on the stereo pairs
		computeDepthsFromStereoPair(lastKey, cameraFrame, cameraParams, position);

		//convert the nodes into a pixel map
		projectDepthNodesToDepthMap(lastKey);
	}

	//run makenewkeyframe check against image quality
	//makeNewKeyframe = !(cv::countNonZero(position.getParameters() != lastKey.cameraTransformationAndScaleS.getParameters()) == 0);
	makeNewKeyframe = false;
	if (makeNewKeyframe)
	{

		KeyFrame newKey;

		newKey.cameraTransformationAndScaleS = position;

		//add image te new  keyframe
		newKey.scaledImageI = cameraFrame;


		//set variance to one
		newKey.inverseDepthD = cv::Mat::zeros(cameraFrame.rows, cameraFrame.cols, CV_64FC1);

		//computes the power tree for the image, allowing for fast analysis 
		ComputeQuadtreeForKeyframe(newKey);

		//construct depth quadtrees based on the stereo pairs
		computeDepthsFromStereoPair(newKey, lastKey.scaledImageI, cameraParams, lastKey.cameraTransformationAndScaleS);

		newKey.depthVarianceV = cv::Mat::ones(cameraFrame.rows, cameraFrame.cols, CV_64FC1);

		//convert the nodes into a pixel map AND create information amtrix
		projectDepthNodesToDepthMap(newKey);

		//loop closure check (TO DO)

		keyframes.V.push_back(newKey);
		keyframes.E.push_back(position.getlieMatrix());

		//add new keyframe and constraints to list

		lastKey = newKey;
	}

	std::cout << "Current estimated camera position:" << std::endl << position.getParameters() << std::endl << std::endl;

	return position;
}

//Sets up matrices and other things
void GraphSLAMer::Initialize_LS_Graph_SLAM(cv::Mat cameraFrame, cv::Mat cameraFrame2)
{
	srand(1111);

	//initialize our pixel scale relative to the incoming frame
	imageScale = (2 * cameraParams.at<double>(0, 2)) / cameraFrame.cols;

	//initialize lastKey
	KeyFrame newKey;

	//set invDepth
	newKey.inverseDepthD = cv::Mat::ones(cameraFrame.rows, cameraFrame.cols, CV_64FC1);

	//set position to 0,0,0

	//add image te new  keyframe

	cvtColor(cameraFrame, newKey.scaledImageI, CV_BGR2GRAY);
	newKey.origImage = cameraFrame;

	//computes the power tree for the image, allowing for fast analysis 
	ComputeQuadtreeForKeyframe(newKey);


	SIM3 position;
	computeDepthsFromStereoPair(newKey, cameraFrame2, cameraParams, position, true);


	projectDepthNodesToDepthMap(newKey);

	//initialize posegraph
	keyframes = PoseGraph();

	lastKey = newKey;

	keyframes.E.push_back(position.getlieMatrix());
	keyframes.V.push_back(lastKey);

	std::ofstream myfile;
	myfile.open("./depths.txt");
	//cycle through the depth maps, converting the depths into points using the camera position
	for (int kfi = 0; kfi < keyframes.V.size(); kfi++)
	{

		cv::Mat depths = 1.0 / keyframes.V[kfi].inverseDepthD;
		for (int px = 0; px < depths.rows; px++)
		{
			for (int py = 0; py < depths.cols; py++)
			{
				double depth = depths.at<double>(px, py);
				myfile << depth << ", ";
			}
			myfile << std::endl;
		}
	}
	myfile.close();

	//initlalize alpha values
	alpha[0] = 0.00001 / 3.141592;
	alpha[1] = 0.00001 / 3.141592;
	alpha[2] = 0.00001 / 3.141592;
	alpha[3] = 0.00001;
	alpha[4] = 0.00001;
	alpha[5] = 0.00001;
	alpha[6] = 0.00001;
}

int cloudOffset = 0;

//passes over keyframes and constraints and returns a list of points
void GraphSLAMer::get3dPointsAndColours(std::vector<cv::Point3d> & pcloud_est, std::vector<cv::Vec3b> & colours)
{

	pcloud_est.clear();
	colours.clear();
	std::ofstream myfile;
	//myfile.open("./depths.txt");
	//cycle through the depth maps, converting the depths into points using the camera position
	for (int kfi = 0; kfi < keyframes.V.size(); kfi++)
	{

		cv::Mat depths = 1.0 / keyframes.V[kfi].inverseDepthD;
		for (int px = 0; px < depths.rows; px++)
		{
			for (int py = 0; py < depths.cols; py++)
			{
				double depth = depths.at<double>(px, py);
				//myfile << depth << ", ";
				if (depth == 1)
				{
					continue;
				}
				Point3d tpoint(py, px, depth * 10); //= projectCameraPointToWorldPointP(cameraParams, keyframes.E[kfi], Point(px, py), depth);
				pcloud_est.push_back(tpoint);
				Vec3b cpoint = (keyframes.V[kfi].origImage.at<Vec3b>(px, py));
				colours.push_back(cpoint);
			}
			//myfile << std::endl;
		}
	}
	//myfile.close();

}

void GraphSLAMer::get3dColours(std::vector<cv::Vec3b> & pcloud_est)
{

	pcloud_est.clear();

	//cycle through the depth maps, converting the depths into points using the camera position
	for (int kfi = 0; kfi < keyframes.V.size(); kfi++)
	{
		cv::Mat depths = 1.0 / keyframes.V[kfi].inverseDepthD;
		for (int px = 0; px < depths.rows; px++)
		{
			for (int py = 0; py < depths.cols; py++)
			{
				Vec3b tpoint = (keyframes.V[kfi].scaledImageI.at<Vec3d>(px, py) * 255);
				pcloud_est.push_back(tpoint);
			}
		}
	}

}