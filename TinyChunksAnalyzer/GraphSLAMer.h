#pragma once
#include "stdafx.h"

class GraphSLAMer
{
public:
	//These are my calculated intrinsic camera parameters (Webcam: Logitech C920)
	//to properly use this program you'll need to find the intrinsic camera parameters for your own webcam/camera, assuming it's a different model
	//google search "opencv camera calibration" for an easy guide
	//(Todo: add autocv::Matic camera calibration)
	cv::Mat cameraParams = (cv::Mat_<double>(3, 3) << 5.7481157594243552e+02, 0.0, 320.0, 0.0, 5.7481157594243552e+02, 240.0, 0.0, 0.0, 1.0);


	cv::Mat lastPos;
	cv::Mat velocity;
	int quadTreeDepth = 4;



	cv::Mat cameraParamsInv = cameraParams.inv();

	//look up "lie groups" for more information
	//contains both the translation and rotation for a given object in 3d space
	//methods can export the lie matrix, its individual components, or it's applied 3x3 extrinsic matrix

	class SE3
	{
	private:
		cv::Mat rotationMat;
		cv::Mat translation;
		cv::Mat lieMat;
		cv::Mat parameters;

	public:


		SE3()
		{
			//create proper size matrices at origin
			translation = cv::Mat::zeros(1, 3, CV_64FC1);
			rotationMat = cv::Mat::eye(3, 3, CV_64FC1);

			lieMat = cv::Mat::zeros(3, 4, CV_64FC1);
			//set rotation
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					lieMat.at<double>(x, y) = rotationMat.at<double>(x, y);
			//set translation
			for (int i = 0; i < 3; i++)
				lieMat.at<double>(i, 3) = translation.at<double>(0, i);

			//contains rotation x y z in decimal (0->1) followed by translation x y z
			parameters = cv::Mat::zeros(1, 6, CV_64FC1);
		}

		SE3(cv::Mat _rotation, cv::Mat _translation, cv::Mat _parameters)
		{
			rotationMat = _rotation.clone();
			translation = _translation.clone();
			parameters = _parameters.clone();

			lieMat = cv::Mat::zeros(3, 4, CV_64FC1);
			//set rotation
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					lieMat.at<double>(x, y) = rotationMat.at<double>(x, y);
			//set translation
			for (int i = 0; i < 3; i++)
				lieMat.at<double>(i, 3) = translation.at<double>(0, i);

			//lieMat.at<double>(3, 3) = 1;
		}

		void setRotation(double x, double y, double z)
		{
			double * paramPointer = parameters.ptr<double>(0);
			paramPointer[0] = x;
			paramPointer[1] = y;
			paramPointer[2] = z;

			//get our row refrences
			double * row1 = rotationMat.ptr<double>(0);
			double * row2 = rotationMat.ptr<double>(1);
			double * row3 = rotationMat.ptr<double>(2);

			//set up our trig values
			double cosa = cos(x);
			double cosb = cos(y);
			double cosc = cos(z);
			double sina = sin(x);
			double sinb = sin(y);
			double sinc = sin(z);

			double sinbcosc = sinb * cosc;
			double sinbsinc = sinb * sinc;

			//set row 1
			row1[0] = cosb * cosc;
			row1[1] = -cosb * sinc;
			row1[2] = sinb;

			//set row 2
			row2[0] = cosa * sinc + sina * sinbcosc;
			row2[1] = cosa * cosc - sina * sinbsinc;
			row2[2] = -sina * cosb;

			//set row 3
			row3[0] = sina * sinc - cosa * sinbcosc;
			row3[1] = sina * cosc + cosa * sinbsinc;
			row3[2] = cosa * cosb;

			//set rotation
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					lieMat.at<double>(x, y) = rotationMat.at<double>(x, y);
		}


		void setTranslation(double x, double y, double z)
		{
			translation.at<double>(0, 0) = x;
			lieMat.at<double>(0, 3) = x;
			parameters.at<double>(0, 3) = x;

			translation.at<double>(0, 1) = y;
			lieMat.at<double>(1, 3) = y;
			parameters.at<double>(0, 4) = y;

			translation.at<double>(0, 2) = z;
			lieMat.at<double>(2, 3) = z;
			parameters.at<double>(0, 5) = z;
		}

		void addSE3(SE3 lieAdd)
		{
			cv::Mat inLieMat = lieAdd.getlieMatrix();
			
			//set translation
			for (int i = 0; i < 3; i++)
			{
				translation.at<double>(0, i) += inLieMat.at<double>(i, 3);
				lieMat.at<double>(i, 3) = translation.at<double>(0, i);
			}

			parameters += lieAdd.getParameters();
			setRotation(parameters.ptr<double>(0)[0], parameters.ptr<double>(0)[1], parameters.ptr<double>(0)[2]);

		}

		void addParameters(cv::Mat _para)
		{
			setParameters(parameters + _para);
		}

		void setParameters(cv::Mat _params)
		{
			parameters = _params.clone();
			double * paraPointer = parameters.ptr<double>(0);
			setRotation(paraPointer[0], paraPointer[1], paraPointer[2]);
			setTranslation(paraPointer[3], paraPointer[4], paraPointer[5]);
		}

		cv::Mat getRotationMat()
		{
			return rotationMat.clone();
		}

		cv::Mat getTranslation()
		{
			return translation.clone();
		}

		//constructs and returns the lie matrix
		cv::Mat getlieMatrix()
		{
			//std::cout << lieMat;
			return lieMat.clone();
		}

		//parameters contains rotation x y z in decimal (0->1) followed by translation x y z
		cv::Mat getParameters()
		{
			return parameters.clone();
		}
	};

	class QuadTreeNode
	{
	public:
		double avgIntensity;
		double xGradient;
		double yGradient;
		double depth;
		double depthVariance;
		int numChildren;
		cv::Point2f position;
		int length;
		int width;
		bool fLeaf;
		bool valid;
		int strikes;
		int layer;
		QuadTreeNode *parent;
		QuadTreeNode *children[4];
	};

	class KeyFrame
	{
	public:
		cv::Mat inverseDepthD;
		cv::Mat scaledImageI;//mean 1
		cv::Mat inverseDepthVarianceV;
		SE3 cameraTransformationAndScaleS; //taken and scaled with repsect to the world frame W aka the first frame. This is an element of Sim(3)
		std::vector<QuadTreeNode> quadTreeLeaves; //contains the significant quad tree nodes
		cv::Mat paramsTimesPose;
		cv::Mat paramsTimesPoseInv;
	};

	class PoseGraph
	{
	public:
		std::vector<KeyFrame> V;
		std::vector<cv::Mat> E; //sim(3) constraints between keyframes
	};

	class Node
	{
	public:
		std::vector<Node> subNodes;
		cv::Point3f location;
	};


	PoseGraph keyframes;
	KeyFrame lastKey;

	//turns the given point into a homogeneouse point
	cv::Mat makeHomo(cv::Mat x);

	cv::Mat deHomo(cv::Mat xbar);

	//simple cholesky decomposition
	cv::Mat MatrixSqrt(cv::Mat a);

	//code roughly from "my math lib"
	void InvertLowerTriangluar(cv::Mat &l);

	//solve for delta x in (H * dX = -b)
	cv::Mat SolveSparseMatrix(cv::Mat H, cv::Mat b);

	double CalcErrorVal(double residuals);


	double derivative(SE3 & cameraPose, KeyFrame & keyframe, cv::Mat & image, double rmean, int bIndexX);

	std::vector<double> ComputeJacobian(SE3 & cameraPose, KeyFrame keyframe, cv::Mat & image, double rmean, int numRes);

	double ComputeResiduals(cv::Mat & cameraPose, KeyFrame keyframe, cv::Mat & image,bool dcheck, double rmean);


	//predicts new position based on dampened approximate velocity
	void forwardPredictPosition(cv::Mat &lastPos, cv::Mat &Velocity);


	//turns the world point into a pixel value
	cv::Point2d projectWorldPointToCameraPointU(cv::Mat & cameraParamsInv, cv::Mat & cameraPoseT, cv::Point3d wPointP);

	//turns a pixel value into a world point
	cv::Point3d projectCameraPointToWorldPointP(cv::Mat & cameraParamsK, cv::Mat & cameraPoseT, cv::Point2d cPointU, double depth);


	double HuberNorm(double x, double epsilon);

	//cv::Mat CalcErrorVec(std::vector<pPixel> pixels);

	void ComputeMedianResidualAndCorrectedPhotometricResiduals(cv::Mat & cameraParams, SE3 cameraPose, cv::Mat & image, KeyFrame kf, double & median);


	//computes the update
	cv::Mat TransformJacobian(cv::Mat & jacobian, cv::Mat & residuals);

	SE3 CalcGNPosOptimization(cv::Mat & image, KeyFrame keyframe);

	void ComputeQuadtreeForKeyframe(KeyFrame &kf);

	//calculates the depths by comparing the image, after plcement into a power of 2 pyramid, against the keyframe quadtree leaves
	void computeDepthsFromStereoPair(KeyFrame kf, cv::Mat & image, cv::Mat & cameraParams, SE3 cameraPos);

	void projectDepthNodesToDepthMap(KeyFrame kf);


	//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
	//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
	//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
	//K: is a 3x3 real mat with the camera parameters
	//pi: perspective projection function
	SE3 LS_Graph_SLAM(cv::Mat cameraFrame);

	//Sets up matrices and other things
	void Initialize_LS_Graph_SLAM(cv::Mat cameraFrame);

	//passes over keyframes and constraints and returns a list of points
	std::vector<cv::Point3d> get3dPoints();
	std::vector<cv::Vec3b> get3dColours();
};