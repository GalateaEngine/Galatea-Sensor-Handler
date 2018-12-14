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
	//get some class-wide pointers and values for camera param rows/entries for fast access
	double * cpPoint0 = cameraParams.ptr<double>(0);
	double * cpPoint1 = cameraParams.ptr<double>(1);
	double fx_d = cpPoint0[0];
	double cx_d = cpPoint0[2];
	double fy_d = cpPoint1[1];
	double cy_d = cpPoint1[2];

	//this creates a scalar to modify indexes when the image is shurnk down from default camera size
	double imageScale;

	cv::Mat lastPos;
	cv::Mat velocity;
	double alpha[7];
	const int quadTreeDepth = 4;

	//std::thread positionThread;
	//std::thread depthThread;
	//std::thread makeKeyframe;

	cv::Mat cameraParamsInv = cameraParams.inv();

	//look up "lie groups" for more information
	//contains both the translation and rotation for a given object in 3d space
	//methods can export the lie matrix, its individual components, or it's applied 3x3 extrinsic matrix

	class SIM3
	{
	private:
		cv::Mat rotationMat;
		cv::Mat translation;
		cv::Mat lieMat;
		cv::Mat parameters;
		double scale = 1;

	public:

		SIM3 operator*(const SIM3 in)
		{
			//verify when more sane that consecutive rotations applied as they are here indeed simply relate to the theta values by stright addition
			//we'll just get theta from the rotation matrix for now, this can be a potential time save later
			//also check if we can replace the pow mess with abs
			cv::Mat trotation = rotationMat * in.rotationMat;
			cv::Mat ttranslation = rotationMat * in.translation.t() + (1.0 / scale) * translation.t();
			cv::Mat tparameters(1, 7, CV_64FC1);
			tparameters = parameters + in.parameters;
			tparameters.at<double>(0, 6) = in.parameters.at<double>(0, 6);
			return SIM3(trotation, ttranslation.t(), tparameters);
		}

		SIM3()
		{
			translation = cv::Mat::zeros(1, 3, CV_64FC1);
			rotationMat = cv::Mat::eye(3, 3, CV_64FC1);
			lieMat = cv::Mat::eye(4, 4, CV_64FC1);
			parameters = cv::Mat::zeros(1, 7, CV_64FC1);

			//create proper size matrices at origin
			scale = parameters.at<double>(0, 6) = 1;
		}

		SIM3(cv::Mat _rotation, cv::Mat _translation, cv::Mat _parameters)
		{
			rotationMat = _rotation.clone();
			translation = _translation.clone();
			parameters = _parameters.clone();

			lieMat = cv::Mat::zeros(4, 4, CV_64FC1);
			//set rotation
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					lieMat.at<double>(x, y) = rotationMat.at<double>(x, y);
			//set translation
			for (int i = 0; i < 3; i++)
				lieMat.at<double>(i, 3) = translation.at<double>(0, i);

			lieMat.at<double>(3, 3) = parameters.at<double>(0,6);
			scale = parameters.at<double>(0, 6);
		}

		SIM3(cv::Mat _parameters)
		{
			//create proper size matrices at origin
			translation = cv::Mat::zeros(1, 3, CV_64FC1);
			rotationMat = cv::Mat::eye(3, 3, CV_64FC1);
			lieMat = cv::Mat::zeros(4, 4, CV_64FC1);

			setParameters(_parameters);
		}

		void setRotation(double x, double y, double z)
		{

			double * paramPointer = parameters.ptr<double>(0);
			if (x > 2 * 3.141592 || x < -2 * 3.141592) paramPointer[0] = 2 * 3.141592;
			else paramPointer[0] = x;
			if (y > 2 * 3.141592 || y < -2 * 3.141592) paramPointer[1] = 2 * 3.141592;
			else paramPointer[1] = y;
			if (z > 2 * 3.141592 || z < -2 * 3.141592) paramPointer[2] = 2 * 3.141592;
			else paramPointer[2] = z;

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

		void addSIM3(SIM3 lieAdd)
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
			scale = lieMat.at<double>(3, 3) = parameters.at<double>(0, 6);
		}

		void addParameters(cv::Mat _para)
		{
			setParameters(parameters + _para);
		}

		void setParameters(cv::Mat _params)
		{
			if (_params.cols == 7)
			{
				parameters = _params.clone();
				scale = lieMat.at<double>(3, 3) = parameters.at<double>(0, 6);
			}
			else
			{
				parameters = cv::Mat(1, 7, CV_64FC1);
				for (int i = 0; i < 6; i++)
				{
					parameters.at<double>(0, i) = _params.at<double>(0, i);
				}
				scale = lieMat.at<double>(3, 3) = parameters.at<double>(0, 6) = 1.0;
			}
			
			double * paraPointer = parameters.ptr<double>(0);
			setRotation(paraPointer[0], paraPointer[1], paraPointer[2]);
			setTranslation(paraPointer[3], paraPointer[4], paraPointer[5]);
			
		}

		cv::Mat getRotationMat()
		{
			return (rotationMat * scale);
		}

		cv::Mat getTranslation()
		{
			return (translation * scale);
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
		double depthDeviation = rand() + 1;
		double mean;
		double updateCount = 1;
		int numChildren;
		cv::Point position;
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
		cv::Mat origImage;
		cv::Mat depthVarianceV;
		SIM3 cameraTransformationAndScaleS; //taken and scaled with repsect to the world frame W aka the first frame. This is an element of Sim(3)
		QuadTreeNode * quadTreeLeaves; //contains the significant quad tree nodes
		int quadTreeNodeCount = 0;
		cv::Mat paramsTimesPose;
		cv::Mat paramsTimesPoseInv;
		std::vector<cv::Mat> pyramid;
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

	double CalcErrorVal(cv::Mat residuals);


	cv::Mat derivative(SIM3 & cameraPose, KeyFrame & keyframe, cv::Mat & image, double rmean, int bIndexX);

	cv::Mat ComputeJacobian(SIM3 & cameraPose, KeyFrame keyframe, cv::Mat & image, double rmean, int numRes);

	cv::Mat ComputeResiduals(cv::Mat & cameraPose, KeyFrame keyframe, cv::Mat & image,bool dcheck, double rmean);

	double ComputeResidualError(cv::Mat & cameraPose, KeyFrame keyframe, cv::Mat & image, bool dcheck, double rmean);


	//predicts new position based on dampened approximate velocity
	void forwardPredictPosition(cv::Mat &lastPos, cv::Mat &Velocity);


	//turns the world point into a pixel value
	void projectWorldPointToCameraPointU(double * poseRow1, double * poseRow2, double * poseRow3, double poseScale, double inX, double inY, double inZ, int & returnX, int & returnY);

	//turns a pixel value into a world point
	void projectCameraPointToWorldPointP(double * poseRow1, double * poseRow2, double * poseRow3, double poseScale, int inX, int inY, double depth, double & returnX, double & returnY, double & returnZ);


	double HuberNorm(double x, double epsilon);

	cv::Mat applyVarianceWeights(cv::Mat & jacobianTranspose, KeyFrame kf);

	//computes the update
	cv::Mat TransformJacobian(cv::Mat & jacobian, KeyFrame kf, cv::Mat residuals);

	SIM3 CalcGNPosOptimization(cv::Mat & image, KeyFrame keyframe);

	void ComputeQuadtreeForKeyframe(KeyFrame &kf);

	//calculates the depths by comparing the image, after plcement into a power of 2 pyramid, against the keyframe quadtree leaves
	void computeDepthsFromStereoPair(KeyFrame & kf, cv::Mat & image, cv::Mat & cameraParams, SIM3 cameraPos, bool initialize = false);

	void projectDepthNodesToDepthMap(KeyFrame & kf);


	//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
	//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
	//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
	//K: is a 3x3 real mat with the camera parameters
	//pi: perspective projection function
	SIM3 LS_Graph_SLAM(cv::Mat cameraFrame);

	//Sets up matrices and other things
	void Initialize_LS_Graph_SLAM(cv::Mat cameraFrame, cv::Mat cameraFrame2);

	//passes over keyframes and constraints and returns a list of points
	void get3dPointsAndColours(std::vector<cv::Point3d> & pcloud_est, std::vector<cv::Vec3b> & colours);
	void get3dColours(std::vector<cv::Vec3b> & pColours);
};