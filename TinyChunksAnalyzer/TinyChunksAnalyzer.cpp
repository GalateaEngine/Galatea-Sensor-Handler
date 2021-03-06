// TinyChunksAnalyzer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TCA_Core.h"
#include <string>
using namespace std;

/** \brief Test all grabbing drivers and fills a vector of all available cameras CAPdrivers+ID
*
* For each CAPdrivers+ID, opens the device. If success, push CAP+ID in \c camIdx
* A grabbing test is done just to inform the user.
* \param camIdx[out] a vector of all readable cameras CAP+ID
* \note remove some cout to use as function
*/
bool EnumerateCameras()//(vector<int> &camIdx)
{
	//camIdx.clear();
	struct CapDriver {
		int enumValue; string enumName; string comment;
	};
	// list of all CAP drivers (see highgui_c.h)
	vector<CapDriver> drivers;
	drivers.push_back({ CV_CAP_MIL, "CV_CAP_MIL", "MIL proprietary drivers" });
	drivers.push_back({ CV_CAP_VFW, "CV_CAP_VFW", "platform native" });
	drivers.push_back({ CV_CAP_FIREWARE, "CV_CAP_FIREWARE", "IEEE 1394 drivers" });
	drivers.push_back({ CV_CAP_STEREO, "CV_CAP_STEREO", "TYZX proprietary drivers" });
	drivers.push_back({ CV_CAP_QT, "CV_CAP_QT", "QuickTime" });
	drivers.push_back({ CV_CAP_UNICAP, "CV_CAP_UNICAP", "Unicap drivers" });
	drivers.push_back({ CV_CAP_DSHOW, "CV_CAP_DSHOW", "DirectShow (via videoInput)" });
	drivers.push_back({ CV_CAP_MSMF, "CV_CAP_MSMF", "Microsoft Media Foundation (via videoInput)" });
	drivers.push_back({ CV_CAP_PVAPI, "CV_CAP_PVAPI", "PvAPI, Prosilica GigE SDK" });
	drivers.push_back({ CV_CAP_OPENNI, "CV_CAP_OPENNI", "OpenNI (for Kinect)" });
	drivers.push_back({ CV_CAP_OPENNI_ASUS, "CV_CAP_OPENNI_ASUS", "OpenNI (for Asus Xtion)" });
	drivers.push_back({ CV_CAP_ANDROID, "CV_CAP_ANDROID", "Android" });
	drivers.push_back({ CV_CAP_ANDROID_BACK, "CV_CAP_ANDROID_BACK", "Android back camera" }),
		drivers.push_back({ CV_CAP_ANDROID_FRONT, "CV_CAP_ANDROID_FRONT","Android front camera" }),
		drivers.push_back({ CV_CAP_XIAPI, "CV_CAP_XIAPI", "XIMEA Camera API" });
	drivers.push_back({ CV_CAP_AVFOUNDATION, "CV_CAP_AVFOUNDATION", "AVFoundation framework for iOS" });
	drivers.push_back({ CV_CAP_GIGANETIX, "CV_CAP_GIGANETIX", "Smartek Giganetix GigEVisionSDK" });
	drivers.push_back({ CV_CAP_INTELPERC, "CV_CAP_INTELPERC", "Intel Perceptual Computing SDK" });

	std::string winName, driverName, driverComment;
	int driverEnum;
	cv::Mat frame;
	bool found;
	std::cout << "Searching for cameras IDs..." << endl << endl;
	for (int drv = 0; drv < drivers.size(); drv++)
	{
		driverName = drivers[drv].enumName;
		driverEnum = drivers[drv].enumValue;
		driverComment = drivers[drv].comment;
		std::cout << "Testing driver " << driverName << "...";
		found = false;

		int maxID = 100; //100 IDs between drivers
		if (driverEnum == CV_CAP_VFW)
			maxID = 10; //VWF opens same camera after 10 ?!?
		else if (driverEnum == CV_CAP_ANDROID)
			maxID = 98; //98 and 99 are front and back cam
		else if ((driverEnum == CV_CAP_ANDROID_FRONT) || (driverEnum == CV_CAP_ANDROID_BACK))
			maxID = 1;

		for (int idx = 0; idx <maxID; idx++)
		{
			cv::VideoCapture cap(driverEnum + idx);  // open the camera
			if (cap.isOpened())                  // check if we succeeded
			{
				found = true;
				//camIdx.push_back(driverEnum + idx);  // vector of all available cameras
				cap >> frame;
				if (frame.empty())
					std::cout << endl << driverEnum + idx << "\t opens: OK \t grabs: FAIL";
				else
					std::cout << endl << driverEnum + idx << "\t opens: OK \t grabs: OK";
				// display the frame
				imshow(driverName + "+" + to_string(idx), frame); cv::waitKey(1);
			}
			cap.release();
		}
		if (!found) cout << "Nothing !" << endl;
		cout << endl;
	}
	//cout << camIdx.size() << " camera IDs has been found ";
	cout << "Press a key..." << endl; cin.get();

	return 1;
	//return (camIdx.size()>0); // returns success
}

int main()
{
	TCA_Core core = TCA_Core(true, false, false);
	//EnumerateCameras();
	while (true)
	{
		core.update();
		if (cv::waitKey(1) == 27)
			break;
	}
    return 0;
}

