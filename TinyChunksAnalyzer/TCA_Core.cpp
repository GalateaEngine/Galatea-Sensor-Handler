#include "stdafx.h"
#include "TCA_Core.h"
#include "TCA_Components.h"


TCA_Core::TCA_Core(bool videoEnabled, bool audioEnabled, bool textEnabled)
{
	if (videoEnabled)
	{
		VidEnabled = true;
		//vid = TCA_Video();
		//enable and set up SLAM algo
		GS.Initialize_LS_Graph_SLAM();
	}

}

void TCA_Core::update()
{
	if (VidEnabled)
	{
		vid.update();
	}
}