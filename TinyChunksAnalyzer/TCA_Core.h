#pragma once
#include "TCA_Components.h"
class TCA_Core
{
public:
	bool VidEnabled = true;
	TCA_Video vid;

	TCA_Core(bool videoEnabled, bool audioEnabled, bool textEnabled);

	void update();
};