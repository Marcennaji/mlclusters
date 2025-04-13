// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMLearningProject.h"

#define CGEDBG(x)cout << "[" << x << "]" << endl

int main(int argc, char** argv)
{


#ifdef _DEBUG

	// Activation du mode Parallele Simule
	PLParallelTask::SetParallelSimulated(true);
	PLParallelTask::SetSimulatedSlaveNumber(4);

	std::cout << "*******COMPILATION********" << std::endl;
	std::cout << "    MLClusters    " << std::endl;
	std::cout << " VERSION ";
	std::cout << " " << INTERNAL_VERSION << std::endl;
	std::cout << __DATE__ << " " << __TIME__ << std::endl;
	std::cout << "**************************" << std::endl;

	Global::SetErrorLogFileName("kmean_exec.log");

#endif

	Global::SetMaxErrorFlowNumber(1000);

	// Parametrage de l'arret de l'allocateur
	MemSetAllocIndexExit(0);
	//MemSetAllocSizeExit(488);
	//MemSetAllocBlockExit((void*)0x00F42044);

	SetLearningVersion(VERSION_FULL);

	KMLearningProject learningProject;
	learningProject.Start(argc, argv);

	return 0;
}
