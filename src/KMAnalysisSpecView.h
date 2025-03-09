// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWAnalysisSpecView.h"

////////////////////////////////////////////////////////////
// Classe KMAnalysisSpecView
///    Version specialisee de KWAnalysisSpecView permettant de preciser les parametres de modelisation pour KMean

class KMAnalysisSpecView : public KWAnalysisSpecView
{
public:

	// Constructeur
	KMAnalysisSpecView();
	~KMAnalysisSpecView();
};
