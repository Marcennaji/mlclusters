// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#define VERSION_FULL "10.1.0"
#define INTERNAL_VERSION "10.1.0.0_i1"

#include "KWLearningProject.h"
#include "KMLearningProblem.h"
#include "KMLearningProblemView.h"

/// Service de lancement du projet Kmeans

class KMLearningProject : public KWLearningProject
{
public:
	// Constructeur
	KMLearningProject();
	~KMLearningProject();

	///////////////////////////////////////////////////////////////////////
	///// Implementation
protected:

	// Reimplementation des methodes virtuelles
	KWLearningProblem* CreateLearningProblem();
	KWLearningProblemView* CreateLearningProblemView();

	/** Initialisation de l'environnement d'apprentissage (enregistrement des regles, predicteurs...) */
	virtual void OpenLearningEnvironnement();
};

