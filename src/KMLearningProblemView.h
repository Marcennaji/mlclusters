// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWLearningBenchmarkView.h"

#include "KMLearningProblem.h"
#include "KWLearningProblemView.h"
#include "KMModelingSpecView.h"
#include "KMPredictorEvaluatorView.h"
#include "KMAnalysisResultsView.h"

////////////////////////////////////////////////////////////////
/// Classe KMLearningProblemView : Vue sur la gestion de l'apprentissage avec k-means

class KMLearningProblemView : public KWLearningProblemView
{
public:
	// Constructeur
	KMLearningProblemView();
	~KMLearningProblemView();


	void SetObject(Object* object);

	/** Acces au probleme d'apprentissage */
	KMLearningProblem* GetLearningProblem();

	/** executer un benchmark de classifieur */
	void ClassifierBenchmark();

protected:

	/** redefinition methode ancetre */
	void EvaluatePredictors();

};

////////////////////////////////////////////////////////////
// Classe KMLearningProblemExtendedActionView
///    Actions d'analyse etendues deportees de KMLearningProblemView
class KMLearningProblemExtendedActionView : public UIObjectView
{
public:

	// Constructeur
	KMLearningProblemExtendedActionView();
	~KMLearningProblemExtendedActionView();


	////////////////////////////////////////////////////////
	// Redefinition des methodes a reimplementer obligatoirement

	// Mise a jour de l'objet par les valeurs de l'interface
	void EventUpdate(Object* object);

	// Mise a jour des valeurs de l'interface par l'objet
	void EventRefresh(Object* object);

	// Actions de menu
	void ClassifierBenchmark();

	// Acces au probleme d'apprentissage
	KMLearningProblem* GetLearningProblem();

	// Acces a la vue principale sur le probleme d'apprentissage
	KMLearningProblemView* GetLearningProblemView();

};

