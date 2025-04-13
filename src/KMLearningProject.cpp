// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMLearningProject.h"
#include "KMClassifierEvaluationTask.h"
#include "KMPredictorEvaluationTask.h"
#include "KMRandomInitialisationTask.h"
#include "KMPredictorKNNView.h"
#include "KMDRRegisterAllRules.h"

KMLearningProject::KMLearningProject()
{
}

KMLearningProject::~KMLearningProject()
{
}

KWLearningProblem* KMLearningProject::CreateLearningProblem()
{
	return new KMLearningProblem;
}

KWLearningProblemView* KMLearningProject::CreateLearningProblemView()
{
	return new KMLearningProblemView;
}

void KMLearningProject::OpenLearningEnvironnement()
{
	// code par defaut
	KWLearningProject::OpenLearningEnvironnement();

	UIObject::SetIconImage("enneade.gif");

	SetLearningApplicationName("Khiops");
	SetLearningModuleName("MLClusters");

	// enregistrements specifiques MLClusters

	KMDRRegisterAllRules(); // regles de derivation
	KWPredictor::RegisterPredictor(new KMPredictor);
	KWPredictor::RegisterPredictor(new KMPredictorKNN);
	KWPredictorView::RegisterPredictorView(new KMPredictorView);
	KWPredictorView::RegisterPredictorView(new KMPredictorKNNView);
	PLParallelTask::RegisterTask(new KMClassifierEvaluationTask);
	PLParallelTask::RegisterTask(new KMPredictorEvaluationTask);
	PLParallelTask::RegisterTask(new KMRandomInitialisationTask);
}

