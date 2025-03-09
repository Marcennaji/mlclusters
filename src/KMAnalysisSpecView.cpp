// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMAnalysisSpecView.h"
#include "KMModelingSpecView.h"

KMAnalysisSpecView::KMAnalysisSpecView()
{
	KMModelingSpecView* modelingSpecView;
	const ALString sModelingSpecIdentifier = "PredictorsSpec";

	//GetFieldAt("AttributeConstructionSpec")->SetVisible(false);
	GetFieldAt("PreprocessingSpec")->SetVisible(false);
	GetFieldAt("RecodersSpec")->SetVisible(false);

	// Specialisation de la fiche des parametres de modelisation,
	// en remplacant l'ancienne version par une sous-classe
	modelingSpecView = new KMModelingSpecView;
	ReplaceCardField(sModelingSpecIdentifier, modelingSpecView);

	//GetFieldAt("TargetAttributeName")->SetHelpText("Name of the target variable."
	//    "\n If the target variable is not specified, the task is unsupervised learning."
	//    "\n If specified, it must be a categorical (symbolic) variable.");

	//GetFieldAt("MainTargetModality")->SetHelpText("Value of the target variable in case of classification,"
	//    "\n for the lift curves in the evaluation reports.");

}


KMAnalysisSpecView::~KMAnalysisSpecView()
{
}
