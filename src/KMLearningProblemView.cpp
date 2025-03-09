// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMLearningProblemView.h"
#include "KMLearningProject.h"
#include "KMAnalysisSpecView.h"

KMLearningProblemView::KMLearningProblemView()
{
	KMAnalysisSpecView* analysisSpecView;
	const ALString sAnalysisSpecIdentifier = "AnalysisSpec";

	KMAnalysisResultsView* analysisResultsView;
	const ALString sAnalysisResultsIdentifier = "AnalysisResults";

	// Specialisation de la fiche des parametres d'analyse,
	// en remplacant l'ancienne version par une sous-classe
	analysisSpecView = new KMAnalysisSpecView;
	ReplaceCardField(sAnalysisSpecIdentifier, analysisSpecView);

	// Specialisation de la fiche des resultats d'analyse,
	// en remplacant l'ancienne version par une sous-classe
	analysisResultsView = new KMAnalysisResultsView;
	ReplaceCardField(sAnalysisResultsIdentifier, analysisResultsView);

	// modifier l'action d'evaluation de predicteurs
	KWLearningProblemActionView* learningProblemActionView = new KWLearningProblemActionView;
	learningProblemActionView->GetActionAt("EvaluatePredictors")->SetActionMethod(
		(ActionMethod)(&KMLearningProblemView::EvaluatePredictors));
	ReplaceCardField("LearningTools", learningProblemActionView);

	// Parametrage de liste d'aide pour le nom de l'attribut cible
	analysisSpecView->GetFieldAt("TargetAttributeName")->SetStyle("HelpedComboBox");
	analysisSpecView->GetFieldAt("TargetAttributeName")->SetParameters(
		"Attributes:Name");

	// Parametrage de liste d'aide pour le nom de la modalite principale
	analysisSpecView->GetFieldAt("MainTargetModality")->SetStyle("HelpedComboBox");
	analysisSpecView->GetFieldAt("MainTargetModality")->SetParameters(
		"TargetValues:Value");

	GetActionAt("ComputeStats")->SetActionMethod((ActionMethod)(&KMLearningProblemView::ComputeStats));

	// Libelles
	SetIdentifier("KMLearningProblem");

	// Fonctionnalites avancees
	if (GetLearningExpertMode())
	{
		AddCardField("Benchmark", "Benchmark",
			new KMLearningProblemExtendedActionView);
	}

	// Info-bulles
	GetActionAt("ComputeStats")->SetHelpText("Analyze the data base and build the clustering prediction model."
		"\n All the preparation, modeling and evaluation reports are produced.");

}

KMLearningProblemView::~KMLearningProblemView()
{
}

KMLearningProblem* KMLearningProblemView::GetLearningProblem()
{
	return cast(KMLearningProblem*, objValue);
}
void KMLearningProblemView::ClassifierBenchmark()
{
	KMLearningBenchmark* classifierBenchmark;
	KWLearningBenchmarkView view;

	// Acces au parametrage du benchmark
	classifierBenchmark = GetLearningProblem()->GetClassifierBenchmark();
	assert(classifierBenchmark->GetTargetAttributeType() == KWType::Symbol);

	// Ouverture de la fenetre
	view.SetObject(classifierBenchmark);
	view.Open();
}


void KMLearningProblemView::SetObject(Object* object)
{
	KMLearningProblem* learningProblem;

	require(object != NULL);

	// Appel de la methode ancetre
	KWLearningProblemView::SetObject(object);

	// Acces a l'objet edite
	learningProblem = cast(KMLearningProblem*, object);

	// Fonctionnalites avancees, disponible uniquement pour l'auteur
	if (GetLearningExpertMode())
	{
		cast(KMLearningProblemExtendedActionView*, GetFieldAt("Benchmark"))->
			SetObject(learningProblem);
	}
}

void KMLearningProblemView::EvaluatePredictors()
{
	ALString sPredictorLabel;
	ALString sTargetAttributeName;
	ALString sMainTargetModality;
	ObjectArray oaClasses;
	ObjectArray oaTrainedPredictors;
	ObjectArray oaEvaluatedPredictors;
	ALString sReferenceTargetAttribute;
	KMPredictorEvaluator* predictorEvaluator;
	KMPredictorEvaluatorView predictorEvaluatorView;

	// Initialisation de l'evaluateur de predicteur
	// On prealimente les champs s'ils sont vides
	predictorEvaluator = GetLearningProblem()->GetPredictorEvaluator();
	if (predictorEvaluator->GetEvaluationDatabase()->GetDatabaseName() == "")
		predictorEvaluator->GetEvaluationDatabase()->CopyFrom(GetLearningProblem()->GetTrainDatabase());
	if (predictorEvaluator->GetMainTargetModality() == "")
		predictorEvaluator->SetMainTargetModality(GetLearningProblem()->GetAnalysisSpec()->GetMainTargetModality());
	predictorEvaluator->FillEvaluatedPredictorSpecs();
	predictorEvaluator->SetEvaluationFileName(ALString("EvaluationReport.xls"));

	// Ouverture
	predictorEvaluatorView.SetObject(predictorEvaluator);
	predictorEvaluatorView.Open();
}

//////////// classe KMLearningProblemExtendedActionView //////////////

KMLearningProblemExtendedActionView::KMLearningProblemExtendedActionView()
{
	// Libelles
	SetIdentifier("KMLearningExtendedProblemAction");
	SetLabel("Benchmark");

	// Benchmarks
	AddAction("ClassifierBenchmark", "Evaluate classifiers...",
		(ActionMethod)(&KMLearningProblemExtendedActionView::ClassifierBenchmark));
}


KMLearningProblemExtendedActionView::~KMLearningProblemExtendedActionView()
{
}


void KMLearningProblemExtendedActionView::EventUpdate(Object* object)
{
	KMLearningProblem* editedObject;

	require(object != NULL);

	editedObject = cast(KMLearningProblem*, object);
}


void KMLearningProblemExtendedActionView::EventRefresh(Object* object)
{
	KMLearningProblem* editedObject;

	require(object != NULL);

	editedObject = cast(KMLearningProblem*, object);
}


void KMLearningProblemExtendedActionView::ClassifierBenchmark()
{
	GetLearningProblemView()->ClassifierBenchmark();
}


KMLearningProblem* KMLearningProblemExtendedActionView::GetLearningProblem()
{
	require(objValue != NULL);

	return cast(KMLearningProblem*, objValue);
}


KMLearningProblemView* KMLearningProblemExtendedActionView::GetLearningProblemView()
{
	require(GetParent() != NULL);

	return cast(KMLearningProblemView*, GetParent());
}




