// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMLearningProblem.h"
#include "KMLearningProject.h"
#include "KMClassStats.h"

KMLearningProblem::KMLearningProblem()
{
	// Specialisation des specifications d'analyse,
	// en detruisant le sous-objet cree dans la classe ancetre et en le remplacant par une version dediee
	delete analysisSpec;
	analysisSpec = new KMAnalysisSpec;

	delete predictorEvaluator;
	predictorEvaluator = new KMPredictorEvaluator;

	// Specialisation des resultats d'analyse
	delete analysisResults;
	analysisResults = new KMAnalysisResults;

	classifierBenchmark = new KMLearningBenchmark;
	classifierBenchmark->SetTargetAttributeType(KWType::Symbol);
}


KMLearningProblem::~KMLearningProblem()
{
	delete classifierBenchmark;
}

void KMLearningProblem::ComputeStats()
{
	KWLearningSpec learningSpec;
	KMClassStats* classStats;
	boolean bStatsOk;
	KWClass* kwcClass;
	KWClass* constructedClass;
	KWClassDomain* constructedClassDomain;
	KWClassDomain* initialClassDomain;
	KWClassDomain trainedClassDomain;
	boolean bIsSpecificRegressionLearningSpecNecessary;
	KWDatabase initialDatabase;
	KWDatabase specificRegressionDatabase;
	ObjectArray oaTrainedPredictors;
	ObjectArray oaTrainedPredictorReports;
	ALString sModelingDictionaryFileName;
	ALString sModelingReportName;
	KMPredictorEvaluator localPredictorEvaluator;
	ObjectArray oaTrainPredictorEvaluations;
	ObjectArray oaTestPredictorEvaluations;
	int i;
	ALString sTmp;

	require(FileService::CheckApplicationTmpDir());
	require(CheckClass());
	require(CheckTargetAttribute());
	require(CheckTrainDatabaseName());
	require(CheckResultFileNames());
	require(GetTrainDatabase()->CheckSelectionValue(GetTrainDatabase()->GetSelectionValue()));
	require(GetTestDatabase()->CheckSelectionValue(GetTestDatabase()->GetSelectionValue()));
	require(GetAnalysisSpec()->GetRecoderSpec()->GetRecodingSpec()->Check());
	require(CheckRecodingSpecs());
	require(CheckPreprocessingSpecs());
	require(GetAnalysisSpec()->GetModelingSpec()->GetAttributeConstructionSpec()->GetMaxConstructedAttributeNumber() == 0 or
		not GetAnalysisSpec()->GetModelingSpec()->GetAttributeConstructionSpec()->GetConstructionDomain()->GetImportAttributeConstructionCosts());
	require(not TaskProgression::IsStarted());

	KMModelingSpec* modelingSpec = cast(KMModelingSpec*, GetAnalysisSpec()->GetModelingSpec());
	assert(modelingSpec != NULL);

	if (not (modelingSpec->IsKmeanActivated() or modelingSpec->IsKNNActivated()))
		// si pas de predicteur de clustering selectionne, on execute le code par defaut
		return KWLearningProblem::ComputeStats();

	// Demarage du suivi de la tache
	TaskProgression::SetTitle("Train model " + GetClassName() + " " + GetTargetAttributeName());
	TaskProgression::SetDisplayedLevelNumber(2);
	TaskProgression::Start();

	KWGrouperSpec* grouperSpec = GetPreprocessingSpec()->GetGrouperSpec();
	KWDiscretizerSpec* discretizerSpec = GetPreprocessingSpec()->GetDiscretizerSpec();
	KMPredictor* kmPredictor = modelingSpec->GetClusteringPredictor();

	// reinitialiser valeurs par defaut, pour ne pas conserver les valeurs d'une precedente execution :
	grouperSpec->SetSupervisedMethodName("MODL");
	discretizerSpec->SetSupervisedMethodName("MODL");
	grouperSpec->SetUnsupervisedMethodName("BasicGrouping");
	discretizerSpec->SetUnsupervisedMethodName("EqualFrequency");

	//  "basic grouping" des variables categorielles, si non supervis�, OU si option volontairement choisie
	if (GetTargetAttributeName() == "" or
		kmPredictor->GetKMParameters()->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::BasicGrouping) {

		grouperSpec->SetSupervisedMethodName("BasicGrouping");
		grouperSpec->SetUnsupervisedMethodName("BasicGrouping");
		grouperSpec->SetMaxGroupNumber(kmPredictor->GetKMParameters()->GetPreprocessingMaxGroupNumber());
	}

	// EqualFreq des continuous, si Rank Normalization (cas "automatique" si non supervis�, OU si option volontairement choisie)
	if ((GetTargetAttributeName() == "" and
		kmPredictor->GetKMParameters()->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed)
		or kmPredictor->GetKMParameters()->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::RankNormalization) {

		discretizerSpec->SetSupervisedMethodName("EqualFrequency");
		discretizerSpec->SetUnsupervisedMethodName("EqualFrequency");
		discretizerSpec->SetMaxIntervalNumber(kmPredictor->GetKMParameters()->GetPreprocessingMaxIntervalNumber());
	}

	// mode supervise : nombre de groupes et d'intervalles max
	if (GetTargetAttributeName() != "") {
		grouperSpec->SetMaxGroupNumber(kmPredictor->GetKMParameters()->GetPreprocessingSupervisedMaxGroupNumber());
		discretizerSpec->SetMaxIntervalNumber(kmPredictor->GetKMParameters()->GetPreprocessingSupervisedMaxIntervalNumber());
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	// Initialisations

	// Creation d'un objet de calcul des stats
	classStats = new KMClassStats;

	// Initialisation des specifications d'apprentissage avec la classe de depart
	InitializeLearningSpec(&learningSpec, KWClassDomain::GetCurrentDomain()->LookupClass(GetClassName()));

	// Debut de la gestion des erreurs dediees a l'apprentissage
	KWLearningErrorManager::BeginErrorCollection();
	KWLearningErrorManager::AddTask("Data preparation");


	// Destruction de tous les rapports potentiels
	DeleteAllOutputFiles();

	// Memorisation du domaine initial
	initialClassDomain = KWClassDomain::GetCurrentDomain();

	// Import des couts des attributs stockes dans les meta-donnees du dictionnaire si demande
	if (GetAnalysisSpec()->GetModelingSpec()->GetAttributeConstructionSpec()->GetConstructionDomain()->GetImportAttributeConstructionCosts())
		bStatsOk = ImportAttributeMetaDataCosts(&learningSpec, constructedClass);
	// Creation d'une classe avec prise en compte eventuelle de construction de variables
	else
		bStatsOk = BuildConstructedClass(&learningSpec, constructedClass,
			classStats->GetMultiTableConstructionSpec(), classStats->GetTextConstructionSpec());

	constructedClassDomain = NULL;
	if (bStatsOk)
	{
		assert(constructedClass != NULL);
		constructedClassDomain = constructedClass->GetDomain();
		KWClassDomain::SetCurrentDomain(constructedClassDomain);
	}
	assert(bStatsOk or constructedClass == NULL);

	// Recherche de la classe
	kwcClass = learningSpec.GetClass();
	check(kwcClass);
	assert(constructedClass == NULL or kwcClass == constructedClass);
	assert(kwcClass == KWClassDomain::GetCurrentDomain()->LookupClass(GetClassName()));

	// Initialisation de domaine des predcicteurs appris
	trainedClassDomain.SetName("Train");

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Calcul des statistiques et ecriture des rapports de preparation

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Calcul des statistiques et ecriture des rapports

	// si on utilise un predicteur KMean ou KNN, le preciser a l'objet classStats, en lui transmettant le parametrage de clustering --> necessaire pour gerer les libelles specifiques MLClusters des pretraitements
	modelingSpec->GetClusteringPredictor()->GetKMParameters()->SetSupervisedMode(GetTargetAttributeName() == "" ? false : true);
	classStats->SetKMParameters(modelingSpec->GetClusteringPredictor()->GetKMParameters());

	// Initialisation des objets de calculs des statistiques
	if (bStatsOk and not TaskProgression::IsInterruptionRequested())
		InitializeClassStats(classStats, &learningSpec);

	// Calcul des statistiques
	if (bStatsOk and not TaskProgression::IsInterruptionRequested())
		classStats->ComputeStats();

	// Memorisation de l'eventuelle selection en cours, dont on a besoin potentiellement par la suite
	if (bStatsOk)
	{
		initialDatabase.CopySamplingAndSelectionFrom(learningSpec.GetDatabase());
		specificRegressionDatabase.CopySamplingAndSelectionFrom(learningSpec.GetDatabase());
	}

	// Test du cas particulier de la regression, si la classe cible contient des valeurs Missing
	// Dans ce cas, la variable cible est disponible, mais on a pas pu calcule les stats, et
	// on va le faire cette fois en filtrant les valeur cible manquantes
	bIsSpecificRegressionLearningSpecNecessary = false;
	if (bStatsOk and not classStats->IsStatsComputed() and not TaskProgression::IsInterruptionRequested())
		bIsSpecificRegressionLearningSpecNecessary = IsSpecificRegressionLearningSpecNecessary(&learningSpec);
	if (bIsSpecificRegressionLearningSpecNecessary)
	{
		// Parametrage des learning spec en creant un attribut de filtrage
		PrepareLearningSpecForRegression(&learningSpec);
		specificRegressionDatabase.CopySamplingAndSelectionFrom(learningSpec.GetDatabase());

		// Ajout d'un message indiquant  que l'on va filtrer les valeurs cibles manquantes
		Global::AddWarning("", "",
			"The missing values of target variable " + GetTargetAttributeName() + " are now filtered in a new attempt to train a model");

		// Recalcul des stats avec les valeurs cibles manquantes filtrees
		classStats->ComputeStats();
		bStatsOk = classStats->IsStatsComputed();
		assert(not learningSpec.IsTargetStatsComputed() or
			cast(KWDescriptiveContinuousStats*, learningSpec.GetTargetDescriptiveStats())->GetMissingValueNumber() == 0);

		// On remet la selection de base initiale, pour ne pas perturber l'ecriture des rapports
		learningSpec.GetDatabase()->CopySamplingAndSelectionFrom(&initialDatabase);
	}

	// On conditionne la suite par la validite des classStats, maintenant que l'on a essaye de le calcule deux fois si necessaire
	if (bStatsOk)
		bStatsOk = classStats->IsStatsComputed();

	// Ecriture des rapports de preparation
	if (bStatsOk and not TaskProgression::IsInterruptionRequested())
		WritePreparationReports(classStats);

	// Creation d'une classe de recodage
	if (bStatsOk and analysisSpec->GetRecoderSpec()->GetRecoder() and
		not TaskProgression::IsInterruptionRequested())
		BuildRecodingClass(initialClassDomain, classStats, &trainedClassDomain);

	// Cas particulier de la regression: on remet la selection de base specific si necessaire, le temps de l'apprentissage
	if (bIsSpecificRegressionLearningSpecNecessary)
		learningSpec.GetDatabase()->CopySamplingAndSelectionFrom(&specificRegressionDatabase);

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Apprentissage

	// Apprentissage
	KWLearningErrorManager::AddTask("Modeling");
	if (bStatsOk and not TaskProgression::IsInterruptionRequested())
	{
		// Base vide
		if (classStats->GetInstanceNumber() == 0)
			Global::AddWarning("", "", sTmp + "No training: database is empty");
		// Apprentissage non supervise
		else if (classStats->GetTargetAttributeType() == KWType::None)
			TrainPredictors(initialClassDomain, classStats, &oaTrainedPredictors);
		// L'attribut cible n'a qu'une seule valeur
		else if (classStats->GetTargetDescriptiveStats()->GetValueNumber() < 2)
		{
			if (learningSpec.GetTargetAttributeType() == KWType::Continuous and
				cast(KWDescriptiveContinuousStats*, learningSpec.GetTargetDescriptiveStats())->GetMissingValueNumber() > 0)
				Global::AddWarning("", "", sTmp + "No training: target variable has only missing values");
			else
				Global::AddWarning("", "", sTmp + "No training: target variable has only one value");
		}
		// Apprentissage en classification
		else if (classStats->GetTargetAttributeType() == KWType::Symbol)
			TrainPredictors(initialClassDomain, classStats, &oaTrainedPredictors);
		// Apprentissage en regression
		else if (classStats->GetTargetAttributeType() == KWType::Continuous)
			TrainPredictors(initialClassDomain, classStats, &oaTrainedPredictors);
	}

	// Cas particulier de la regression: restitution des learning spec initiales si necessaire
	if (bIsSpecificRegressionLearningSpecNecessary)
		RestoreInitialLearningSpec(&learningSpec, &initialDatabase);

	// Ecriture du rapport de modelisation
	if (bStatsOk and not TaskProgression::IsInterruptionRequested() and
		analysisResults->GetModelingFileName() != "" and oaTrainedPredictors.GetSize() > 0)
	{
		// Collecte des rapport d'apprentissage
		for (i = 0; i < oaTrainedPredictors.GetSize(); i++)
		{
			KWPredictor* predictor = cast(KWPredictor*, oaTrainedPredictors.GetAt(i));
			if (predictor->IsTrained())
				oaTrainedPredictorReports.Add(predictor->GetPredictorReport());
		}

		// Ecriture du rapport
		if (oaTrainedPredictorReports.GetSize() == 0)
			Global::AddWarning("", "", "Modeling and preparation reports are not written since no predictor was trained");
		else
		{
			assert(oaTrainedPredictors.GetSize() == 1);
			kmPredictor = cast(KMPredictor*, oaTrainedPredictors.GetAt(0));

			// Ecriture du rapport de preparation a ce moment, car on veut pouvoir y indiquer le nombre de variables utilisees par le clustering
			classStats->SetClusteringVariablesNumber(kmPredictor->GetClusteringVariablesNumber());
			WritePreparationReports(classStats);

			sModelingReportName = analysisResults->BuildOutputFilePathName(analysisResults->GetModelingFileName());
			AddSimpleMessage("Write modeling report " + sModelingReportName);

			const ALString asPredictorName = cast(KWPredictor*, oaTrainedPredictors.GetAt(0))->GetName();
			assert(asPredictorName == KMPredictor::PREDICTOR_NAME or asPredictorName == KMPredictorKNN::PREDICTOR_NAME);


			KMPredictorReport* predictorReport = cast(KMPredictorReport*, oaTrainedPredictorReports.GetAt(0));
			predictorReport->SetPredictor(kmPredictor);
			predictorReport->WriteFullReportFile(sModelingReportName, &oaTrainedPredictorReports);

			if (kmPredictor->GetLocalModelsPredictors().GetSize() > 0) {

				// parcourir les predicteurs de modeles locaux appris, et referencer leurs rapports
				const ObjectArray& localModelsPredictors = kmPredictor->GetLocalModelsPredictors();

				for (int iLocalModel = 0; iLocalModel < localModelsPredictors.GetSize(); iLocalModel++) {

					KWPredictor* localModelPredictor = cast(KWPredictor*, localModelsPredictors.GetAt(iLocalModel));

					const KMCluster* cluster = cast(const KMCluster*, kmPredictor->GetBestTrainedClustering()->GetClusters()->GetAt(iLocalModel));

					// assigner un domaine a la classe du predicteur local, afin de pouvoir ecrire le rapport
					KWClass* localModelClass = localModelPredictor->GetLearningSpec()->GetClass();
					KWClassDomain::GetCurrentDomain()->RemoveClass(localModelClass->GetName());
					KWClassDomain::GetCurrentDomain()->InsertClass(localModelClass);

					// ecrire le rapport de preparation du modele local
					ALString sLocalModelPreparationReportName = analysisResults->BuildOutputFilePathName("cluster_" + cluster->GetLabel() + "_" + analysisResults->GetPreparationFileName());
					AddSimpleMessage("Writing preparation report for cluster " + cluster->GetLabel() + " local model : " + sLocalModelPreparationReportName);
					localModelPredictor->GetClassStats()->SetWriteOptionStats2D(true);
					localModelPredictor->GetClassStats()->WriteReportFile(sLocalModelPreparationReportName);

					KWPredictorReport* localModelPredictorReport = localModelPredictor->GetPredictorReport();

					// restitution etat initial : reaffecter la classe du LearningSpec au domaine courant
					KWPredictor* predictor = cast(KWPredictor*, oaTrainedPredictors.GetAt(0));
					KWClassDomain::GetCurrentDomain()->RemoveClass(predictor->GetLearningSpec()->GetClass()->GetName());
					KWClassDomain::GetCurrentDomain()->InsertClass(predictor->GetLearningSpec()->GetClass());

					const ALString sLocalModelsModelingReportName = analysisResults->BuildOutputFilePathName("cluster_" + cluster->GetLabel() + "_ModelingReport.xls");
					AddSimpleMessage("Writing modeling report for cluster " + cluster->GetLabel() + " local model : " + sLocalModelsModelingReportName);
					ObjectArray oa;
					oa.Add(localModelPredictorReport);
					localModelPredictorReport->WriteFullReportFile(sLocalModelsModelingReportName, &oa);
				}
			}
		}
	}

	// Evaluation dans le cas supervise ET non supervise (evaluation en non supervise = specifique Enenade)
	if (bStatsOk and not TaskProgression::IsInterruptionRequested() and oaTrainedPredictors.GetSize() > 0)
	{
		// Evaluation des predicteurs sur la base d'apprentissage
		if (analysisResults->GetTrainEvaluationFileName() != "" and GetTrainDatabase()->GetDatabaseName() != "" and
			not GetTrainDatabase()->IsEmptySampling())
		{
			localPredictorEvaluator.EvaluatePredictors(&oaTrainedPredictors,
				GetTrainDatabase(), "Train", &oaTrainPredictorEvaluations);

			// Ecriture du rapport d'evaluation
			localPredictorEvaluator.WriteEvaluationReport(analysisResults->BuildOutputFilePathName(analysisResults->GetTrainEvaluationFileName()),
				"Train", &oaTrainPredictorEvaluations);
		}

		// Evaluation des predicteurs sur la base de test
		if (analysisResults->GetTestEvaluationFileName() != "" and GetTestDatabase()->GetDatabaseName() != "" and
			not GetTrainDatabase()->IsEmptySampling())
		{
			localPredictorEvaluator.EvaluatePredictors(&oaTrainedPredictors,
				GetTestDatabase(), "Test", &oaTestPredictorEvaluations);

			// Ecriture du rapport d'evaluation
			localPredictorEvaluator.WriteEvaluationReport(analysisResults->BuildOutputFilePathName(analysisResults->GetTestEvaluationFileName()),
				"Test", &oaTestPredictorEvaluations);
		}
	}

	// Ecriture du rapport JSON : specifique KMean, il faut les ecrire ici, avant de dereferencer les classes des predicteurs
	if (bStatsOk and not TaskProgression::IsInterruptionRequested())
		WriteJSONAnalysisReport(classStats, &oaTrainedPredictorReports,
			&oaTrainPredictorEvaluations, &oaTestPredictorEvaluations);

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Gestion du fichier de dictionnaires appris

	// Collecte des classes de prediction de predicteurs dans un domaine de classe
	// Les classes des predicteurs sont transferees dans le domaine de classe en sortie, et dereferencees des predicteurs
	CollectTrainedPredictorClasses(&oaTrainedPredictors, &trainedClassDomain);

	// Sauvegarde du fichier des dictionnaires appris si necessaire
	if (trainedClassDomain.GetClassNumber() > 0)
	{
		// Ecriture du fichier des dictionnaires de modelisation
		if (GetAnalysisResults()->GetModelingDictionaryFileName() != "" and not TaskProgression::IsInterruptionRequested())
		{
			KWLearningErrorManager::AddTask("Write modeling dictionary file");
			sModelingDictionaryFileName = analysisResults->BuildOutputFilePathName(GetAnalysisResults()->GetModelingDictionaryFileName());
			AddSimpleMessage("Write modeling dictionary file " + sModelingDictionaryFileName);

			// Sauvegarde des dictionnaires de modelisation
			trainedClassDomain.WriteFile(sModelingDictionaryFileName);
		}
	}

	// Ajout de log memoire
	MemoryStatsManager::AddLog("ComputeStats .Clean Begin");

	// Nettoyage des predicteurs et des classes apprises, et des evaluations
	trainedClassDomain.DeleteAllClasses();
	oaTrainedPredictors.DeleteAll();
	oaTrainPredictorEvaluations.DeleteAll();
	oaTestPredictorEvaluations.DeleteAll();

	// Nettoyage du domaine
	if (constructedClassDomain != NULL)
	{
		delete constructedClassDomain;
		KWClassDomain::SetCurrentDomain(initialClassDomain);
	}

	// Nettoyage
	delete classStats;

	// Fin de la gestion des erreurs dediees a l'apprentissage
	KWLearningErrorManager::EndErrorCollection();

	// Ajout de log memoire
	MemoryStatsManager::AddLog("ComputeStats .Clean End");

	// Fin du suivi de la tache
	TaskProgression::Stop();

	ensure(not TaskProgression::IsStarted());
}

void KMLearningProblem::CleanClass(KWClass* kwc) {

	StringVector oaDeletedAttributes;

	KWAttribute* attribute = kwc->GetHeadAttribute();

	while (attribute != NULL) {

		if (not attribute->GetUsed())
			oaDeletedAttributes.Add(attribute->GetName());

		kwc->GetNextAttribute(attribute);
	}

	if (oaDeletedAttributes.GetSize() == 0)
		return;

	for (int i = 0; i < oaDeletedAttributes.GetSize(); i++) {

		attribute = kwc->LookupAttribute(oaDeletedAttributes.GetAt(i));
		if (attribute != NULL) {
			kwc->DeleteAttribute(attribute->GetName());
		}
	}

	kwc->Compile();
}

KMLearningBenchmark* KMLearningProblem::GetClassifierBenchmark()
{
	return classifierBenchmark;
}

void KMLearningProblem::CollectPredictors(KWClassStats* classStats, ObjectArray* oaPredictors)
{
	KMPredictor* predictorKmean;
	KMPredictorKNN* predictorKNN;
	KMModelingSpec* modelingSpec;

	require(classStats != NULL);
	require(classStats->IsStatsComputed());
	require(oaPredictors != NULL);
	require(oaPredictors->GetSize() == 0);

	// Acces a la version specialisee des specification de modelisation
	modelingSpec = cast(KMModelingSpec*, analysisSpec->GetModelingSpec());

	if (modelingSpec->IsKmeanActivated()) {

		if (modelingSpec->IsKNNActivated()) {
			AddError("KMean predictor and KNN predictor can't be both selected. Please choose only one of them.");
			return;
		}
		else {

			// Predicteur KMean
			predictorKmean = cast(KMPredictor*, KMPredictor::ClonePredictor(KMPredictor::PREDICTOR_NAME, classStats->GetTargetAttributeType()));

			if (predictorKmean != NULL) {
				predictorKmean->CopyFrom(modelingSpec->GetClusteringPredictor());
				oaPredictors->Add(predictorKmean);
			}
			else
				AddWarning("K-Means predictor " + KWType::GetPredictorLabel(classStats->GetTargetAttributeType()) +
					" is not available");
		}
	}
	if (modelingSpec->IsKNNActivated()) {

		// Predicteur KNN
		predictorKNN = cast(KMPredictorKNN*, KMPredictorKNN::ClonePredictor(KMPredictorKNN::PREDICTOR_NAME, classStats->GetTargetAttributeType()));

		if (predictorKNN != NULL) {
			predictorKNN->CopyFrom(modelingSpec->GetClusteringPredictor());
			oaPredictors->Add(predictorKNN);
		}
		else
			AddWarning("KNN predictor " + KWType::GetPredictorLabel(classStats->GetTargetAttributeType()) +
				" is not available");
	}

	// Appel de la methode ancetre pour completer la liste
	KWLearningProblem::CollectPredictors(classStats, oaPredictors);
}


boolean KMLearningProblem::CheckTargetAttribute() const
{
	if (!CheckClass())
		return false;

	if (!KWLearningProblem::CheckTargetAttribute()) // appel methode ancetre
		return false;

	boolean bOk = true;
	KWClass* kwcClass;
	KWAttribute* attribute;

	if (GetTargetAttributeName() != "")
	{
		// Recherche de la classe
		kwcClass = KWClassDomain::GetCurrentDomain()->LookupClass(GetClassName());
		assert(kwcClass != NULL);

		// Recherche de l'attribut cible
		attribute = kwcClass->LookupAttribute(GetTargetAttributeName());
		assert(attribute != NULL);

		if (attribute->GetType() != KWType::Symbol and attribute->GetType() != KWType::None)
		{
			bOk = false;
			Global::AddError("", "", "Incorrect type for target variable " + GetTargetAttributeName() + ", should be Symbolic");
		}
	}
	return bOk;
}



KMPredictorEvaluator* KMLearningProblem::GetPredictorEvaluator()
{
	return cast(KMPredictorEvaluator*, predictorEvaluator);
}

////////////////////////////////////////////////////////////////////////
//// Classe KMAnalysisSpec
//
KMAnalysisSpec::KMAnalysisSpec()
{
	// Specialisation des specifications dde modelisation
	// en detruisant le sous-objet cree dans la classe ancetre et en le remplacant par une version dediee
	delete modelingSpec;
	modelingSpec = new KMModelingSpec;
}

KMAnalysisSpec::~KMAnalysisSpec()
{
}
