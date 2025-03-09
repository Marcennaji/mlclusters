// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMLearningBenchmark.h"
#include "KMLearningProblem.h"
#include "KMClusteringQuality.h"
#include "KMParameters.h"
#include "KMClassifierEvaluation.h"
#include "KMClassifierEvaluationTask.h"


// NB. pas de methode SetPredictorFilter dans la biblio --> obligé de dériver une classe juste pour modifier le filtre ?!
const ALString KMLearningBenchmark::GetPredictorFilter() const
{
	return "Naive Bayes;Selective Naive Bayes;Baseline;" + ALString(KMPredictor::PREDICTOR_NAME) + ";" + ALString(KMPredictorKNN::PREDICTOR_NAME);
}

void KMLearningBenchmark::CreateClassifierCriterions()
{
	// appel methode ancetre
	KWLearningBenchmark::CreateClassifierCriterions();

	AddCriterion("TrainEVA", "Train EVA", true);
	AddCriterion("TestEVA", "Test EVA", true);
	AddCriterion("RatioEVA", "Ratio EVA", true);

	AddCriterion("TrainARIByClusters", "Train ARI By Clusters", true);
	AddCriterion("TestARIByClusters", "Test ARI By Clusters", true);
	AddCriterion("RatioARIByClusters", "Ratio ARI By Clusters", true);

	AddCriterion("TrainARIByClasses", "Train ARI By Classes", true);
	AddCriterion("TestARIByClasses", "Test ARI By Classes", true);
	AddCriterion("RatioARIByClasses", "Ratio ARI By Classes", true);

	AddCriterion("TrainVariationOfInformation", "Train Variation Of Information", true);
	AddCriterion("TestVariationOfInformation", "Test Variation Of Information", true);
	AddCriterion("RatioVariationOfInformation", "Ratio Variation Of Information", true);

	AddCriterion("TrainPredictiveClustering", "Train Predictive Clustering", true);
	AddCriterion("TestPredictiveClustering", "Test Predictive Clustering", true);
	AddCriterion("RatioPredictiveClustering", "Ratio Predictive Clustering", true);

	AddCriterion("TrainDistance", "Train dist", true);
	AddCriterion("TestDistance", "Test dist", true);
	AddCriterion("RatioDistance", "Ratio dist", true);

	AddCriterion("TrainLEVA", "Train LEVA", true);
	AddCriterion("TestLEVA", "Test LEVA", true);
	AddCriterion("RatioLEVA", "Ratio LEVA", true);

	AddCriterion("TrainDaviesBouldin", "Train Davies Bouldin", true);
	AddCriterion("TestDaviesBouldin", "Test Davies Bouldin", true);
	AddCriterion("RatioDaviesBouldin", "Ratio Davies Bouldin", true);

	AddCriterion("TrainInertyIntra", "Train Inerty Intra", true);
	AddCriterion("TestInertyIntra", "Test Inerty Intra", true);
	AddCriterion("RatioInertyIntra", "Ratio Inerty Intra", true);

	AddCriterion("TrainInertyInter", "Train Inerty Inter", true);
	AddCriterion("TestInertyInter", "Test Inerty Inter", true);
	AddCriterion("RatioInertyInter", "Ratio Inerty Inter", true);

	AddCriterion("TrainInertyTotal", "Train Inerty Total", true);
	AddCriterion("TestInertyTotal", "Test Inerty Total", true);
	AddCriterion("RatioInertyTotal", "Ratio Inerty Total", true);

	AddCriterion("TrainNormalizedMutualInformationByClusters", "Train NMI by clusters", true);
	AddCriterion("TestNormalizedMutualInformationByClusters", "Test NMI by clusters", true);
	AddCriterion("RatioNormalizedMutualInformationByClusters", "Ratio NMI by clusters", true);

	AddCriterion("TrainNormalizedMutualInformationByClasses", "Train NMI by classes", true);
	AddCriterion("TestNormalizedMutualInformationByClasses", "Test NMI by classes", true);
	AddCriterion("RatioNormalizedMutualInformationByClasses", "Ratio NMI by classes", true);

}

void KMLearningBenchmark::EvaluateExperiment(int nBenchmark, int nPredictor,
	int nValidation, int nFold, IntVector* ivFoldIndexes)
{
	KWStatisticalEvaluation* totalComputingTimeEvaluation;
	KWStatisticalEvaluation* preprocessingComputingTimeEvaluation;
	KWBenchmarkSpec* benchmarkSpec;
	KWPredictorSpec* predictorSpec;
	KWLearningSpec* learningSpec;
	KWPredictor* predictor;
	KWPredictorEvaluation* predictorEvaluation;
	KWClass* constructedClass;
	ObjectDictionary odMultiTableConstructedAttributes;
	ObjectDictionary odTextConstructedAttributes;
	KWClassDomain* initialDomain;
	KWClassStats* classStats;
	int nRun;
	ALString sMainLabel;
	int nTotalExperimentNumber;
	int nExperimentIndex;
	double dTotalComputingTime;
	double dPreprocessingComputingTime;
	clock_t tBegin;
	clock_t tPreprocessingEnd;
	clock_t tTotalEnd;

	require(0 <= nBenchmark and nBenchmark < GetBenchmarkSpecs()->GetSize());
	require(0 <= nPredictor and nPredictor < GetPredictorSpecs()->GetSize());
	require(0 <= nValidation and nValidation < GetCrossValidationNumber());
	require(0 <= nFold and nFold < GetFoldNumber());
	require(ivFoldIndexes != NULL);

	//////////////////////////////////////////////////////////////////
	// Acces aux parametres de l'experience

	// Specifications du benchmark
	benchmarkSpec = cast(KWBenchmarkSpec*,
		GetBenchmarkSpecs()->GetAt(nBenchmark));
	assert(benchmarkSpec->Check());
	assert(benchmarkSpec->IsLearningSpecValid());

	// Probleme d'apprentissage
	learningSpec = benchmarkSpec->GetLearningSpec();

	// Specifications du predicteur
	predictorSpec = cast(KWPredictorSpec*, GetPredictorSpecs()->GetAt(nPredictor));
	assert(predictorSpec->Check());

	// Classifieur
	predictor = predictorSpec->GetPredictor();

	// Suivi de tache
	sMainLabel = benchmarkSpec->GetClassName();
	if (GetCrossValidationNumber() > 1)
		sMainLabel = sMainLabel + " Iter " + IntToString(nValidation);
	sMainLabel = sMainLabel + " " + predictorSpec->GetObjectLabel();
	sMainLabel = sMainLabel + " Fold " + IntToString(nFold + 1);
	TaskProgression::DisplayMainLabel(sMainLabel);
	nTotalExperimentNumber = GetBenchmarkSpecs()->GetSize() *
		GetPredictorSpecs()->GetSize() *
		GetCrossValidationNumber() * GetFoldNumber();
	nExperimentIndex = nBenchmark * GetCrossValidationNumber() *
		GetPredictorSpecs()->GetSize() * GetFoldNumber() +
		nValidation * GetPredictorSpecs()->GetSize() * GetFoldNumber() +
		nPredictor * GetFoldNumber() +
		nFold + 1;
	TaskProgression::DisplayProgression((nExperimentIndex * 100) / nTotalExperimentNumber);

	//////////////////////////////////////////////////////////
	// Apprentissage

	// Parametrage des specifications d'apprentissage par le
	// preprocessing du predicteur
	TaskProgression::DisplayLabel("Train");
	learningSpec->GetPreprocessingSpec()->CopyFrom(predictorSpec->GetPreprocessingSpec());

	// Parametrage des instances a garder en apprentissage
	benchmarkSpec->ComputeDatabaseSelectedInstance(ivFoldIndexes, nFold, true);

	// Creation d'un objet de calcul des stats
	classStats = new KWClassStats;

	// Parametrage du nombre initial d'attributs
	learningSpec->SetInitialAttributeNumber(learningSpec->GetClass()->ComputeInitialAttributeNumber(GetTargetAttributeType() != KWType::None));

	// Construction d'une classe avec de nouvelles variables si necessaire
	initialDomain = KWClassDomain::GetCurrentDomain();
	constructedClass = BuildLearningSpecConstructedClass(learningSpec, predictorSpec,
		classStats->GetMultiTableConstructionSpec(), classStats->GetTextConstructionSpec());


	// On prend le domaine de construction comme domaine de travail
	if (constructedClass != NULL)
	{
		KWClassDomain::SetCurrentDomain(constructedClass->GetDomain());
		learningSpec->SetClass(constructedClass);
	}
	// Sinon, on change quand meme de domaine de travail (pour la creation potentielle de nouvelles variables)
	else
	{
		KWClassDomain::SetCurrentDomain(KWClassDomain::GetCurrentDomain()->Clone());
		KWClassDomain::GetCurrentDomain()->Compile();
		learningSpec->SetClass(KWClassDomain::GetCurrentDomain()->LookupClass(learningSpec->GetClass()->GetName()));
	}
	assert(learningSpec->Check());

	// Prise en compte des specification de construction d'arbres
	if (KDDataPreparationAttributeCreationTask::GetGlobalCreationTask())
	{
		// On recopie les specifications de creation d'attributs
		KDDataPreparationAttributeCreationTask::GetGlobalCreationTask()->CopyAttributeCreationSpecFrom(
			predictorSpec->GetAttributeConstructionSpec()->GetAttributeCreationParameters());

		// On recopie le nombre d'attributs a construire, qui est specifie dans au niveau au dessus
		KDDataPreparationAttributeCreationTask::GetGlobalCreationTask()->SetMaxCreatedAttributeNumber(
			predictorSpec->GetAttributeConstructionSpec()->GetMaxTreeNumber());
	}

	// Prise en compte des paires de variables demandees par le predicteur
	predictorSpec->GetAttributeConstructionSpec()->GetAttributePairsSpec()->SetClassName(learningSpec->GetClass()->GetName());
	classStats->SetAttributePairsSpec(predictorSpec->GetAttributeConstructionSpec()->GetAttributePairsSpec());

	// Calcul des stats descriptives
	tBegin = clock();
	classStats->SetLearningSpec(learningSpec);
	classStats->ComputeStats();
	tPreprocessingEnd = clock();
	dPreprocessingComputingTime = (double)(tPreprocessingEnd - tBegin) / CLOCKS_PER_SEC;

	// Apprentissage
	if (classStats->IsStatsComputed())
	{

		// Debut de suivi de la tache d'apprentissage
		TaskProgression::BeginTask();
		TaskProgression::DisplayMainLabel(predictorSpec->GetObjectLabel());

		// Apprentissage
		predictor->SetLearningSpec(learningSpec);
		predictor->SetClassStats(classStats);

		// ============= debut de code specifique kmean

		if (predictorSpec->GetPredictor()->GetName() == ALString(KMPredictor::PREDICTOR_NAME) or
			predictorSpec->GetPredictor()->GetName() == ALString(KMPredictorKNN::PREDICTOR_NAME)) {

			KMPredictor* kmpredictor = cast(KMPredictor*, predictorSpec->GetPredictor());

			KWGrouperSpec* grouperSpec = kmpredictor->GetPreprocessingSpec()->GetGrouperSpec();
			KWDiscretizerSpec* discretizerSpec = kmpredictor->GetPreprocessingSpec()->GetDiscretizerSpec();

			grouperSpec->SetSupervisedMethodName("MODL");
			discretizerSpec->SetSupervisedMethodName("MODL");
			grouperSpec->SetUnsupervisedMethodName("BasicGrouping");
			discretizerSpec->SetUnsupervisedMethodName("EqualFrequency");

			//  "basic grouping" des variables categorielles, si non supervisé, ou si option volontairement choisie
			if (learningSpec->GetTargetAttributeType() == KWType::None or
				kmpredictor->GetKMParameters()->GetCategoricalPreprocessingType() == KMParameters::PreprocessingType::BasicGrouping) {

				grouperSpec->SetSupervisedMethodName("BasicGrouping");
				grouperSpec->SetUnsupervisedMethodName("BasicGrouping");
				grouperSpec->SetMaxGroupNumber(kmpredictor->GetKMParameters()->GetPreprocessingMaxGroupNumber());
			}
			// EqualFreq des continuous, si Rank Normalization (cas "automatique" non supervisé, ou ou si option volontairement choisie)
			if ((learningSpec->GetTargetAttributeType() == KWType::None and
				kmpredictor->GetKMParameters()->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::AutomaticallyComputed)
				or kmpredictor->GetKMParameters()->GetContinuousPreprocessingType() == KMParameters::PreprocessingType::RankNormalization) {

				discretizerSpec->SetSupervisedMethodName("EqualFrequency");
				discretizerSpec->SetUnsupervisedMethodName("EqualFrequency");
				discretizerSpec->SetMaxIntervalNumber(kmpredictor->GetKMParameters()->GetPreprocessingMaxIntervalNumber());
			}

			// mode supervise : nombre de groupes et d'intervalles max
			if (learningSpec->GetTargetAttributeType() != KWType::None) {
				grouperSpec->SetMaxGroupNumber(kmpredictor->GetKMParameters()->GetPreprocessingSupervisedMaxGroupNumber());
				discretizerSpec->SetMaxIntervalNumber(kmpredictor->GetKMParameters()->GetPreprocessingSupervisedMaxIntervalNumber());
			}

			classStats->ComputeStats();
		}

		// ============= fin de code specifique kmean


		predictor->Train();

		// Nettoyage de la classe du predicteur, en gardant tous les attributs de prediction
		if (predictor->IsTrained())
			predictor->GetTrainedPredictor()->CleanPredictorClass(initialDomain);

		// Fin de suivi de tache
		TaskProgression::EndTask();
	}
	tTotalEnd = clock();
	dTotalComputingTime = (double)(tTotalEnd - tBegin) / CLOCKS_PER_SEC;

	///////////////////////////////////////////////////////////////////
	// Collecte des resultats

	// Collecte si l'apprentissage a reussi
	if (classStats->IsStatsComputed() and predictor->IsTrained())
	{
		TaskProgression::DisplayLabel("Evaluation");

		// Index de l'experience
		nRun = nValidation * GetFoldNumber() + nFold;

		// Mise a jour des resultats d'evaluation sur tous les criteres en apprentissage
		if (not TaskProgression::IsInterruptionRequested())
		{
			assert(learningSpec->GetDatabase()->GetObjects()->GetSize() == 0);
			predictorEvaluation = predictor->Evaluate(learningSpec->GetDatabase());
			CollectAllResults(true, nBenchmark, nPredictor, nBenchmark, nRun,
				predictor, predictorEvaluation);
			delete predictorEvaluation;
			assert(learningSpec->GetDatabase()->GetObjects()->GetSize() == 0);

			// Memorisation des temps de calcul
			totalComputingTimeEvaluation = GetUpdatableEvaluationAt(
				GetCriterionIndexAt("TotalComputingTime"), nPredictor);
			preprocessingComputingTimeEvaluation = GetUpdatableEvaluationAt(
				GetCriterionIndexAt("PreprocessingComputingTime"), nPredictor);
			totalComputingTimeEvaluation->SetResultAt(nBenchmark, nRun, dTotalComputingTime);
			preprocessingComputingTimeEvaluation->SetResultAt(nBenchmark, nRun, dPreprocessingComputingTime);
		}

		// Mise a jour des resultats d'evaluation sur tous les criteres en test
		if (not TaskProgression::IsInterruptionRequested())
		{
			// Parametrage des instances a garder en test
			benchmarkSpec->ComputeDatabaseSelectedInstance(ivFoldIndexes, nFold, false);

			// Mise a jour des resultats
			assert(learningSpec->GetDatabase()->GetObjects()->GetSize() == 0);
			predictorEvaluation = predictor->Evaluate(learningSpec->GetDatabase());
			CollectAllResults(false, nBenchmark, nPredictor, nBenchmark, nRun,
				predictor, predictorEvaluation);
			delete predictorEvaluation;
			assert(learningSpec->GetDatabase()->GetObjects()->GetSize() == 0);
		}
	}

	// Restitution du domaine initial
	if (initialDomain != KWClassDomain::GetCurrentDomain())
	{
		learningSpec->SetClass(initialDomain->LookupClass(learningSpec->GetClass()->GetName()));
		delete KWClassDomain::GetCurrentDomain();
		KWClassDomain::SetCurrentDomain(initialDomain);
	}

	// Nettoyage
	predictorSpec->GetPredictor()->SetClassStats(NULL);
	predictorSpec->GetPredictor()->SetLearningSpec(NULL);
	delete classStats;

}


void KMLearningBenchmark::CollectAllClassifierResults(boolean bTrain,
	int nBenchmark, int nPredictor,
	int nExperiment, int nRun,
	KWPredictor* trainedPredictor,
	KWPredictorEvaluation* predictorEvaluation)
{

	// appel methode ancetre
	KWLearningBenchmark::CollectAllClassifierResults(bTrain,
		nBenchmark, nPredictor,
		nExperiment, nRun,
		trainedPredictor,
		predictorEvaluation);

	if (trainedPredictor->GetName() != KMPredictor::PREDICTOR_NAME and
		trainedPredictor->GetName() != KMPredictorKNN::PREDICTOR_NAME)
		return;

	// traitement specifique predicteur kmean

	KMClassifierEvaluation* classifierEvaluation;

	KWStatisticalEvaluation* trainEVAEvaluation;
	KWStatisticalEvaluation* testEVAEvaluation;
	KWStatisticalEvaluation* ratioEVAEvaluation;

	KWStatisticalEvaluation* trainARIByClustersEvaluation;
	KWStatisticalEvaluation* testARIByClustersEvaluation;
	KWStatisticalEvaluation* ratioARIByClustersEvaluation;

	KWStatisticalEvaluation* trainARIByClassesEvaluation;
	KWStatisticalEvaluation* testARIByClassesEvaluation;
	KWStatisticalEvaluation* ratioARIByClassesEvaluation;

	KWStatisticalEvaluation* trainVariationOfInformationEvaluation;
	KWStatisticalEvaluation* testVariationOfInformationEvaluation;
	KWStatisticalEvaluation* ratioVariationOfInformationEvaluation;

	KWStatisticalEvaluation* trainPredictiveClusteringEvaluation;
	KWStatisticalEvaluation* testPredictiveClusteringEvaluation;
	KWStatisticalEvaluation* ratioPredictiveClusteringEvaluation;

	KWStatisticalEvaluation* trainDistanceEvaluation;
	KWStatisticalEvaluation* testDistanceEvaluation;
	KWStatisticalEvaluation* ratioDistanceEvaluation;

	KWStatisticalEvaluation* trainLEVAEvaluation;
	KWStatisticalEvaluation* testLEVAEvaluation;
	KWStatisticalEvaluation* ratioLEVAEvaluation;

	KWStatisticalEvaluation* trainDaviesBouldinEvaluation;
	KWStatisticalEvaluation* testDaviesBouldinEvaluation;
	KWStatisticalEvaluation* ratioDaviesBouldinEvaluation;

	KWStatisticalEvaluation* trainInertyIntraEvaluation;
	KWStatisticalEvaluation* testInertyIntraEvaluation;
	KWStatisticalEvaluation* ratioInertyIntraEvaluation;

	KWStatisticalEvaluation* trainInertyInterEvaluation;
	KWStatisticalEvaluation* testInertyInterEvaluation;
	KWStatisticalEvaluation* ratioInertyInterEvaluation;

	KWStatisticalEvaluation* trainInertyTotalEvaluation;
	KWStatisticalEvaluation* testInertyTotalEvaluation;
	KWStatisticalEvaluation* ratioInertyTotalEvaluation;

	KWStatisticalEvaluation* trainNormalizedMutualInformationByClustersEvaluation;
	KWStatisticalEvaluation* testNormalizedMutualInformationByClustersEvaluation;
	KWStatisticalEvaluation* ratioNormalizedMutualInformationByClustersEvaluation;

	KWStatisticalEvaluation* trainNormalizedMutualInformationByClassesEvaluation;
	KWStatisticalEvaluation* testNormalizedMutualInformationByClassesEvaluation;
	KWStatisticalEvaluation* ratioNormalizedMutualInformationByClassesEvaluation;

	require(GetTargetAttributeType() == KWType::Symbol);

	// Acces a l'evaluation specialisee
	classifierEvaluation = cast(KMClassifierEvaluation*, predictorEvaluation);

	// Acces aux evaluations des criteres en train
	trainEVAEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainEVA"), nPredictor);
	trainARIByClustersEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainARIByClusters"), nPredictor);
	trainARIByClassesEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainARIByClasses"), nPredictor);
	trainVariationOfInformationEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainVariationOfInformation"), nPredictor);
	trainPredictiveClusteringEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainPredictiveClustering"), nPredictor);
	trainDistanceEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainDistance"), nPredictor);
	trainLEVAEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainLEVA"), nPredictor);
	trainDaviesBouldinEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainDaviesBouldin"), nPredictor);
	trainInertyIntraEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainInertyIntra"), nPredictor);
	trainInertyInterEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainInertyInter"), nPredictor);
	trainInertyTotalEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainInertyTotal"), nPredictor);
	trainNormalizedMutualInformationByClustersEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainNormalizedMutualInformationByClusters"), nPredictor);
	trainNormalizedMutualInformationByClassesEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TrainNormalizedMutualInformationByClasses"), nPredictor);

	// Acces aux evaluations des criteres en test
	testEVAEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestEVA"), nPredictor);
	testARIByClustersEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestARIByClusters"), nPredictor);
	testARIByClassesEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestARIByClasses"), nPredictor);
	testVariationOfInformationEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestVariationOfInformation"), nPredictor);
	testPredictiveClusteringEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestPredictiveClustering"), nPredictor);
	testDistanceEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestDistance"), nPredictor);
	testLEVAEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestLEVA"), nPredictor);
	testDaviesBouldinEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestDaviesBouldin"), nPredictor);
	testInertyIntraEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestInertyIntra"), nPredictor);
	testInertyInterEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestInertyInter"), nPredictor);
	testInertyTotalEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestInertyTotal"), nPredictor);
	testNormalizedMutualInformationByClustersEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestNormalizedMutualInformationByClusters"), nPredictor);
	testNormalizedMutualInformationByClassesEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("TestNormalizedMutualInformationByClasses"), nPredictor);

	// Acces aux criteres de type ratio test/train
	ratioEVAEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioEVA"), nPredictor);
	ratioARIByClustersEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioARIByClusters"), nPredictor);
	ratioARIByClassesEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioARIByClasses"), nPredictor);
	ratioVariationOfInformationEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioVariationOfInformation"), nPredictor);
	ratioPredictiveClusteringEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioPredictiveClustering"), nPredictor);
	ratioDistanceEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioDistance"), nPredictor);
	ratioLEVAEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioLEVA"), nPredictor);
	ratioDaviesBouldinEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioDaviesBouldin"), nPredictor);
	ratioInertyIntraEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioInertyIntra"), nPredictor);
	ratioInertyInterEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioInertyInter"), nPredictor);
	ratioInertyTotalEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioInertyTotal"), nPredictor);
	ratioNormalizedMutualInformationByClustersEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioNormalizedMutualInformationByClusters"), nPredictor);
	ratioNormalizedMutualInformationByClassesEvaluation = GetUpdatableEvaluationAt(
		GetCriterionIndexAt("RatioNormalizedMutualInformationByClasses"), nPredictor);



	// calcul des inerties (train ou test)

	double totalInertyInter = 0;
	double totalInertyIntra = 0;


	KMParameters::DistanceType distanceType = classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetParameters()->GetDistanceType();
	ObjectArray* clusters = classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusters();

	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));

		totalInertyInter += c->GetInertyInter(distanceType);
		totalInertyIntra += c->GetInertyIntra(distanceType);

	}

	double total = (1.0 / (double)classifierEvaluation->GetEvaluationInstanceNumber()) * classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetGlobalCluster()->GetDistanceSum(distanceType);


	// Memorisation des resultats d'evaluation en train
	if (bTrain)
	{
		trainEVAEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetEVA());
		trainARIByClustersEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetARIByClusters());
		trainARIByClassesEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetARIByClasses());
		trainNormalizedMutualInformationByClustersEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetNormalizedMutualInformationByClusters());
		trainNormalizedMutualInformationByClassesEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetNormalizedMutualInformationByClasses());
		trainVariationOfInformationEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetVariationOfInformation());
		trainPredictiveClusteringEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetPredictiveClustering());
		trainDistanceEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetMeanDistance());
		trainLEVAEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetLEVA());
		trainDaviesBouldinEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetDaviesBouldin());
		trainInertyIntraEvaluation->SetResultAt(nExperiment, nRun, totalInertyIntra);
		trainInertyInterEvaluation->SetResultAt(nExperiment, nRun, totalInertyInter);
		trainInertyTotalEvaluation->SetResultAt(nExperiment, nRun, total);
	}

	// Memorisation des resultats d'evaluation en test
	if (not bTrain)
	{
		testEVAEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetEVA());
		testARIByClustersEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetARIByClusters());
		testARIByClassesEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetARIByClasses());
		testNormalizedMutualInformationByClustersEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetNormalizedMutualInformationByClusters());
		testNormalizedMutualInformationByClassesEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetNormalizedMutualInformationByClasses());
		testVariationOfInformationEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetVariationOfInformation());
		testPredictiveClusteringEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetPredictiveClustering());
		testDistanceEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetMeanDistance());
		testLEVAEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetLEVA());
		testDaviesBouldinEvaluation->SetResultAt(nExperiment, nRun,
			classifierEvaluation->GetClassifierEvaluationTask()->GetClustering()->GetClusteringQuality()->GetDaviesBouldin());
		testInertyIntraEvaluation->SetResultAt(nExperiment, nRun, totalInertyIntra);
		testInertyInterEvaluation->SetResultAt(nExperiment, nRun, totalInertyInter);
		testInertyTotalEvaluation->SetResultAt(nExperiment, nRun, total);

	}

	// Memorisation des criteres de type ratio test/train
	if (not bTrain)
	{
		if (trainEVAEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioEVAEvaluation->SetResultAt(nExperiment, nRun,
				testEVAEvaluation->GetResultAt(nExperiment, nRun) /
				trainEVAEvaluation->GetResultAt(nExperiment, nRun));

		if (trainARIByClustersEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioARIByClustersEvaluation->SetResultAt(nExperiment, nRun,
				testARIByClustersEvaluation->GetResultAt(nExperiment, nRun) /
				trainARIByClustersEvaluation->GetResultAt(nExperiment, nRun));

		if (trainNormalizedMutualInformationByClustersEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioNormalizedMutualInformationByClustersEvaluation->SetResultAt(nExperiment, nRun,
				testNormalizedMutualInformationByClustersEvaluation->GetResultAt(nExperiment, nRun) /
				trainNormalizedMutualInformationByClustersEvaluation->GetResultAt(nExperiment, nRun));

		if (trainNormalizedMutualInformationByClassesEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioNormalizedMutualInformationByClassesEvaluation->SetResultAt(nExperiment, nRun,
				testNormalizedMutualInformationByClassesEvaluation->GetResultAt(nExperiment, nRun) /
				trainNormalizedMutualInformationByClassesEvaluation->GetResultAt(nExperiment, nRun));

		if (trainARIByClassesEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioARIByClassesEvaluation->SetResultAt(nExperiment, nRun,
				testARIByClassesEvaluation->GetResultAt(nExperiment, nRun) /
				trainARIByClassesEvaluation->GetResultAt(nExperiment, nRun));

		if (trainVariationOfInformationEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioVariationOfInformationEvaluation->SetResultAt(nExperiment, nRun,
				testVariationOfInformationEvaluation->GetResultAt(nExperiment, nRun) /
				trainVariationOfInformationEvaluation->GetResultAt(nExperiment, nRun));

		if (trainPredictiveClusteringEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioPredictiveClusteringEvaluation->SetResultAt(nExperiment, nRun,
				testPredictiveClusteringEvaluation->GetResultAt(nExperiment, nRun) /
				trainPredictiveClusteringEvaluation->GetResultAt(nExperiment, nRun));

		if (trainDistanceEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioDistanceEvaluation->SetResultAt(nExperiment, nRun,
				testDistanceEvaluation->GetResultAt(nExperiment, nRun) /
				trainDistanceEvaluation->GetResultAt(nExperiment, nRun));

		if (trainLEVAEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioLEVAEvaluation->SetResultAt(nExperiment, nRun,
				testLEVAEvaluation->GetResultAt(nExperiment, nRun) /
				trainLEVAEvaluation->GetResultAt(nExperiment, nRun));

		if (trainDaviesBouldinEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioDaviesBouldinEvaluation->SetResultAt(nExperiment, nRun,
				testDaviesBouldinEvaluation->GetResultAt(nExperiment, nRun) /
				trainDaviesBouldinEvaluation->GetResultAt(nExperiment, nRun));

		if (trainInertyIntraEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioInertyIntraEvaluation->SetResultAt(nExperiment, nRun,
				testInertyIntraEvaluation->GetResultAt(nExperiment, nRun) /
				trainInertyIntraEvaluation->GetResultAt(nExperiment, nRun));

		if (trainInertyInterEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioInertyInterEvaluation->SetResultAt(nExperiment, nRun,
				testInertyInterEvaluation->GetResultAt(nExperiment, nRun) /
				trainInertyInterEvaluation->GetResultAt(nExperiment, nRun));

		if (trainInertyTotalEvaluation->GetResultAt(nExperiment, nRun) != 0)
			ratioInertyTotalEvaluation->SetResultAt(nExperiment, nRun,
				testInertyTotalEvaluation->GetResultAt(nExperiment, nRun) /
				trainInertyTotalEvaluation->GetResultAt(nExperiment, nRun));

	}
}
