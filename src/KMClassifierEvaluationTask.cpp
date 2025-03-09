// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClassifierEvaluationTask.h"
#include "KMClusteringQuality.h"
#include "KMLearningProject.h"


KMClassifierEvaluationTask::KMClassifierEvaluationTask()
{
	classifierEvaluation = NULL;
	kmClassifierEvaluation = NULL;
	kmGlobalCluster = NULL;
	lReadInstancesForMedianComputation = 0;
	evaluationInstances = NULL;
	lInstancesWithMissingValues = 0;
	lInstanceEvaluationNumber = 0;
	odAttributesPartitions = NULL;
	odAtomicModalities = NULL;
	kmEvaluationClustering = NULL;
	targetAttribute = NULL;
}

KMClassifierEvaluationTask::~KMClassifierEvaluationTask()
{
	odGroupedModalitiesFrequencyTables.DeleteAll();
	odAtomicModalitiesFrequencyTables.DeleteAll();

	if (evaluationInstances != NULL) {
		evaluationInstances->DeleteAll();
		delete evaluationInstances;
	}
	if (kmEvaluationClustering != NULL)
		delete kmEvaluationClustering;
}


boolean KMClassifierEvaluationTask::Evaluate(KWPredictor* predictor,
	KWDatabase* evaluationDatabase,
	KWPredictorEvaluation* requesterPredictorEvaluation)
{
	boolean bOk;
	const longint lMinNecessaryMemory = 16 * 1024 * 1024;
	KWObject* kwoObject;
	longint nObject;
	KWLearningSpec currentLearningSpec;
	ALString sTmp;
	KMTrainedClassifier* trainedPredictor;

	require(predictor != NULL);
	require(predictor->IsTrained());
	require(evaluationDatabase != NULL);
	require(evaluationDatabase->GetObjects()->GetSize() == 0);

	// l'evaluation ne fait pas appel a la parallelisation, et se deroule entierement dans le master (i.e, on ne fait pas appel a RunDatabaseTask()  )

	Timer timer;
	timer.Start();


	// Initialisation des variables necessaires pour l'evaluation

	trainedPredictor = cast(KMTrainedClassifier*, predictor->GetTrainedClassifier());

	// on recupere le modele K-Means a partir du dico de deploiement, et on enrichit le dico de deploiement
	KMClustering* clustering = trainedPredictor->CreateModelingClustering();

	if (clustering == NULL)
		return false;

	kmEvaluationClustering = clustering->Clone();

	predictorEvaluation = requesterPredictorEvaluation;
	InitializePredictorSharedVariables(predictor);

	bOk = MasterInitialize(); // on n'utilise pas le service d'execution parallele des taches, donc on initialise nous-memes directement le maitre
	if (not bOk) {
		CleanPredictorSharedVariables();
		return false;
	}

	kmGlobalCluster = kmEvaluationClustering->GetGlobalCluster();

	assert(kmEvaluationClustering->GetParameters()->GetIdClusterAttribute() != NULL);
	assert(kmEvaluationClustering->GetParameters()->GetIdClusterAttribute()->GetLoadIndex().IsValid());

	lInstancesWithMissingValues = 0;

	odAttributesPartitions = ((ObjectDictionary*)&kmEvaluationClustering->GetAttributesPartitioningManager()->GetPartitions());
	odAtomicModalities = ((ObjectDictionary*)&kmEvaluationClustering->GetAttributesPartitioningManager()->GetAtomicModalities());

	lInstanceEvaluationNumber = 0;

	/////////////////////////////////////////////////////////////////////
	// Chargement de la base pour evaluation des criteres specifiques

	AddSimpleMessage(ALString("Evaluate database ") + evaluationDatabase->GetDatabaseName() +
		ALString(" with predictor ") + predictor->GetObjectLabel());

	// Debut de suivi de tache
	TaskProgression::BeginTask();
	TaskProgression::DisplayMainLabel("Evaluate database " + evaluationDatabase->GetDatabaseName());

	targetAttribute = predictor->GetClass()->LookupAttribute(predictor->GetTargetAttributeName());
	assert(targetAttribute != NULL);

	const longint estimatedObjectsNumber = evaluationDatabase->GetEstimatedObjectNumber();

	// Ouverture de la base en lecture
	bOk = evaluationDatabase->OpenForRead();

	const bool updateModalitiesProbs = (kmEvaluationClustering->GetAttributesPartitioningManager()->GetPartitions().GetCount() > 0 ? true : false);

	if (updateModalitiesProbs)
		InitializeModalitiesProbs();

	// Lecture d'objets dans la base
	if (bOk)
	{
		boolean bComputeMedians = kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics();

		int iReadPercentageForMedianComputation = KMPredictorEvaluation::ComputeReadPercentageForMedianComputation(kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics(),
			estimatedObjectsNumber, trainedPredictor->GetPredictorClass());
		if (GetLearningExpertMode() and bComputeMedians and iReadPercentageForMedianComputation < 100) {
			AddWarning("Not enough memory : can't store 100% of database instances for median values computing. Median will be computed on "
				+ ALString(IntToString(iReadPercentageForMedianComputation)) + "% of database. Other statistics will still be computed on 100% of database instances.");
		}

		lReadInstancesForMedianComputation = 0;

		Global::ActivateErrorFlowControl();

		nObject = 0;
		while (not evaluationDatabase->IsEnd())
		{
			if (nObject % 100 == 0)
			{
				// Arret si plus assez de memoire
				if (RMResourceManager::GetRemainingAvailableMemory() < lMinNecessaryMemory)
				{
					bOk = false;
					AddError(sTmp + "Not enough memory: interrupted after evaluation of " + LongintToString(nObject) + " instances (remaining available memory = "
						+ LongintToHumanReadableString(RMResourceManager::GetRemainingAvailableMemory()) + ", min necessary memory = " + LongintToHumanReadableString(lMinNecessaryMemory));
					break;
				}
			}

			// controle memoire additionnel pour plus de securite (tenir compte de l'evolution de l'occupation memoire due a d'autres applications)
			// Arret du stockage des instances en vue du calcul des medianes, si le seuil de memoire dispo est devenu dangereusement bas
			if (bComputeMedians and nObject % 5 == 0 and RMResourceManager::GetRemainingAvailableMemory() < lMinNecessaryMemory * 2) {

				if (GetLearningExpertMode()) {
					AddWarning("Not enough memory : can't store any more database instances for median values computing. Instances number stored so far : "
						+ ALString(LongintToString(lReadInstancesForMedianComputation)) +
						", total number of read instances : " + ALString(LongintToString(nObject)));
				}
				bComputeMedians = false;
			}

			// Traitement d'un nouvel objet
			kwoObject = evaluationDatabase->Read();

			if (kwoObject != NULL)
			{
				nObject++;

				// Mise a jour de l'evaluation : affecte l'instance au cluster correspondant, et maj, le cas echeant, des centroides d'evaluation MOYENNES (sans toucher aux centroides initiaux, issus du modele)
				KMCluster* cluster = UpdateEvaluationFirstDatabaseRead(predictor, kwoObject, updateModalitiesProbs);

				if (cluster == NULL) {
					// pas d'affectation possible, passer a l'instance suivante
					delete kwoObject;
					continue;
				}

				bool bKeepInstanceForMedianComputing = false;

				if (bComputeMedians) {

					// On teste si on garde ou non l'exemple afin de calculer les medianes (valeurs par attributs)
					// NB. si le cluster est encore vide a ce stade, on lui affecte l'instance sans faire de tirage aleatoire

					if (cluster->GetCount() == 0)
						bKeepInstanceForMedianComputing = true;
					else {
						const int nRandom = 1 + IthRandomInt(nObject, 99);
						bKeepInstanceForMedianComputing = (nRandom <= iReadPercentageForMedianComputation ? true : false);
					}
				}

				if (not bKeepInstanceForMedianComputing)
					delete kwoObject;
				else {
					// stocker cette instance dans le cluster, afin d'effectuer par la suite le calcul des medianes
					lReadInstancesForMedianComputation++;
					cluster->AddInstance(kwoObject);
					kmGlobalCluster->AddInstance(kwoObject);
				}
			}

			// Arret si erreur ou interruption
			if (evaluationDatabase->IsError() or
				(nObject % 100 == 0 and TaskProgression::IsInterruptionRequested()))
			{
				bOk = false;
				break;
			}
		}

		if (bOk) {

			if (kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics() and kmGlobalCluster->GetCount() > 0)
				kmGlobalCluster->ComputeNativeAttributesContinuousMedianValues();

			kmGlobalCluster->ComputeMajorityTargetValue(kmEvaluationClustering->GetTargetAttributeValues());

			for (int i = 0; i < kmEvaluationClustering->GetClusters()->GetSize(); i++)
			{
				KMCluster* c = cast(KMCluster*, kmEvaluationClustering->GetClusters()->GetAt(i));

				if (c->GetCount() > 0) {
					if (kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics())
						c->ComputeNativeAttributesContinuousMedianValues();
					c->DeleteAll(); // suppression des KWObject du cluster, ayant servi au calcul des medianes (gain memoire, car ils sont maintenant devenus inutiles)
				}

				c->ComputeMajorityTargetValue(kmEvaluationClustering->GetTargetAttributeValues());

			}

			// recalculer les distances entre clusters, sur la base des centroides d'evaluation qui viennent d'etre calcules
			kmEvaluationClustering->ComputeClustersCentersDistances(true);

			// 2eme lecture de base, afin de mettre a jour les stats qui dependent des centroides d'evaluation

			evaluationDatabase->Close();
			evaluationDatabase->OpenForRead();

			nObject = 0;
			while (not evaluationDatabase->IsEnd())
			{
				kwoObject = evaluationDatabase->Read();

				if (kwoObject != NULL)
				{
					nObject++;

					// Mise a jour de l'evaluation : affecte l'instance au cluster correspondant, et maj la somme des distances ainsi que les inerties intra, en fonction des nouveaux centroides
					UpdateEvaluationSecondDatabaseRead(kwoObject);

					delete kwoObject;

				}

				// Arret si erreur ou interruption
				if (evaluationDatabase->IsError() or
					(nObject % 100 == 0 and TaskProgression::IsInterruptionRequested()))
				{
					bOk = false;
					break;
				}
			}
		}

		Global::DesactivateErrorFlowControl();

		TaskProgression::EndTask();

		AddSimpleMessage(ALString("Evaluation instances number (with no missing values after preprocessing) : ") + LongintToString(lInstanceEvaluationNumber));
		AddSimpleMessage(ALString("Instances with missing values : ") + LongintToString(kmEvaluationClustering->GetInstancesWithMissingValues()));

		// Fermeture
		bOk = evaluationDatabase->Close() and bOk;
	}

	if (lInstanceEvaluationNumber > 0) {

		kmGlobalCluster->FinalizeStatisticsUpdateFromInstances();

		for (int i = 0; i < kmEvaluationClustering->GetClusters()->GetSize(); i++)
		{
			KMCluster* c = cast(KMCluster*, kmEvaluationClustering->GetClusters()->GetAt(i));
			c->FinalizeStatisticsUpdateFromInstances();
			c->ComputeInertyInter(KMParameters::L2Norm, kmGlobalCluster->GetEvaluationCentroidValues(), kmGlobalCluster->GetFrequency(), true);
			c->ComputeInertyInter(KMParameters::L1Norm, kmGlobalCluster->GetEvaluationCentroidValues(), kmGlobalCluster->GetFrequency(), true);
			c->ComputeInertyInter(KMParameters::CosineNorm, kmGlobalCluster->GetEvaluationCentroidValues(), kmGlobalCluster->GetFrequency(), true);
		}

		kmEvaluationClustering->UpdateGlobalDistancesSum();

		TaskProgression::DisplayLabel("Computing clusters quality indicators");

		kmEvaluationClustering->GetClusteringQuality()->ComputeARIByClusters(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues());
		kmEvaluationClustering->GetClusteringQuality()->ComputeDaviesBouldin();
		kmEvaluationClustering->GetClusteringQuality()->ComputePredictiveClustering(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues(), targetAttribute, true);

		if ((GetLearningExpertMode() and kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics()) or
			kmEvaluationClustering->GetParameters()->GetReplicateChoice() == KMParameters::EVA)
			kmEvaluationClustering->GetClusteringQuality()->ComputeEVA(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues().GetSize());

		if ((GetLearningExpertMode() and kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics()) or
			kmEvaluationClustering->GetParameters()->GetReplicateChoice() == KMParameters::LEVA)
			kmEvaluationClustering->GetClusteringQuality()->ComputeLEVA(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues());

		if ((GetLearningExpertMode() and kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics()) or
			kmEvaluationClustering->GetParameters()->GetReplicateChoice() == KMParameters::ARIByClasses)
			kmEvaluationClustering->GetClusteringQuality()->ComputeARIByClasses(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues(), kmEvaluationClustering->GetConfusionMatrix());

		if ((GetLearningExpertMode() and kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics()) or
			kmEvaluationClustering->GetParameters()->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters)
			kmEvaluationClustering->GetClusteringQuality()->ComputeNormalizedMutualInformationByClusters(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues());

		if ((GetLearningExpertMode() and kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics()) or
			kmEvaluationClustering->GetParameters()->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses)
			kmEvaluationClustering->GetClusteringQuality()->ComputeNormalizedMutualInformationByClasses(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues(), kmEvaluationClustering->GetConfusionMatrix());

		if ((GetLearningExpertMode() and kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics()) or
			kmEvaluationClustering->GetParameters()->GetReplicateChoice() == KMParameters::VariationOfInformation)
			kmEvaluationClustering->GetClusteringQuality()->ComputeVariationOfInformation(kmEvaluationClustering->GetGlobalCluster(), kmEvaluationClustering->GetTargetAttributeValues());

	}

	bOk = MasterFinalize(bOk); // appele directement, car on n'utilise pas le service d'execution parallele des taches

	CleanPredictorSharedVariables();

	return bOk;
}


const ALString KMClassifierEvaluationTask::GetTaskName() const
{
	return "Enneade Classifier evaluation";
}

PLParallelTask* KMClassifierEvaluationTask::Create() const
{
	return new KMClassifierEvaluationTask;
}


boolean KMClassifierEvaluationTask::MasterInitialize()
{
	// reimplemente car n'utilise pas le service de parallelisation de taches

	boolean bOk = true;
	int nTargetValue;
	ALString sTmp;

	require(masterConfMatrixEvaluation == NULL);
	require(slaveConfusionMatrixEvaluation == NULL);
	require(masterAucEvaluation == NULL);
	require(masterInstanceEvaluationSampler == NULL);
	require(evaluationInstances == NULL);

	AddSimpleMessage("Khiops Enneade internal version is " + ALString(INTERNAL_VERSION));

	// Memorisation de la specialisation du rapport d'evaluation demandeur
	classifierEvaluation = cast(KWClassifierEvaluation*, predictorEvaluation);
	kmClassifierEvaluation = cast(KMClassifierEvaluation*, predictorEvaluation);
	kmClassifierEvaluation->dCompressionRate = 0;

	// Initialisation du service d'evaluation de la matrice de confusion
	masterConfMatrixEvaluation = new KWConfusionMatrixEvaluation;
	masterConfMatrixEvaluation->Initialize();
	for (nTargetValue = 0; nTargetValue < shared_nTargetValueNumber; nTargetValue++)
		masterConfMatrixEvaluation->AddPredictedTarget(shared_svPredictedModalities.GetAt(nTargetValue));


	// Initialisation des courbes de lift pour l'ensemble des modalites
	assert(kmClassifierEvaluation->oaAllLiftCurveValues.GetSize() == 0);
	for (nTargetValue = 0; nTargetValue < shared_nTargetValueNumber; nTargetValue++)
	{
		// Arret et warning si le maximum de courbes est atteint
		if (nTargetValue == nMaxLiftEvaluationNumber)
		{
			AddWarning(sTmp
				+ "The lift curves will be computed only for " + IntToString(nMaxLiftEvaluationNumber)
				+ " values (among " + IntToString(shared_nTargetValueNumber) + ")");
			break;
		}
		kmClassifierEvaluation->oaAllLiftCurveValues.Add(new DoubleVector);
	}

	// Initialisation du compteur des echantillons d'evaluation des esclaves pour le calcul d'AUC et courbes de lift
	nCurrentSample = 0;

	// Initialisation du service de calcul de l'AUC
	evaluationInstances = new ObjectArray;
	bIsAucEvaluated = shared_livProbAttributes.GetSize() > 0 and shared_nTargetValueNumber > 0;
	masterAucEvaluation = new KWAucEvaluation;
	masterAucEvaluation->SetTargetValueNumber(shared_nTargetValueNumber);

	ensure(Check());

	return bOk;
}

boolean KMClassifierEvaluationTask::MasterFinalize(boolean bProcessEndedCorrectly)
{
	// reimplemente car n'utilise pas le service de parallelisation de taches

	const int nPartileNumber = 1000;
	boolean bOk = true;
	int nLiftCurve;
	int nPredictorTarget;
	DoubleVector* dvLiftCurveValues;

	assert(kmClassifierEvaluation != NULL);
	kmClassifierEvaluation->SetInstanceEvaluationNumber(lInstanceEvaluationNumber);

	// Memorisation de la matrice de confusion
	assert(masterConfMatrixEvaluation->Check());
	masterConfMatrixEvaluation->ExportDataGridStats(&kmClassifierEvaluation->dgsConfusionMatrix);
	kmClassifierEvaluation->dgsConfusionMatrix.ExportAttributePartFrequenciesAt(1, &kmClassifierEvaluation->ivActualModalityFrequencies);

	// Calcul et memorisation des taux de prediction
	kmClassifierEvaluation->dAccuracy = masterConfMatrixEvaluation->ComputeAccuracy();
	kmClassifierEvaluation->dBalancedAccuracy = masterConfMatrixEvaluation->ComputeBalancedAccuracy();
	kmClassifierEvaluation->dMajorityAccuracy = masterConfMatrixEvaluation->ComputeMajorityAccuracy();
	kmClassifierEvaluation->dTargetEntropy = masterConfMatrixEvaluation->ComputeTargetEntropy();

	// Calcul et memorisation du taux de compression
	if (shared_livProbAttributes.GetSize() > 0 and kmClassifierEvaluation->lInstanceEvaluationNumber > 0)
	{
		// Normalisation par rapport a l'entropie cible
		if (kmClassifierEvaluation->dTargetEntropy > 0) {
			kmClassifierEvaluation->dCompressionRate = 1.0
				- kmClassifierEvaluation->dCompressionRate
				/ (kmClassifierEvaluation->lInstanceEvaluationNumber * kmClassifierEvaluation->dTargetEntropy);
		}
		else
			kmClassifierEvaluation->dCompressionRate = 0;

		// On arrondit si necessaire a 0
		if (fabs(kmClassifierEvaluation->dCompressionRate)
			< kmClassifierEvaluation->dTargetEntropy / kmClassifierEvaluation->lInstanceEvaluationNumber)
			kmClassifierEvaluation->dCompressionRate = 0;
	}

	// Calcul de l'AUC s'il y y a des instances en evaluation
	if (bIsAucEvaluated and lInstanceEvaluationNumber > 0)
	{
		masterAucEvaluation->SetInstanceEvaluations(evaluationInstances);
		if (shared_livProbAttributes.GetSize() > 0 and masterAucEvaluation->GetTargetValueNumber() > 0)
			kmClassifierEvaluation->dAUC = masterAucEvaluation->ComputeGlobalAUCValue();

		// Calcul des courbes de lift
		for (nLiftCurve = 0; nLiftCurve < kmClassifierEvaluation->oaAllLiftCurveValues.GetSize(); nLiftCurve++)
		{
			dvLiftCurveValues = cast(DoubleVector*, kmClassifierEvaluation->oaAllLiftCurveValues.GetAt(nLiftCurve));

			// L'index de lift de la modalite est celui de la modalite directement, sauf si la derniere
			// courbe memorise la courbe pour la modalite cible principale
			nPredictorTarget = GetPredictorTargetIndexAtLiftCurveIndex(nLiftCurve);

			// Si l'on a avant la modalite cible est au debut
			masterAucEvaluation->ComputeLiftCurveAt(nPredictorTarget, nPartileNumber, dvLiftCurveValues);
		}
	}


	// Nettoyage
	kmClassifierEvaluation = NULL;
	delete masterConfMatrixEvaluation;
	masterConfMatrixEvaluation = NULL;
	delete masterAucEvaluation;
	masterAucEvaluation = NULL;
	delete masterInstanceEvaluationSampler;
	masterInstanceEvaluationSampler = NULL;
	dMasterSamplingProb = -1.0;
	nCurrentSample = -1;
	oaAllInstanceEvaluationSamples.DeleteAll();

	if (evaluationInstances != NULL) {
		evaluationInstances->DeleteAll();
		delete evaluationInstances;
		evaluationInstances = NULL;
	}

	return bOk;
}

KMCluster* KMClassifierEvaluationTask::UpdateEvaluationFirstDatabaseRead(KWPredictor* predictor, KWObject* kwoObject, const bool updateModalitiesProbs)
{
	const Continuous cEpsilon = (Continuous)1e-6;
	int nActualValueIndex;
	int i;
	Symbol sActualTargetValue;
	Symbol sPredictedTargetValue;
	Continuous cActualTargetValueProb;
	KWClassifierInstanceEvaluation* instanceEvaluation;

	require(predictor != NULL);
	require(kwoObject != NULL);
	require(targetAttribute != NULL);
	require(kmClassifierEvaluation != NULL);

	if (kmEvaluationClustering->GetParameters()->HasMissingKMeanValue(kwoObject)) {
		kmEvaluationClustering->IncrementInstancesWithMissingValuesNumber();
		return NULL;
	}

	// si une valeur cible est presente dans le fichier evalue mais pas dans le modele, alors il faut la referencer dans tous les clusters, afin de produire des stats dessus
	kmEvaluationClustering->AddTargetAttributeValueIfNotExists(targetAttribute, kwoObject);

	const int idCluster = (int)kwoObject->GetContinuousValueAt(kmEvaluationClustering->GetParameters()->GetIdClusterAttribute()->GetLoadIndex()) - 1;

	if (idCluster > kmEvaluationClustering->GetClusters()->GetSize()) {
		AddError("UpdateEvaluation : Cluster number " + ALString(IntToString(idCluster + 1)) + " does not exist.");
		// ne doit pas arriver, sauf si on a utilis� par erreur un dico de modelisation en mode benchmark, a la place d'un dico natif
		return NULL;
	}

	KMCluster* cluster = kmEvaluationClustering->GetCluster(idCluster);

	if (kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics()) {
		if (kmEvaluationClustering->GetParameters()->HasMissingNativeValue(kwoObject)) {
			cluster->IncrementInstancesWithMissingNativeValuesNumber(kwoObject);
			kmGlobalCluster->IncrementInstancesWithMissingNativeValuesNumber(kwoObject);
		}
	}

	kmGlobalCluster->SetFrequency(kmGlobalCluster->GetFrequency() + 1);
	kmGlobalCluster->UpdateMeanCentroidValues(kwoObject, (ContinuousVector&)kmGlobalCluster->GetEvaluationCentroidValues());
	kmGlobalCluster->UpdateNativeAttributesContinuousMeanValues(kwoObject);
	kmGlobalCluster->UpdateTargetProbs((ObjectArray&)kmEvaluationClustering->GetTargetAttributeValues(), targetAttribute, kwoObject);

	cluster->SetFrequency(cluster->GetFrequency() + 1);
	cluster->UpdateMeanCentroidValues(kwoObject, (ContinuousVector&)cluster->GetEvaluationCentroidValues());
	cluster->UpdateNativeAttributesContinuousMeanValues(kwoObject);
	cluster->UpdateTargetProbs((ObjectArray&)kmEvaluationClustering->GetTargetAttributeValues(), targetAttribute, kwoObject);

	lInstanceEvaluationNumber++;

	// Obtention des modalites predites et effectives
	assert(shared_liTargetAttribute.GetValue().IsValid());
	assert(shared_liPredictionAttribute.GetValue().IsValid());
	sActualTargetValue = kwoObject->GetSymbolValueAt(shared_liTargetAttribute.GetValue());
	sPredictedTargetValue = kwoObject->GetSymbolValueAt(shared_liPredictionAttribute.GetValue());

	// mise a jour de la matrice specifique kmean, qui servira a calculer les ARI par classes et NMI par classes
	kmEvaluationClustering->UpdateConfusionMatrix(sPredictedTargetValue, sActualTargetValue);

	// Mise a jour de la matrice de confusion (directement dans la matrice du maitre, puisqu'on n'utilise pas d'esclaves)
	masterConfMatrixEvaluation->AddInstanceEvaluation(sPredictedTargetValue, sActualTargetValue);

	// Recherche de l'index en apprentissage de la modalite effective
	// Par defaut: le nombre de modalites cible en apprentissage
	// (signifie valeur cible inconnue en apprentissage)
	nActualValueIndex = shared_nTargetValueNumber;
	if (shared_livProbAttributes.GetSize() > 0)
	{
		for (i = 0; i < shared_nTargetValueNumber; i++)
		{
			if (shared_svPredictedModalities.GetAt(i) == sActualTargetValue)
			{
				nActualValueIndex = i;
				break;
			}
		}
	}

	// Mise a jour du taux de compression si pertinente
	if (shared_livProbAttributes.GetSize() > 0)
	{
		// Recherche de la probabilite predite pour la valeur cible reelle
		cActualTargetValueProb = 0;
		if (nActualValueIndex < shared_livProbAttributes.GetSize())
		{
			assert(kwoObject->GetSymbolValueAt(shared_liTargetAttribute.GetValue()) ==
				shared_svPredictedModalities.GetAt(nActualValueIndex));
			cActualTargetValueProb =
				kwoObject->GetContinuousValueAt(shared_livProbAttributes.GetAt(nActualValueIndex));

			// On projete sur [0, 1] pour avoir une probabilite quoi qu'il arrive
			if (cActualTargetValueProb < cEpsilon)
				cActualTargetValueProb = cEpsilon;
			if (cActualTargetValueProb > 1)
				cActualTargetValueProb = 1;
		}
		// Si la valeur etait inconnue en apprentissage, on lui associe une probabilite minimale
		else
			cActualTargetValueProb = cEpsilon;

		// Ajout du log negatif de cette probabilite a l'evaluation des scores
		kmClassifierEvaluation->dCompressionRate -= log(cActualTargetValueProb);
	}

	// Collecte des informations necessaires a l'estimation de l'AUC et aux courbes de lift
	if (bIsAucEvaluated)
	{
		instanceEvaluation = new KWClassifierInstanceEvaluation;
		instanceEvaluation->SetTargetValueNumber(shared_nTargetValueNumber);
		instanceEvaluation->SetActualTargetIndex(nActualValueIndex);
		for (i = 0; i < shared_nTargetValueNumber; i++)
		{
			instanceEvaluation->SetTargetProbAt(i,
				kwoObject->GetContinuousValueAt(shared_livProbAttributes.GetAt(i)));
		}
		evaluationInstances->Add(instanceEvaluation);
	}

	if (updateModalitiesProbs)
		UpdateModalitiesProbs(kwoObject, idCluster);

	return cluster;
}


KMCluster* KMClassifierEvaluationTask::UpdateEvaluationSecondDatabaseRead(KWObject* kwoObject)
{
	require(kwoObject != NULL);
	assert(kmEvaluationClustering != NULL);

	if (kmEvaluationClustering->GetParameters()->HasMissingKMeanValue(kwoObject)) {
		return NULL;
	}

	const int idCluster = (int)kwoObject->GetContinuousValueAt(kmEvaluationClustering->GetParameters()->GetIdClusterAttribute()->GetLoadIndex()) - 1;

	if (idCluster > kmEvaluationClustering->GetClusters()->GetSize()) {
		AddError("UpdateEvaluation : Cluster number " + ALString(IntToString(idCluster + 1)) + " does not exist.");
		// ne doit pas arriver, sauf si on a utilis� par erreur un dico de modelisation en mode benchmark, a la place d'un dico natif
		return NULL;
	}

	KMCluster* cluster = kmEvaluationClustering->GetCluster(idCluster);

	if (cluster->GetEvaluationCentroidValues().GetSize() > 0) {

		kmGlobalCluster->UpdateDistanceSum(KMParameters::L1Norm, kwoObject, kmGlobalCluster->GetEvaluationCentroidValues());
		kmGlobalCluster->UpdateDistanceSum(KMParameters::L2Norm, kwoObject, kmGlobalCluster->GetEvaluationCentroidValues());
		kmGlobalCluster->UpdateDistanceSum(KMParameters::CosineNorm, kwoObject, kmGlobalCluster->GetEvaluationCentroidValues());

		cluster->UpdateInertyIntra(KMParameters::L1Norm, kwoObject, cluster->GetEvaluationCentroidValues());
		cluster->UpdateInertyIntra(KMParameters::L2Norm, kwoObject, cluster->GetEvaluationCentroidValues());
		cluster->UpdateInertyIntra(KMParameters::CosineNorm, kwoObject, cluster->GetEvaluationCentroidValues());

		cluster->UpdateDistanceSum(KMParameters::L1Norm, kwoObject, cluster->GetEvaluationCentroidValues());
		cluster->UpdateDistanceSum(KMParameters::L2Norm, kwoObject, cluster->GetEvaluationCentroidValues());
		cluster->UpdateDistanceSum(KMParameters::CosineNorm, kwoObject, cluster->GetEvaluationCentroidValues());

		cluster->UpdateCompactness(kwoObject, kmEvaluationClustering->GetTargetAttributeValues(), targetAttribute, cluster->GetEvaluationCentroidValues());
	}

	return cluster;
}

void KMClassifierEvaluationTask::UpdateModalitiesProbs(const KWObject* kwoObject, const int idCluster) {

	// mise a jour pour modalites groupees :
	POSITION position = odGroupedModalitiesFrequencyTables.GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		odGroupedModalitiesFrequencyTables.GetNextAssoc(position, key, oCurrent);

		KWFrequencyTable* table = cast(KWFrequencyTable*, oCurrent);

		if (table != NULL) {

			const KWAttribute* attribute = kwoObject->GetClass()->LookupAttribute(key);
			assert(attribute != NULL);
			assert(attribute->GetLoadIndex().IsValid());

			const Continuous value = kwoObject->GetContinuousValueAt(attribute->GetLoadIndex());

			const int modalityIndex = (int)value - 1;

			assert(modalityIndex != -1 and modalityIndex < table->GetFrequencyVectorNumber());

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(modalityIndex));
			fv->GetFrequencyVector()->SetAt(idCluster, fv->GetFrequencyVector()->GetAt(idCluster) + 1);
		}
	}

	// idem pour modalit�s non group�es :

	position = odAtomicModalitiesFrequencyTables.GetStartPosition();

	while (position != NULL) {

		odAtomicModalitiesFrequencyTables.GetNextAssoc(position, key, oCurrent);

		const KWAttribute* attribute = kwoObject->GetClass()->LookupAttribute(key);
		assert(attribute != NULL);

		if (attribute->GetDerivationRule() == NULL)
			continue;

		KWFrequencyTable* table = cast(KWFrequencyTable*, odAtomicModalitiesFrequencyTables.Lookup(attribute->GetName()));

		assert(table != NULL);

		// determiner l'index de la modalite lue
		// Si la valeur n'est pas r�pertori�e, on affecte � la modalit� "Other"
		KWAttribute* nativeAttribute = attribute->GetDerivationRule()->GetSecondOperand()->GetOriginAttribute();
		ObjectArray* atomicModalities = cast(ObjectArray*, odAtomicModalities->Lookup(attribute->GetName()));
		assert(atomicModalities != NULL);

		const ALString targetModality = kwoObject->GetSymbolValueAt(nativeAttribute->GetLoadIndex()).GetValue();

		bool found = false;

		for (int i = 0; i < atomicModalities->GetSize(); i++) {
			StringObject* s = cast(StringObject*, atomicModalities->GetAt(i));
			if (s->GetString() == targetModality) {
				KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(i));
				fv->GetFrequencyVector()->SetAt(idCluster, fv->GetFrequencyVector()->GetAt(idCluster) + 1);
				found = true;
				break;
			}
		}

		if (not found) {
			// incrementer le dernier poste ("Unseen")
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(atomicModalities->GetSize() - 1));
			fv->GetFrequencyVector()->SetAt(idCluster, fv->GetFrequencyVector()->GetAt(idCluster) + 1);
		}
	}
}

void KMClassifierEvaluationTask::InitializeModalitiesProbs() {

	// initialiser le dictionnaire contenant les probas de modalit�s : chaque poste pointe sur un objet KWFrequencyTable, correspondant aux intervalles d'un attribut)

	odGroupedModalitiesFrequencyTables.DeleteAll();

	POSITION position = odAttributesPartitions->GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		odAttributesPartitions->GetNextAssoc(position, key, oCurrent);

		ObjectArray* oaModalities = cast(ObjectArray*, oCurrent);
		KWFrequencyTable* table = new KWFrequencyTable;
		table->SetFrequencyVectorNumber(oaModalities->GetSize());
		for (int i = 0; i < table->GetFrequencyVectorNumber(); i++) {
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(i));
			fv->GetFrequencyVector()->SetSize(kmEvaluationClustering->GetClusters()->GetSize());
		}
		odGroupedModalitiesFrequencyTables.SetAt(key, table);
	}


	// idem pour les modalit�s non group�es

	odAtomicModalitiesFrequencyTables.DeleteAll();

	position = odAtomicModalities->GetStartPosition();

	while (position != NULL) {

		odAtomicModalities->GetNextAssoc(position, key, oCurrent);

		ObjectArray* oaModalities = cast(ObjectArray*, oCurrent);
		KWFrequencyTable* table = new KWFrequencyTable;
		table->SetFrequencyVectorNumber(oaModalities->GetSize());
		for (int i = 0; i < table->GetFrequencyVectorNumber(); i++) {
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(i));
			fv->GetFrequencyVector()->SetSize(kmEvaluationClustering->GetClusters()->GetSize());
		}
		odAtomicModalitiesFrequencyTables.SetAt(key, table);
	}
}


