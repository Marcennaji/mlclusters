// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictorEvaluationTask.h"
#include "KMClusteringQuality.h"
#include "KMLearningProject.h"

////////////////////////////////////////////////////////////////////////////////
// Classe KMPredictorEvaluationTask

KMPredictorEvaluationTask::KMPredictorEvaluationTask()
{
	predictorEvaluation = NULL;
	kmGlobalCluster = NULL;
	iReadInstancesForMedianComputation = 0;
	kmEvaluationClustering = NULL;
	lInstancesWithMissingValues = 0;
	lInstanceEvaluationNumber = 0;
	odAttributesPartitions = NULL;
	odAtomicModalities = NULL;
}

KMPredictorEvaluationTask::~KMPredictorEvaluationTask()
{
	odGroupedModalitiesFrequencyTables.DeleteAll();
	odAtomicModalitiesFrequencyTables.DeleteAll();

	if (kmEvaluationClustering != NULL)
		delete kmEvaluationClustering;
}

boolean KMPredictorEvaluationTask::MasterInitialize()
{

	AddSimpleMessage("MLClusters internal version is " + ALString(INTERNAL_VERSION));
	return KWPredictorEvaluationTask::MasterInitialize();

}


boolean KMPredictorEvaluationTask::Evaluate(KWPredictor* predictor,
	KWDatabase* evaluationDatabase,
	KWPredictorEvaluation* requesterPredictorEvaluation)
{
	boolean bOk;
	const double dMinNecessaryMemory = 16 * 1024 * 1024;
	KWObject* kwoObject;
	longint nObject;
	KWLearningSpec currentLearningSpec;
	ALString sTmp;
	KMTrainedPredictor* trainedPredictor;

	require(predictor != NULL);
	require(predictor->IsTrained());
	require(evaluationDatabase != NULL);
	require(evaluationDatabase->GetObjects()->GetSize() == 0);

	// l'evaluation ne fait pas appel a la parallelisation, et se deroule entierement dans le master (i.e, on ne fait pas appel a RunDatabaseTask()  )

	Timer timer;
	timer.Start();


	// Initialisation des variables necessaires pour l'evaluation

	trainedPredictor = cast(KMTrainedPredictor*, predictor->GetTrainedPredictor());

	// on recupere le modele K-Means a partir du dico de deploiement, et on enrichit le dico de deploiement
	KMClustering* clustering = trainedPredictor->CreateModelingClustering();

	if (clustering == NULL)
		return false;

	kmEvaluationClustering = clustering->Clone();

	predictorEvaluation = requesterPredictorEvaluation;
	InitializePredictorSharedVariables(predictor);

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

		iReadInstancesForMedianComputation = 0;

		Global::ActivateErrorFlowControl();

		nObject = 0;
		while (not evaluationDatabase->IsEnd())
		{
			if (nObject % 100 == 0)
			{
				// Arret si plus assez de memoire
				if (RMResourceManager::GetRemainingAvailableMemory() < dMinNecessaryMemory)
				{
					bOk = false;
					AddError(sTmp + "Not enough memory: interrupted after evaluation of " + LongintToString(nObject) + " instances (remaining available memory = "
						+ LongintToHumanReadableString(RMResourceManager::GetRemainingAvailableMemory()) + ", min necessary memory = " + LongintToHumanReadableString(dMinNecessaryMemory));
					break;
				}
			}

			// controle memoire additionnel pour plus de securite (tenir compte de l'evolution de l'occupation memoire due a d'autres applications)
			// Arret du stockage des instances en vue du calcul des medianes, si le seuil de memoire dispo est devenu dangereusement bas
			if (bComputeMedians and nObject % 5 == 0 and RMResourceManager::GetRemainingAvailableMemory() < dMinNecessaryMemory * 2) {

				if (GetLearningExpertMode()) {
					AddWarning("Not enough memory : can't store any more database instances for median values computing. Instances number stored so far : "
						+ ALString(LongintToString(iReadInstancesForMedianComputation)) +
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
					iReadInstancesForMedianComputation++;
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

			for (int i = 0; i < kmEvaluationClustering->GetClusters()->GetSize(); i++)
			{
				KMCluster* c = cast(KMCluster*, kmEvaluationClustering->GetClusters()->GetAt(i));

				if (c->GetCount() > 0) {
					if (kmEvaluationClustering->GetParameters()->GetWriteDetailedStatistics())
						c->ComputeNativeAttributesContinuousMedianValues();
					c->DeleteAll(); // suppression des KWObject du cluster, ayant servi au calcul des medianes (gain memoire, car ils sont maintenant devenus inutiles)
				}
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

		AddSimpleMessage(ALString("Evaluation instances number (with no missing values after preprocessing) : ") + IntToString(lInstanceEvaluationNumber));
		AddSimpleMessage(ALString("Instances with missing values : ") + IntToString(kmEvaluationClustering->GetInstancesWithMissingValues()));

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

		kmEvaluationClustering->GetClusteringQuality()->ComputeDaviesBouldin();

	}

	bOk = MasterFinalize(bOk); // appele directement car on n'utilise pas le service d'execution parallele des taches

	CleanPredictorSharedVariables();

	return bOk;
}

KMCluster* KMPredictorEvaluationTask::UpdateEvaluationFirstDatabaseRead(KWPredictor* predictor, KWObject* kwoObject, const bool updateModalitiesProbs)
{
	require(predictor != NULL);
	require(kwoObject != NULL);

	if (kmEvaluationClustering->GetParameters()->HasMissingKMeanValue(kwoObject)) {
		kmEvaluationClustering->IncrementInstancesWithMissingValuesNumber();
		return NULL;
	}

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

	cluster->SetFrequency(cluster->GetFrequency() + 1);
	cluster->UpdateMeanCentroidValues(kwoObject, (ContinuousVector&)cluster->GetEvaluationCentroidValues());
	cluster->UpdateNativeAttributesContinuousMeanValues(kwoObject);

	lInstanceEvaluationNumber++;

	if (updateModalitiesProbs)
		UpdateModalitiesProbs(kwoObject, idCluster);

	return cluster;
}

KMCluster* KMPredictorEvaluationTask::UpdateEvaluationSecondDatabaseRead(KWObject* kwoObject)
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

	}

	return cluster;
}
void KMPredictorEvaluationTask::UpdateModalitiesProbs(const KWObject* kwoObject, const int idCluster) {

	// mise a jour pour modalites groupees

	POSITION position = odGroupedModalitiesFrequencyTables.GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		odGroupedModalitiesFrequencyTables.GetNextAssoc(position, key, oCurrent);

		KWFrequencyTable* table = cast(KWFrequencyTable*, oCurrent);
		assert(table != NULL);

		const KWAttribute* attribute = kwoObject->GetClass()->LookupAttribute(key);
		assert(attribute != NULL);

		if (attribute->GetDerivationRule() == NULL)
			continue;

		assert(attribute->GetLoadIndex().IsValid());

		int modalityIndex = -1;
		const Continuous value = kwoObject->GetContinuousValueAt(attribute->GetLoadIndex());

		// determiner l'index du groupement de modalit�s
		if (attribute->GetName().GetLength() >= 3 and attribute->GetName().Left(3) == "NRP")
			modalityIndex = (int)floor(value * table->GetFrequencyVectorNumber());
		else
			if (attribute->GetName().GetLength() >= 10 and attribute->GetName().Left(10) == "CellIndexP")
				modalityIndex = (int)value - 1;

		assert(modalityIndex != -1 and modalityIndex < table->GetFrequencyVectorNumber());

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(modalityIndex));
		fv->GetFrequencyVector()->SetAt(idCluster, fv->GetFrequencyVector()->GetAt(idCluster) + 1);
	}

	// idem pour modalites non groupees :

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
		ObjectArray* atomicModalities = cast(ObjectArray*, kmEvaluationClustering->GetAttributesPartitioningManager()->GetAtomicModalities().Lookup(attribute->GetName()));
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
			// incrementer le dernier poste ("Unseen values")
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(atomicModalities->GetSize() - 1));
			fv->GetFrequencyVector()->SetAt(idCluster, fv->GetFrequencyVector()->GetAt(idCluster) + 1);
		}
	}
}
boolean KMPredictorEvaluationTask::MasterFinalize(boolean bProcessEndedCorrectly)
{
	// reimplemente, car on ne fait pas appel a l'execution parallele des taches

	KMPredictorEvaluation* kmPredictorEvaluation = cast(KMPredictorEvaluation*, predictorEvaluation);
	kmPredictorEvaluation->SetInstanceEvaluationNumber(lInstanceEvaluationNumber);
	return bProcessEndedCorrectly;
}

void KMPredictorEvaluationTask::InitializeModalitiesProbs() {

	// initialiser le dictionnaire contenant les probas de modalit�s : chaque poste pointe sur un objet KWFrequencyTable, correspondant aux intervalles d'un attribut)

	odGroupedModalitiesFrequencyTables.DeleteAll();

	POSITION position = kmEvaluationClustering->GetAttributesPartitioningManager()->GetPartitions().GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		kmEvaluationClustering->GetAttributesPartitioningManager()->GetPartitions().GetNextAssoc(position, key, oCurrent);

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

	position = kmEvaluationClustering->GetAttributesPartitioningManager()->GetAtomicModalities().GetStartPosition();

	while (position != NULL) {

		kmEvaluationClustering->GetAttributesPartitioningManager()->GetAtomicModalities().GetNextAssoc(position, key, oCurrent);

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


const ALString KMPredictorEvaluationTask::GetTaskName() const
{
	return "MLClusters Predictor evaluation";
}

PLParallelTask* KMPredictorEvaluationTask::Create() const
{
	return new KMPredictorEvaluationTask;
}


