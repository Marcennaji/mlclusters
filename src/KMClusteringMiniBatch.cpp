// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClusteringMiniBatch.h"
#include "KMClusteringQuality.h"

KMClusteringMiniBatch::KMClusteringMiniBatch(KMParameters* p) : KMClustering(p)
{
}

KMClusteringMiniBatch::~KMClusteringMiniBatch(void)
{
}

bool KMClusteringMiniBatch::ComputeReplicate(KWDatabase* database, const KWAttribute* targetAttribute, const int iMiniBatchesNumber,
	const int originDatabaseSamplePercentage, const int miniBatchDatabaseSamplePercentage)
{
	Timer timer;
	timer.Start();

	if (kmGlobalCluster->GetFrequency() == 0) {
		// NB. ne pas utiliser GetCount(), car les instances ne sont pas gard�es dans le cluster, seuls les stats et centroides sont gard�s
		AddWarning("All database instances have at least one missing value. Try to preprocess the values.");
		return false;
	}

	if (parameters->GetVerboseMode() and GetInstancesWithMissingValues() > 0)
		AddSimpleMessage(ALString("Instances with missing values, detected during clusters initialization : ") + LongintToString(GetInstancesWithMissingValues()));

	ContinuousVector cvClustersTotalCount;

	// modifier le sample en fonction du nombre d'instances de chaque mini batch (calcule precedemment)
	database->SetSampleNumberPercentage(miniBatchDatabaseSamplePercentage);
	database->SetSilentMode(true);

	for (int iIteration = 0; iIteration < iMiniBatchesNumber; iIteration++) {

		database->DeleteAll();
		database->ReadAll();// lecture partielle de la base
		ObjectArray* miniBatchInstances = database->GetObjects();
		miniBatchInstances->Shuffle();
		const int nbInstances = miniBatchInstances->GetSize();

		if (parameters->GetKValue() > nbInstances) {
			AddWarning("K parameter (" + ALString(IntToString(parameters->GetKValue())) +
				") is greater than the number of instances in mini-batch (" + ALString(IntToString(nbInstances)) +
				"), setting K value to " + ALString(IntToString(nbInstances)));
			parameters->SetKValue(nbInstances);
		}

		if (iIteration == 0) {
			// a la premiere iteration : repartition initiale des instances du mini-batch entre les clusters, selon la methode parametree par l'utilisateur, et
			// calcul des centroides initiaux
			if (not InitializeClusters(parameters->GetClustersCentersInitializationMethod(), miniBatchInstances, targetAttribute)) {
				AddMessage("Failed to initialize clusters");
				return false;
			}
			cvClustersTotalCount.SetSize(kmClusters->GetSize());
			cvClustersTotalCount.Initialize();
		}
		else
			// vider les clusters de leurs anciennes instances, et affecter les instances du minibatch aux centroides ayant ete modifies a l'iteration precedente
			AddInstancesToClusters(miniBatchInstances);

		// parcourir les instances de chaque cluster et faire evoluer la valeur des centroides
		for (int idxCluster = 0; idxCluster < kmClusters->GetSize(); idxCluster++) {

			KMCluster* cluster = cast(KMCluster*, kmClusters->GetAt(idxCluster));

			NUMERIC key;
			Object* oCurrent;
			POSITION position = cluster->GetStartPosition();

			// parcours des instances du cluster
			while (position != NULL) {

				cluster->GetNextAssoc(position, key, oCurrent);
				KWObject* currentInstance = static_cast<KWObject *>(oCurrent);

				cvClustersTotalCount.SetAt(idxCluster, cvClustersTotalCount.GetAt(idxCluster) + 1);

				const Continuous cLearningRate = 1.0 / (double)cvClustersTotalCount.GetAt(idxCluster);

				ContinuousVector cvUpdatedCentroidValues;
				cvUpdatedCentroidValues.CopyFrom(&cluster->GetInitialCentroidValues());

				for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++) {
					const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
					if (loadIndex.IsValid()) {
						cvUpdatedCentroidValues.SetAt(i, ((1 - cLearningRate) * cvUpdatedCentroidValues.GetAt(i)) + (cLearningRate * currentInstance->GetContinuousValueAt(loadIndex)));
					}
				}
				cluster->SetModelingCentroidValues(cvUpdatedCentroidValues);
			}
			cluster->SetStatisticsUpToDate(true);// on n'a pas besoin de rafraichir les autres stats
		}
	}

	// a partir des instances de l'ensemble de la base, mise a jour finale des stats des clusters (sans toucher aux centroides) :
	database->SetSampleNumberPercentage(originDatabaseSamplePercentage);
	FinalizeReplicateComputing(database, targetAttribute);

	int iDroppedClusters = ManageEmptyClusters(false);// supprimer les clusters qui seraient devenus vides
	if (iDroppedClusters > 0) {
		AddWarning(ALString(IntToString(iDroppedClusters)) + " empty cluster(s) have been dropped during this replicate.");
	}

	if (targetAttribute != NULL) {

		clusteringQuality->ComputeARIByClusters(kmGlobalCluster, oaTargetAttributeValues);
		clusteringQuality->ComputePredictiveClustering(kmGlobalCluster, oaTargetAttributeValues, targetAttribute);

		if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::EVA)
			clusteringQuality->ComputeEVA(kmGlobalCluster, oaTargetAttributeValues.GetSize());
		if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::LEVA)
			clusteringQuality->ComputeLEVA(kmGlobalCluster, oaTargetAttributeValues);
		if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters)
			clusteringQuality->ComputeNormalizedMutualInformationByClusters(kmGlobalCluster, oaTargetAttributeValues);
		if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses)
			clusteringQuality->ComputeNormalizedMutualInformationByClasses(kmGlobalCluster, oaTargetAttributeValues, kwftConfusionMatrix);
		if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::ARIByClasses)
			clusteringQuality->ComputeARIByClasses(kmGlobalCluster, oaTargetAttributeValues, kwftConfusionMatrix);
		if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::VariationOfInformation)
			clusteringQuality->ComputeVariationOfInformation(kmGlobalCluster, oaTargetAttributeValues);

	}

	clusteringQuality->ComputeDaviesBouldin(); // calcul de l'indice DB, tous attributs confondus


	//cout << endl << "minibatch DB : " << clusteringQuality->GetDaviesBouldin() << endl;

	//for (int i = 0; i < GetClusters()->GetSize(); i++){

	//	KMCluster * c = cast(KMCluster *, GetClusters()->GetAt(i));
	//	cout << c->GetLabel() << endl;
	//	for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++){
	//		const int loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
	//		if (loadIndex >= 0){
	//			cout << "minibatch Inerty intra for attribute " << loadIndex << " : " << c->GetInertyIntraForAttribute(loadIndex, parameters->GetDistanceType()) << endl;
	//		}
	//	}
	//}

	for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++) {
		const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
		if (loadIndex.IsValid()) {
			clusteringQuality->ComputeDaviesBouldinForAttribute(i);
			//cout << "minibatch DB for attribute " << loadIndex << " : " << clusteringQuality->GetDaviesBouldinForAttribute(loadIndex) << endl;
		}
	}

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage(" ");
		if (targetAttribute != NULL) {
			AddSimpleMessage("ARI by clusters is " + ALString(DoubleToString(clusteringQuality->GetARIByClusters())));
			AddSimpleMessage("Predictive clustering is " + ALString(DoubleToString(clusteringQuality->GetPredictiveClustering())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::ARIByClasses)
				AddSimpleMessage("ARI by classes is " + ALString(DoubleToString(clusteringQuality->GetARIByClasses())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::EVA)
				AddSimpleMessage("EVA is " + ALString(DoubleToString(clusteringQuality->GetEVA())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::LEVA)
				AddSimpleMessage("LEVA is " + ALString(DoubleToString(clusteringQuality->GetLEVA())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::VariationOfInformation)
				AddSimpleMessage("Variation of information is " + ALString(DoubleToString(clusteringQuality->GetVariationOfInformation())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters)
				AddSimpleMessage("NMI by clusters is " + ALString(DoubleToString(clusteringQuality->GetNormalizedMutualInformationByClusters())));
			if ((GetLearningExpertMode() and parameters->GetWriteDetailedStatistics()) or parameters->GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses)
				AddSimpleMessage("NMI by classes is " + ALString(DoubleToString(clusteringQuality->GetNormalizedMutualInformationByClasses())));
		}
		AddSimpleMessage("Davies Bouldin index is " + ALString(DoubleToString(clusteringQuality->GetDaviesBouldin())));
	}

	timer.Stop();

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage("Number of clusters : " + ALString(IntToString(kmClusters->GetSize())));
	}

	if (parameters->GetVerboseMode())
		AddSimpleMessage("Replicate compute time : " + ALString(SecondsToString(timer.GetElapsedTime())));

	database->SetSilentMode(false);

	return true;
}

void KMClusteringMiniBatch::UpdateTrainingConfusionMatrix(const KWObject* instance, const KMCluster* cluster, const KWAttribute* targetAttribute) {

	assert(oaTargetAttributeValues.GetSize() > 0);

	int idxMajorityTarget = cluster->GetMajorityTargetIndex();
	assert(idxMajorityTarget >= 0);

	ALString actualTarget = instance->GetSymbolValueAt(targetAttribute->GetLoadIndex()).GetValue();

	// rechercher l'index correspondant a la valeur de la modalite, pour renseigner notre tableau d'occurences
	int idxActualTarget = 0;
	for (; idxActualTarget < oaTargetAttributeValues.GetSize(); idxActualTarget++) {
		StringObject* s = cast(StringObject*, oaTargetAttributeValues.GetAt(idxActualTarget));
		if (actualTarget == s->GetString())
			break;
	}

	assert(idxActualTarget != oaTargetAttributeValues.GetSize());

	KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(idxMajorityTarget));
	fv->GetFrequencyVector()->SetAt(idxActualTarget, fv->GetFrequencyVector()->GetAt(idxActualTarget) + 1);
}

// calculer les statistiques globales
void KMClusteringMiniBatch::ComputeGlobalClusterStatistics(KWDatabase* allInstances, const KWAttribute* targetAttribute) {

	assert(allInstances != NULL);
	assert(allInstances->GetSampleEstimatedObjectNumber() > 0);

	kmGlobalCluster = CreateGlobalCluster();

	ResetInstancesWithMissingValuesNumber();

	// calcul des centroides et (si supervise) des stats sur la modalite cible
	ComputeGlobalClusterStatisticsFirstDatabaseRead(allInstances, targetAttribute);

	// calcul des stats dependant de la valeur finale du centroide (distances, ...)
	ComputeGlobalClusterStatisticsSecondDatabaseRead(allInstances, targetAttribute);

	kmGlobalCluster->FinalizeStatisticsUpdateFromInstances();
}

// calculer les statistiques globales, premiere passe
void KMClusteringMiniBatch::ComputeGlobalClusterStatisticsFirstDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute) {

	assert(allInstances != NULL);
	assert(allInstances->GetSampleEstimatedObjectNumber() > 0);
	const double dMinNecessaryMemory = 16 * 1024 * 1024;
	ALString sTmp;

	// stocker dans un tableau les modalites existantes pour l'attribut cible
	// il faut mettre la modalite cible (s'il y en a une) en premier, afin de pouvoir retrouver cette information dans la suite du traitement

	assert(oaTargetAttributeValues.GetSize() == 0);

	KWLoadIndex targetIndex;

	if (targetAttribute != NULL)
		targetIndex = targetAttribute->GetLoadIndex();

	boolean bHasMainTargetModality = parameters->GetMainTargetModality() != "" ? true : false;
	int iMainTargetModalityIndex = -1;

	// Ouverture de la base en lecture
	boolean bOk = allInstances->OpenForRead();

	// Lecture d'objets dans la base :
	if (bOk)
	{
		Global::ActivateErrorFlowControl();

		int nObject = 0;

		// 1ere passe pour calculer les centroides
		while (not allInstances->IsEnd())
		{
			if (nObject % 100 == 0)
			{
				// Arret si plus assez de memoire
				if (RMResourceManager::GetRemainingAvailableMemory() < dMinNecessaryMemory)
				{
					bOk = false;
					AddError(sTmp + "Not enough memory: interrupted after having read " + IntToString(nObject) + " instances (remaining available memory = "
						+ LongintToHumanReadableString(RMResourceManager::GetRemainingAvailableMemory()) + ", min necessary memory = " + LongintToHumanReadableString(dMinNecessaryMemory));
					break;
				}
			}

			// Traitement d'un nouvel objet
			KWObject* kwoObject = allInstances->Read();

			if (kwoObject != NULL)
			{
				nObject++;

				if (GetParameters()->GetWriteDetailedStatistics()) {
					if (GetParameters()->HasMissingNativeValue(kwoObject)) {
						kmGlobalCluster->IncrementInstancesWithMissingNativeValuesNumber(kwoObject);
					}
				}
				if (parameters->HasMissingKMeanValue(kwoObject)) {
					IncrementInstancesWithMissingValuesNumber();
					delete kwoObject;
					continue;
				}

				if (targetIndex.IsValid()) {

					// mode supervise : referencer la valeur cible, dans le cas ou elle n'a pas encore ete rencontree :
					const ALString sTarget = kwoObject->GetSymbolValueAt(targetIndex).GetValue();

					bool found = false;
					for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {
						if (cast(StringObject*, oaTargetAttributeValues.GetAt(i))->GetString() == sTarget)
							found = true;
					}
					if (not found) {
						StringObject* value = new StringObject;
						value->SetString(sTarget);
						oaTargetAttributeValues.Add(value);
					}
					// detecter si la valeur cible principale, parametree via l'IHM, est bien presente au moins une fois dans la base
					if (bHasMainTargetModality and iMainTargetModalityIndex == -1) {
						if (parameters->GetMainTargetModality() == sTarget)
							iMainTargetModalityIndex = oaTargetAttributeValues.GetSize() - 1;
					}
				}

				kmGlobalCluster->SetFrequency(kmGlobalCluster->GetFrequency() + 1);
				kmGlobalCluster->UpdateMeanCentroidValues(kwoObject, (ContinuousVector&)kmGlobalCluster->GetModelingCentroidValues());
				kmGlobalCluster->UpdateNativeAttributesContinuousMeanValues(kwoObject);

				delete kwoObject;
			}
		}


		Global::DesactivateErrorFlowControl();
	}

	// si la modalite cible principale est presente en base de donnees, alors il faut qu'elle figure en premier dans le tableau des valeurs cibles presentes en base
	if (targetIndex.IsValid() and bHasMainTargetModality and iMainTargetModalityIndex != -1) {

		ObjectArray oaNewTargetAttributeValues;
		oaNewTargetAttributeValues.Add(oaTargetAttributeValues.GetAt(iMainTargetModalityIndex));

		for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {
			StringObject* modality = cast(StringObject*, oaTargetAttributeValues.GetAt(i));
			if (modality->GetString() != parameters->GetMainTargetModality())
				oaNewTargetAttributeValues.Add(oaTargetAttributeValues.GetAt(i));
		}
		oaTargetAttributeValues.CopyFrom(&oaNewTargetAttributeValues);
	}
	allInstances->Close();
}

// calculer les statistiques globales, seconde passe
void KMClusteringMiniBatch::ComputeGlobalClusterStatisticsSecondDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute) {

	assert(allInstances != NULL);
	assert(allInstances->GetSampleEstimatedObjectNumber() > 0);
	const double dMinNecessaryMemory = 16 * 1024 * 1024;
	ALString sTmp;

	assert(kmGlobalCluster != NULL);
	assert(kmGlobalCluster->GetFrequency() > 0);

	// Ouverture de la base en lecture
	boolean bOk = allInstances->OpenForRead();

	// Lecture d'objets dans la base :
	if (bOk)
	{
		Global::ActivateErrorFlowControl();

		int nObject = 0;

		// 1ere passe pour calculer les centroides
		while (not allInstances->IsEnd())
		{
			if (nObject % 100 == 0)
			{
				// Arret si plus assez de memoire
				if (RMResourceManager::GetRemainingAvailableMemory() < dMinNecessaryMemory)
				{
					bOk = false;
					AddError(sTmp + "Not enough memory: interrupted after having read " + IntToString(nObject) + " instances (remaining available memory = "
						+ DoubleToString(RMResourceManager::GetRemainingAvailableMemory() / 1024 / 1024) + "Mo, min necessary memory = " + DoubleToString(dMinNecessaryMemory / 1024 / 1024) + "Mo)");
					break;
				}
			}

			// Traitement d'un nouvel objet
			KWObject* kwoObject = allInstances->Read();

			if (kwoObject != NULL)
			{
				kmGlobalCluster->UpdateDistanceSum(KMParameters::L1Norm, kwoObject, kmGlobalCluster->GetModelingCentroidValues());
				kmGlobalCluster->UpdateDistanceSum(KMParameters::L2Norm, kwoObject, kmGlobalCluster->GetModelingCentroidValues());
				kmGlobalCluster->UpdateDistanceSum(KMParameters::CosineNorm, kwoObject, kmGlobalCluster->GetModelingCentroidValues());
				kmGlobalCluster->UpdateInstanceNearestToCentroid(parameters->GetDistanceType(), kwoObject, kmGlobalCluster->GetModelingCentroidValues());

				delete kwoObject;
			}
		}

		Global::DesactivateErrorFlowControl();
	}
	allInstances->Close();
}

void KMClusteringMiniBatch::FinalizeReplicateComputing(KWDatabase* allInstances, const KWAttribute* targetAttribute) {

	assert(allInstances != NULL);
	assert(allInstances->GetSampleEstimatedObjectNumber() > 0);

	// remise a zero des stats des clusters et suppression des instances des clusters, sans toucher aux centroides
	for (int i = 0; i < GetClusters()->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		c->RemoveAll();
		c->InitializeStatistics();
	}
	// colonne = classe reelle, ligne = classe predite
	kwftConfusionMatrix->SetFrequencyVectorNumber(oaTargetAttributeValues.GetSize());
	for (int i = 0; i < kwftConfusionMatrix->GetFrequencyVectorNumber(); i++) {
		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(i));
		fv->GetFrequencyVector()->SetSize(oaTargetAttributeValues.GetSize());
	}

	ResetInstancesWithMissingValuesNumber();

	FinalizeReplicateComputingFirstDatabaseRead(allInstances, targetAttribute);

	if (targetAttribute != NULL) {
		for (int i = 0; i < GetClusters()->GetSize(); i++)
		{
			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
			c->ComputeMajorityTargetValue(oaTargetAttributeValues);// afin de pouvoir mettre a jour ensuite les matrices de confusion servant au calcul des levels de clustering
		}
	}

	FinalizeReplicateComputingSecondDatabaseRead(allInstances, targetAttribute);

	for (int i = 0; i < GetClusters()->GetSize(); i++)
	{
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		c->FinalizeStatisticsUpdateFromInstances();
		c->SetStatisticsUpToDate(true);
	}

	UpdateGlobalDistancesSum();
}

void KMClusteringMiniBatch::FinalizeReplicateComputingFirstDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute) {

	assert(allInstances != NULL);
	assert(allInstances->GetSampleEstimatedObjectNumber() > 0);
	const double dMinNecessaryMemory = 16 * 1024 * 1024;
	ALString sTmp;

	// Ouverture de la base en lecture
	boolean bOk = allInstances->OpenForRead();

	// Lecture d'objets dans la base, premiere passe pour calculer les frequences des clusters
	if (bOk)
	{
		Global::ActivateErrorFlowControl();

		int nObject = 0;

		while (not allInstances->IsEnd())
		{
			if (nObject % 100 == 0)
			{
				// Arret si plus assez de memoire
				if (RMResourceManager::GetRemainingAvailableMemory() < dMinNecessaryMemory)
				{
					bOk = false;
					AddError(sTmp + "Not enough memory: interrupted after having read " + IntToString(nObject) + " instances (remaining available memory = "
						+ LongintToHumanReadableString(RMResourceManager::GetRemainingAvailableMemory()) + ", min necessary memory = " + LongintToHumanReadableString(dMinNecessaryMemory));
					break;
				}
			}

			// Traitement d'un nouvel objet
			KWObject* kwoObject = allInstances->Read();

			if (kwoObject != NULL)
			{
				nObject++;

				if (parameters->HasMissingKMeanValue(kwoObject)) {
					IncrementInstancesWithMissingValuesNumber();
					delete kwoObject;
					continue;
				}

				KMCluster* cluster = FindNearestCluster(kwoObject);
				cluster->SetFrequency(cluster->GetFrequency() + 1);
				cluster->UpdateInertyIntra(parameters->GetDistanceType(), kwoObject, cluster->GetModelingCentroidValues());// necessaire pour calculer l'indice Davies Bouldin

				if (targetAttribute != NULL)
					cluster->UpdateTargetProbs(oaTargetAttributeValues, targetAttribute, kwoObject); // mode supervis� : calculer la repartition des valeurs de l'attribut cible, dans chaque cluster

				delete kwoObject;
			}
		}

		Global::DesactivateErrorFlowControl();
	}

	allInstances->Close();

}

void KMClusteringMiniBatch::FinalizeReplicateComputingSecondDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute) {

	assert(allInstances != NULL);
	assert(allInstances->GetSampleEstimatedObjectNumber() > 0);
	const longint dMinNecessaryMemory = 16 * 1024 * 1024;
	ALString sTmp;

	// Ouverture de la base en lecture
	boolean bOk = allInstances->OpenForRead();

	// Lecture d'objets dans la base, 2eme passe pour calculer les stats dependant des valeurs finales des centroides

	if (bOk)
	{
		Global::ActivateErrorFlowControl();

		int nObject = 0;

		while (not allInstances->IsEnd())
		{
			if (nObject % 100 == 0)
			{
				// Arret si plus assez de memoire
				if (RMResourceManager::GetRemainingAvailableMemory() < dMinNecessaryMemory)
				{
					bOk = false;
					AddError(sTmp + "Not enough memory: interrupted after having read " + IntToString(nObject) + " instances (remaining available memory = "
						+ LongintToHumanReadableString(RMResourceManager::GetRemainingAvailableMemory()) + ", min necessary memory = " + LongintToHumanReadableString(dMinNecessaryMemory));
					break;
				}
			}

			// Traitement d'un nouvel objet
			KWObject* kwoObject = allInstances->Read();

			if (kwoObject != NULL)
			{
				nObject++;

				KMCluster* cluster = FindNearestCluster(kwoObject);

				cluster->UpdateDistanceSum(KMParameters::L1Norm, kwoObject, cluster->GetModelingCentroidValues());
				cluster->UpdateDistanceSum(KMParameters::L2Norm, kwoObject, cluster->GetModelingCentroidValues());
				cluster->UpdateDistanceSum(KMParameters::CosineNorm, kwoObject, cluster->GetModelingCentroidValues());
				cluster->UpdateInstanceNearestToCentroid(parameters->GetDistanceType(), kwoObject, cluster->GetModelingCentroidValues());

				if (targetAttribute != NULL)
					cluster->UpdateCompactness(kwoObject, oaTargetAttributeValues, targetAttribute, cluster->GetModelingCentroidValues());

				// mise a jour des inerties intra par attribut et par cluster (necessaire pour calcul Davies Bouldin) :
				for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++) {
					const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
					if (loadIndex.IsValid())
						cluster->UpdateInertyIntraForAttribute(kwoObject, i, parameters->GetDistanceType());
				}

				if (targetAttribute != NULL)
					UpdateTrainingConfusionMatrix(kwoObject, cluster, targetAttribute);

				delete kwoObject;
			}
		}

		Global::DesactivateErrorFlowControl();
	}

	allInstances->Close();

}




