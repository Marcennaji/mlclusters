// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClustering.h"
#include "KMClusteringQuality.h"
#include "KMClusteringInitializer.h"
#include <cmath>

KMClustering::KMClustering(KMParameters* p)
{
	require(p != NULL);

	parameters = p;
	kmClusters = new ObjectArray();
	kmBestClusters = new ObjectArray();
	iIterationsDone = 0;
	iDroppedClustersNumber = 0;
	dUsedSampleNumberPercentage = 100.0;
	cvClustersDistancesSum.SetSize(3); // 3 normes : L1, L2 et Cosinus
	cvClustersDistancesSum.Initialize();
	kmGlobalCluster = NULL;
	clustersCentersDistances = new Continuous * [KMParameters::K_MAX_VALUE];
	for (int i = 0; i < KMParameters::K_MAX_VALUE; i++)
		clustersCentersDistances[i] = NULL;
	instancesToClusters = new NumericKeyDictionary;
	clusteringQuality = new KMClusteringQuality(kmClusters, parameters);
	clusteringInitializer = new KMClusteringInitializer(this);
	attributesPartitioningManager = new KMAttributesPartitioningManager;
	kwftConfusionMatrix = new KWFrequencyTable;
}


KMClustering::~KMClustering(void)
{
	kmClusters->DeleteAll();
	delete kmClusters;

	kmBestClusters->DeleteAll();
	delete kmBestClusters;

	if (kmGlobalCluster != NULL)
		delete kmGlobalCluster;

	oaTargetAttributeValues.DeleteAll();

	for (int i = 0; i < KMParameters::K_MAX_VALUE; i++)
		if (clustersCentersDistances[i] != NULL)
			delete[] clustersCentersDistances[i];
	delete[] clustersCentersDistances;

	if (instancesToClusters != NULL) {
		instancesToClusters->RemoveAll();
		delete instancesToClusters;
	}

	delete clusteringQuality;
	delete clusteringInitializer;
	delete attributesPartitioningManager;
	delete kwftConfusionMatrix;
	nkdClusteringLevels.DeleteAll();
	odGroupedModalitiesFrequencyTables.DeleteAll();
}

void KMClustering::SetTargetAttributeValues(const ObjectArray& source) {
	oaTargetAttributeValues.CopyFrom(&source);
}

void KMClustering::SetUsedSampleNumberPercentage(const double sampleNumberPercentage) {
	dUsedSampleNumberPercentage = sampleNumberPercentage;
}

const double  KMClustering::GetUsedSampleNumberPercentage() const {
	return dUsedSampleNumberPercentage;
}

bool KMClustering::ComputeReplicate(ObjectArray* instances, const KWAttribute* targetAttribute)
{
	Timer timer;
	timer.Start();

	if (instances->GetSize() == 0) {
		// ne pas faire un "assert", pour g�rer correctement le cas ou une interruption de lecture de la base a et� demand�e par l'utilisateur
		AddError("database not read");
		return false;
	}

	instances->Shuffle();

	// affecter les instances a un cluster 'fictif' unique, et calculer les statistiques correspondantes
	// (uniquement dans le cas ou ces stats n'auraient pas deja �t� recuperees a partir d'un autre resultat)
	if (kmGlobalCluster == NULL)
		ComputeGlobalClusterStatistics(instances);

	if (kmGlobalCluster->GetFrequency() == 0) {
		// NB. ne pas utiliser GetCount(), car les instances ne sont pas gard�es dans le cluster, seuls les stats et centroides sont gard�s
		AddWarning("All database instances have at least one missing value. Try to preprocess the values.");
		return false;
	}

	// lecture des modalites cibles (necessaire pour certaines initialisations de clusters)
	if (oaTargetAttributeValues.GetSize() == 0 and targetAttribute != NULL)
		ReadTargetAttributeValues(instances, targetAttribute);

	// repartition initiale des instances entre les clusters, selon la methode parametree par l'utilisateur
	if (not InitializeClusters(parameters->GetClustersCentersInitializationMethod(), instances, targetAttribute))
		return false;

	if (parameters->GetVerboseMode() and GetInstancesWithMissingValues() > 0)
		AddSimpleMessage(ALString("Instances with missing values, detected during clusters initialization : ") + LongintToString(GetInstancesWithMissingValues()));

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage("");
		AddSimpleMessage("Convergence :");
		AddSimpleMessage("--------------------------------------------------------------------------------------------------------------------------------------------------------------");
		AddSimpleMessage(" Iter. \tMovements \tMean distance \tImprovement \t\tBest distance \t\tEpsil. iter. \tEmpty clusters ");
	}

	if (not DoClusteringIterations(instances, instances->GetSize()))// iterations jusqu'� convergence
		return false;

	const bool recomputeCentroids = (parameters->GetMaxIterations() == -1 ? false : true); // doit-on recalculer les centroides apres convergence, ou garder ceux qui sont issus de la phase d'initialisation ?

	if (recomputeCentroids)
		AddInstancesToClusters(instances);// affecte les instances aux clusters en fonction des centroides prec�demment calcul�s, et met a jour les matrices de distances inter-clusters

	FinalizeReplicateComputing(recomputeCentroids);// NB : certaines instances ont pu changer de cluster, suite au recalcul des centroides

	if (recomputeCentroids)
		ManageEmptyClusters(false);// gestion des eventuels clusters devenus vides apres reaffectation des instances

	if (targetAttribute != NULL) {

		ComputeTrainingTargetProbs(targetAttribute); // mode supervis� : calculer la repartition des valeurs de l'attribut cible, dans chaque cluster

		clusteringQuality->ComputeARIByClusters(kmGlobalCluster, oaTargetAttributeValues);
		clusteringQuality->ComputeCompactness(oaTargetAttributeValues, targetAttribute);
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
	//cout << endl << "DB : " << clusteringQuality->GetDaviesBouldin() << endl;

	// calcul de l'indice DB pour chaque attribut separement (auparavant, il est necessaire de calculer les inerties intra par attribut et par cluster) :
	for (int i = 0; i < GetClusters()->GetSize(); i++) {

		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));

		for (int iLoadIndex = 0; iLoadIndex < parameters->GetKMeanAttributesLoadIndexes().GetSize(); iLoadIndex++) {
			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(iLoadIndex);
			if (loadIndex.IsValid()) {
				c->ComputeInertyIntraForAttribute(iLoadIndex, parameters->GetDistanceType());
			}
		}
	}
	for (int iLoadIndex = 0; iLoadIndex < parameters->GetKMeanAttributesLoadIndexes().GetSize(); iLoadIndex++) {
		const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(iLoadIndex);
		if (loadIndex.IsValid()) {
			clusteringQuality->ComputeDaviesBouldinForAttribute(iLoadIndex);
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

	return true;
}

boolean KMClustering::DoClusteringIterations(const ObjectArray* instances, const longint maxInstances) {

	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(maxInstances > 0 and maxInstances <= instances->GetSize());
	assert(kmClusters->GetSize() > 0); // une premiere initialisation des clusters doit avoir �t� faite auparavant

	int epsilonIterations = 0;
	int movements = 0;
	double distancesSum = 0.0;
	double minDistanceSum = 0.0;
	double newDistancesSum = 0.0;
	boolean interruptRequest = false;

	iIterationsDone = 0;
	iDroppedClustersNumber = 0;

	// calcul de la distance initiale, tous clusters confondus
	for (int i = 0; i < kmClusters->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		newDistancesSum += c->GetDistanceSum(parameters->GetDistanceType());
	}

	minDistanceSum = newDistancesSum;

	if (parameters->GetVerboseMode() and maxInstances == instances->GetSize())
		AddSimpleMessage(KMGetDisplayString(iIterationsDone) +
			KMGetDisplayString(0) +
			KMGetDisplayString(newDistancesSum / maxInstances) +
			KMGetDisplayString(newDistancesSum / maxInstances) +
			KMGetDisplayString(minDistanceSum / maxInstances) +
			KMGetDisplayString(0) +
			KMGetDisplayString(0));

	TaskProgression::BeginTask();
	TaskProgression::SetTitle("Clustering");

	boolean continueClustering = true;

	while (continueClustering)
	{
		interruptRequest = UpdateProgressionBar(maxInstances, iIterationsDone, movements);

		if (interruptRequest)
			break;

		distancesSum = 0.0;
		movements = 0;

		if (parameters->GetMaxIterations() != -1) {

			// (re)initaliser la matrice des distances inter-clusters, ainsi que la correspondance entre chaque cluster et son plus proche cluster
			ComputeClustersCentersDistances();

			// balayer tous les clusters, et calculer les sommes des distances de tous les clusters, avant reaffectation des instances aux clusters
			for (int i = 0; i < kmClusters->GetSize(); i++) {
				KMCluster* currentCluster = cast(KMCluster*, kmClusters->GetAt(i));
				distancesSum += currentCluster->GetDistanceSum(parameters->GetDistanceType());
			}

			// effectuer les mouvements d'instances entre clusters
			for (int i = 0; i < maxInstances; i++) {

				KWObject* instance = cast(KWObject*, instances->GetAt(i));

				KMCluster* currentCluster = cast(KMCluster*, instancesToClusters->Lookup(instance));

				if (currentCluster == NULL)
					continue; // cas d'une instance ayant des valeurs K-Means manquantes, et qui n'a donc jamais ete affectee precedemment a un cluster

				KMCluster* newCluster = FindNearestCluster(instance);

				if (newCluster != NULL and newCluster != currentCluster) {
					// l'instance change de cluster
					currentCluster->RemoveInstance(instance);
					newCluster->AddInstance(instance);
					instancesToClusters->SetAt(instance, newCluster);
					movements += 1;
				}
			}

			iIterationsDone++;
		}

		newDistancesSum = 0.0;

		if ((iIterationsDone <= parameters->GetMaxIterations() or parameters->GetMaxIterations() == 0) and parameters->GetMaxIterations() != -1) {

			// mise a jour des stats de chaque cluster (seulement les stats necessaires a la poursuite des iterations)
			for (int i = 0; i < kmClusters->GetSize(); i++) {
				KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
				c->ComputeIterationStatistics();
				newDistancesSum += c->GetDistanceSum(parameters->GetDistanceType());
			}
		}

		if (parameters->GetMaxIterations() == -1) {
			continueClustering = false; // on se contente dans ce cas de l'initialisation des clusters qui a �t� r�alis�e, selon la methode choisie
		}
		else {
			// evaluer si le clustering doit se poursuivre
			// attention : minDistanceSum et epsilonIterations peuvent etre modifi�s par l'appel
			continueClustering = ManageConvergence(movements, iIterationsDone,
				distancesSum, newDistancesSum, maxInstances, minDistanceSum, epsilonIterations);
		}

		// en fin de clustering, on garde la meilleure iteration effectuee (qui n'est pas forcement la derniere)
		// NB. la meilleure iteration retenue peut contenir des clusters vides

		if (not continueClustering and kmBestClusters->GetSize() > 0) {

			for (int i = 0; i < kmBestClusters->GetSize(); i++)
			{
				KMCluster* source = cast(KMCluster*, kmBestClusters->GetAt(i));
				KMCluster* target = cast(KMCluster*, kmClusters->GetAt(i));
				target->CopyFrom(source);
			}
			kmBestClusters->DeleteAll();
		}

		// gestion des clusters devenus vides apres une iteration
		int emptyClusters = ManageEmptyClusters(continueClustering);

		if (parameters->GetVerboseMode() && parameters->GetMaxIterations() != -1) {
			AddSimpleMessage(KMGetDisplayString(iIterationsDone) +
				KMGetDisplayString(movements) +
				KMGetDisplayString(newDistancesSum / maxInstances) +
				KMGetDisplayString((distancesSum - newDistancesSum) / maxInstances) +
				KMGetDisplayString(minDistanceSum / maxInstances) +
				KMGetDisplayString(epsilonIterations) +
				KMGetDisplayString(emptyClusters));

			if (not continueClustering and emptyClusters > 0)
				AddSimpleMessage(ALString(IntToString(emptyClusters)) + " empty cluster(s) dropped");
		}

	} // fin de la boucle d'affectation des instances aux clusters

	TaskProgression::EndTask();

	return (interruptRequest == true ? false : true);

}

bool KMClustering::ManageConvergence(const int movements, const int iterationsDone,
	const double distancesSum, const double newDistancesSum,
	const longint instancesCount,
	double& minDistanceSum, int& epsilonIterations) {

	bool continueClustering = true;

	assert(parameters->GetMaxIterations() >= 0);

	if ((movements == 0) or (iterationsDone >= parameters->GetMaxIterations() and parameters->GetMaxIterations() != 0))
		continueClustering = false;

	if (movements > 0) {

		if (fabs((distancesSum - newDistancesSum) / instancesCount) >= parameters->GetEpsilonValue() and
			(newDistancesSum < minDistanceSum)) {
			// --> si l'ecart entre la distance calcul�e a la precedente iteration et l'iteration courante est superieur
			// a epsilon, et que la nouvelle distance est inferieure a la plus petite distance jamais observee, alors
			// on memorise le modele courant

			epsilonIterations = 0; // Remise a 0 du compteur d'iterations sans amelioration
			minDistanceSum = newDistancesSum; // Memorisation de la distance mini observ�e, toutes iterations confondues

			// Memorisation du mod�le courant, car il est le meilleur
			CloneBestClusters();
		}
		else {

			// pas d'amelioration significative observee depuis la precedente iteration

			if (parameters->GetEpsilonValue() > 0.0) {

				epsilonIterations++;

				if (epsilonIterations >= parameters->GetEpsilonMaxIterations())
					continueClustering = false;
			}
		}
	}

	return continueClustering;

}

KMCluster* KMClustering::FindNearestCluster(KWObject* instance)
{
	assert(parameters->GetDistanceType() == KMParameters::L1Norm or
		parameters->GetDistanceType() == KMParameters::L2Norm or
		parameters->GetDistanceType() == KMParameters::CosineNorm);
	assert(clustersCentersDistances != NULL);

	if (kmClusters == NULL or kmClusters->GetSize() == 0) {
		return NULL;
	}

	if (parameters->GetDistanceType() == KMParameters::L1Norm)
		return FindNearestClusterL1(instance);
	else
		if (parameters->GetDistanceType() == KMParameters::L2Norm)
			return FindNearestClusterL2(instance);
		else
			if (parameters->GetDistanceType() == KMParameters::CosineNorm)
				return FindNearestClusterCosinus(instance);

	return NULL;

}

KMCluster* KMClustering::FindNearestClusterL1(KWObject* instance) {

	if (kmClusters == NULL or kmClusters->GetSize() == 0)
		return NULL;

	const int nbClusters = kmClusters->GetSize();

	const int size = parameters->GetKMeanAttributesLoadIndexes().GetSize();
	int	nearestClusterIndex = 0;
	Continuous minimumDistance = 0.0;

	// recuperer le cluster auquel appartient actuellement cette instance (NB. : en cours  de premi�re initialisation des clusters, il n'y en a pas encore)
	KMCluster* firstClusterToCheck = cast(KMCluster*, instancesToClusters->Lookup(instance));

	if (firstClusterToCheck == NULL) {

		// cas de la premiere initialisation des clusters. On calcule dans ce cas la distance au premier cluster de la liste, pour
		// minimiser les tests a effectuer par la suite, et optimiser ainsi la vitesse d'execution

		firstClusterToCheck = cast(KMCluster*, kmClusters->GetAt(0));

		for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
			if (not loadIndex.IsValid())
				continue;
			assert(firstClusterToCheck->GetModelingCentroidValues().GetSize() > idxAttribut);
			const Continuous d = firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut) - instance->GetContinuousValueAt(loadIndex);
			minimumDistance += fabs(d); // premiere initialisation de la distance minimale de reference
		}


	}
	else {

		// pour optimiser la vitesse d'execution : comparer la distance entre l'instance et son cluster, avec la distance entre
		// le cluster de l'instance et son cluster le plus proche. En fonction du r�sultat, on pourra se passer de calculer la distance pour
		// les autres clusters

		nearestClusterIndex = firstClusterToCheck->GetIndex();

		// calcul de distance entre l'instance et le centroide de son cluster courant

		for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
			if (not loadIndex.IsValid())
				continue;
			assert(firstClusterToCheck->GetModelingCentroidValues().GetSize() > idxAttribut);
			const Continuous d = firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut) - instance->GetContinuousValueAt(loadIndex);
			minimumDistance += fabs(d); // premiere initialisation de la distance minimale de reference
		}

		KMCluster* nearestToCurrentCluster = firstClusterToCheck->GetNearestCluster();

		assert(nearestToCurrentCluster != NULL);
		assert(nearestToCurrentCluster->GetIndex() >= 0);
		assert(firstClusterToCheck->GetIndex() >= 0);

		// distance entre le cluster de l'instance, et le cluster le plus proche de ce cluster
		Continuous distanceBetweenClusters = clustersCentersDistances[nearestToCurrentCluster->GetIndex()][firstClusterToCheck->GetIndex()];

		if (distanceBetweenClusters * 0.5 > minimumDistance) {
			return firstClusterToCheck; // l'instance ne changera pas de cluster, inutile de verifier pour les autres clusters
		}
	}

	// calculer la distance aux autres centroides de clusters :

	for (int idxCluster = 0; idxCluster < nbClusters; idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, kmClusters->GetAt(idxCluster));

		if (cluster == firstClusterToCheck)
			continue; // cluster deja trait�

		Continuous distance = 0.0;
		bool distanceComputed = false;

		if (0.5 * clustersCentersDistances[nearestClusterIndex][idxCluster] < minimumDistance) {

			distanceComputed = true;

			for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {
				const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
				if (not loadIndex.IsValid())
					continue;
				assert(cluster->GetModelingCentroidValues().GetSize() > idxAttribut);
				const Continuous d = cluster->GetModelingCentroidValues().GetAt(idxAttribut) - instance->GetContinuousValueAt(loadIndex);

				distance += fabs(d);

				if (distance > minimumDistance)
					break; // pas la peine de continuer a calculer, la distance entre l'instance et ce cluster sera superieure a la distance minimale deja trouvee
			}
		}

		if (distanceComputed and minimumDistance > distance) {
			minimumDistance = distance;
			nearestClusterIndex = idxCluster;
		}
	}

	return cast(KMCluster*, kmClusters->GetAt(nearestClusterIndex));

}

KMCluster* KMClustering::FindNearestClusterL2(KWObject* instance) {

	if (kmClusters == NULL or kmClusters->GetSize() == 0)
		return NULL;

	const int nbClusters = kmClusters->GetSize();

	const int size = parameters->GetKMeanAttributesLoadIndexes().GetSize();
	int	nearestClusterIndex = 0;
	Continuous minimumDistance = 0.0;

	// recuperer le cluster auquel appartient actuellement cette instance (NB. : en cours  de premi�re initialisation des clusters, il n'y en a pas encore)
	KMCluster* firstClusterToCheck = cast(KMCluster*, instancesToClusters->Lookup(instance));

	if (firstClusterToCheck == NULL) {

		// cas de la premiere initialisation des clusters. On calcule dans ce cas la distance au premier cluster de la liste, pour
		// minimiser les tests a effectuer par la suite, et optimiser ainsi la vitesse d'execution

		firstClusterToCheck = cast(KMCluster*, kmClusters->GetAt(0));

		for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
			if (not loadIndex.IsValid())
				continue;
			assert(firstClusterToCheck->GetModelingCentroidValues().GetSize() > idxAttribut);
			const Continuous d = firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut) - instance->GetContinuousValueAt(loadIndex);
			minimumDistance += (d * d); // premiere initialisation de la distance minimale de reference
		}

	}
	else {

		// pour optimiser la vitesse d'execution : comparer la distance entre l'instance et son cluster, avec la distance entre
		// l'instance et le cluster le plus proche de son cluster. En fonction du r�sultat, on pourra se passer de calculer la distance pour
		// les autres clusters

		nearestClusterIndex = firstClusterToCheck->GetIndex();

		// calcul de distance entre l'instance et le centroide de son cluster courant

		for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
			if (not loadIndex.IsValid())
				continue;
			assert(firstClusterToCheck->GetModelingCentroidValues().GetSize() > idxAttribut);
			const Continuous d = firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut) - instance->GetContinuousValueAt(loadIndex);
			minimumDistance += (d * d); // premiere initialisation de la distance minimale de reference
		}

		KMCluster* nearestToCurrentCluster = firstClusterToCheck->GetNearestCluster();

		assert(nearestToCurrentCluster != NULL);
		assert(nearestToCurrentCluster->GetIndex() >= 0);
		assert(firstClusterToCheck->GetIndex() >= 0);

		// distance entre le cluster de l'instance, et le cluster le plus proche de ce cluster
		Continuous distanceBetweenClusters = clustersCentersDistances[nearestToCurrentCluster->GetIndex()][firstClusterToCheck->GetIndex()];

		if (sqrt(distanceBetweenClusters) * 0.5 > sqrt(minimumDistance))
			return firstClusterToCheck; // l'instance ne changera pas de cluster, inutile de verifier pour les autres clusters
	}

	// calculer la distance aux autres centroides de clusters :

	for (int idxCluster = 0; idxCluster < nbClusters; idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, kmClusters->GetAt(idxCluster));

		if (cluster == firstClusterToCheck)
			continue; // cluster deja trait�

		Continuous distance = 0.0;
		bool distanceComputed = false;

		if (0.5 * sqrt(clustersCentersDistances[nearestClusterIndex][idxCluster]) < sqrt(minimumDistance)) {

			distanceComputed = true;

			for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

				const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
				if (not loadIndex.IsValid())
					continue;
				assert(cluster->GetModelingCentroidValues().GetSize() > idxAttribut);
				const Continuous d = cluster->GetModelingCentroidValues().GetAt(idxAttribut) - instance->GetContinuousValueAt(loadIndex);

				distance += (d * d);

				if (distance > minimumDistance)
					break; // pas la peine de continuer a calculer, la distance entre l'instance et ce cluster sera superieure a la distance minimale deja trouvee
			}
		}

		if (distanceComputed and minimumDistance > distance) {
			minimumDistance = distance; // todo distance2
			nearestClusterIndex = idxCluster;
		}
	}

	return cast(KMCluster*, kmClusters->GetAt(nearestClusterIndex));

}

KMCluster* KMClustering::FindNearestClusterCosinus(KWObject* instance) {

	if (kmClusters == NULL or kmClusters->GetSize() == 0)
		return NULL;

	const int nbClusters = kmClusters->GetSize();

	const int size = parameters->GetKMeanAttributesLoadIndexes().GetSize();
	int	nearestClusterIndex = 0;
	Continuous minimumDistance = 0.0;
	Continuous numeratorCosinus = 0.0;
	Continuous denominatorInstanceCosinus = 0.0;
	Continuous denominatorCentroidCosinus = 0.0;
	Continuous denominator = 0.0;

	// recuperer le cluster auquel appartient actuellement cette instance (NB. : en cours  de premi�re initialisation des clusters, il n'y en a pas encore)
	KMCluster* firstClusterToCheck = cast(KMCluster*, instancesToClusters->Lookup(instance));

	if (firstClusterToCheck == NULL) {

		// cas de la premiere initialisation des clusters. On calcule dans ce cas la distance au premier cluster de la liste, pour
		// minimiser les tests a effectuer par la suite, et optimiser ainsi la vitesse d'execution

		firstClusterToCheck = cast(KMCluster*, kmClusters->GetAt(0));

		for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
			if (not loadIndex.IsValid())
				continue;

			numeratorCosinus += firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut) * instance->GetContinuousValueAt(loadIndex);
			denominatorInstanceCosinus += pow(instance->GetContinuousValueAt(loadIndex), 2);
			denominatorCentroidCosinus += pow(firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut), 2);
		}
		denominator = sqrt(denominatorInstanceCosinus) * sqrt(denominatorCentroidCosinus);
		minimumDistance = 1 - (denominator == 0 ? 0 : numeratorCosinus / denominator);;

	}
	else {

		// pour optimiser la vitesse d'execution : comparer la distance entre l'instance et son cluster, avec la distance entre
		// l'instance et le cluster le plus proche de son cluster. En fonction du r�sultat, on pourra se passer de calculer la distance pour
		// les autres clusters

		nearestClusterIndex = firstClusterToCheck->GetIndex();

		// calcul de distance entre l'instance et le centroide de son cluster courant

		for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
			if (not loadIndex.IsValid())
				continue;
			assert(firstClusterToCheck->GetModelingCentroidValues().GetSize() > idxAttribut);
			numeratorCosinus += firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut) * instance->GetContinuousValueAt(loadIndex);
			denominatorInstanceCosinus += pow(instance->GetContinuousValueAt(loadIndex), 2);
			denominatorCentroidCosinus += pow(firstClusterToCheck->GetModelingCentroidValues().GetAt(idxAttribut), 2);
		}
		denominator = sqrt(denominatorInstanceCosinus) * sqrt(denominatorCentroidCosinus);
		minimumDistance = 1 - (denominator == 0 ? 0 : numeratorCosinus / denominator);;

		KMCluster* nearestToCurrentCluster = firstClusterToCheck->GetNearestCluster();

		assert(nearestToCurrentCluster != NULL);
		assert(nearestToCurrentCluster->GetIndex() >= 0);
		assert(firstClusterToCheck->GetIndex() >= 0);

		// distance entre le cluster de l'instance, et le cluster le plus proche de ce cluster
		Continuous distanceBetweenClusters = clustersCentersDistances[nearestToCurrentCluster->GetIndex()][firstClusterToCheck->GetIndex()];

		if (distanceBetweenClusters * 0.5 > minimumDistance)
			return firstClusterToCheck; // l'instance ne changera pas de cluster, inutile de verifier pour les autres clusters

	}

	// calculer la distance aux autres centroides de clusters :

	for (int idxCluster = 0; idxCluster < nbClusters; idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, kmClusters->GetAt(idxCluster));

		if (cluster == firstClusterToCheck)
			continue; // cluster deja trait�

		Continuous distance = 0.0;
		bool distanceComputed = false;

		if (0.5 * clustersCentersDistances[nearestClusterIndex][idxCluster] < minimumDistance) {

			distanceComputed = true;

			numeratorCosinus = 0.0;
			denominatorInstanceCosinus = 0.0;
			denominatorCentroidCosinus = 0.0;

			for (int idxAttribut = 0; idxAttribut < size; idxAttribut++) {

				const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(idxAttribut);
				if (not loadIndex.IsValid())
					continue;

				numeratorCosinus += cluster->GetModelingCentroidValues().GetAt(idxAttribut) * instance->GetContinuousValueAt(loadIndex);
				denominatorInstanceCosinus += pow(instance->GetContinuousValueAt(loadIndex), 2);
				denominatorCentroidCosinus += pow(cluster->GetModelingCentroidValues().GetAt(idxAttribut), 2);
			}
			denominator = sqrt(denominatorInstanceCosinus) * sqrt(denominatorCentroidCosinus);
			distance = 1 - (denominator == 0 ? 0 : numeratorCosinus / denominator);
		}

		if (distanceComputed and minimumDistance > distance) {
			minimumDistance = distance;
			nearestClusterIndex = idxCluster;
		}
	}

	return cast(KMCluster*, kmClusters->GetAt(nearestClusterIndex));

}

/** calculer les distances entre les diff�rents centres des clusters, afin de produire une matrice des distances,
dont l'utilisation permettra une optimisation des performances */
void KMClustering::ComputeClustersCentersDistances(const boolean bUseEvaluationCentroids) {

	const int nbClusters = GetClusters()->GetSize();
	assert(clustersCentersDistances != NULL);

	// suppression des anciennes distances
	for (int i = 0; i < KMParameters::K_MAX_VALUE; i++) {
		if (clustersCentersDistances[i] != NULL) {
			delete[] clustersCentersDistances[i];
			clustersCentersDistances[i] = NULL;
		}
	}

	for (int i = 0; i < nbClusters; i++) {

		KMCluster* cluster1 = cast(KMCluster*, kmClusters->GetAt(i));

		clustersCentersDistances[i] = new Continuous[nbClusters];

		cluster1->SetIndex(i); // pour acceder a cette distance ulterieurement, garder la memoire de l'index de chaque cluster

		for (int j = 0; j < nbClusters; j++) {

			if (i == j) {
				clustersCentersDistances[i][j] = 0.0;
				continue;
			}
			else {
				if (i > j) {
					// la distance entre les 2 centres de clusters a deja ete calcul�e, la reporter
					clustersCentersDistances[i][j] = clustersCentersDistances[j][i];
					continue;
				}
			}

			KMCluster* cluster2 = cast(KMCluster*, kmClusters->GetAt(j));

			const ContinuousVector& cluster1Centroids = (bUseEvaluationCentroids ? cluster1->GetEvaluationCentroidValues() : cluster1->GetModelingCentroidValues());
			const ContinuousVector& cluster2Centroids = (bUseEvaluationCentroids ? cluster2->GetEvaluationCentroidValues() : cluster2->GetModelingCentroidValues());

			if (cluster1Centroids.GetSize() == 0 or cluster2Centroids.GetSize() == 0)
				clustersCentersDistances[i][j] = 0; // cas de clusters devenus vides lors de l'evaluation de test
			else
				clustersCentersDistances[i][j] = GetDistanceBetween(cluster1Centroids, cluster2Centroids, parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());
		}
	}


	// pour chaque cluster, repertorier quel est le cluster qui en est le plus proche
	// (aux fins d'optimisation de la vitesse d'execution, lors des affectations aux clusters)

	for (int i = 0; i < nbClusters; i++) {

		double minimumDistance = -1;

		KMCluster* cluster = cast(KMCluster*, kmClusters->GetAt(i));
		assert(cluster != NULL);

		// pour chaque cluster, m�moriser le pointeur du cluster qui lui est le plus proche
		for (int j = 0; j < nbClusters; j++) {

			if (i == j and nbClusters > 1)
				continue;

			if (minimumDistance == -1 or minimumDistance > clustersCentersDistances[i][j]) {

				minimumDistance = clustersCentersDistances[i][j];
				cluster->SetNearestCluster(cast(KMCluster*, kmClusters->GetAt(j)));
			}
		}
	}

	//for (int i = 0; i < nbClusters; i++){
	//	if (clustersCentersDistances[i] != NULL){
	//		cout << "cluster " << i << " : " << endl;
	//		for (int j = 0; j < nbClusters; j++){
	//			cout << "\tcluster " << j << " : distance = " << clustersCentersDistances[i][j] << endl;
	//		}
	//	}
	//}

}

Continuous KMClustering::GetSimilarityBetween(const ContinuousVector& v1, const ContinuousVector& v2,
	const ALString& targetModality1, const ALString& targetModality2, const KMParameters* parameters) {

	if (v1.GetSize() == 0 or v1.GetSize() != v2.GetSize())
		// comparaison avec un cluster devenu vide ?
		return KWContinuous::GetMaxValue();

	const Continuous distance = GetDistanceBetween(v1, v2, parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());

	double denominator = (parameters->GetDistanceType() == KMParameters::L2Norm ? distance : pow(distance, 2));// car en L2, la distance est deja au carre
	denominator = denominator / parameters->GetKMeanAttributesLoadIndexes().GetSize();
	denominator += 1;

	double numerator = (targetModality1 == targetModality2 ? 1 : exp(-1));

	Continuous result = 1 - (numerator / denominator);

	return result;
}


Continuous KMClustering::GetDistanceBetween(const ContinuousVector& v1, const ContinuousVector& v2, const KMParameters::DistanceType distanceType, const KWLoadIndexVector& kmeanAttributesLoadIndexes) {

	if (v1.GetSize() == 0 or v1.GetSize() != v2.GetSize())
		// comparaison avec un cluster devenu vide ?
		return KWContinuous::GetMaxValue();

	Continuous result = 0.0;
	const int size = kmeanAttributesLoadIndexes.GetSize();

	// calcul de distance norme L2
	if (distanceType == KMParameters::L2Norm) {

		for (int i = 0; i < size; i++) {
			const KWLoadIndex loadIndex = kmeanAttributesLoadIndexes.GetAt(i);
			if (not loadIndex.IsValid())
				continue;
			const Continuous d = v1.GetAt(i) - v2.GetAt(i);
			result += (d * d);
		}
	}
	// norme L1
	else {
		if (distanceType == KMParameters::L1Norm) {
			for (int i = 0; i < size; i++) {
				const KWLoadIndex loadIndex = kmeanAttributesLoadIndexes.GetAt(i);
				if (not loadIndex.IsValid())
					continue;
				result += fabs(v1.GetAt(i) - v2.GetAt(i));
			}
		}
		else {

			if (distanceType == KMParameters::CosineNorm) {
				// norme cosinus

				Continuous numeratorCosinus = 0.0;
				Continuous denominatorInstanceCosinus = 0.0;
				Continuous denominatorCentroidCosinus = 0.0;

				for (int i = 0; i < size; i++) {
					const KWLoadIndex loadIndex = kmeanAttributesLoadIndexes.GetAt(i);
					if (not loadIndex.IsValid())
						continue;
					numeratorCosinus += v1.GetAt(i) * v2.GetAt(i);
					denominatorInstanceCosinus += pow(v1.GetAt(i), 2);
					denominatorCentroidCosinus += pow(v2.GetAt(i), 2);
				}

				Continuous denominator = sqrt(denominatorInstanceCosinus) * sqrt(denominatorCentroidCosinus);
				result = 1 - (denominator == 0 ? 0 : numeratorCosinus / denominator);
			}
			else
				assert(false);
		}
	}

	return result;

}

Continuous KMClustering::GetDistanceBetweenForAttribute(const int attributeLoadIndex, const ContinuousVector& v1, const ContinuousVector& v2,
	const KMParameters::DistanceType distanceType) {

	if (v1.GetSize() == 0 or v1.GetSize() != v2.GetSize())
		return KWContinuous::GetMaxValue();

	Continuous result = 0.0;

	// calcul de distance norme L2
	if (distanceType == KMParameters::L2Norm) {
		const Continuous d = v1.GetAt(attributeLoadIndex) - v2.GetAt(attributeLoadIndex);
		result = (d * d);
	}
	// norme L1
	else {
		if (distanceType == KMParameters::L1Norm) {
			result = fabs(v1.GetAt(attributeLoadIndex) - v2.GetAt(attributeLoadIndex));
		}
		else {

			if (distanceType == KMParameters::CosineNorm) {
				// norme cosinus
				Continuous numeratorCosinus = v1.GetAt(attributeLoadIndex) * v2.GetAt(attributeLoadIndex);
				Continuous denominatorInstanceCosinus = pow(v1.GetAt(attributeLoadIndex), 2);
				Continuous denominatorCentroidCosinus = pow(v2.GetAt(attributeLoadIndex), 2);
				Continuous denominator = sqrt(denominatorInstanceCosinus) * sqrt(denominatorCentroidCosinus);
				result = 1 - (denominator == 0 ? 0 : numeratorCosinus / denominator);
			}
		}
	}

	return result;

}

const Continuous  KMClustering::GetMeanDistance() const {

	assert(kmClusters->GetSize() > 0);

	Continuous distanceSum = 0.0;
	longint instancesNumber = 0;

	for (int i = 0; i < kmClusters->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		distanceSum += c->GetDistanceSum(parameters->GetDistanceType());
		instancesNumber += c->GetFrequency();
	}

	if (instancesNumber == 0)
		return 0;

	return distanceSum / instancesNumber;
}


int KMClustering::ManageEmptyClusters(const boolean continueClustering) {

	int emptyClusters = 0;

	for (int i = 0; i < GetClusters()->GetSize(); i++)
	{
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));

		if (c->GetFrequency() == 0)
			// ne pas se baser sur GetCount(), car les clusters clon�s ne contiennent pas r�ellement les instances
			emptyClusters++;
	}

	if (emptyClusters == 0)
		return 0;

	if (not continueClustering) {

		// en fin de clustering, faire un drop des clusters vides

		for (int i = 0; i < GetClusters()->GetSize(); i++)
		{
			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));

			if (c->GetFrequency() == 0) {
				iDroppedClustersNumber++;
				DeleteClusterAt(i);
				i = -1; // refaire la boucle
			}
		}

	}
	else {

		// en cours de clustering : recuperer autant de "points les moins bien construits", tous clusters confondus,
		// qu'il y a de clusters vides, et attribuer ensuite ces points aux clusters vides

		ObjectArray oaInstances;

		// etablir la liste compl�te des instances et de leurs distances
		for (int i = 0; i < kmClusters->GetSize(); i++)
		{
			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));

			POSITION position = c->GetStartPosition();
			NUMERIC key;
			Object* oCurrent;

			while (position != NULL) {

				c->GetNextAssoc(position, key, oCurrent);
				KWObject* object = static_cast<KWObject *>(oCurrent);

				if (object != NULL) {
					double d = c->FindDistanceFromCentroid(object, c->GetModelingCentroidValues(), parameters->GetDistanceType());
					oaInstances.Add(new KMInstance(object, i, d));
				}
			}
		}

		oaInstances.SetCompareFunction(KMClusteringDistanceCompareDesc);
		oaInstances.Sort(); // tri des instances, par distances decroissantes

		int instanceIdx = 0;

		for (int i = 0; i < kmClusters->GetSize(); i++)
		{
			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));

			if (c->GetFrequency() == 0) {

				KMInstance* inst = cast(KMInstance*, oaInstances.GetAt(instanceIdx++));
				c->AddInstance(cast(KWObject*, inst->instance));

				// enlever l'instance de l'ancien cluster :
				KMCluster* oldCluster = cast(KMCluster*, GetClusters()->GetAt(inst->idCluster));
				oldCluster->RemoveInstance((KWObject*)inst->instance);

				// maj de la correspondance entre instances et clusters
				instancesToClusters->SetAt((KWObject*)inst->instance, c);
			}
		}

		oaInstances.DeleteAll();

		// remettre a jour les stats, pour les clusters qui ont �t� modifi�s
		for (int i = 0; i < kmClusters->GetSize(); i++)
		{
			KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
			c->ComputeIterationStatistics();
		}

	}

	return emptyClusters;

}

boolean KMClustering::InitializeClusters(const KMParameters::ClustersCentersInitMethod initMethod, const ObjectArray* instances, const KWAttribute* targetAttribute)
{
	assert(kmClusters->GetSize() == 0);
	assert(kmGlobalCluster != NULL);
	assert(kmGlobalCluster->GetFrequency() > 0);

	TaskProgression::BeginTask();
	TaskProgression::DisplayMainLabel("Clusters initialization");

	boolean bOk = true;
	clusteringInitializer->ResetInstancesWithMissingValuesNumber();

	assert(instances->GetSize() > 0);

	instancesToClusters->RemoveAll();

	TaskProgression::DisplayProgression(5);

	// verifier s'il y a assez d'instances completes pour initialiser les clusters
	for (int i = 0; i < instances->GetSize(); i++) {

		KWObject* kwo = cast(KWObject*, instances->GetAt(i));
		if (parameters->HasMissingKMeanValue(kwo))
			clusteringInitializer->IncrementInstancesWithMissingValuesNumber();
	}
	TaskProgression::DisplayProgression(10);

	if (instances->GetSize() - clusteringInitializer->GetInstancesWithMissingValues() < parameters->GetKValue()) {
		AddWarning("Clusters initialization failed : too many missing values");
		TaskProgression::EndTask();
		bOk = false;
	}
	if (clusteringInitializer->GetInstancesWithMissingValues() > instances->GetSize() / 2)
		AddWarning("Clusters initialization : many missing values, initialization may take some time. Please wait.");

	if (bOk) {

		if (initMethod == KMParameters::Random)
			bOk = clusteringInitializer->InitializeRandomCentroids(instances);
		else

			if (initMethod == KMParameters::MinMaxRandom)
				bOk = clusteringInitializer->InitializeMinMaxCentroids(instances, false);
			else

				if (initMethod == KMParameters::MinMaxDeterministic)
					bOk = clusteringInitializer->InitializeMinMaxCentroids(instances, true);
				else

					if (initMethod == KMParameters::VariancePartitioning)
						bOk = clusteringInitializer->InitializeVariancePartitioningCentroids(instances);
					else

						if (initMethod == KMParameters::ClassDecomposition) {
							if (targetAttribute == NULL) {
								AddWarning("Clusters initialization : Class Decomposition is available ony in supervised mode");
								bOk = false;
							}
							if (bOk)
								bOk = clusteringInitializer->InitializeClassDecompositionCentroids(instances, targetAttribute);
						}
						else

							if (initMethod == KMParameters::Sample) {

								// tirage al�atoire des k centroides de clusters, puis convergence sur un echantillon de la base, puis initialisation des centres a partir des resultats de cette convergence

								if (not clusteringInitializer->InitializeRandomCentroids(instances))
									bOk = false;

								if (bOk) {

									AddInstancesToClusters(instances);

									const int x = instances->GetSize();
									longint maxInstances = x * (1 / (pow((2 * x), 0.23)));  // ((x)*(1 / ((2 * (x))**0.23)));

									if (maxInstances < parameters->GetKValue())
										maxInstances = parameters->GetKValue();

									if (parameters->GetVerboseMode())
										AddSimpleMessage("Clusters initialization : sample size is " + ALString(LongintToString(maxInstances)) + " instances");

									// iterations jusqu'� convergence, sur le % retenu de la base, pour d�terminer des centroides initiaux :

									for (int i = 0; i < kmClusters->GetSize(); i++) {
										KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
										c->ComputeIterationStatistics(); // mise a jour des stats pour permettre les iterations
									}

									const boolean bOldverboseMode = parameters->GetVerboseMode();
									parameters->SetVerboseMode(false); // desactiver temporairement le mode verbose

									bOk = DoClusteringIterations(instances, maxInstances); // iterations jusqu'a convergence

									parameters->SetVerboseMode(bOldverboseMode); // retablir valeur d'origine

									if (parameters->GetVerboseMode() and iDroppedClustersNumber > 0)
										AddWarning("Clusters initialization : sample convergence has " + ALString(IntToString(iDroppedClustersNumber)) + " dropped cluster(s)");


								}
							}
							else

								if (initMethod == KMParameters::KMeanPlusPlus)
									bOk = clusteringInitializer->InitializeKMeanPlusPlusCentroids(instances);
								else

									if (initMethod == KMParameters::KMeanPlusPlusR) {

										if (parameters->GetKValue() <= 1) {
											AddWarning("Clusters initialization : KMean++R is possible only if K is > 1 ");
											bOk = false;
										}
										if (targetAttribute == NULL) {
											AddWarning("Clusters initialization : KMean++R is available only in supervised mode");
											bOk = false;
										}
										if (bOk)
											bOk = clusteringInitializer->InitializeKMeanPlusPlusRCentroids(instances, targetAttribute);
									}
									else

										if (initMethod == KMParameters::RocchioThenSplit) {

											// algo : creer les clusters correspondant aux modalites cibles, puis diviser le cluster ayant la plus grande inertie intra en 2 clusters.
											// Recalculer les inerties, puis refaire cette division jusqu'a obtenir les K clusters.
											if (parameters->GetKValue() <= 1) {
												AddWarning("Clusters initialization : Rocchio then Split algorithm is possible only if K is > 1 ");
												bOk = false;
											}
											if (targetAttribute == NULL) {
												AddWarning("Clusters initialization : Rocchio then Split algorithm is available only in supervised mode");
												bOk = false;
											}
											if (bOk)
												bOk = clusteringInitializer->InitializeRocchioThenSplitCentroids(instances, targetAttribute);
										}
										else

											if (initMethod == KMParameters::Bisecting) {

												if (parameters->GetKValue() <= 1) {
													AddWarning("Clusters initialization : Bisecting algorithm is possible only if K is > 1 ");
													bOk = false;
												}
												if (bOk)
													bOk = clusteringInitializer->InitializeBisectingCentroids(instances, targetAttribute);
											}

	}

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	if (bOk) {

		if (parameters->GetVerboseMode() and parameters->GetClusteringType() == KMParameters::KMeans and kmClusters->GetSize() < parameters->GetKValue()) {
			// impossible de creer autant de centres que demande, avant reaffectation
			AddWarning("Clusters initialization failed before instances re-assigment : only " + ALString(IntToString(kmClusters->GetSize())) + " cluster(s) centroid(s) could be created with this initialization method.");
			bOk = false;
		}

		// sauvegarder la valeur initiale des centroides, aux fins de reporting
		for (int i = 0; i < kmClusters->GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
			c->SetInitialCentroidValues(c->GetModelingCentroidValues());
		}

		TaskProgression::DisplayLabel("Clusters initialization : assigning instances to created clusters....");

		// (re)affecter les instances en fonction des centroides determines
		AddInstancesToClusters(instances);

		// y a t il des clusters vides ?
		iDroppedClustersNumber = 0;

		for (int i = 0; i < kmClusters->GetSize(); i++) {

			KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
			if (c->GetCount() == 0) {
				DeleteClusterAt(i);
				iDroppedClustersNumber++;
				i = -1; // recommencer la boucle
			}
		}

		if (iDroppedClustersNumber > 0) {
			if (parameters->GetClusteringType() == KMParameters::KNN) {
				if (parameters->GetMinKValuePostOptimization() > kmClusters->GetSize()) {
					AddWarning("Clusters initialization failed after reassigning instances to created clusters : unable to initialize KNN clustering with the requested minimal value for K.");
					AddSimpleMessage("Possible reasons : too many instances with missing values, or maybe too many instances have the same values.");
					AddSimpleMessage("Hint : decrease K value, or try changing preprocessing parameters.");
					bOk = false;
				}
			}
			else {
				AddWarning("Clusters initialization failed after reassigning instances to created clusters : " + ALString(IntToString(iDroppedClustersNumber)) + " empty cluster(s) have been dropped.");
				AddSimpleMessage("Hint : decrease K value, or try changing preprocessing parameters.");
				bOk = false;
			}
		}

		if (bOk) {

			// mise a jour initiale des stats de chaque cluster apres initialisation
			for (int i = 0; i < kmClusters->GetSize(); i++) {

				if (TaskProgression::IsInterruptionRequested()) {
					bOk = false;
					break;
				}

				TaskProgression::DisplayLabel("Clusters initialization : computing initial statistics for cluster " + ALString(IntToString(i + 1)) + " on " + ALString(IntToString(parameters->GetKValue())));

				KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));

				if (parameters->GetMaxIterations() != -1) {
					c->ComputeIterationStatistics(); // mise a jour des stats ET centroides, suite a l'ajout des instances aux clusters
				}
				else {
					// on met a jour les stats des clusters, mais sans toucher aux centroides
					c->ComputeDistanceSum(KMParameters::L2Norm);
					c->ComputeDistanceSum(KMParameters::CosineNorm);
					c->ComputeDistanceSum(KMParameters::L1Norm);
					c->SetFrequency(c->GetCount());
					c->ComputeInstanceNearestToCentroid(parameters->GetDistanceType());
					c->ComputeInertyIntra(parameters->GetDistanceType());
					c->SetStatisticsUpToDate(true);
				}
			}
		}

		if (bOk) {

			TaskProgression::DisplayLabel("Clusters initialization : computing initial clusters centers distance...");

			// gestion des eventuels clusters droppes : (re)initaliser la matrice des distances inter-clusters, ainsi que la correspondance entre chaque cluster et son plus proche cluster
			ComputeClustersCentersDistances();

			// mise a jour des labels
			if (initMethod != KMParameters::Bisecting and initMethod != KMParameters::ClassDecomposition) { // car gestion particuliere dans ces cas la
				for (int i = 0; i < kmClusters->GetSize(); i++) {
					KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
					c->SetLabel(IntToString(i + 1));
				}
			}
		}
	}

	if (TaskProgression::IsInterruptionRequested()) {
		AddWarning("Interruption requested by user");
		bOk = false;
	}

	TaskProgression::DisplayLabel("");
	TaskProgression::EndTask();

	return bOk;

}


bool KMClustering::UpdateProgressionBar(const longint instancesNumber, const int iterationsDone, const int movements) {

	assert(instancesNumber > 0);

	if (instancesNumber == 0)
		return false;

	// Suivi de progression
	double progression = ((instancesNumber - movements) * 100) / instancesNumber;

	// ponderation grossi�re du % de progression
	if (iterationsDone < 2)
		progression /= 10;
	else if (iterationsDone < 7)
		progression /= 7;
	else if (iterationsDone < 10)
		progression /= 4;
	else if (iterationsDone < 20)
		progression /= 2;
	else if (iterationsDone < 30)
		progression /= 1.8;
	else if (iterationsDone < 40)
		progression /= 1.5;
	else if (iterationsDone < 50)
		progression /= 1.3;
	else if (iterationsDone < 60)
		progression /= 1.2;

	TaskProgression::DisplayProgression((int)progression);
	TaskProgression::DisplayLabel("Current clustering progression");

	if (TaskProgression::IsInterruptionRequested())
		return true;
	else
		return false;

}

void KMClustering::ComputeTrainingTargetProbs(const KWAttribute* targetAttribute) {

	assert(oaTargetAttributeValues.GetSize() > 0);

	for (int i = 0; i < GetClusters()->GetSize(); i++)
	{
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		c->ComputeTrainingTargetProbs(oaTargetAttributeValues, targetAttribute);
	}

	// en fonction des probas, calculer la classe majoritaire
	ComputeTrainingConfusionMatrix(targetAttribute);
}

void KMClustering::ComputeTrainingConfusionMatrix(const KWAttribute* targetAttribute) {

	assert(oaTargetAttributeValues.GetSize() > 0);
	assert(instancesToClusters != NULL);
	assert(instancesToClusters->GetCount() > 0);

	// colonne = classe reelle, ligne = classe predite
	kwftConfusionMatrix->SetFrequencyVectorNumber(oaTargetAttributeValues.GetSize());
	for (int i = 0; i < kwftConfusionMatrix->GetFrequencyVectorNumber(); i++) {
		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(i));
		fv->GetFrequencyVector()->SetSize(oaTargetAttributeValues.GetSize());
	}
	POSITION position = instancesToClusters->GetStartPosition();
	NUMERIC key;
	Object* oCurrent;

	while (position != NULL) {

		instancesToClusters->GetNextAssoc(position, key, oCurrent);
		KWObject* currentInstance = static_cast<KWObject *>(oCurrent);
		KMCluster* cluster = cast(KMCluster*, oCurrent);

		int idxMajorityTarget = cluster->GetMajorityTargetIndex();
		assert(idxMajorityTarget >= 0);

		ALString actualTarget = currentInstance->GetSymbolValueAt(targetAttribute->GetLoadIndex()).GetValue();

		// rechercher l'index correspondant a la valeur de la modalite, pour renseigner notre tableau d'occurences
		int idxActualTarget = 0;
		for (; idxActualTarget < oaTargetAttributeValues.GetSize(); idxActualTarget++) {
			StringObject* s = cast(StringObject*, oaTargetAttributeValues.GetAt(idxActualTarget));
			if (actualTarget == s->GetString())
				break;
		}

		assert(idxActualTarget != oaTargetAttributeValues.GetSize());

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(idxMajorityTarget));

		fv->GetFrequencyVector()->SetAt(idxActualTarget,
			fv->GetFrequencyVector()->GetAt(idxActualTarget) + 1);
	}

}


void KMClustering::UpdateConfusionMatrix(const Symbol& sPredictedTarget, const Symbol& sActualTarget) {

	assert(oaTargetAttributeValues.GetSize() > 0);
	assert(kwftConfusionMatrix != NULL);

	if (kwftConfusionMatrix->GetFrequencyVectorNumber() != oaTargetAttributeValues.GetSize() or
		kwftConfusionMatrix->GetFrequencyVectorAt(0)->GetSize() != oaTargetAttributeValues.GetSize()) {

		/* deux cas possibles :
			- soit c'est la premiere initialisation de la table de contingence
			- soit une valeur cible inconnue en apprentissage figure dans le fichier de test
		*/

		if (kwftConfusionMatrix->GetFrequencyVectorNumber() == 0 or kwftConfusionMatrix->GetFrequencyVectorAt(0)->GetSize() == 0) {
			// premiere initialisation
			kwftConfusionMatrix->SetFrequencyVectorNumber(oaTargetAttributeValues.GetSize());
			for (int i = 0; i < kwftConfusionMatrix->GetFrequencyVectorNumber(); i++) {
				KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(i));
				fv->GetFrequencyVector()->SetSize(oaTargetAttributeValues.GetSize());
			}
		}
		else {

			// agrandir la matrice de confusion en preservant ses valeurs existantes, pour prendre en compte la valeur cible qui etait inconnue jusqu'ici
			KWFrequencyTable tmp;
			tmp.CopyFrom(kwftConfusionMatrix);
			delete kwftConfusionMatrix;
			kwftConfusionMatrix = new KWFrequencyTable;
			kwftConfusionMatrix->SetFrequencyVectorNumber(oaTargetAttributeValues.GetSize());
			for (int i = 0; i < kwftConfusionMatrix->GetFrequencyVectorNumber(); i++) {
				KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(i));
				fv->GetFrequencyVector()->SetSize(oaTargetAttributeValues.GetSize());
			}

			for (int iSource = 0; iSource < tmp.GetFrequencyVectorNumber(); iSource++) {
				KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, tmp.GetFrequencyVectorAt(iSource));
				KWDenseFrequencyVector* fv2 = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(iSource));
				for (int iTarget = 0; iTarget < fv->GetSize(); iTarget++) {
					fv2->GetFrequencyVector()->SetAt(iTarget, fv->GetFrequencyVector()->GetAt(iTarget));
				}
			}
		}
	}

	// rechercher les index "predit" et "reel", pour renseigner notre tableau d'occurences

	int idxActualTarget = -1;
	int idxPredictedTarget = -1;

	for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {

		StringObject* s = cast(StringObject*, oaTargetAttributeValues.GetAt(i));

		if (sActualTarget == s->GetString())
			idxActualTarget = i;

		if (sPredictedTarget == s->GetString())
			idxPredictedTarget = i;
	}

	assert(idxActualTarget != -1);
	assert(idxPredictedTarget != -1);

	KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwftConfusionMatrix->GetFrequencyVectorAt(idxPredictedTarget));
	fv->GetFrequencyVector()->SetAt(idxActualTarget,
		fv->GetFrequencyVector()->GetAt(idxActualTarget) + 1);

}


void KMClustering::ReadTargetAttributeValues(const ObjectArray* instances, const KWAttribute* targetAttribute) {

	// stocker dans un tableau les modalites existantes pour l'attribut cible
	// il faut mettre la modalite cible (s'il y en a une) en premier, afin de pouvoir retrouver cette information dans la suite du traitement

	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(oaTargetAttributeValues.GetSize() == 0);
	assert(targetAttribute != NULL);

	const KWLoadIndex targetIndex = targetAttribute->GetLoadIndex();
	assert(targetIndex.IsValid());

	boolean bHasMainTargetModality = parameters->GetMainTargetModality() != "" ? true : false;
	int iMainTargetModalityIndex = -1;

	for (int i = 0; i < instances->GetSize(); i++)
	{
		KWObject* instance = cast(KWObject*, instances->GetAt(i));

		if (parameters->HasMissingKMeanValue(instance))
			continue;

		const ALString sTarget = instance->GetSymbolValueAt(targetIndex).GetValue();

		bool found = false;
		for (int iTarget = 0; iTarget < oaTargetAttributeValues.GetSize(); iTarget++) {
			if (cast(StringObject*, oaTargetAttributeValues.GetAt(iTarget))->GetString() == sTarget)
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
	// si la modalite cible principale est presente en base de donnees, alors il faut qu'elle figure en premier dans le tableau des valeurs cibles presentes en base
	if (bHasMainTargetModality and iMainTargetModalityIndex != -1) {

		ObjectArray oaNewTargetAttributeValues;
		oaNewTargetAttributeValues.Add(oaTargetAttributeValues.GetAt(iMainTargetModalityIndex));

		for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {
			StringObject* modality = cast(StringObject*, oaTargetAttributeValues.GetAt(i));
			if (modality->GetString() != parameters->GetMainTargetModality())
				oaNewTargetAttributeValues.Add(oaTargetAttributeValues.GetAt(i));
		}
		oaTargetAttributeValues.CopyFrom(&oaNewTargetAttributeValues);
	}
}

void KMClustering::AddInstancesToClusters(const ObjectArray* instances) {

	// affecter des instances de DB a des clusters dont les centroides ont deja et� calcul�s
	// le readAll() de la base doit avoir deja �t� fait

	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(instancesToClusters != NULL);

	clusteringInitializer->ResetInstancesWithMissingValuesNumber();

	// (re)initaliser la matrice des distances inter-clusters
	ComputeClustersCentersDistances();

	// s'assurer qu'il n'y a aucune instance dans les clusters, et supprimer les instances existantes, le cas echeant
	for (int i = 0; i < GetClusters()->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		assert(c->GetModelingCentroidValues().GetSize() > 0); // les centroides doivent avoir ete calcul�s (on ne fait qu'affecter de nouvelles instances a des centres existants)
		c->RemoveAll(); // suppression des pointeurs sur instances
		c->SetStatisticsUpToDate(false); // forcer le recalcul des stats, y compris sur les clusters clones (qui ne contiennent pas d'instances)

	}
	instancesToClusters->RemoveAll();

	// reaffecter les instances aux clusters, en fonction de leurs centroides precedemment calcul�s
	for (int i = 0; i < instances->GetSize(); i++) {
		KWObject* instance = cast(KWObject*, instances->GetAt(i));
		if (parameters->HasMissingKMeanValue(instance)) {
			clusteringInitializer->IncrementInstancesWithMissingValuesNumber();
			continue;
		}
		KMCluster* c = FindNearestCluster(instance);

		if (c != NULL) {
			c->AddInstance(instance);
			instancesToClusters->SetAt(instance, c);
		}
	}

	// mettre a jour les frequences des clusters (valeur persistante, meme si on supprime les instances pour ne conserver que les centroides)
	for (int i = 0; i < GetClusters()->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		c->SetFrequency(c->GetCount());
	}

}

void KMClustering::FinalizeReplicateComputing(bool recomputeCentroids) {

	// recalculer les stats et centroides, au cas ou quelques instances n'ont pas �t� reaffect�es � leur cluster d'origine

	for (int i = 0; i < GetClusters()->GetSize(); i++) {

		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));

		if (recomputeCentroids)
			c->ComputeIterationStatistics(); // maj des centroides et de la somme des distances, pour la norme choisie. Cette methode doit etre ex�cut�e y compris si un cluster est devenu vide
		else {
			// conserver les centroides existants
			c->ComputeDistanceSum(GetParameters()->GetDistanceType());
			c->SetFrequency(c->GetCount());
			c->SetStatisticsUpToDate(true);
		}

		if (c->GetFrequency() == 0) // teste si le cluster est devenu vide
			continue;

		c->ComputeInstanceNearestToCentroid(parameters->GetDistanceType());

		// calculer les sommes des distances au centre du cluster (uniquement pour les normes pas encore calcul�es)
		if (GetParameters()->GetDistanceType() == KMParameters::L1Norm) {
			c->ComputeDistanceSum(KMParameters::L2Norm);
			c->ComputeDistanceSum(KMParameters::CosineNorm);
		}
		else {
			if (GetParameters()->GetDistanceType() == KMParameters::L2Norm) {
				c->ComputeDistanceSum(KMParameters::L1Norm);
				c->ComputeDistanceSum(KMParameters::CosineNorm);
			}
			else {
				c->ComputeDistanceSum(KMParameters::L1Norm);
				c->ComputeDistanceSum(KMParameters::L2Norm);
			}
		}
		c->ComputeInertyIntra(parameters->GetDistanceType()); // necessaire pour calculer l'indice Davies Bouldin
	}

	UpdateGlobalDistancesSum();

}

void KMClustering::UpdateGlobalDistancesSum() {

	// parcourir les clusters et cumuler la somme des distances des instances par rapport � leurs clusters respectifs

	cvClustersDistancesSum.Initialize(); // remise a zero d'eventuels precedents resultats

	for (int i = 0; i < GetClusters()->GetSize(); i++) {

		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));

		cvClustersDistancesSum.SetAt(KMParameters::L2Norm, cvClustersDistancesSum.GetAt(KMParameters::L2Norm) +
			c->GetDistanceSum(KMParameters::L2Norm));

		cvClustersDistancesSum.SetAt(KMParameters::L1Norm, cvClustersDistancesSum.GetAt(KMParameters::L1Norm) +
			c->GetDistanceSum(KMParameters::L1Norm));

		cvClustersDistancesSum.SetAt(KMParameters::CosineNorm, cvClustersDistancesSum.GetAt(KMParameters::CosineNorm) +
			c->GetDistanceSum(KMParameters::CosineNorm));
	}
}


// construire le cluster contenant toutes les instances, et calculer les statistiques correspondantes
void KMClustering::ComputeGlobalClusterStatistics(ObjectArray* instances) {

	// affecter des instances de DB a un cluster unique "fictif", afin de calculer les statistiques globales

	assert(instances != NULL);
	assert(instances->GetSize() > 0);

	kmGlobalCluster = CreateGlobalCluster();

	// construire le cluster global
	for (int i = 0; i < instances->GetSize(); i++) {
		KWObject* instance = cast(KWObject*, instances->GetAt(i));
		if (parameters->HasMissingKMeanValue(instance))
			continue;
		kmGlobalCluster->AddInstance(instance);
	}

	if (kmGlobalCluster->GetCount() == 0)
		return;

	kmGlobalCluster->ComputeIterationStatistics();
	kmGlobalCluster->ComputeInstanceNearestToCentroid(parameters->GetDistanceType());

	// calculer la distance dans les normes pas encore trait�es
	if (GetParameters()->GetDistanceType() == KMParameters::L1Norm) {
		kmGlobalCluster->ComputeDistanceSum(KMParameters::L2Norm);
		kmGlobalCluster->ComputeDistanceSum(KMParameters::CosineNorm);
	}
	else {
		if (GetParameters()->GetDistanceType() == KMParameters::L2Norm) {
			kmGlobalCluster->ComputeDistanceSum(KMParameters::L1Norm);
			kmGlobalCluster->ComputeDistanceSum(KMParameters::CosineNorm);
		}
		else {
			kmGlobalCluster->ComputeDistanceSum(KMParameters::L1Norm);
			kmGlobalCluster->ComputeDistanceSum(KMParameters::L2Norm);
		}
	}
}

void KMClustering::AddTargetAttributeValueIfNotExists(const KWAttribute* targetAttribute, const KWObject* instance) {

	assert(kmGlobalCluster != NULL);

	ALString value = instance->GetSymbolValueAt(targetAttribute->GetLoadIndex()).GetValue();

	// rechercher l'index correspondant a la valeur de la modalite, pour renseigner notre tableau d'occurences
	int idx = 0;
	for (; idx < oaTargetAttributeValues.GetSize(); idx++) {
		StringObject* s = cast(StringObject*, oaTargetAttributeValues.GetAt(idx));
		if (value == s->GetString())
			break;
	}

	if (idx == oaTargetAttributeValues.GetSize()) {

		// si valeur inexistante : la referencer dans tous les clusters

		StringObject* s = new StringObject;
		s->SetString(value);
		oaTargetAttributeValues.Add(s);

		for (int i = 0; i < GetClusters()->GetSize(); i++) {

			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
			ContinuousVector cv;
			cv.CopyFrom(&c->GetTargetProbs());
			cv.SetSize(oaTargetAttributeValues.GetSize());
			c->SetTargetProbs(cv);
		}
	}

	ContinuousVector cv;
	cv.CopyFrom(&kmGlobalCluster->GetTargetProbs());
	cv.SetSize(oaTargetAttributeValues.GetSize());
	kmGlobalCluster->SetTargetProbs(cv);

}

KMCluster* KMClustering::CreateGlobalCluster()
{
	if (kmGlobalCluster != NULL)
		delete kmGlobalCluster;

	kmGlobalCluster = new KMCluster(parameters);
	kmGlobalCluster->SetLabel("global");

	return kmGlobalCluster;
}

void KMClustering::SetGlobalCluster(KMCluster* c) {

	assert(c != NULL);

	if (kmGlobalCluster != NULL)
		delete kmGlobalCluster;

	kmGlobalCluster = c;
}


KMClustering* KMClustering::Clone()
{

	KMClustering* aClone;

	aClone = new KMClustering(parameters);
	aClone->CopyFrom(this);

	return aClone;
}

void KMClustering::CopyFrom(const KMClustering* aSource)
{
	require(aSource != NULL);

	// copie des derniers clusters calcul�s :
	kmClusters->DeleteAll();
	for (int i = 0; i < aSource->kmClusters->GetSize(); i++)
	{
		KMCluster* c = cast(KMCluster*, aSource->kmClusters->GetAt(i));
		kmClusters->Add(c->Clone());
	}

	// copie des meilleurs clusters observ�s :
	kmBestClusters->DeleteAll();
	for (int i = 0; i < aSource->kmBestClusters->GetSize(); i++)
	{
		KMCluster* c = cast(KMCluster*, aSource->kmBestClusters->GetAt(i));
		kmBestClusters->Add(c->Clone());
	}

	// copie du cluster global (contenant toutes les instances)
	if (kmGlobalCluster != NULL) {
		delete kmGlobalCluster;
		kmGlobalCluster = NULL;
	}
	if (aSource->kmGlobalCluster != NULL)
		kmGlobalCluster = aSource->kmGlobalCluster->Clone();

	parameters = aSource->parameters;
	iIterationsDone = aSource->iIterationsDone;
	dUsedSampleNumberPercentage = aSource->dUsedSampleNumberPercentage;
	cvClustersDistancesSum.CopyFrom(&aSource->cvClustersDistancesSum);
	iDroppedClustersNumber = aSource->iDroppedClustersNumber;

	// copier les StringObjects de oaTargetAttributeValues
	oaTargetAttributeValues.DeleteAll();
	for (int i = 0; i < aSource->oaTargetAttributeValues.GetSize(); i++) {
		StringObject* value = new StringObject;
		value->SetString(cast(StringObject*, aSource->oaTargetAttributeValues.GetAt(i))->GetString());
		oaTargetAttributeValues.Add(value);
	}

	ComputeClustersCentersDistances();

	// copie de l'initialiseur de clustering
	if (clusteringInitializer != NULL)
		delete clusteringInitializer;
	clusteringInitializer = new KMClusteringInitializer();
	clusteringInitializer->CopyFrom(aSource->clusteringInitializer);

	// copie du gestionnaire des criteres de qualite de clustering
	if (clusteringQuality != NULL)
		delete clusteringQuality;
	clusteringQuality = new KMClusteringQuality();
	clusteringQuality->CopyFrom(aSource->clusteringQuality);
	clusteringQuality->SetClusters(kmClusters); // car les clusters de la source pointent desormais sur des clusters detruits, cf. plus haut

	// copie du gestionnaire d'intervalles et modalites
	if (attributesPartitioningManager != NULL)
		delete attributesPartitioningManager;
	attributesPartitioningManager = new KMAttributesPartitioningManager;
	attributesPartitioningManager->CopyFrom(aSource->attributesPartitioningManager);
}

void KMClustering::CloneBestClusters() {

	kmBestClusters->DeleteAll();

	for (int i = 0; i < GetClusters()->GetSize(); i++)
	{
		KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
		kmBestClusters->Add(c->Clone());	// NB. on ne clone pas les instances elles-m�mes, mais uniquement les centroides et les stats
	}
}


boolean KMClustering::PostOptimize(const ObjectArray* instances, const KWAttribute* targetAttribute) {

	assert(instances != NULL);
	assert(kmClusters != NULL);
	assert(kmClusters->GetSize() > 0);

	boolean bOk = true;

#ifdef DEBUG_POST_OPTIMIZATION
	// pendant les tests, logguer les resultats de clustering dans un fichier
	fstream fsPostOptimizationFile;
	ALString sPostOptimizationFileName = FileService::BuildFilePathName(RMResourceManager::GetTmpDir(), "enneade_PostOptimization.txt");

	if (not FileService::OpenOutputFile(sPostOptimizationFileName, fsPostOptimizationFile)) {
		AddWarning("Can't open post-optimization log file '" + ALString(sPostOptimizationFileName) + "'");
		return false;
	}
#endif

	TaskProgression::BeginTask();
	TaskProgression::DisplayMainLabel("Clustering post-optimization, initial clustering size is " + ALString(IntToString(kmClusters->GetSize())));

	// calculer le nombre de clustering a faire, afin d'afficher une barre de progression
	int nbClusteringsDone = 0;
	int nbClusteringsToDo = 0;
	for (int i = 1; i <= kmClusters->GetSize(); i++)
		nbClusteringsToDo += i;

	// mise a jour des stats, necessaire apres reaffectation des instances
	if (parameters->GetVerboseMode())
		AddSimpleMessage("Re-computing stats after instances re-affectation...");

	ComputeTrainingTargetProbs(targetAttribute);
	GetClusteringQuality()->ComputeEVA(kmGlobalCluster, oaTargetAttributeValues.GetSize());

	double overAllBestEVA = GetClusteringQuality()->GetEVA();
	int iBestK = kmClusters->GetSize();

	NumericKeyDictionary* instancesToClustersByAscDistance = ComputeInstancesToClustersByAscDistance();// pour chaque instance, creer une liste des clusters tries par ordre de distance croissante
	KWFrequencyTable* currentClusteringModalitiesFrequenciesByClusters = CreateModalitiesFrequenciesByClusters(kmClusters);
	KWFrequencyTable overallBestModalitiesFrequenciesByClusters;
	NumericKeyDictionary* removedInstancesNewClusters = new NumericKeyDictionary;

	const double evaOneCluster =
		GetClusteringQuality()->ComputeEVAFirstTerm(1, currentClusteringModalitiesFrequenciesByClusters) +
		GetClusteringQuality()->ComputeEVASecondTerm(1, currentClusteringModalitiesFrequenciesByClusters) +
		GetClusteringQuality()->ComputeEVAThirdTerm(1, currentClusteringModalitiesFrequenciesByClusters);

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage("\nPost-optimization for the best replicate :");
		AddSimpleMessage("--------------------------------------------------------------------------------------------------------------------------------------------------------------");
		AddSimpleMessage("K value\tBest EVA\tCluster to remove\tOverall best K\t\tOverall best EVA");
		AddSimpleMessage(KMGetDisplayString(kmClusters->GetSize()) +
			KMGetDisplayString(overAllBestEVA) + "\t\t" +
			KMGetDisplayString(iBestK) +
			KMGetDisplayString(overAllBestEVA)
		);
	}


#ifdef DEBUG_POST_OPTIMIZATION
	fsPostOptimizationFile << endl << "Initial EVA value is " << overAllBestEVA << endl << endl;
#endif

	KMClustering* currentClustering = Clone();

	// boucle de test de toutes les valeurs de K, de la valeur max a la valeur min :

	while (currentClustering->GetClusters()->GetSize() > parameters->GetMinKValuePostOptimization()) {

		if (TaskProgression::IsInterruptionRequested())
			break;

		ALString sClusterToRemove;
		double currentClusteringBestEVA = KWContinuous::GetMinValue();

#ifdef DEBUG_POST_OPTIMIZATION
		fsPostOptimizationFile << "Current clustering (K = " << currentClustering->GetClusters()->GetSize() - 1 << ")\tEVA" << endl;
#endif

		const int K = currentClustering->GetClusters()->GetSize() - 1;
		const double eVAallClustersFirstTerm = GetClusteringQuality()->ComputeEVAFirstTerm(K, currentClusteringModalitiesFrequenciesByClusters);

		KWFrequencyTable currentClusteringBestLocalFrequencies;

		TaskProgression::DisplayLabel("Looking for best EVA when K = " + ALString(IntToString(currentClustering->GetClusters()->GetSize() - 1)) +
			" (so far, best EVA is " + ALString(DoubleToString(overAllBestEVA)) + ", optimal K value is " + ALString(IntToString(iBestK)) + ")");

		// rechercher le cluster dont la suppression produit la meilleure valeur d'EVA, pour la valeur de K en cours de test
		KMCluster* clusterToRemove = PostOptimizationSearchClusterToRemove(currentClustering,
			currentClusteringModalitiesFrequenciesByClusters,
			instancesToClustersByAscDistance,
			targetAttribute,
			K,
			eVAallClustersFirstTerm,
			evaOneCluster,
			currentClusteringBestLocalFrequencies,
			removedInstancesNewClusters,
			nbClusteringsDone,
			currentClusteringBestEVA
#ifdef DEBUG_POST_OPTIMIZATION
			, fsPostOptimizationFile
#endif
		);

		bOk = (clusterToRemove == NULL ? false : true);

		if (not bOk)
			break;

		PostOptimizationMoveInstancesToNextClusters(removedInstancesNewClusters); // affecter les instances du cluster a supprimer, a leurs nouveaux clusters

		sClusterToRemove = clusterToRemove->GetLabel();

		//  sur le clustering courant, supprimer le cluster dont la suppression a produit la meilleure augmentation de l'EVA, et continuer avec une valeur de K inferieure
		for (int i = 0; i < currentClustering->GetClusters()->GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, currentClustering->GetClusters()->GetAt(i));
			if (c->GetIndex() == clusterToRemove->GetIndex()) {
				currentClustering->DeleteClusterAt(i);
				break;
			}
		}

		// le prochain clustering (avec une valeur de K inferieure) doit repartir de la meilleure solution locale observee au clustering precedent :
		currentClusteringModalitiesFrequenciesByClusters->CopyFrom(&currentClusteringBestLocalFrequencies);

		// si ce clustering est le meilleur observe jusqu'ici, toutes valeurs de K confondues : on le garde en memoire
		if (currentClusteringBestEVA >= overAllBestEVA) {

			overAllBestEVA = currentClusteringBestEVA;
			overallBestModalitiesFrequenciesByClusters.CopyFrom(currentClusteringModalitiesFrequenciesByClusters);

			// pouvoir reperer les clusters definitivement ecartes de la solution optimisee, et sans les supprimer physiquement pour le moment :
			for (int i = 0; i < kmClusters->GetSize(); i++) {
				KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
				KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, overallBestModalitiesFrequenciesByClusters.GetFrequencyVectorAt(c->GetIndex()));
				longint sourceFrequency = 0;
				for (int iTarget = 0; iTarget < fv->GetSize(); iTarget++) {
					sourceFrequency += fv->GetFrequencyVector()->GetAt(iTarget);
				}
				if (sourceFrequency == 0) {
					c->SetFrequency(0);
				}
			}
			iBestK = currentClustering->GetClusters()->GetSize();
		}

		if (parameters->GetVerboseMode())
			AddSimpleMessage(KMGetDisplayString(currentClustering->GetClusters()->GetSize()) +
				KMGetDisplayString(currentClusteringBestEVA) +
				sClusterToRemove + "\t\t" +
				KMGetDisplayString(iBestK) +
				KMGetDisplayString(overAllBestEVA)
			);

#ifdef DEBUG_POST_OPTIMIZATION
		if (currentClustering->GetClusters()->GetSize() > 1)
			fsPostOptimizationFile << endl << "Now computing EVA when cluster " << sClusterToRemove << " is removed from current clustering :" << endl;
#endif

		TaskProgression::DisplayProgression(nbClusteringsDone / nbClusteringsToDo * 100);
	}

	if (TaskProgression::IsInterruptionRequested()) {
		AddWarning("Interruption requested by user");
		bOk = false;
	}

	if (bOk) {

		// suppression physique des clusters ne faisant pas partie de la solution optimisee :
		for (int i = 0; i < kmClusters->GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, kmClusters->GetAt(i));
			if (c->GetFrequency() == 0) {
				DeleteClusterAt(i);
				i--;
			}
		}

		AddInstancesToClusters(instances); // reaffectation finale des instances aux clusters faisant partie de la solution optimisee :

#ifdef DEBUG_POST_OPTIMIZATION
		fsPostOptimizationFile << endl << endl << "Optimized clustering : ";
#endif

		// mise a jour finale des stats des clusters (sans toucher aux centroides), pour le meilleur clustering obtenu :
		for (int i = 0; i < kmClusters->GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
			c->ComputeDistanceSum(KMParameters::L2Norm);
			c->ComputeDistanceSum(KMParameters::CosineNorm);
			c->ComputeDistanceSum(KMParameters::L1Norm);
			c->SetFrequency(c->GetCount());
			c->ComputeInstanceNearestToCentroid(parameters->GetDistanceType());
			c->ComputeInertyIntra(parameters->GetDistanceType());
			c->SetStatisticsUpToDate(true);
#ifdef DEBUG_POST_OPTIMIZATION
			fsPostOptimizationFile << c->GetLabel() << " ";
#endif
		}
#ifdef DEBUG_POST_OPTIMIZATION
		fsPostOptimizationFile << ", EVA is " << overAllBestEVA << endl;
#endif

		ComputeTrainingTargetProbs(targetAttribute); // calculer la repartition des valeurs de l'attribut cible, dans chaque cluster
		GetClusteringQuality()->ComputeEVA(kmGlobalCluster, oaTargetAttributeValues.GetSize());// pour tests

		if (parameters->GetVerboseMode()) {
			AddSimpleMessage("Best EVA is " + ALString(DoubleToString(overAllBestEVA)) + ", optimal K value is " + ALString(IntToString(kmClusters->GetSize())));
			AddSimpleMessage("EVA recomputed after instances re-affectation is " + ALString(DoubleToString(GetClusteringQuality()->GetEVA())));
#ifdef DEBUG_POST_OPTIMIZATION
			AddSimpleMessage("Post-optimization is done. Please check file " + sPostOptimizationFileName + " for detailed log");
#else
			AddSimpleMessage("Post-optimization is done.");
#endif
		}
	}

	delete currentClustering;

	TaskProgression::DisplayMainLabel("");
	TaskProgression::EndTask();

	delete currentClusteringModalitiesFrequenciesByClusters;
	delete removedInstancesNewClusters;

	if (parameters->GetPostOptimizationVnsLevel() > 0)
		bOk = PostOptimizeVns(instances, targetAttribute);

	instancesToClustersByAscDistance->DeleteAll();
	delete instancesToClustersByAscDistance;

	return bOk;

}

KMCluster* KMClustering::PostOptimizationSearchClusterToRemove(const KMClustering* currentClustering,
	const KWFrequencyTable* currentClusteringModalitiesFrequenciesByClusters,
	const NumericKeyDictionary* instancesToClustersByAscDistance,
	const KWAttribute* targetAttribute,
	const int K,
	const double eVAallClustersFirstTerm,
	const double evaOneCluster,
	KWFrequencyTable& currentClusteringBestLocalFrequencies,
	NumericKeyDictionary* removedInstancesNewClusters,
	int& nbClusteringsDone,
	double& currentClusteringBestEVA
#ifdef DEBUG_POST_OPTIMIZATION
	, fstream& fsPostOptimizationFile
#endif
) {

	KMCluster* result = NULL;

	// pour chacun des centres de clusters toujours en cours de selection, calculer l'EVA du clustering, dans l'hypothese ou ce centre serait supprime
	for (int idxCluster = 0; idxCluster < currentClustering->GetClusters()->GetSize(); idxCluster++) {

		KMCluster* c = cast(KMCluster*, currentClustering->GetClusters()->GetAt(idxCluster));
		KMCluster* removedCluster = cast(KMCluster*, kmClusters->GetAt(c->GetIndex())); // ce cluster contient reellement des instances, contrairement au cluster clone contenu dans currentClustering
		assert(removedCluster != NULL);
		assert(removedCluster->GetFrequency() > 0);

		if (TaskProgression::IsInterruptionRequested())
			break;

		NumericKeyDictionary* removedInstancesNewClustersAfterClusterRemoval = new NumericKeyDictionary;// memoriser les nouveaux clusters qui seraient affectes aux instances du cluster dont on teste la suppression

		// parcourir les instances du cluster dont on teste la suppression, et pour chaque instance, mettre a jour l' effectif du cluster le plus proche de cette instance
		KWFrequencyTable frequenciesAfterClusterRemoval;
		frequenciesAfterClusterRemoval.CopyFrom(currentClusteringModalitiesFrequenciesByClusters);
		boolean bOk = PostOptimizationUpdateFrequencies(removedCluster, instancesToClustersByAscDistance, targetAttribute, oaTargetAttributeValues, frequenciesAfterClusterRemoval, removedInstancesNewClustersAfterClusterRemoval);

		if (not bOk)
			return NULL;

		// en utilisant la table des frequences par modalites resultant de l'hypothese de la suppression du cluster, calculer l'EVA
		const double eVAallClusters =
			eVAallClustersFirstTerm +
			GetClusteringQuality()->ComputeEVASecondTerm(K, &frequenciesAfterClusterRemoval) +
			GetClusteringQuality()->ComputeEVAThirdTerm(K, &frequenciesAfterClusterRemoval);

		const double currentEVA = 1 - (eVAallClusters / evaOneCluster);

		nbClusteringsDone++;

#ifdef DEBUG_POST_OPTIMIZATION
		for (int i = 0; i < currentClustering->GetClusters()->GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, currentClustering->GetClusters()->GetAt(i));
			if (c->GetIndex() != removedCluster->GetIndex())
				fsPostOptimizationFile << c->GetLabel() << " ";
		}
		fsPostOptimizationFile << "\t" << currentEVA << endl;
#endif

		// si, pour la valeur de K en cours de test, le critere EVA a ete ameliore "localement" en supprimant ce cluster, alors garder la memoire de l'index de ce cluster, ainsi que des frequences
		if (currentEVA > currentClusteringBestEVA) {
			currentClusteringBestEVA = currentEVA;
			result = removedCluster;
			currentClusteringBestLocalFrequencies.CopyFrom(&frequenciesAfterClusterRemoval);
			removedInstancesNewClusters->CopyFrom(removedInstancesNewClustersAfterClusterRemoval);
		}

		delete removedInstancesNewClustersAfterClusterRemoval;
	}

	assert(result != NULL);

	return result;
}


boolean KMClustering::PostOptimizationUpdateFrequencies(const KMCluster* removedCluster, const NumericKeyDictionary* instancesToClustersByAscDistance,
	const KWAttribute* targetAttribute, const ObjectArray& targetAttributeValues, KWFrequencyTable& frequenciesAfterClusterRemoval, NumericKeyDictionary* removedInstancesNewClusters) {

	assert(removedCluster != NULL);
	assert(removedCluster->GetFrequency() > 0);
	assert(removedCluster->GetFrequency() == removedCluster->GetCount());
	assert(instancesToClustersByAscDistance != NULL);
	assert(targetAttribute != NULL);
	assert(targetAttributeValues.GetSize() > 0);
	assert(frequenciesAfterClusterRemoval.GetTotalFrequency() == kmGlobalCluster->GetFrequency());

	KWDenseFrequencyVector* fvRemovedCluster = cast(KWDenseFrequencyVector*, frequenciesAfterClusterRemoval.GetFrequencyVectorAt(removedCluster->GetIndex()));
	int nbInstancesRemoved = 0;
	for (int idxTarget = 0; idxTarget < fvRemovedCluster->GetSize(); idxTarget++) {
		nbInstancesRemoved += fvRemovedCluster->GetFrequencyVector()->GetAt(idxTarget);
		fvRemovedCluster->GetFrequencyVector()->SetAt(idxTarget, 0);
	}
	assert(nbInstancesRemoved == removedCluster->GetCount());

	// parcourir les instances du cluster dont on veut tester la suppression, et les affecter a leurs clusters suivants les plus proches, dans leurs listes respectives
	POSITION position = removedCluster->GetStartPosition();
	NUMERIC key;
	Object* oCurrent;

	while (position != NULL) {

		removedCluster->GetNextAssoc(position, key, oCurrent);
		KWObject* currentInstance = static_cast<KWObject *>(oCurrent);
		Object* o = instancesToClustersByAscDistance->Lookup(currentInstance);
		assert(o != NULL);

		// acces a la liste des clusters tries par distances croissantes, pour cette instance, afin de trouver le cluster suivant, par ordre de distance croissante a l'instance
		// NB. ce cluster ne doit pas avoir ete ecarte auparavant de la liste des clusters faisant partie de la solution optimisee
		ObjectArray* oaClustersList = cast(ObjectArray*, o);
		assert(oaClustersList->GetSize() == kmClusters->GetSize());

		// chercher le cluster le plus proche dans la liste, qui soit encore disponible :

		KMCluster* nextCluster = NULL;

		for (int idxCluster = 0; idxCluster < oaClustersList->GetSize(); idxCluster++) {

			KMCluster* c = cast(KMCluster*, oaClustersList->GetAt(idxCluster));

			if (c->GetIndex() == removedCluster->GetIndex())
				continue;
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, frequenciesAfterClusterRemoval.GetFrequencyVectorAt(c->GetIndex()));
			longint sourceFrequency = 0;
			for (int iTarget = 0; iTarget < fv->GetSize(); iTarget++) {
				sourceFrequency += fv->GetFrequencyVector()->GetAt(iTarget);
			}

			if (sourceFrequency == 0)
				continue; // car si la frequence est a 0, cela veut dire qu'il a deja ete ecarte de la solution optimisee

			nextCluster = c;
			break;
		}

		if (nextCluster == NULL) {
			// ne devrait en principe jamais arriver, mais....
			AddError("Nearest available cluster not found for a database instance. Aborting post-optimization....");
			return false;
		}

		assert(nextCluster->GetFrequency() > 0);
		removedInstancesNewClusters->SetAt(currentInstance, nextCluster);

		// recuperer l'index de la modalite cible pour cette instance
		ALString value = currentInstance->GetSymbolValueAt(targetAttribute->GetLoadIndex()).GetValue();
		int idxTarget = 0;
		for (; idxTarget < targetAttributeValues.GetSize(); idxTarget++) {
			StringObject* s = cast(StringObject*, targetAttributeValues.GetAt(idxTarget));
			if (value == s->GetString())
				break;
		}

		assert(idxTarget < targetAttributeValues.GetSize());

		// Mettre a jour la frequence du cluster qui recupere cette instance, pour la modalite concernee
		KWDenseFrequencyVector* fvNextCluster = cast(KWDenseFrequencyVector*, frequenciesAfterClusterRemoval.GetFrequencyVectorAt(nextCluster->GetIndex()));
		fvNextCluster->GetFrequencyVector()->SetAt(idxTarget, fvNextCluster->GetFrequencyVector()->GetAt(idxTarget) + 1);
	}

	require(frequenciesAfterClusterRemoval.GetTotalFrequency() == kmGlobalCluster->GetFrequency());

	return true;
}


void KMClustering::PostOptimizationMoveInstancesToNextClusters(const NumericKeyDictionary* removedInstancesNewClusters) {

	// parcourir les instances du dictionnaire et les affecter a leurs clusters correspondants
	POSITION position = removedInstancesNewClusters->GetStartPosition();
	NUMERIC key;
	Object* oCurrent;

	while (position != NULL) {

		removedInstancesNewClusters->GetNextAssoc(position, key, oCurrent);
		KWObject* currentInstance = static_cast<KWObject *>(oCurrent);
		KMCluster* cluster = cast(KMCluster*, oCurrent);
		cluster->AddInstance(currentInstance);
	}

	// synchroniser les frequences des clusters toujours retenus par la solution optimisee, avec leur nouveau nombre d'instances
	for (int idxCluster = 0; idxCluster < kmClusters->GetSize(); idxCluster++) {
		KMCluster* cluster = cast(KMCluster*, kmClusters->GetAt(idxCluster));
		if (cluster->GetFrequency() != 0)
			cluster->SetFrequency(cluster->GetCount());
	}

}

KWFrequencyTable* KMClustering::CreateModalitiesFrequenciesByClusters(const ObjectArray* _clusters) {

	assert(oaTargetAttributeValues.GetSize() > 0);
	assert(_clusters != NULL);
	assert(_clusters->GetSize() > 0);

	KWFrequencyTable* modalityFrequencyByCluster = new KWFrequencyTable;
	modalityFrequencyByCluster->SetFrequencyVectorNumber(_clusters->GetSize());
	for (int i = 0; i < modalityFrequencyByCluster->GetFrequencyVectorNumber(); i++) {
		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(i));
		fv->GetFrequencyVector()->SetSize(oaTargetAttributeValues.GetSize());
	}

	// boucle sur les clusters de 1 � k, pour initialiser les valeurs de la table de contingence avec les effectifs des clusters pour chaque modalit� cible
	for (int idxCluster = 0; idxCluster < kmClusters->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, kmClusters->GetAt(idxCluster));
		assert(cluster->GetIndex() >= 0);
		assert(cluster->GetFrequency() == cluster->GetCount());

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));

		for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) { // boucle sur les modalit�s de la variable cible

			if (cluster->GetFrequency() == 0)
				fv->GetFrequencyVector()->SetAt(idxTargetvalue, 0);
			else {
				const int modalityFrequency = (int)((cluster->GetTargetProbs().GetAt(idxTargetvalue) * cluster->GetFrequency()) + 0.5); // le 0.5 sert a arrondir a l'entier le plus proche
				fv->GetFrequencyVector()->SetAt(idxTargetvalue, modalityFrequency);
			}
		}
	}

#ifdef DEBUG_POST_OPTIMIZATION

	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));
		assert(cluster->GetFrequency() == cluster->GetCount());
		assert(cluster->GetFrequency() == modalityFrequencyByCluster->GetSourceFrequencyAt(cluster->GetIndex()));
	}

#endif

	return modalityFrequencyByCluster;
}


NumericKeyDictionary* KMClustering::ComputeInstancesToClustersByAscDistance() const {

	NumericKeyDictionary* instancesToClustersByAscDistance = new NumericKeyDictionary; // cle = instance KWObject *, valeur = ObjectArray * contenant la liste de KMCluster *, tries par distance croissante � cette instance

	if (parameters->GetVerboseMode())
		AddSimpleMessage("Computing clusters sorted list, for each database instance...");


	// par commodite, on part de la liste existante associant chaque instance a son cluster
	assert(instancesToClusters != NULL);
	assert(instancesToClusters->GetCount() > 0);

	POSITION position = instancesToClusters->GetStartPosition();
	NUMERIC key;
	Object* oCurrent;

	while (position != NULL) {

		instancesToClusters->GetNextAssoc(position, key, oCurrent);
		KWObject* currentInstance = static_cast<KWObject *>(oCurrent);

		// recenser toutes les distances entre cette instance et les clusters
		ObjectArray* oaDistances = new ObjectArray;

		for (int i = 0; i < kmClusters->GetSize(); i++)
		{
			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
			double d = c->FindDistanceFromCentroid(currentInstance, c->GetModelingCentroidValues(), parameters->GetDistanceType());
			oaDistances->Add(new KMInstance(currentInstance, i, d));
		}
		oaDistances->SetCompareFunction(KMClusteringDistanceCompareAsc);
		oaDistances->Sort(); // tri par distances croissantes

		ObjectArray* oaClustersList = new ObjectArray;

		for (int i = 0; i < oaDistances->GetSize(); i++) {

			KMInstance* instance = cast(KMInstance*, oaDistances->GetAt(i));
			oaClustersList->Add(kmClusters->GetAt(instance->idCluster));
		}
		oaDistances->DeleteAll();
		delete oaDistances;

		instancesToClustersByAscDistance->SetAt(currentInstance, oaClustersList);
	}

	if (parameters->GetVerboseMode())
		AddSimpleMessage("Done.");

	return instancesToClustersByAscDistance;

}

boolean KMClustering::PostOptimizeVns(const ObjectArray* instances, const KWAttribute* targetAttribute) {

	assert(kmGlobalCluster != NULL);
	assert(kmGlobalCluster->GetFrequency() > 0);
	assert(kmClusters != NULL);
	assert(kmClusters->GetSize() > 0);
	assert(parameters->GetPostOptimizationVnsLevel() >= 0);

	boolean bOk = true;

	// calcul de KMax :

	const longint N = kmGlobalCluster->GetFrequency();
	const int KMaxZero = N / log(N); // N / Ln(N)
	const double maxLevel = round(log(N) + 0.5);
	if (parameters->GetPostOptimizationVnsLevel() > maxLevel) {
		AddWarning("Post-optimisation Vns level is set too high, will be reset to " + ALString(IntToString(maxLevel)));
		parameters->SetPostOptimizationVnsLevel(maxLevel);
	}

	double numerator = 0;
	for (int i = 0; i < parameters->GetPostOptimizationVnsLevel(); i++) {
		numerator += pow(2, i);
	}
	double denominator = 0;
	for (int i = 0; i < maxLevel; i++) {
		denominator += pow(2, i);
	}
	const longint KMax = ((numerator / denominator) * (N - KMaxZero)) + KMaxZero;

	const longint maxDegree = pow(2, parameters->GetPostOptimizationVnsLevel());
	int currentDegree = 1;

	if (parameters->GetVerboseMode()) {
		AddSimpleMessage("VNS post-optimization (KMax = " + ALString(IntToString(KMax)) + ", max degree = " + ALString(IntToString(maxDegree)));
		AddSimpleMessage("--------------------------------------------------------------------------------------------------------------------------------------------------------------");
		AddSimpleMessage("Degree\tInitial K\tFinal K\tChallenged clusters\tChallenged instances\tOverall best K\tEVA\tOverall best EVA");
	}

	KMClustering* bestClustering = Clone();
	double overAllBestEVA = GetClusteringQuality()->GetEVA();

	const int vnsLevelOldValue = parameters->GetPostOptimizationVnsLevel();// sauvegarder la valeur originale afin de la retablir a la fin de la procedure de VNS
	parameters->SetPostOptimizationVnsLevel(0);// eviter la recursivite infinie lorqu'on fera appel a la post-optimisation
	boolean verboseModeOldValue = parameters->GetVerboseMode();
	parameters->SetVerboseMode(false);

	while (currentDegree < maxDegree) {

		double challengedPercentage = (double)currentDegree / (double)maxDegree; // donne le % qui sera remis en question pour le clustering en cours de test

		const int nbChallengedClusters = round(challengedPercentage * (double)kmClusters->GetSize() + 0.5);

		// on tire au hasard le nombre de clusters remis en cause (challenged)
		IntVector idxChallengedClusters;

		while (idxChallengedClusters.GetSize() < nbChallengedClusters) {
			int idxCluster = RandomInt(kmClusters->GetSize() - 1);
			// verifier que ce cluster n'a pas deja ete tire au sort, et dans le cas contraire, le referencer
			boolean found = false;
			for (int i = 0; i < idxChallengedClusters.GetSize(); i++) {
				if (idxChallengedClusters.GetAt(i) == idxCluster) {
					// deja tire au sort, recommencer
					found = true;
					break;
				}
			}
			if (not found)
				idxChallengedClusters.Add(idxCluster);
		}

#ifdef DEBUG_POST_OPTIMIZATION_VNS
		cout << endl << endl << "currentDegree = " << currentDegree << endl;
		cout << "challengedPercentage = " << challengedPercentage << endl;
		cout << "nbChallengedClusters = " << nbChallengedClusters << endl;
		cout << "Randomly chosen clusters : " << endl;
		for (int i = 0; i < idxChallengedClusters.GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, clusters->GetAt(i));
			cout << c->GetIndex() << " ";
		}
		cout << endl;
#endif

		// copier les instances des clusters concernes dans un tableau de travail
		ObjectArray* oaChallengedClustersInstances = new ObjectArray;
		for (int i = 0; i < idxChallengedClusters.GetSize(); i++) {
			const int idxChallenged = idxChallengedClusters.GetAt(i);
			KMCluster* removedCluster = cast(KMCluster*, kmClusters->GetAt(idxChallenged));
			NUMERIC key;
			Object* oCurrent;
			POSITION position = removedCluster->GetStartPosition();
			while (position != NULL) {
				removedCluster->GetNextAssoc(position, key, oCurrent);
				KWObject* currentInstance = static_cast<KWObject *>(oCurrent);
				oaChallengedClustersInstances->Add(currentInstance);
			}
		}

		oaChallengedClustersInstances->Shuffle(); // melanger aleatoirement
		int newClustersNumber = round((challengedPercentage * (double)oaChallengedClustersInstances->GetSize()) + 0.5);
		if (newClustersNumber >= KMax)
			newClustersNumber = KMax;

		// supprimer les clusters destines a etre remplaces par les nouveaux
		for (int i = 0; i < idxChallengedClusters.GetSize(); i++) {
			for (int j = 0; j < kmClusters->GetSize(); j++) {
				KMCluster* c = cast(KMCluster*, kmClusters->GetAt(j));
				if (c->GetIndex() == idxChallengedClusters.GetAt(i)) {
					DeleteClusterAt(j);
					break;
				}
			}
		}

		// creer les nouveaux clusters, a partir des premieres instances du tableau de travail
		for (int i = 0; i < newClustersNumber; i++) {
			KWObject* currentInstance = cast(KWObject*, oaChallengedClustersInstances->GetAt(i));
			KMCluster* newCluster = new KMCluster(parameters);
			newCluster->InitializeModelingCentroidValues(currentInstance);
			newCluster->SetInitialCentroidValues(newCluster->GetModelingCentroidValues());
			newCluster->SetLabel("VNS_degree_" + ALString(IntToString(currentDegree)) + "_number_" + ALString(IntToString(i)));
			kmClusters->Add(newCluster);
		}

		AddInstancesToClusters(instances); // reaffecter toutes les instances

		int emptyClusters = ManageEmptyClusters(false);// car il est possible qu'il y ait des clusters vides, dans le cas ou on a utilise des instances qui ont des valeurs identiques, comme nouveaux centres de clusters

		if (emptyClusters > 0)
			AddInstancesToClusters(instances); // reaffecter toutes les instances, uniquement sur les clusters qui n'etaient pas vides

		for (int i = 0; i < kmClusters->GetSize(); i++) {
			KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
			assert(c->GetCount() > 0);
			c->ComputeDistanceSum(KMParameters::L2Norm);
			c->ComputeDistanceSum(KMParameters::CosineNorm);
			c->ComputeDistanceSum(KMParameters::L1Norm);
			c->SetFrequency(c->GetCount());
			c->ComputeInstanceNearestToCentroid(parameters->GetDistanceType());
			c->ComputeInertyIntra(parameters->GetDistanceType());
			c->SetStatisticsUpToDate(true);
		}

		const int initialKValue = kmClusters->GetSize();

		bOk = PostOptimize(instances, targetAttribute); // reffectuer une post-optimisation complete a partir des nouveaux clusters

		AddSimpleMessage(
			KMGetDisplayString(currentDegree) +
			KMGetDisplayString(initialKValue) +
			KMGetDisplayString(kmClusters->GetSize()) +
			KMGetDisplayString(nbChallengedClusters) +
			KMGetDisplayString(oaChallengedClustersInstances->GetSize()) +
			KMGetDisplayString(bestClustering->GetClusters()->GetSize()) +
			KMGetDisplayString(GetClusteringQuality()->GetEVA()) +
			KMGetDisplayString(overAllBestEVA)
		);

		if (GetClusteringQuality()->GetEVA() > overAllBestEVA) {
			currentDegree = 1;
			bestClustering->CopyFrom(this);
			overAllBestEVA = GetClusteringQuality()->GetEVA();
		}
		else
			currentDegree++;

		delete oaChallengedClustersInstances;

		if (not bOk)
			break;
	}

	CopyFrom(bestClustering);

	AddInstancesToClusters(instances);

	// mise a jour des stats, necessaire apres reaffectation des instances

	for (int i = 0; i < kmClusters->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, GetClusters()->GetAt(i));
		c->ComputeDistanceSum(KMParameters::L2Norm);
		c->ComputeDistanceSum(KMParameters::CosineNorm);
		c->ComputeDistanceSum(KMParameters::L1Norm);
		c->SetFrequency(c->GetCount());
		c->ComputeInstanceNearestToCentroid(parameters->GetDistanceType());
		c->ComputeInertyIntra(parameters->GetDistanceType());
		c->SetStatisticsUpToDate(true);
	}

	ComputeTrainingTargetProbs(targetAttribute);
	GetClusteringQuality()->ComputeEVA(kmGlobalCluster, oaTargetAttributeValues.GetSize());

	AddSimpleMessage("VNS post-optimization is done. EVA is now " + ALString(DoubleToString(GetClusteringQuality()->GetEVA())) + ", optimal K value is now " + ALString(IntToString(kmClusters->GetSize())) + ".");

	parameters->SetPostOptimizationVnsLevel(vnsLevelOldValue);
	parameters->SetVerboseMode(verboseModeOldValue);

	delete bestClustering;

	return bOk;

}

void KMClustering::ComputeClusteringLevels(KWClass* kwcModeling, ObjectArray* attributesStats, ObjectArray* clusters) {

	assert(clusters != NULL);
	assert(clusters->GetSize() > 0);

	InitializeClusteringLevelFrequencyTables(clusters->GetSize());

	// parcourir les instances presentes dans les clusters, et mettre a jour les tables de contingences necessaires au calcul du level

	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++)
	{
		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));

		assert(cluster->GetCount() > 0); // les instances doivent etre presentes dans les clusters (et pas seulement les centroides)

		POSITION position = cluster->GetStartPosition();
		NUMERIC key;
		Object* oCurrent;

		while (position != NULL) {

			cluster->GetNextAssoc(position, key, oCurrent);
			KWObject* instance = static_cast<KWObject *>(oCurrent);
			UpdateClusteringLevelFrequencyTables(instance, idxCluster);
		}
	}

	// a partir des tables de contingence obtenues, calculer les levels de clustering, et les stocker dans une structure pour utilisation
	// lors de l'ecriture du rapport de modelisation
	// Le level est d�fini par Level = 1 � (Cost/NullCost), sachant que NullCost est calcul� a partir d'une table de frequence "nulle"

	// construction d'une table "nulle", ayant une seule ligne avec les effectifs totaux par colonnes (correspondant aux clusters).
	KWFrequencyTable nullTable;
	nullTable.SetFrequencyVectorNumber(1);
	KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, nullTable.GetFrequencyVectorAt(0));
	fv->GetFrequencyVector()->SetSize(clusters->GetSize());
	fv->SetModalityNumber(clusters->GetSize());

	for (int i = 0; i < clusters->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, clusters->GetAt(i));
		fv->GetFrequencyVector()->SetAt(i, c->GetFrequency());
	}

	KWDiscretizerMODL discretizer;
	const Continuous discretizerNullCost = discretizer.ComputeDiscretizationCost(&nullTable);

	// mise a jour pour modalit�s group�es :
	POSITION position = odGroupedModalitiesFrequencyTables.GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		odGroupedModalitiesFrequencyTables.GetNextAssoc(position, key, oCurrent);

		KWFrequencyTable* table = cast(KWFrequencyTable*, oCurrent);
		assert(table != NULL);

		const KWAttribute* attribute = kwcModeling->LookupAttribute(key);
		assert(attribute != NULL);

		KWAttribute* cellIndexAttribute = kwcModeling->GetAttributeAtLoadIndex(attribute->GetLoadIndex());
		assert(cellIndexAttribute != NULL);

		KWAttribute* nativeAttribute = cellIndexAttribute->GetDerivationRule()->GetSecondOperand()->GetOriginAttribute();

		Symbol sNativeName(nativeAttribute->GetName());
		svNativeAttributesNames.Add(sNativeName);

		// retouver les stats correspondant a l'attribut
		KWAttributeStats* attributeStats = NULL;
		for (int i = 0; i < attributesStats->GetSize(); i++) {
			attributeStats = cast(KWAttributeStats*, attributesStats->GetAt(i));
			if (attributeStats->GetAttributeName() == nativeAttribute->GetName())
				break;
		}
		assert(attributeStats != NULL);

		table->SetGranularity(attributeStats->GetPreparedDataGridStats()->GetGranularity());

		if (nativeAttribute->GetType() == KWType::Continuous) {

			Continuous attributeCost = discretizer.ComputeDiscretizationCost(table);
			ContinuousObject* clusteringLevel = new ContinuousObject;

			if (discretizerNullCost == 0)
				clusteringLevel->SetContinuous(0);
			else {
				clusteringLevel->SetContinuous(1 - (attributeCost / discretizerNullCost));
				if (clusteringLevel->GetContinuous() < 0)
					clusteringLevel->SetContinuous(0);
			}

			nkdClusteringLevels.SetAt(sNativeName.GetNumericKey(), clusteringLevel);

		}
		else {

			// cas des variables categorielles
			KWGrouperMODL grouper;

			const Continuous grouperNullCost = grouper.ComputeGroupingCost(&nullTable, attributeStats->GetDescriptiveStats()->GetValueNumber());

			Continuous attributeCost = grouper.ComputeGroupingCost(table, attributeStats->GetDescriptiveStats()->GetValueNumber());
			ContinuousObject* clusteringLevel = new ContinuousObject;

			if (grouperNullCost == 0)
				clusteringLevel->SetContinuous(0);
			else {
				clusteringLevel->SetContinuous(1 - (attributeCost / grouperNullCost));
				if (clusteringLevel->GetContinuous() < 0)
					clusteringLevel->SetContinuous(0);
			}
			nkdClusteringLevels.SetAt(sNativeName.GetNumericKey(), clusteringLevel);
		}
	}
}


void KMClustering::ComputeClusteringLevels(KWDatabase* instances, KWClass* kwcModeling, ObjectArray* attributesStats, ObjectArray* clusters) {

	assert(instances != NULL);
	assert(instances->GetSampleEstimatedObjectNumber() > 0);
	assert(clusters != NULL);
	assert(clusters->GetSize() > 0);

	const double dMinNecessaryMemory = 16 * 1024 * 1024;
	ALString sTmp;

	InitializeClusteringLevelFrequencyTables(clusters->GetSize());

	// Ouverture de la base en lecture
	boolean bOk = instances->OpenForRead();

	// Lecture d'objets dans la base
	if (bOk)
	{
		Global::ActivateErrorFlowControl();

		int nObject = 0;

		while (not instances->IsEnd())
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
			KWObject* kwoObject = instances->Read();

			if (kwoObject != NULL)
			{
				if (parameters->HasMissingKMeanValue(kwoObject)) {
					delete kwoObject;
					continue;
				}

				KMCluster* cluster = FindNearestCluster(kwoObject);
				UpdateClusteringLevelFrequencyTables(kwoObject, cluster->GetIndex());
				delete kwoObject;
			}
		}
	}

	Global::DesactivateErrorFlowControl();

	instances->Close();

	// a partir des tables de contingence obtenues, calculer les levels de clustering, et les stocker dans une structure pour utilisation
	// lors de l'ecriture du rapport de modelisation
	// Le level est d�fini par Level = 1 � (Cost/NullCost), sachant que NullCost est calcul� a partir d'une table de contingence "nulle"

	// construction d'une table "nulle", ayant une seule ligne avec les effectifs totaux par colonne.
	KWFrequencyTable nullTable;
	nullTable.SetFrequencyVectorNumber(1);
	KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, nullTable.GetFrequencyVectorAt(0));
	fv->GetFrequencyVector()->SetSize(clusters->GetSize());

	for (int i = 0; i < clusters->GetSize(); i++) {
		KMCluster* c = cast(KMCluster*, clusters->GetAt(i));
		fv->GetFrequencyVector()->SetAt(i, c->GetFrequency());
	}

	KWDiscretizerMODL discretizer;
	const Continuous discretizerNullCost = discretizer.ComputeDiscretizationCost(&nullTable);

	POSITION position = odGroupedModalitiesFrequencyTables.GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		odGroupedModalitiesFrequencyTables.GetNextAssoc(position, key, oCurrent);

		KWFrequencyTable* table = cast(KWFrequencyTable*, oCurrent);
		assert(table != NULL);

		const KWAttribute* attribute = kwcModeling->LookupAttribute(key);
		assert(attribute != NULL);

		KWAttribute* cellIndexAttribute = kwcModeling->GetAttributeAtLoadIndex(attribute->GetLoadIndex());
		assert(cellIndexAttribute != NULL);

		KWAttribute* nativeAttribute = cellIndexAttribute->GetDerivationRule()->GetSecondOperand()->GetOriginAttribute();

		Symbol sNativeName(nativeAttribute->GetName());
		svNativeAttributesNames.Add(sNativeName);

		// retouver les stats correspondant a l'attribut
		KWAttributeStats* attributeStats = NULL;
		for (int i = 0; i < attributesStats->GetSize(); i++) {
			attributeStats = cast(KWAttributeStats*, attributesStats->GetAt(i));
			if (attributeStats->GetAttributeName() == nativeAttribute->GetName())
				break;
		}
		assert(attributeStats != NULL);

		table->SetGranularity(attributeStats->GetPreparedDataGridStats()->GetGranularity());

		if (nativeAttribute->GetType() == KWType::Continuous) {

			Continuous attributeCost = discretizer.ComputeDiscretizationCost(table);
			ContinuousObject* clusteringLevel = new ContinuousObject;

			if (discretizerNullCost == 0)
				clusteringLevel->SetContinuous(0);
			else {
				clusteringLevel->SetContinuous(1 - (attributeCost / discretizerNullCost));
				if (clusteringLevel->GetContinuous() < 0)
					clusteringLevel->SetContinuous(0);
			}

			nkdClusteringLevels.SetAt(sNativeName.GetNumericKey(), clusteringLevel);

		}
		else {

			// cas des variables categorielles
			KWGrouperMODL grouper;

			const Continuous grouperNullCost = grouper.ComputeGroupingCost(&nullTable, attributeStats->GetDescriptiveStats()->GetValueNumber());

			Continuous attributeCost = grouper.ComputeGroupingCost(table, attributeStats->GetDescriptiveStats()->GetValueNumber());
			ContinuousObject* clusteringLevel = new ContinuousObject;

			if (grouperNullCost == 0)
				clusteringLevel->SetContinuous(0);
			else {
				clusteringLevel->SetContinuous(1 - (attributeCost / grouperNullCost));
				if (clusteringLevel->GetContinuous() < 0)
					clusteringLevel->SetContinuous(0);
			}
			nkdClusteringLevels.SetAt(sNativeName.GetNumericKey(), clusteringLevel);
		}
	}
}



void KMClustering::UpdateClusteringLevelFrequencyTables(const KWObject* kwoObject, const int idCluster) {

	// mise a jour pour modalit�s group�es (calcul du level de clustering ) :

	POSITION position = odGroupedModalitiesFrequencyTables.GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		odGroupedModalitiesFrequencyTables.GetNextAssoc(position, key, oCurrent);

		const KWAttribute* attribute = kwoObject->GetClass()->LookupAttribute(key);
		assert(attribute != NULL);

		KWFrequencyTable* table = cast(KWFrequencyTable*, oCurrent);

		if (table != NULL) {

			const Continuous value = kwoObject->GetContinuousValueAt(attribute->GetLoadIndex());
			const int modalityIndex = (int)value - 1;
			assert(modalityIndex != -1 and modalityIndex < table->GetFrequencyVectorNumber());
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(modalityIndex));
			fv->GetFrequencyVector()->SetAt(idCluster, fv->GetFrequencyVector()->GetAt(idCluster) + 1);
		}
	}
}


void KMClustering::InitializeClusteringLevelFrequencyTables(const int nbClusters) {

	// initialiser les tables de contingences servant au calcul du level de clustering :
	// chaque poste pointe sur un objet KWFrequencyTable, correspondant aux intervalles d'un attribut)

	assert(attributesPartitioningManager != NULL);
	odGroupedModalitiesFrequencyTables.DeleteAll();

	POSITION position = attributesPartitioningManager->GetPartitions().GetStartPosition();
	ALString key;
	Object* oCurrent;

	while (position != NULL) {

		attributesPartitioningManager->GetPartitions().GetNextAssoc(position, key, oCurrent);

		ObjectArray* oaModalities = cast(ObjectArray*, oCurrent);
		KWFrequencyTable* table = new KWFrequencyTable;
		table->SetFrequencyVectorNumber(oaModalities->GetSize());
		for (int i = 0; i < table->GetFrequencyVectorNumber(); i++) {
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, table->GetFrequencyVectorAt(i));
			fv->GetFrequencyVector()->SetSize(nbClusters);
			fv->SetModalityNumber(nbClusters);
		}
		odGroupedModalitiesFrequencyTables.SetAt(key, table);
	}
}


void KMClustering::DeleteClusterAt(int idx) {

	assert(kmClusters != NULL);
	assert(idx < kmClusters->GetSize() and idx >= 0);
	KMCluster* c = cast(KMCluster*, kmClusters->GetAt(idx));
	kmClusters->RemoveAt(idx);
	delete c;
}

const double KMClustering::GetClustersDistanceSum(KMParameters::DistanceType d) const
{
	return cvClustersDistancesSum.GetAt(d);
}


const ObjectArray& KMClustering::GetTargetAttributeValues() const {

	return oaTargetAttributeValues;

}

const int  KMClustering::GetDroppedClustersNumber() const {
	return iDroppedClustersNumber;
}

const longint KMClustering::GetInstancesWithMissingValues() const {
	return clusteringInitializer->GetInstancesWithMissingValues();
}


void KMClustering::IncrementInstancesWithMissingValuesNumber() {
	clusteringInitializer->IncrementInstancesWithMissingValuesNumber();
}

void KMClustering::ResetInstancesWithMissingValuesNumber() {
	clusteringInitializer->ResetInstancesWithMissingValuesNumber();
}


//////////////////////////////////////////////////////////
// Classe PLShared_Clustering
/// Serialisation de la classe KMClustering

PLShared_Clustering::PLShared_Clustering()
{
}

PLShared_Clustering::~PLShared_Clustering()
{
}

void PLShared_Clustering::SetClustering(KMClustering* c)
{
	require(c != NULL);
	SetObject(c);
}

KMClustering* PLShared_Clustering::GetClustering()
{
	return cast(KMClustering*, GetObject());
}

void PLShared_Clustering::SerializeObject(PLSerializer* serializer,
	const Object* object) const
{
	KMClustering* clustering;
	PLShared_Cluster sharedCluster;

	require(serializer != NULL);
	require(serializer->IsOpenForWrite());
	require(object != NULL);

	clustering = cast(KMClustering*, object);
	sharedCluster.SerializeObject(serializer, clustering->kmGlobalCluster);
}

void PLShared_Clustering::DeserializeObject(PLSerializer* serializer, Object* object) const
{
	KMClustering* clustering;
	PLShared_Cluster sharedCluster;

	require(serializer->IsOpenForRead());

	clustering = cast(KMClustering*, object);

	// Deserialization des attributs
	sharedCluster.DeserializeObject(serializer, clustering->kmGlobalCluster);
}


Object* PLShared_Clustering::Create() const
{
	return new KMClustering(NULL);
}

////////////////////////////////////////////////////////////////

// fonction de comparaison pour tri de tableau, tri par distances decroissantes
int KMClusteringDistanceCompareDesc(const void* elem1, const void* elem2) {

	KMInstance* i1 = (KMInstance*) * (Object**)elem1;
	KMInstance* i2 = (KMInstance*) * (Object**)elem2;

	if (i1->distance > i2->distance)
		return -1;
	else if (i1->distance < i2->distance)
		return 1;
	else
		return 0;

}
// fonction de comparaison pour tri de tableau, tri par distances croissantes
int KMClusteringDistanceCompareAsc(const void* elem1, const void* elem2) {

	KMInstance* i1 = (KMInstance*) * (Object**)elem1;
	KMInstance* i2 = (KMInstance*) * (Object**)elem2;

	if (i1->distance < i2->distance)
		return -1;
	else if (i1->distance > i2->distance)
		return 1;
	else
		return 0;

}

const ALString KMGetDisplayString(const double d) {

	ALString s(DoubleToString(d));

	return s + (s.GetLength() < 12 ? "\t\t" : "\t");
}

const ALString KMGetDisplayString(const int d) {

	ALString s(IntToString(d));

	return s + "\t";
}








