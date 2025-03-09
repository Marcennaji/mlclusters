// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#ifndef KMCLUSTERING_INITIALIZER_H
#define KMCLUSTERING_INITIALIZER_H


#include "KWObject.h"
#include "KMParameters.h"
#include "KMClustering.h"
#include "KMCluster.h"

/////////////////////////////////////////////////
/// Classe de gestion de l'initialisation d'un clustering

class KMClusteringInitializer : public Object {

public:

	KMClusteringInitializer();
	KMClusteringInitializer(KMClustering* _clustering);
	~KMClusteringInitializer();

	/** initialise les centroides de clusters a partir d'instances tirees au hasard. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeRandomCentroids(const ObjectArray* instances);

	/** initialise les centroides de clusters selon l'algorithme kmean++. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeKMeanPlusPlusCentroids(const ObjectArray* instances);

	/** initialiser les centroides de  clusters selon l'algorithme kmean++R. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeKMeanPlusPlusRCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** initialiser les centroides de  clusters selon l'algorithme rocchio + split. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeRocchioThenSplitCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** initialiser les centroides de  clusters selon l'algorithme Min-Max.
	Principe : pour chaque instance, on calcule quelle est la distance a son centre le plus proche. Puis on choisit comme nouveau centre, l'instance
	dont la distance precedemment calculee est la plus grande. Si methode deterministe, alors le premier centre choisi est le centre de gravite des donnees.
	Sinon, le premier centre est choisi au hasard. Retourne True si l'initialisation a reussi, sinon False.  */
	boolean InitializeMinMaxCentroids(const ObjectArray* instances, const boolean isDeterministic);

	/** initialiser les centroides de  clusters selon l'algorithme PCA Part
	Principe :
		- On selectionne le cluster qui a la plus grande variance intra
		- puis on divise ce cluster en 2, en fonction de la variable qui a la plus grande variance dans ce cluster
	. Retourne True si l'initialisation a reussi, sinon False.
	*/
	boolean InitializeVariancePartitioningCentroids(const ObjectArray* instances);

	/** Initialisation selon l'algorithme de decomposition des classes. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeClassDecompositionCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** initialiser les centroides de  clusters selon l'algorithme de bisecting. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeBisectingCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** retourne le nombre d'instances qui ont au moins une valeur manquante dans leurs attributs */
	const longint GetInstancesWithMissingValues() const;

	/** incrementer de 1 le nombre d'instances avec valeurs manquantes */
	void IncrementInstancesWithMissingValuesNumber();

	/** remettre a 0 le nombre d'instances manquantes */
	void ResetInstancesWithMissingValuesNumber();

	void CopyFrom(const KMClusteringInitializer* aSource);

	/// Comptage des modalites cibles
	class TargetModalityCount : public Object {
	public:
		/** valeur de la modalite cible */
		Symbol sModality;
		/** compte associe a la modalite cible */
		int iCount;
	};

protected:

	/** creer les "C" clusters initiaux (correspondant aux modalites cibles) */
	void CreateTargetModalitiesClusters(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** initialiser les centres suivants, en KMean++ ou KMean++R */
	void InitializeKMeanPlusPlusNextCenters(const ObjectArray* instances, const int nbCenters);

	/** initialiser les centres suivants, en MinMax */
	void InitializeMinMaxNextCenters(const ObjectArray* instances);

	/**  creer le cluster correspondant a une  modalite cible */
	void CreateClusterForTargetModality(const ALString modalityValue, const ObjectArray* instances, const KWAttribute* targetAttribute);

	/* retourne un tableau d'objets TargetModalityCount, tries par frequences de modalites decroissantes */
	ObjectArray* ComputeTargetModalitiesCount(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** initialisation par decomposition de classe, a partir d'un cluster. Retourne True si l'initialisation a reussi, sinon False. */
	boolean ClassDecompositionCreateClustersFrom(const KMCluster* originCluster, const int nbClustersToCreate);

	/** initialisation des centroides bisecting en mode non supervise. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeBisectingCentroidsUnsupervised(const ObjectArray* instances);

	/** initialisation des centroides bisecting en mode supervise. Retourne True si l'initialisation a reussi, sinon False. */
	boolean InitializeBisectingCentroidsSupervised(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** convergence bisecting. Retourne True si la convergence a reussi, sinon False. */
	boolean DoBisecting(KMParameters& bisectingParameters, const KWAttribute* targetAttribute);

	/** execution des replicates bisecting et selection du meilleur replicate */
	KMClustering* BisectingComputeAllReplicates(ObjectArray* instances, KMParameters&, const KWAttribute* targetAttribute, const ALString sLabel);

	/** effectuer une convergence a partir du cluster de modalite recu en parametre. Retourne True si la convergence a reussi, sinon False. */
	boolean DoClassDecomposition(KMParameters& bisectingParameters, const KMCluster* modalityCluster);

	/** Initialisation random en mode parallele, si l'option "Parallel mode" a ete cochee (mode expert) */
	boolean InitializeRandomCentroidsParallelized(const ObjectArray* instances);

	/** Initialisation random en mode sequentiel, si l'option "Parallel mode" n'a pas ete cochee (mode expert) */
	boolean InitializeRandomCentroidsNotParallelized(const ObjectArray* instances);

	/** Creation d'une database a partir d'une liste d'instances (liste de KWObject* ) */
	KWDatabase* CreateDatabaseFromInstances(const ObjectArray* instances);

	//////////////// attributs ///////////////////

	/** nombre d'instances de la base qui ont au moins une valeur manquante parmi leurs attributs */
	longint lInstancesWithMissingValues;

	/** clustering dont depend cette instance de KMClusteringInitializer (passee au constructeur) */
	KMClustering* clustering;

};


inline void KMClusteringInitializer::IncrementInstancesWithMissingValuesNumber() {
	lInstancesWithMissingValues++;
}

inline void KMClusteringInitializer::ResetInstancesWithMissingValuesNumber() {
	lInstancesWithMissingValues = 0;
}

inline const longint KMClusteringInitializer::GetInstancesWithMissingValues() const {
	return lInstancesWithMissingValues;
}

#endif



