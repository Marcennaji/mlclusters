// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWObject.h"
#include "KWClassStats.h"
#include "KMCluster.h"
#include "KMParameters.h"
#include "KMAttributesPartitioningManager.h"

// #define DEBUG_POST_OPTIMIZATION
// #define DEBUG_POST_OPTIMIZATION_VNS

class KMClusteringInitializer;
class KMClusteringQuality;

///////////////////////////////////////////////////////////////////////////
///  Classe representant un clustering (ensemble de clusters), son parametrage, et les services associes, qui sont principalement : l' execution des "replicates", le calcul des indicateurs de qualite associes a chaque resultat de replicate, la selection du meilleur replicate en fonction du critere choisi (EVA, etc), la post optimisation du resultat du meilleur replicate

class KMClustering : public Object
{
public:

	KMClustering(KMParameters*);
	~KMClustering(void);

	/** tableau d'objets KMCluster */
	ObjectArray* GetClusters() const;

	KMCluster* GetCluster(const int clusterId) const;

	void DeleteClusterAt(int idx);

	/** cluster contenant toutes les instances, et stats associees */
	KMCluster* GetGlobalCluster() const;

	/** (re)creer un cluster global (utilisé en évaluation test ou train) */
	KMCluster* CreateGlobalCluster();

	/** calcul K-Means : boucle principale d'un traitement de clustering */
	bool ComputeReplicate(ObjectArray* instances, const KWAttribute* targetAttribute);

	/** retourne le cluster dont le centre est le plus proche de l'objet passé en parametre */
	KMCluster* FindNearestCluster(KWObject*);

	/** parametrage du clustering kmean */
	const KMParameters* GetParameters() const;

	/** retourne le nombre d'iterations effectuees lors d'un clustering */
	const int  GetIterationsDone() const;

	/** retourne le nombre final de clusters vides supprimes lors d'un clustering */
	const int  GetDroppedClustersNumber() const;

	/** initialise la liste des modalités de la variable cible (mode supervisé) */
	void SetTargetAttributeValues(const ObjectArray&);

	/** initialise le cluster global, a partir d'un autre resultat */
	void SetGlobalCluster(KMCluster*);

	/** retourne la liste des modalités de la variable cible (mode supervisé) */
	const ObjectArray& GetTargetAttributeValues() const;

	/** matrice de confusion du clustering */
	const KWFrequencyTable* GetConfusionMatrix() const;

	KMClustering* Clone();

	void CopyFrom(const KMClustering* aSource);

	/** initialiser la valeur du pourcentage devant etre lu, de la base de donnees devant etre lu (sert uniquement en cas de memoire insuffisante) */
	void SetUsedSampleNumberPercentage(const double sampleNumberPercentage);

	/** retourne la valeur du pourcentage devant etre lu, de la base de donnees devant etre lu (sert uniquement en cas de memoire insuffisante) */
	const double GetUsedSampleNumberPercentage() const;

	/** somme totale des distances des instances, par rapport à leurs clusters respectifs */
	const double GetClustersDistanceSum(KMParameters::DistanceType) const;

	/** distance moyenne des instances de clusters à leur centre */
	const Continuous  GetMeanDistance() const;

	/** distance entre deux instances de clusters, tous attributs confondus */
	static Continuous GetDistanceBetween(const ContinuousVector& v1, const ContinuousVector& v2, const KMParameters::DistanceType, const KWLoadIndexVector& kmeanAttributesLoadIndexes);

	/** distance entre deux instances de clusters, pour un attribut particulier */
	static Continuous GetDistanceBetweenForAttribute(const int attributeLoadIndex, const ContinuousVector& v1, const ContinuousVector& v2,
		const KMParameters::DistanceType);

	/** mesure de similarite entre deux instances de clusters, tous attributs confondus */
	static Continuous GetSimilarityBetween(const ContinuousVector& v1, const ContinuousVector& v2,
		const ALString& targetModality1, const ALString& targetModality2, const KMParameters* parameters);

	/** calculer les distances entre les différents centres des clusters, afin de produire une matrice des distances,
	dont l'utilisation permettra une optimisation des performances */
	void ComputeClustersCentersDistances(const boolean useEvaluationCentroids = false);

	// mise a jour de la somme des distances des clusters, pour toutes les normes
	void UpdateGlobalDistancesSum();

	/** maj, lors de l'evaluation, de la matrice de confusion "classes majoritaires / classes reelles" */
	void UpdateConfusionMatrix(const Symbol& sPredictedTarget, const Symbol& sActualTarget);

	/** acces a la classe gerant les indicateurs de la qualite d'un clustering */
	KMClusteringQuality* GetClusteringQuality() const;

	/** acces a la classe gerant les intervalles et modalites (groupees ou non) des attributs */
	KMAttributesPartitioningManager* GetAttributesPartitioningManager() const;

	/** matrice des distances entre clusters */
	Continuous** GetClustersCentersDistances() const;

	/** dictionnaire des instances et de leurs clusters associes */
	NumericKeyDictionary* GetInstancesToClusters() const;

	/** retourne le nombre d'instances qui ont au moins une valeur manquante dans leurs attributs */
	const longint GetInstancesWithMissingValues() const;

	/** incrementer de 1 le nombre d'instances avec valeurs manquantes */
	void IncrementInstancesWithMissingValuesNumber();

	/** remettre a 0 le nombre d'instances manquantes */
	void ResetInstancesWithMissingValuesNumber();

	/** ajouter des instances a des clusters, en fonction de leurs centres deja calcules */
	void AddInstancesToClusters(const ObjectArray* instances);

	void AddTargetAttributeValueIfNotExists(const KWAttribute* targetAttribute, const KWObject* kwoObject);

	/** supprimer certains centres de clusters, si cela a pour effet d'ameliorer l'EVA du clustering */
	boolean PostOptimize(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** rechercher le cluster dont la suppression produirait le meilleur EVA, pour un clustering donne */
	KMCluster* PostOptimizationSearchClusterToRemove(const KMClustering* currentClustering,
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
	);


	/** post-optimisation a base de recherche a voisinage variable (Variable Neighborhood Search) */
	boolean PostOptimizeVns(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** construction des tables de contingence servant au calcul des levels de clustering, a l'issue de l'apprentissage, et a partir des instances contenues dans les clusters */
	void ComputeClusteringLevels(KWClass* modelingClass, ObjectArray* attributesStats, ObjectArray* clusters);

	/** construction "a la volee" (en parcourant sequentiellement la base) des tables de contingence servant au calcul des levels de clustering, a l'issue de l'apprentissage */
	void ComputeClusteringLevels(KWDatabase*, KWClass* modelingClass, ObjectArray* attributesStats, ObjectArray* clusters);

	/** levels de clustering (clé = nom de l'attribut natif, valeur = level) */
	NumericKeyDictionary& GetClusteringLevelsDictionary() const;

protected:

	/** initialise les clusters, avant de proceder aux iterations */
	boolean InitializeClusters(const KMParameters::ClustersCentersInitMethod, const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** deroulement des iterations d'1 replicate, jusqu'a convergence */
	boolean DoClusteringIterations(const ObjectArray* instances, const longint maxInstances);

	/** retourne le cluster dont le centre est le plus proche de l'objet passé en parametre (norme L1) */
	KMCluster* FindNearestClusterL1(KWObject*);

	/** retourne le cluster dont le centre est le plus proche de l'objet passé en parametre (norme L2) */
	KMCluster* FindNearestClusterL2(KWObject*);

	/** retourne le cluster dont le centre est le plus proche de l'objet passé en parametre (norme Cosinus) */
	KMCluster* FindNearestClusterCosinus(KWObject*);

	/** construire un cluster 'fictif' contenant toutes les instances, et calculer les statistiques correspondantes */
	void ComputeGlobalClusterStatistics(ObjectArray* instances);

	/** gestion de la convergence en cours d'iterations : determine s'il faut continuer le clustering */
	bool ManageConvergence(const int movements, const int iterationsDone,
		const double distancesSum, const double newDistancesSum,
		const longint instancesCount,
		double& minDistanceSum, int& epsilonIterations);

	/** detection et gestion des clusters devenus vides apres une iteration */
	int ManageEmptyClusters(const boolean continueClustering);

	/** calcul de certaines stats et indicateurs, a la fin de chaque replicate */
	void FinalizeReplicateComputing(bool recomputeCentroids);

	bool UpdateProgressionBar(const longint instancesNumber, const int iterationsDone, const int movements);

	/** calcul des probabilites correspondant aux modalites de la variable cible (mode supervisé) */
	void ComputeTrainingTargetProbs(const KWAttribute* targetAttribute);

	/** en apprentissage, calcul de la matrice de confusion "classes majoritaires / classes reelles" */
	void ComputeTrainingConfusionMatrix(const KWAttribute* targetAttribute);

	/** determiner quelles sont les modalites de la variable cible (mode supervisé) */
	void ReadTargetAttributeValues(const ObjectArray* instances, const KWAttribute* targetAttribute);

	/** sauvegarder les meilleurs clusters observés */
	void CloneBestClusters();

	/** initialiser les tables de contingence permettant de caculer un level de clustering */
	void InitializeClusteringLevelFrequencyTables(const int nbClusters);

	/** mettre a jour les tables de contingence permettant de caculer un level de clustering */
	void UpdateClusteringLevelFrequencyTables(const KWObject* kwoObject, const int idCluster);

	/** post optimisation d'un clustering : creation d'une structure faisant correspondre, pour chaque instance de la base, la liste des clusters tries par ordre de distance croissante
	(cle = KWObject *, valeur =  ObjectArray contenant des KMClusters *)   */
	NumericKeyDictionary* ComputeInstancesToClustersByAscDistance() const;

	/** post optimisation d'un clustering : lignes = clusters, colonnes = modalites cibles, valeurs = frequences pour 1 cluster et 1 modalite cible donnes  */
	KWFrequencyTable* CreateModalitiesFrequenciesByClusters(const ObjectArray* clusters);

	/** post optimisation d'un clustering : dans le cas ou on supprimerait un cluster, calculer les frequences qui seraient produites si on
	affectait les instances du cluster supprime, a leurs plus proches clusters suivants */
	boolean PostOptimizationUpdateFrequencies(const KMCluster* removedCluster, const NumericKeyDictionary* instancesToClustersByAscDistance,
		const KWAttribute* targetAttribute, const ObjectArray& oaTargetAttributeValues,
		KWFrequencyTable& frequenciesAfterClusterRemoval, NumericKeyDictionary* removedInstancesNewClusters);

	/** post optimisation d'un clustering : deplacer les instances d'un cluster supprime, a leurs clusters les plus proches */
	void PostOptimizationMoveInstancesToNextClusters(const NumericKeyDictionary* removedInstancesNewClusters);

	// ==============================================  attributs de la classe ===============================================

	/** contient des objets KMCluster * , et représente l'etat courant des clusters, au fil des iterations */
	ObjectArray* kmClusters;

	/** contient des objets KMCluster * , et sauvegarde le meilleur état observé au cours des iterations */
	ObjectArray* kmBestClusters;

	/** cluster global : permet la construction des statistiques globales, calculées à partir de toutes les instances */
	KMCluster* kmGlobalCluster;

	/** somme des distances des instances, par rapport à leurs centres de clusters respectifs */
	ContinuousVector cvClustersDistancesSum;

	/** contient des StringObject *, et représente les modalités de l'attribut cible, en mode supervisé */
	ObjectArray oaTargetAttributeValues;

	/** parametres du traitement de clustering */
	KMParameters* parameters;

	/** classe gerant les indicateurs de la qualite d'un clustering (EVA, ARI, etc) */
	KMClusteringQuality* clusteringQuality;

	/* classe gerant la premiere initialisation des clusters, avant convergence, selon un choix de methodes possibles (KMean++, etc) */
	KMClusteringInitializer* clusteringInitializer;

	/* classe servant a calculer les levels de clustering */
	KMAttributesPartitioningManager* attributesPartitioningManager;

	/** nombre d'iterations effectuees au cours du clustering */
	int iIterationsDone;

	/** nombre de clusters vides supprimés */
	int iDroppedClustersNumber;

	/** pourcentage de la base qui a ete lue (utile en cas de memoire insuffisante) */
	double dUsedSampleNumberPercentage;

	/** matrice 2 dimensions (ligne = n° de cluster, colonne = n° de cluster) qui contient les distances entre chaque centre de cluster */
	Continuous** clustersCentersDistances;

	/** correspondance, à un instant T, entre une instance et son cluster d'appartenance. Clé = pointeur sur KWObject. Valeur = pointeur sur KMCluster */
	NumericKeyDictionary* instancesToClusters;

	/** matrice de confusion "classes predites (ou majoritaires) versus classes reelles", mode supervise et phase de train
	colonne = classe reelle, ligne = classe predite */
	KWFrequencyTable* kwftConfusionMatrix;

	/** dictionnaire contenant les levels de clustering. Clé = nom de l'attribut natif, valeur = level */
	NumericKeyDictionary nkdClusteringLevels;

	/**  tables de contingences pour le calcul les levels sur le clustering : Cle = lnom d'attribut, Valeur = KWFrequencyTable * --> comptage des modalités groupées ou d'intervalles pour un attribut donné */
	ObjectDictionary odGroupedModalitiesFrequencyTables;

	/** stocker les noms d'attributs natifs, afin de garantir la persistance memoire des SymbolData utilisés comme clés, dans le dictionnaire nkdClusteringLevels */
	SymbolVector svNativeAttributesNames;

	friend class PLShared_Clustering;

};

//////////////////////////////////////////////////////////
// Classe PLShared_Clustering
/// Serialisation de la classe KMClustering

class PLShared_Clustering : public PLSharedObject
{

public:

	PLShared_Clustering();
	~PLShared_Clustering();

	// Acces au clustering
	void SetClustering(KMClustering*);
	KMClustering* GetClustering();

	// Reimplementation des methodes virtuelles
	void DeserializeObject(PLSerializer*, Object*) const;
	void SerializeObject(PLSerializer*, const Object*) const override;

	//////////////////////////////////////////////////////////////////
	///// Implementation
protected:

	// Creation d'un objet (type d'objet a serialiser)
	Object* Create() const;
};

inline ObjectArray* KMClustering::GetClusters() const {
	return kmClusters;
}

inline const KMParameters* KMClustering::GetParameters() const {
	return parameters;
}

inline KMClusteringQuality* KMClustering::GetClusteringQuality() const {
	return clusteringQuality;
}

inline KMAttributesPartitioningManager* KMClustering::GetAttributesPartitioningManager() const {
	return attributesPartitioningManager;
}

inline KMCluster* KMClustering::GetGlobalCluster() const
{
	return kmGlobalCluster;
}

inline KMCluster* KMClustering::GetCluster(const int idx) const
{
	assert(idx < GetClusters()->GetSize());
	return cast(KMCluster*, GetClusters()->GetAt(idx));
}

inline Continuous** KMClustering::GetClustersCentersDistances() const {

	assert(clustersCentersDistances != NULL);
	return clustersCentersDistances;
}

inline NumericKeyDictionary* KMClustering::GetInstancesToClusters() const {
	assert(instancesToClusters != NULL);
	return instancesToClusters;
}


inline const int  KMClustering::GetIterationsDone() const {
	return iIterationsDone;
}

inline const KWFrequencyTable* KMClustering::GetConfusionMatrix() const {
	return kwftConfusionMatrix;
}


inline NumericKeyDictionary& KMClustering::GetClusteringLevelsDictionary() const {
	return (NumericKeyDictionary&)nkdClusteringLevels;
}


//***********************************************************************

/// classe d'implementation, representant une instance de cluster correspondant a un objet de database present en memoire, et sa distance au centre de son cluster. Cette classe interne est utilisee principalement dans le cadre de l'algo de post-optimisation d'un clustering.
class KMInstance : public Object
{
public:

	KMInstance(KWObject* inst, int idClus, double dist) : idCluster(idClus), distance(dist), instance(inst) {}

	const int idCluster;
	const double distance;
	const KWObject* instance;
};

int KMClusteringDistanceCompareDesc(const void* elem1, const void* elem2);

int KMClusteringDistanceCompareAsc(const void* elem1, const void* elem2);

int KMClusteringTargetCountCompare(const void* elem1, const void* elem2);

const ALString KMGetDisplayString(const int d);

const ALString KMGetDisplayString(const double d);





