// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "Object.h"
#include "KWClass.h"
#include "KWPredictorReport.h"

int KMCompareLabels(const void* elem1, const void* elem2);

////////////////////////////////////////////////////////////
// Classe KMParameters
///    Parametrage d'un traitement de clustering
//

class KMParameters : public Object
{
public:
	// Constructeur
	KMParameters();
	~KMParameters();

	// Copie et duplication
	void CopyFrom(const KMParameters* aSource);

	KMParameters* Clone() const;

	/** type de clustering */
	enum ClusteringType {
		KMeans,
		KNN
	};

	/** mode de calcul de la distance au centre du cluster */
	enum DistanceType {
		L1Norm,				// distance selon la norme L1
		L2Norm,				// distance selon la norme L2
		CosineNorm		// distance selon la norme cosinus
	};

	/** type de model local */
	enum LocalModelType {
		None,// pas de modele local (genere pour chaque cluster)
		NB,	// modele local Naive Bayes pour chaque cluster
		SNB// modele local Selective Naive Bayes pour chaque cluster
	};

	/** type de pre-traitement */
	enum PreprocessingType {
		UnusedVariable,
		NoPreprocessing,
		RankNormalization,
		Normalization,
		CenterReduction,
		BasicGrouping,
		Binarization,
		HammingConditionalInfo,
		ConditionaInfoWithPriors,
		Entropy,
		EntropyWithPriors,
		AutomaticallyComputed,
		SourceConditionalInfo
	};

	/** mode d'initialisation des centres de clusters, avant le calcul kmean */
	enum ClustersCentersInitMethod {
		ClustersCentersInitMethodAutomaticallyComputed, // si supervise : KMean++R, si non supervise : KMean++
		Random,				// K centres sont choisis aleatoirement, parmi les instances de la base entiere
		Sample, // K centres sont choisis aleatoirement, parmi les instances d'un echantillon de la base
		KMeanPlusPlus,		// algorithme kmean++
		KMeanPlusPlusR,		// algo KMean++R
		RocchioThenSplit,	// rocchio, puis split du cluster ayant l'intra la plus grande pour en obtenir 2 nouveaux clusters
		Bisecting,			// bisecting KMean
		MinMaxRandom,		// algo Min-Max (1er centre tire aleatoirement)
		MinMaxDeterministic,// algo Min-Max (1er centre = centre de gravite global)
		VariancePartitioning,	// algo base sur la variance des clusters et des variables
		ClassDecomposition	// les premiers centres correspondent aux modalites cibles, puis on applique un KMean++ sur chacune de ces partitions
	};

	/** type des centroides  */
	enum CentroidType {
		CentroidVirtual,		// centroide virtuel
		CentroidRealInstance    // instance reelle (la plus proche du centre virtuel)
	};

	/** mode de selection du meilleur replicate */
	enum ReplicateChoice {
		Distance,		// le meilleur clustering est celui qui la distance totale la plus petite
		EVA,			// le meilleur clustering est celui qui l'EVA le plus grand
		ARIByClusters,			// idem, avec ARI par clusters le plus eleve
		ARIByClasses,			// idem, avec ARI par clusters le plus élevé
		VariationOfInformation, // critere Variation of information le plus petit
		LEVA,			// variante de EVA
		DaviesBouldin,  // le meilleur clustering est celui qui a l'indice DaviesBouldin le plus petit
		PredictiveClustering, // mesure le compromis entre les classes induites par le clustering, et
							  // la compacite / separabilite des clusters (mode supervisé uniquement)
							  ReplicateChoiceAutomaticallyComputed,		// ARI si supervisé, sinon Distance
							  NormalizedMutualInformationByClusters, // NMI par clusters
							  NormalizedMutualInformationByClasses   // NMI par classes
	};

	/** post-optimisation du meilleur replicate */
	enum ReplicatePostOptimization {
		FastOptimization,
		NoOptimization
	};

	////////////////////////////////////////////////////////
	// Acces aux attributs


	/** pre-traitement des attributs continus */
	const PreprocessingType GetContinuousPreprocessingType() const;
	void SetContinuousPreprocessingType(const PreprocessingType);
	void SetContinuousPreprocessingType(const ALString preprocessingTypelabel);

	/** pre-traitement des attributs categoriels */
	const PreprocessingType GetCategoricalPreprocessingType() const;
	void SetCategoricalPreprocessingType(const PreprocessingType);
	void SetCategoricalPreprocessingType(const ALString preprocessingTypelabel);

	/** methode d'initialisation des centres de clusters */
	const ClustersCentersInitMethod GetClustersCentersInitializationMethod() const;
	void SetClustersCentersInitializationMethod(const ClustersCentersInitMethod);
	void SetClustersCentersInitializationMethod(const ALString label);

	/** type de distance a utiliser */
	const DistanceType GetDistanceType() const;
	void SetDistanceType(const DistanceType);
	void SetDistanceType(const ALString label);

	/** type de clustering */
	const ClusteringType GetClusteringType() const;
	void SetClusteringType(const ClusteringType);

	/** critere sur lequel selectionner le meilleur replicate */
	const ReplicateChoice GetReplicateChoice() const;
	void SetReplicateChoice(const ReplicateChoice);
	void SetReplicateChoice(const ALString label);

	/** type de modele local */
	const LocalModelType GetLocalModelType() const;
	void SetLocalModelType(const LocalModelType);
	void SetLocalModelType(const ALString label);

	/** forcer l'utilisation de MODL pour les modeles locaux */
	const boolean GetLocalModelUseMODL() const;
	void SetLocalModelUseMODL(boolean);

	/** type de centriode (reel ou virtuel) */
	const CentroidType GetCentroidType() const;
	void SetCentroidType(const CentroidType);
	void SetCentroidType(const ALString label);

	/** nombre de clusters  */
	const int GetKValue() const;
	void SetKValue(const int);

	/** nombre plancher de K lors de la post-optimisation */
	const int GetMinKValuePostOptimization() const;
	void SetMinKValuePostOptimization(const int);

	/** nombre maxi d'iterations lors du clustering (quelle que soit l'amelioration) */
	const int GetMaxIterations() const;
	void SetMaxIterations(int nValue);

	/** nombre maxi d'iterations lors de l'initialisation bisecting */
	const int GetBisectingMaxIterations() const;
	void SetBisectingMaxIterations(int nValue);

	/** gestion de la convergence : valeur de epsilon */
	const double GetEpsilonValue() const;
	void SetEpsilonValue(double nValue);

	/** gestion de la convergence. Lorsque l'amelioration entre 2 iterations est inferieure a epsilon, on fait au maximum
	 * 'n' iteration --> eviter les boucles infinies */
	const int GetEpsilonMaxIterations() const;
	void SetEpsilonMaxIterations(int nValue);

	/** parametre P du preprocessing "rank normalization" */
	const int GetPreprocessingMaxIntervalNumber() const;
	void SetPreprocessingMaxIntervalNumber(int nValue);

	/** parametre Q du preprocessing "basic grouping" (nombre de groupes) */
	const int GetPreprocessingMaxGroupNumber() const;
	void SetPreprocessingMaxGroupNumber(int nValue);

	/** nombre de groupes max en mode supervise */
	const int GetPreprocessingSupervisedMaxGroupNumber() const;
	void SetPreprocessingSupervisedMaxGroupNumber(int nValue);

	/** nombre d'intervalles max en mode supervise */
	const int GetPreprocessingSupervisedMaxIntervalNumber() const;
	void SetPreprocessingSupervisedMaxIntervalNumber(int nValue);

	/** nbre de fois que l'apprentissage doit être executé. Au terme de ces clustering, on ne retiendra que le meilleur resultat */
	const int GetLearningNumberOfReplicates() const;
	void SetLearningNumberOfReplicates(int nValue);

	/** si pas assez de memoire disponible, ou si explicitement demande, alors l'apprentissage KMean se fera par mini-batchs successifs, dont la taille (en nombre d'instances) est specifiee par ce parametre */
	const int GetMiniBatchSize() const;
	void SetMiniBatchSize(int nValue);

	const boolean GetMiniBatchMode() const;
	void SetMiniBatchMode(boolean);

	const int GetPostOptimizationVnsLevel() const;
	void SetPostOptimizationVnsLevel(int nValue);

	/** nbre de fois que les replicates bisecting doivent etre executes */
	const int GetBisectingNumberOfReplicates() const;
	void SetBisectingNumberOfReplicates(int nValue);

	/** flag mode supervisé */
	const boolean GetSupervisedMode() const;
	void SetSupervisedMode(boolean nValue);

	/** flag mode verbeux */
	const boolean GetVerboseMode() const;
	void SetVerboseMode(boolean nValue);

	/** flag mode parallele */
	const boolean GetParallelMode() const;
	void SetParallelMode(boolean nValue);

	/** post-optimisation de replicate */
	const ReplicatePostOptimization GetReplicatePostOptimization() const;
	void SetReplicatePostOptimization(ReplicatePostOptimization);
	void SetReplicatePostOptimization(const ALString label);

	/** flag mode verbeux pour initialisation bisecting */
	const boolean GetBisectingVerboseMode() const;
	void SetBisectingVerboseMode(boolean nValue);

	/** flag : produire ou non des statistiques detailles dans les rapports d'apprentissage et d'evaluation */
	const boolean GetWriteDetailedStatistics() const;
	void SetWriteDetailedStatistics(boolean nValue);

	/** nombre max de variables (tri en fonction du level) */
	const int GetMaxEvaluatedAttributesNumber() const;
	void SetMaxEvaluatedAttributesNumber(int nValue);

	/** flag : variables a level nul a utiliser ou pas, lors du KMean ou du KNN (ne s'applique eventuellement, que si les pretraitements sont non supervises).  */
	const boolean GetKeepNulLevelVariables() const;
	void SetKeepNulLevelVariables(boolean nValue);

	/** garder la liste des attributs utilisés lors du clustering (reperés a partir de leur label) */
	void AddAttributes(const KWClass* kwc);

	/** garder la liste des noms d'attributs natifs et recodés, aux fins d'affichage dans les rapports */
	void AddRecodedAttribute(const KWAttribute* nativeAttribute, const KWAttribute* recodedAtttribute);

	/** determiner si un attribut sert au calcul kmean */
	bool IsKMeanAttributeLoadIndex(const KWLoadIndex& i) const;

	/** determiner si un attribut sert au calcul kmean, en fonction de son nom */
	bool IsKMAttributeName(const ALString& name) const;

	/** ecriture des rapports : obtenir un nom d'attribut a partir de son index de chargement */
	ALString GetLoadedAttributeNameByRank(const int idx) const;

	/** ecriture des rapports : obtenir un nom d'attribut natif, a partir de son nom recodé */
	ALString GetNativeAttributeName(const ALString recodedAttributeName) const;

	/** liste des attributs servant au calcul K-Means : clé = nom, valeur = index de chargement de l'attribut */
	const ObjectDictionary& GetKMAttributeNames() const;

	/** liste des attributs : clé = nom, valeur = indice permettant de retrouver l'index de chargement de l'attribut, dans la structure correspondante */
	const ObjectDictionary& GetLoadedAttributesNames() const;

	/** clé = nom attribut recodé, valeur = nom attribut natif */
	const ObjectDictionary& GetRecodedAttributesNames() const;

	/** liste des index de chargement des attributs servant au calcul K-Means. Les attributs non kmeans sont egalement presents, et leur index n'est pas renseigne  (KWLoadIndex::IsValid() renvoie faux )*/
	const KWLoadIndexVector& GetKMeanAttributesLoadIndexes() const;

	/** liste des index de chargement des attributs natifs. Les attributs non natifs sont egalement presents, et leur index n'est pas renseigne  (KWLoadIndex::IsValid() renvoie faux ) */
	const KWLoadIndexVector& GetNativeAttributesLoadIndexes() const;

	/** liste des index de chargement de tous les attributs charges */
	const KWLoadIndexVector& GetLoadedAttributesLoadIndexes() const;

	/** retrouver le rang d'un attribut (dans les centroides) a partir de son index de chargement valide */
	const int GetAttributeRankFromLoadIndex(const KWLoadIndex&) const;

	/** determine si l'objet passé en parametre a une valeur manquante parmi ses attributs après recodage */
	bool HasMissingKMeanValue(const KWObject*) const;

	/** determine si l'objet passé en parametre a une valeur manquante parmi ses attributs natifs */
	bool HasMissingNativeValue(const KWObject*) const;

	// modifier le parametrage des attributs de la classe, en fonction des libelles
	void PrepareDeploymentClass(KWClass* modelingClass);

	/** attribut id de cluster */
	const KWAttribute* GetIdClusterAttribute() const;
	void SetIdClusterAttributeFromClass(const KWClass* kwc);

	const ALString& GetMainTargetModality() const;
	void SetMainTargetModality(const ALString&);

	/** recuperer une chaine pourvue d'un suffixe numérique, en evitant les doublons eventuels */
	static StringObject* GetUniqueLabel(const ObjectArray& existingLabels, const ALString prefix);

	////////////////////////////////////////////////////////
	// Divers

	// Ecriture
	void Write(ostream& ost) const;
	void WriteJSON(JSONFile* fJSON) const;

	// Libelles
	const ALString GetClassLabel() const;
	const ALString GetObjectLabel() const;
	const ALString GetDistanceTypeLabel() const;
	const ALString GetClustersCentersInitializationMethodLabel() const;
	const ALString GetCategoricalPreprocessingTypeLabel(const bool bTranslateAutomaticallyComputed = false) const;
	const ALString GetContinuousPreprocessingTypeLabel(const bool bTranslateAutomaticallyComputed = false) const;
	const ALString GetCentroidTypeLabel() const;
	const ALString GetReplicateChoiceLabel() const;
	const ALString GetLocalModelTypeLabel() const;
	const ALString GetReplicatePostOptimizationLabel() const;

	// variables statiques
	static const int K_MAX_VALUE;
	static const int K_DEFAULT_VALUE;
	static const int REPLICATE_NUMBER_MAX_VALUE;
	static const int MINI_BATCH_SIZE_MAX_VALUE;
	static const int MAX_ITERATIONS;
	static const int EPSILON_MAX_ITERATIONS;
	static const int EPSILON_MAX_ITERATIONS_DEFAULT_VALUE;
	static const double EPSILON_DEFAULT_VALUE;
	static const int PREPROCESSING_MAX_INTERVAL_DEFAULT_VALUE;
	static const int PREPROCESSING_MAX_GROUP_DEFAULT_VALUE;
	static const int REPLICATE_NUMBER_DEFAULT_VALUE;
	static const int MINI_BATCH_SIZE_DEFAULT_VALUE;

	// libelles
	static const ALString KM_ATTRIBUTE_LABEL;
	static const ALString SELECTED_NATIVE_ATTRIBUTE_LABEL;
	static const char* AUTO_COMPUTED_LABEL;
	static const char* CENTROID_REAL_INSTANCE_LABEL;
	static const char* CENTROID_VIRTUAL_LABEL;

	// pretraitements
	static const char* MODL_LABEL;
	static const char* BASIC_GROUPING_LABEL;
	static const char* SOURCE_CONDITIONAL_INFO_LABEL;
	static const char* RANK_NORMALIZATION_LABEL;
	static const char* HAMMING_CONDITIONAL_INFO_CONTINUOUS_LABEL;
	static const char* HAMMING_CONDITIONAL_INFO_CATEGORICAL_LABEL;
	static const char* CONDITIONAL_INFO_WITH_PRIORS_CONTINUOUS_LABEL;
	static const char* CONDITIONAL_INFO_WITH_PRIORS_CATEGORICAL_LABEL;
	static const char* ENTROPY_CONTINUOUS_LABEL;
	static const char* ENTROPY_CATEGORICAL_LABEL;
	static const char* ENTROPY_WITH_PRIORS_CONTINUOUS_LABEL;
	static const char* ENTROPY_WITH_PRIORS_CATEGORICAL_LABEL;
	static const char* CENTER_REDUCTION_LABEL;
	static const char* BINARIZATION_LABEL;
	static const char* NORMALIZATION_LABEL;
	static const char* UNUSED_VARIABLE_LABEL;
	static const char* NO_PREPROCESSING_LABEL;
	static const char* REPLICATE_DISTANCE_LABEL;
	static const char* REPLICATE_EVA_LABEL;
	static const char* REPLICATE_ARI_BY_CLUSTERS_LABEL;
	static const char* REPLICATE_ARI_BY_CLASSES_LABEL;
	static const char* REPLICATE_VARIATION_OF_INFORMATION_LABEL;
	static const char* REPLICATE_PREDICTIVE_CLUSTERING_LABEL;
	static const char* REPLICATE_LEVA_LABEL;
	static const char* REPLICATE_DAVIES_BOULDIN_LABEL;
	static const char* REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLUSTERS_LABEL;
	static const char* REPLICATE_NORMALIZED_MUTUAL_INFORMATION_BY_CLASSES_LABEL;

	// Verification globale
	boolean Check();

	////////////////////////////////////////////////////////
	//// Implementation
protected:

	// Attributs de la classe
	int nMaxIterations;
	int nBisectingMaxIterations;
	double dEpsilonValue;
	int nEpsilonMaxIterations;
	boolean bSupervisedMode;
	boolean bVerboseMode;
	boolean bParallelMode;
	boolean bMiniBatchMode;
	boolean bBisectingVerboseMode;
	boolean bWriteDetailedStatistics;
	boolean bLocalModelUseMODL;
	boolean bKeepNulLevelVariables;
	int iMaxEvaluatedAttributesNumber;
	int iKValue;
	int iMinKValuePostOptimization;
	int iPreprocessingMaxIntervalNumber;
	int iPreprocessingMaxGroupNumber;
	int iPreprocessingSupervisedMaxIntervalNumber;
	int iPreprocessingSupervisedMaxGroupNumber;
	int iLearningNumberOfReplicates;
	int iMiniBatchSize;
	int iPostOptimizationVnsLevel;
	int iBisectingNumberOfReplicates;
	ClusteringType clusteringType;
	DistanceType distanceType;
	CentroidType centroidType;
	ClustersCentersInitMethod clustersCentersInitMethod;
	PreprocessingType categoricalPreprocessingType;
	PreprocessingType continuousPreprocessingType;
	ReplicateChoice replicateChoice;
	ReplicatePostOptimization replicatePostOptimization;
	LocalModelType localModelType;

	/** index de chargement de tous les attributs charges du dico */
	KWLoadIndexVector livLoadedAttributesLoadIndexes;

	/** index de chargement des attributs utilises pour le calcul kmean. Les attributs non kmeans sont egalement presents, et leur index n'est pas renseigne  (KWLoadIndex::IsValid() renvoie faux ) */
	KWLoadIndexVector livKMeanAttributesLoadIndexes;

	/** index de chargement des attributs natifs. Les attributs non natifs sont egalement presents, et leur index n'est pas renseigne  (KWLoadIndex::IsValid() renvoie faux ) */
	KWLoadIndexVector livNativeAttributesLoadIndexes;

	/** clé = nom, valeur = IntegerObject * : poste concerne de l'attribut K-Means, dans la structure livLoadedAttributesLoadIndexes */
	ObjectDictionary odKMeanAttributesNames;

	/** clé = nom d'attribut, valeur = IntegerObject * : poste concerne de l'attribut (K-Means ou non), dans la structure livLoadedAttributesLoadIndexes */
	ObjectDictionary odLoadedAttributesNames;

	/** clé = nom attribut recodé, valeur = StringObject *, contenant le nom de l'attribut natif */
	ObjectDictionary odRecodedAttributesNames;

	ALString asMainTargetModality;

	KWAttribute* idClusterAttribute;
};




////////////////////////////////////////////////////////////
// Implementations inline

inline bool KMParameters::IsKMeanAttributeLoadIndex(const KWLoadIndex& idx) const {

	assert(idx.IsValid());

	for (int i = 0; i < livKMeanAttributesLoadIndexes.GetSize(); i++) {
		if (livKMeanAttributesLoadIndexes.GetAt(i) == idx)
			return true;
	}
	return false;
}

inline 	const int KMParameters::GetAttributeRankFromLoadIndex(const KWLoadIndex& idx) const {

	assert(idx.IsValid());

	for (int i = 0; i < livLoadedAttributesLoadIndexes.GetSize(); i++) {
		if (livLoadedAttributesLoadIndexes.GetAt(i) == idx)
			return i;
	}
	return -1;
}


inline bool KMParameters::IsKMAttributeName(const ALString& s) const {

	return (odKMeanAttributesNames.Lookup(s) == NULL ? false : true);
}

inline 	const KWLoadIndexVector& KMParameters::GetKMeanAttributesLoadIndexes() const {

	return livKMeanAttributesLoadIndexes;
}
inline 	const KWLoadIndexVector& KMParameters::GetNativeAttributesLoadIndexes() const {

	return livNativeAttributesLoadIndexes;
}
inline 	const KWLoadIndexVector& KMParameters::GetLoadedAttributesLoadIndexes() const {

	return livLoadedAttributesLoadIndexes;
}

inline 	const ObjectDictionary& KMParameters::GetRecodedAttributesNames() const {
	return odRecodedAttributesNames;
}

inline const int KMParameters::GetMaxIterations() const
{
	return nMaxIterations;
}

inline void KMParameters::SetMaxIterations(int nValue)
{
	nMaxIterations = nValue;
}
inline const int KMParameters::GetBisectingMaxIterations() const
{
	return nBisectingMaxIterations;
}

inline void KMParameters::SetBisectingMaxIterations(int nValue)
{
	nBisectingMaxIterations = nValue;
}
inline const double KMParameters::GetEpsilonValue() const
{
	return dEpsilonValue;
}
inline void KMParameters::SetEpsilonValue(double nValue)
{
	dEpsilonValue = nValue;
}
inline const int KMParameters::GetPreprocessingMaxIntervalNumber() const
{
	return iPreprocessingMaxIntervalNumber;
}
inline void KMParameters::SetPreprocessingMaxIntervalNumber(int nValue)
{
	iPreprocessingMaxIntervalNumber = nValue;
}
inline const int KMParameters::GetPreprocessingMaxGroupNumber() const
{
	return iPreprocessingMaxGroupNumber;
}
inline void KMParameters::SetPreprocessingMaxGroupNumber(int nValue)
{
	iPreprocessingMaxGroupNumber = nValue;
}
inline const int KMParameters::GetPreprocessingSupervisedMaxIntervalNumber() const
{
	return iPreprocessingSupervisedMaxIntervalNumber;
}
inline void KMParameters::SetPreprocessingSupervisedMaxIntervalNumber(int nValue)
{
	iPreprocessingSupervisedMaxIntervalNumber = nValue;
}
inline const int KMParameters::GetPreprocessingSupervisedMaxGroupNumber() const
{
	return iPreprocessingSupervisedMaxGroupNumber;
}
inline void KMParameters::SetPreprocessingSupervisedMaxGroupNumber(int nValue)
{
	iPreprocessingSupervisedMaxGroupNumber = nValue;
}
inline const int KMParameters::GetLearningNumberOfReplicates() const
{
	return iLearningNumberOfReplicates;
}
inline void KMParameters::SetLearningNumberOfReplicates(int nValue)
{
	iLearningNumberOfReplicates = nValue;
}
inline const int KMParameters::GetMiniBatchSize() const
{
	return iMiniBatchSize;
}
inline void KMParameters::SetMiniBatchSize(int nValue)
{
	iMiniBatchSize = nValue;
}
inline const int KMParameters::GetPostOptimizationVnsLevel() const
{
	return iPostOptimizationVnsLevel;
}
inline void KMParameters::SetPostOptimizationVnsLevel(int nValue)
{
	iPostOptimizationVnsLevel = nValue;
}
inline const int KMParameters::GetBisectingNumberOfReplicates() const
{
	return iBisectingNumberOfReplicates;
}
inline void KMParameters::SetBisectingNumberOfReplicates(int nValue)
{
	iBisectingNumberOfReplicates = nValue;
}
inline const int KMParameters::GetEpsilonMaxIterations() const
{
	return nEpsilonMaxIterations;
}
inline void KMParameters::SetEpsilonMaxIterations(int nValue)
{
	nEpsilonMaxIterations = nValue;
}

inline void KMParameters::SetCentroidType(const CentroidType d)
{
	centroidType = d;
}
inline const KMParameters::CentroidType KMParameters::GetCentroidType() const
{
	return centroidType;
}
inline void KMParameters::SetReplicateChoice(const KMParameters::ReplicateChoice d)
{
	replicateChoice = d;
}
inline const KMParameters::ReplicateChoice KMParameters::GetReplicateChoice() const
{
	return replicateChoice;
}
inline void KMParameters::SetLocalModelType(const KMParameters::LocalModelType d)
{
	localModelType = d;
}
inline const KMParameters::LocalModelType KMParameters::GetLocalModelType() const
{
	return localModelType;
}
inline void KMParameters::SetDistanceType(const DistanceType d)
{
	distanceType = d;
}
inline const KMParameters::DistanceType KMParameters::GetDistanceType() const
{
	return distanceType;
}
inline void KMParameters::SetClusteringType(const ClusteringType d)
{
	clusteringType = d;
}
inline const KMParameters::ClusteringType KMParameters::GetClusteringType() const
{
	return clusteringType;
}
inline void KMParameters::SetCategoricalPreprocessingType(const PreprocessingType d)
{
	categoricalPreprocessingType = d;
}
inline const KMParameters::PreprocessingType KMParameters::GetCategoricalPreprocessingType() const
{
	return categoricalPreprocessingType;
}
inline void KMParameters::SetContinuousPreprocessingType(const PreprocessingType d)
{
	continuousPreprocessingType = d;
}
inline const KMParameters::PreprocessingType KMParameters::GetContinuousPreprocessingType() const
{
	return continuousPreprocessingType;
}
inline void KMParameters::SetClustersCentersInitializationMethod(const ClustersCentersInitMethod d)
{
	clustersCentersInitMethod = d;
}
inline const KMParameters::ClustersCentersInitMethod KMParameters::GetClustersCentersInitializationMethod() const
{
	return clustersCentersInitMethod;
}
inline void KMParameters::SetKValue(const int i)
{
	iKValue = i;
}
inline const int KMParameters::GetKValue() const
{
	return iKValue;
}

inline void KMParameters::SetMinKValuePostOptimization(const int i)
{
	iMinKValuePostOptimization = i;
}
inline const int KMParameters::GetMinKValuePostOptimization() const
{
	return iMinKValuePostOptimization;
}

inline const ALString& KMParameters::GetMainTargetModality() const {
	return asMainTargetModality;
}

inline void KMParameters::SetMainTargetModality(const ALString& s) {
	asMainTargetModality = s;
}

inline const KWAttribute* KMParameters::GetIdClusterAttribute() const {
	return idClusterAttribute;
}

inline bool KMParameters::HasMissingKMeanValue(const KWObject* o) const {

	// controle des valeurs manquantes, parmi les attributs servant à calculer le K-Means
	// (ce sont des continuous, en principe recodés, sauf si on a choisi de ne pas faire de pretraitement)

	const int size = GetKMeanAttributesLoadIndexes().GetSize();

	for (int i = 0; i < size; i++) {
		KWLoadIndex loadIndex = GetKMeanAttributesLoadIndexes().GetAt(i);
		if (loadIndex.IsValid() and o->GetContinuousValueAt(loadIndex) == KWContinuous::GetMissingValue())
			return true;
	}

	return false;
}

inline bool KMParameters::HasMissingNativeValue(const KWObject* o) const {

	// controle des valeurs manquantes, parmi les attributs natifs :

	const int size = GetNativeAttributesLoadIndexes().GetSize();
	assert(size > 0);

	for (int i = 0; i < size; i++) {

		KWLoadIndex loadIndex = GetNativeAttributesLoadIndexes().GetAt(i);
		if (not loadIndex.IsValid())
			continue;

		KWAttribute* native = o->GetClass()->GetAttributeAtLoadIndex(loadIndex);

		if (native->GetType() == KWType::Symbol and o->GetSymbolValueAt(loadIndex) == Symbol(""))
			return true;
		else
			if (native->GetType() == KWType::Continuous and o->GetContinuousValueAt(loadIndex) == KWContinuous::GetMissingValue())
				return true;
	}

	return false;
}
