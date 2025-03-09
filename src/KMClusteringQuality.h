// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWObject.h"
#include "KMParameters.h"
#include "KMClustering.h"
#include "KMCluster.h"

///////////////////////////////////
/// classe de mesure de la qualite d'un clustering, en fonction de divers criteres (ARI, EVA, NMI, Davies Bouldin....)
//

class KMClusteringQuality : public Object {

public:

	KMClusteringQuality();
	KMClusteringQuality(const ObjectArray* clusters, const KMParameters* param);
	~KMClusteringQuality();

	/** calcul du critere EVA, en fonction des clusters existants */
	void ComputeEVA(KMCluster* globalCluster, const int nbTargetModalities);

	/** calcul du critere EVA, en fonction d'une table des frequences de clusters par modalites (lignes = clusters, colonnes = modalites cibles) */
	void ComputeEVA(KWFrequencyTable* clustersFrequenciesByModalities);

	/** calcul du 1er terme de l'EVA (ne dependant pas des frequences par clusters et modalites cibles) */
	double ComputeEVAFirstTerm(const int clustersNumber, KWFrequencyTable* clustersFrequenciesByModalities);

	/** calcul du 2eme terme de l'EVA  */
	double ComputeEVASecondTerm(const int clustersNumber, KWFrequencyTable* clustersFrequenciesByModalities);

	/** calcul du 3eme terme de l'EVA  */
	double ComputeEVAThirdTerm(const int clustersNumber, KWFrequencyTable* clustersFrequenciesByModalities);

	/** calcul du critere LEVA */
	void ComputeLEVA(KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues);

	/** calcul de l'Adjusted Rand Index par clusters, pour le clustering obtenu */
	void ComputeARIByClusters(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues);

	/** calcul de l'Adjusted Rand Index par classes, pour le clustering obtenu */
	void ComputeARIByClasses(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues, const KWFrequencyTable* kwctFrequencyByPredictedClass);

	/** calcul du NMI par clusters, pour le clustering obtenu */
	void ComputeNormalizedMutualInformationByClusters(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues);

	/** calcul du NMI par classes pour le clustering obtenu */
	void ComputeNormalizedMutualInformationByClasses(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues, const KWFrequencyTable* kwctFrequencyByPredictedClass);

	/** calcul du PCC pour le clustering obtenu */
	void ComputePredictiveClustering(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues,
		const KWAttribute* targetAttribute, const boolean useEvaluationCentroids = false);

	/** calcul de la compacite des clusters */
	void ComputeCompactness(const ObjectArray& oaTargetAttributeValues, const KWAttribute* targetAttribute);

	/** calcul de la variation de l'information pour le clustering obtenu */
	void ComputeVariationOfInformation(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues);

	/** calcul de l'indice Davies Bouldin pour le clustering obtenu */
	void ComputeDaviesBouldin(const boolean useEvaluationCentroids = false);

	/** calcul de l'indice Davies Bouldin pour un attribut particulier*/
	void ComputeDaviesBouldinForAttribute(const int attributeRank);

	/** verifier qu'un clustering obtenu satisfait bien le theoreme de Huygens */
	boolean CheckHuygensTheoremCorrectness(KMCluster* globalCluster) const;

	/** gain EVA des clusters */
	const double GetEVA() const;

	/** gain LEVA des clusters */
	const double GetLEVA() const;

	/** Adjusted Rand Index par clusters */
	const double GetARIByClusters() const;

	/** Adjusted Rand Index par classes */
	const double GetARIByClasses() const;

	/** NMI */
	const double GetNormalizedMutualInformationByClusters() const;

	/** NMI */
	const double GetNormalizedMutualInformationByClasses() const;

	/** variation de l'information */
	const double GetVariationOfInformation() const;

	/** valeur PCC */
	const double GetPredictiveClustering() const;

	/** Davies Bouldin Index */
	const double GetDaviesBouldin() const;

	/** Davies Bouldin Index, pour un attribut particulier */
	const double GetDaviesBouldinForAttribute(const int attributeLoadIndex) const;

	const ObjectArray* GetClusters() const;
	void SetClusters(const ObjectArray*);

	const KMParameters* GetParameters() const;
	void SetParameters(const KMParameters*);

	void CopyFrom(const KMClusteringQuality* aSource);


protected:

	/** calcul de l'EVA */
	double ComputeEVA(const int clustersNumber, KMCluster* globalCluster, const int nbTargetModalities);

	/** calcul du LEVA */
	double ComputeLEVA(const int clustersNumber, KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues);

	/* calcul de factorielle utilisé dans le calcul de l'Adjusted Rand Index */
	double ComputeARIFactorial(long int n, int k);

	/** initialiser les probas des valeurs cibles pour le cluster global, à partir des clusters construits */
	void InitializeGlobalTargetProbs(KMCluster* globalCluster, const int nbTargetModalities);

	/** clustering courant */
	const ObjectArray* clusters;

	const KMParameters* parameters;

	/** gain EVA */
	double dEVA;

	/** gain LEVA */
	double dLEVA;

	/** variation de l'information */
	double dVariationOfInformation;

	/** Adjusted Rand Index, by clusters */
	double dARIByClusters;

	/** Adjusted Rand Index, by classes */
	double dARIByClasses;

	/** NMI par clusters */
	double dNormalizedMutualInformationByClusters;

	/** NMI par classes */
	double dNormalizedMutualInformationByClasses;

	/* indice Predictive clustering */
	double dPredictiveClustering;

	/* indice Davies Bouldin, tous attributs confondus */
	double dDaviesBouldin;

	/* indice Davies Bouldin, par attribut  */
	ContinuousVector cvDaviesBouldin;

};

inline const double KMClusteringQuality::GetEVA() const {
	return dEVA;
}

inline const double KMClusteringQuality::GetLEVA() const {
	return dLEVA;
}

inline const double KMClusteringQuality::GetVariationOfInformation() const {
	return dVariationOfInformation;
}

inline const double KMClusteringQuality::GetPredictiveClustering() const {
	return dPredictiveClustering;
}
inline const double KMClusteringQuality::GetARIByClusters() const {
	return dARIByClusters;
}

inline const double KMClusteringQuality::GetARIByClasses() const {
	return dARIByClasses;
}

inline const double KMClusteringQuality::GetNormalizedMutualInformationByClusters() const {
	return dNormalizedMutualInformationByClusters;
}

inline const double KMClusteringQuality::GetNormalizedMutualInformationByClasses() const {
	return dNormalizedMutualInformationByClasses;
}

inline const double KMClusteringQuality::GetDaviesBouldin() const {
	return dDaviesBouldin;
}
inline const double KMClusteringQuality::GetDaviesBouldinForAttribute(const int attributeLoadIndex) const {
	return cvDaviesBouldin.GetAt(attributeLoadIndex);
}

inline const ObjectArray* KMClusteringQuality::GetClusters() const {
	return clusters;
}


inline const KMParameters* KMClusteringQuality::GetParameters() const {
	return parameters;
}


