// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClusteringQuality.h"

KMClusteringQuality::KMClusteringQuality() {

	dEVA = 0.0;
	dLEVA = 0.0;
	dARIByClusters = 0.0;
	dARIByClasses = 0.0;
	dNormalizedMutualInformationByClusters = 0.0;
	dNormalizedMutualInformationByClasses = 0.0;
	dVariationOfInformation = 0.0;
	dPredictiveClustering = 0.0;
	dDaviesBouldin = 0.0;
	clusters = NULL;
	parameters = NULL;
}

KMClusteringQuality::KMClusteringQuality(const ObjectArray* _clusters, const KMParameters* _param) : clusters(_clusters), parameters(_param) {

	assert(_clusters != NULL);
	assert(_param != NULL);

	dEVA = 0.0;
	dLEVA = 0.0;
	dARIByClusters = 0.0;
	dNormalizedMutualInformationByClusters = 0.0;
	dNormalizedMutualInformationByClasses = 0.0;
	dARIByClasses = 0.0;
	dDaviesBouldin = 0.0;
	dVariationOfInformation = 0.0;
	dPredictiveClustering = 0.0;
}

KMClusteringQuality::~KMClusteringQuality() {

}

void KMClusteringQuality::ComputeEVA(KMCluster* globalCluster, const int nbTargetModalities) {

	double dEVANotNormalized = ComputeEVA(clusters->GetSize(), globalCluster, nbTargetModalities);

	if (dEVANotNormalized == KWContinuous::GetMinValue()) {
		// n'arrive que si on a des valeurs de modalites cible en test qui etaient inconnues en train
		dEVA = 0.0;
		return;
	}

	double eva1 = ComputeEVA(1, globalCluster, nbTargetModalities);
	if (eva1 == KWContinuous::GetMinValue()) {
		// n'arrive que si on a des valeurs de modalites cible en test qui etaient inconnues en train
		dEVA = 0.0;
		return;
	}

	dEVA = 1 - (dEVANotNormalized / eva1);

	//AddSimpleMessage("EVA value is 1 - (" +
	//	ALString(DoubleToString(dEVANotNormalized)) + " / " +
	//	ALString(DoubleToString(eva1)) + ") = " +
	//	ALString(DoubleToString(dEVA)));
}

void KMClusteringQuality::ComputeEVA(KWFrequencyTable* clustersFrequenciesByModalities) {

	/* formule de calcul :

	EVA(K) = 1er terme + 2eme terme + 3eme terme

	avec :

	1er terme = log(N) + logf(N+K-1) - logf(K) - logf(N-1)
	2eme terme = somme(k=1 � K) [logf(Nk + J - 1) - logf (J - 1) - logf(Nk)]
	3eme terme = somme(k=1 � K) [logf(Nk) - [somme(j=1 � J) logf(Nkj)] ]

	et :

	K = nombre de clusters (non vides)
	N = nombre total d'instances de la base
	J = nombre de classes du mod�le (c'est � dire le nombre de modalit�s diff�rentes pour la variable cible). Attention, en test, les modalites peuvent etre differentes, et l'EVA vaudra alors 0
	Nk = nombre d'instances dans le cluster k
	Nkj = nombre d'instances dans le cluster k, qui sont de la classe j

	*/

	assert(clustersFrequenciesByModalities != NULL);
	assert(clustersFrequenciesByModalities->GetFrequencyVectorNumber() > 0);

	int iNonEmptyClusters = 0;

	for (int i = 0; i < clustersFrequenciesByModalities->GetFrequencyVectorNumber(); i++) {

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, clustersFrequenciesByModalities->GetFrequencyVectorAt(i));

		for (int j = 0; j < fv->GetSize(); j++) {

			if (fv->GetFrequencyVector()->GetAt(j) > 0) {
				iNonEmptyClusters++;
				break;
			}
		}
	}

	double d1 = ComputeEVAFirstTerm(iNonEmptyClusters, clustersFrequenciesByModalities);
	double d2 = ComputeEVASecondTerm(iNonEmptyClusters, clustersFrequenciesByModalities);
	double d3 = ComputeEVAThirdTerm(iNonEmptyClusters, clustersFrequenciesByModalities);

	if (d3 == KWContinuous::GetMinValue()) {
		// n'arrive que si on a des valeurs de modalites cible en test qui etaient inconnues en train
		dEVA = 0.0;
		return;
	}

	const double dEVANotNormalized = d1 + d2 + d3;

	d1 = ComputeEVAFirstTerm(1, clustersFrequenciesByModalities);
	d2 = ComputeEVASecondTerm(1, clustersFrequenciesByModalities);
	d3 = ComputeEVAThirdTerm(1, clustersFrequenciesByModalities);

	const double dEvaOneCluster = d1 + d2 + d3;

	dEVA = 1 - (dEVANotNormalized / dEvaOneCluster);

	//AddSimpleMessage("EVA value is 1 - (" +
	//	ALString(DoubleToString(dEVANotNormalized)) + " / " +
	//	ALString(DoubleToString(dEvaOneCluster)) + ") = " +
	//	ALString(DoubleToString(dEVA)));
}

void KMClusteringQuality::ComputeLEVA(KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues) {

	double evaK = ComputeLEVA(clusters->GetSize(), globalCluster, oaTargetAttributeValues);
	if (evaK == KWContinuous::GetMinValue()) {
		dLEVA = 0.0;
		return;
	}

	double eva1 = ComputeLEVA(1, globalCluster, oaTargetAttributeValues);
	if (eva1 == KWContinuous::GetMinValue() or eva1 == 0) {
		dLEVA = 0.0;
		return;
	}

	dLEVA = 1 - (evaK / eva1);
}

void KMClusteringQuality::ComputeDaviesBouldin(const boolean useEvaluationCentroids) {

	/* calcul de l'index Davies-Bouldin :

	DB = (1 / K ) sum{ forall i in 1: K } max[i != j] { ratioIntraInter }

	et ratioIntraInter = (inertyIntra(Ki) + inertyIntra(Kj)) / distInter(Ki, Kj)

	Avec :
	i, j : indices des clusters provenant d'un meme partitionnement
	distInter(Kl, Km) : distance moyenne entre les clusters Kl and Km,
	inertyIntra(Kn) : inertie intra du cluster Kn,
	K : nombre de clusters

	*/

	assert(clusters->GetSize() > 0);

	// initialiser au passage la structure contenant les valeurs Davies Bouldin par attribut :
	KMCluster* c = cast(KMCluster*, clusters->GetAt(0));
	cvDaviesBouldin.SetSize(useEvaluationCentroids ? c->GetEvaluationCentroidValues().GetSize() : c->GetModelingCentroidValues().GetSize());
	cvDaviesBouldin.Initialize();

	// boucle sur les clusters de 1 � k
	for (int i = 0; i < clusters->GetSize(); i++) {

		double maxRatioIntraInter = 0;

		KMCluster* clusterI = cast(KMCluster*, clusters->GetAt(i));

		if (clusterI->GetFrequency() == 0)
			// cas d'un cluster devenu vide lors de l'evaluation de test
			continue;

		for (int j = 0; j < clusters->GetSize(); j++) {

			if (i == j)
				continue;

			KMCluster* clusterJ = cast(KMCluster*, clusters->GetAt(j));

			if (clusterJ->GetFrequency() == 0)
				// cas d'un cluster devenu vide lors de l'evaluation de test
				continue;

			const ContinuousVector& clusterICentroids = (useEvaluationCentroids ? clusterI->GetEvaluationCentroidValues() : clusterI->GetModelingCentroidValues());
			const ContinuousVector& clusterJCentroids = (useEvaluationCentroids ? clusterJ->GetEvaluationCentroidValues() : clusterJ->GetModelingCentroidValues());

			double ratioIntraInter = (sqrt(clusterI->GetInertyIntra(parameters->GetDistanceType())) + sqrt(clusterJ->GetInertyIntra(parameters->GetDistanceType()))) /
				sqrt(KMClustering::GetDistanceBetween(clusterICentroids, clusterJCentroids, KMParameters::L2Norm, parameters->GetKMeanAttributesLoadIndexes()));

			if (ratioIntraInter > maxRatioIntraInter)
				maxRatioIntraInter = ratioIntraInter;

		}
		dDaviesBouldin = dDaviesBouldin + maxRatioIntraInter;
	}

	dDaviesBouldin = dDaviesBouldin / clusters->GetSize();

}


void KMClusteringQuality::ComputeDaviesBouldinForAttribute(const int attributeRank) {

	// meme principe que l'index Davies Bouldin global, mais calcule pour un attribut particulier

	assert(clusters->GetSize() > 0);
	assert(cvDaviesBouldin.GetSize() > 0);

	cvDaviesBouldin.SetAt(attributeRank, 0);

	// boucle sur les clusters de 1 � k
	for (int i = 0; i < clusters->GetSize(); i++) {

		double maxRatioIntraInter = 0;

		KMCluster* clusterI = cast(KMCluster*, clusters->GetAt(i));

		if (clusterI->GetFrequency() == 0)
			// cas d'un cluster devenu vide lors de l'evaluation de test
			continue;

		for (int j = 0; j < clusters->GetSize(); j++) {

			if (i == j)
				continue;

			KMCluster* clusterJ = cast(KMCluster*, clusters->GetAt(j));

			if (clusterJ->GetFrequency() == 0)
				// cas d'un cluster devenu vide lors de l'evaluation de test
				continue;

			// calcul de ratioIntraInter
			double ratioIntraInter = (sqrt(clusterI->GetInertyIntraForAttribute(attributeRank, parameters->GetDistanceType())) + sqrt(clusterJ->GetInertyIntraForAttribute(attributeRank, parameters->GetDistanceType()))) /
				sqrt(KMClustering::GetDistanceBetweenForAttribute(attributeRank, clusterI->GetModelingCentroidValues(), clusterJ->GetModelingCentroidValues(), KMParameters::L2Norm));

			if (ratioIntraInter > maxRatioIntraInter)
				maxRatioIntraInter = ratioIntraInter;

		}
		cvDaviesBouldin.SetAt(attributeRank, cvDaviesBouldin.GetAt(attributeRank) + maxRatioIntraInter);
	}

	cvDaviesBouldin.SetAt(attributeRank, cvDaviesBouldin.GetAt(attributeRank) / clusters->GetSize());

}

void KMClusteringQuality::ComputeARIByClusters(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues) {

	// bas� sur l'algorithme matlab de Tijl De Bie : http://www.kernel-methods.net/matlab/algorithms/adjrand.m

	assert(globalCluster != NULL);
	assert(globalCluster->GetFrequency() > 0);

	assert(clusters->GetSize() > 0);

	assert(oaTargetAttributeValues.GetSize() > 0);

	KWFrequencyTable* modalityFrequencyByCluster = new KWFrequencyTable;
	modalityFrequencyByCluster->SetFrequencyVectorNumber(clusters->GetSize());// nombre de lignes

	for (int i = 0; i < clusters->GetSize(); i++) {
		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(i));
		fv->GetFrequencyVector()->SetSize(oaTargetAttributeValues.GetSize());// nombre de colonnes ppur chaque ligne
	}

	IntVector totalFrequencyByCluster;
	totalFrequencyByCluster.SetSize(clusters->GetSize());
	totalFrequencyByCluster.Initialize();

	IntVector totalFrequencyByModality;
	totalFrequencyByModality.SetSize(oaTargetAttributeValues.GetSize());
	totalFrequencyByModality.Initialize();

	// boucle sur les clusters de 1 � k, pour initialiser les valeurs de la table de contingence avec les effectifs des clusters pour chaque modalit� cible
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));

		for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) { // boucle sur les modalit�s de la variable cible

			if (cluster->GetFrequency() == 0)
				fv->GetFrequencyVector()->SetAt(idxTargetvalue, 0);
			else
				fv->GetFrequencyVector()->SetAt(idxTargetvalue, ((double)cluster->GetFrequency()) * cluster->GetTargetProbs().GetAt(idxTargetvalue));
		}
	}

	// boucle sur les modalites cibles, pour initialiser les frequences totales par modalite cible
	for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) {

		for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));
			totalFrequencyByModality.SetAt(idxTargetvalue, totalFrequencyByModality.GetAt(idxTargetvalue) + fv->GetFrequencyVector()->GetAt(idxTargetvalue));
		}
	}

	// boucle sur les clusters, pour initialiser les frequences totales par cluster
	// (NB. ces frequences ne comprennent que les individus ayant une modalite cible repertoriee lors de l'apprentissage, ce qui peut
	// etre different de la frequence totale d'un cluster, donnee par KMCluster::GetFrequency())   (si la base de test contient des modalites cibles inconnues)
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));

		for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) {
			totalFrequencyByCluster.SetAt(idxCluster, totalFrequencyByCluster.GetAt(idxCluster) + fv->GetFrequencyVector()->GetAt(idxTargetvalue));
		}
	}

	double a = 0.0;

	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));

		for (int j = 0; j < oaTargetAttributeValues.GetSize(); j++) {

			if (fv->GetFrequencyVector()->GetAt(j) > 1)
				a = a + ComputeARIFactorial(fv->GetFrequencyVector()->GetAt(j), 2);
		}
	}

	double b1 = 0.0;
	for (int i = 0; i < clusters->GetSize(); i++) {

		if (totalFrequencyByCluster.GetAt(i) > 1)
			b1 = b1 + ComputeARIFactorial(totalFrequencyByCluster.GetAt(i), 2);
	}

	double b2 = 0.0;
	for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {

		if (totalFrequencyByModality.GetAt(i) > 1)
			b2 = b2 + ComputeARIFactorial(totalFrequencyByModality.GetAt(i), 2);
	}

	if (globalCluster->GetFrequency() - 2 < 0)
		dARIByClusters = 0;
	else {

		double c = ComputeARIFactorial(globalCluster->GetFrequency(), 2);
		if (c == 0 or ((0.5 * (b1 + b2)) - ((b1 * b2) / c)) == 0)
			dARIByClusters = 0;
		else
			dARIByClusters = (a - ((b1 * b2) / c)) / ((0.5 * (b1 + b2)) - ((b1 * b2) / c));
	}

	delete modalityFrequencyByCluster;

}


void KMClusteringQuality::ComputeARIByClasses(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues, const KWFrequencyTable* kwctFrequencyByPredictedClass) {

	// bas� sur l'algorithme matlab de Tijl De Bie : http://www.kernel-methods.net/matlab/algorithms/adjrand.m

	assert(globalCluster != NULL);
	assert(globalCluster->GetFrequency() > 0);
	assert(oaTargetAttributeValues.GetSize() > 0);
	assert(kwctFrequencyByPredictedClass != NULL);
	assert(kwctFrequencyByPredictedClass->GetFrequencyVectorSize() == oaTargetAttributeValues.GetSize());
	assert(kwctFrequencyByPredictedClass->GetFrequencyVectorNumber() == oaTargetAttributeValues.GetSize());

	IntVector totalFrequencyByPredictedClass;
	totalFrequencyByPredictedClass.SetSize(oaTargetAttributeValues.GetSize());
	totalFrequencyByPredictedClass.Initialize();

	IntVector totalFrequencyByActualClass;
	totalFrequencyByActualClass.SetSize(oaTargetAttributeValues.GetSize());
	totalFrequencyByActualClass.Initialize();

	// initialiser les 2 vecteurs de frequences par modalites (predit et reel) a partir de la table de contingence
	for (int idxPredictedTargetValue = 0; idxPredictedTargetValue < oaTargetAttributeValues.GetSize(); idxPredictedTargetValue++) {

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwctFrequencyByPredictedClass->GetFrequencyVectorAt(idxPredictedTargetValue));

		for (int idxActualTargetvalue = 0; idxActualTargetvalue < oaTargetAttributeValues.GetSize(); idxActualTargetvalue++) {

			totalFrequencyByPredictedClass.SetAt(idxPredictedTargetValue,
				totalFrequencyByPredictedClass.GetAt(idxPredictedTargetValue) + fv->GetFrequencyVector()->GetAt(idxActualTargetvalue));

			totalFrequencyByActualClass.SetAt(idxActualTargetvalue,
				totalFrequencyByActualClass.GetAt(idxActualTargetvalue) + fv->GetFrequencyVector()->GetAt(idxActualTargetvalue));

		}
	}

	double a = 0.0;

	for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwctFrequencyByPredictedClass->GetFrequencyVectorAt(i));

		for (int j = 0; j < oaTargetAttributeValues.GetSize(); j++) {

			if (fv->GetFrequencyVector()->GetAt(j) > 1)
				a = a + ComputeARIFactorial(fv->GetFrequencyVector()->GetAt(j), 2);
		}
	}

	double b1 = 0.0;
	for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {

		if (totalFrequencyByPredictedClass.GetAt(i) > 1)
			b1 = b1 + ComputeARIFactorial(totalFrequencyByPredictedClass.GetAt(i), 2);
	}

	double b2 = 0.0;
	for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {

		if (totalFrequencyByActualClass.GetAt(i) > 1)
			b2 = b2 + ComputeARIFactorial(totalFrequencyByActualClass.GetAt(i), 2);
	}

	if (globalCluster->GetFrequency() - 2 < 0)
		dARIByClasses = 0;
	else {

		double c = ComputeARIFactorial(globalCluster->GetFrequency(), 2);
		if (c == 0 or ((0.5 * (b1 + b2)) - ((b1 * b2) / c)) == 0)
			dARIByClasses = 0;
		else
			dARIByClasses = (a - ((b1 * b2) / c)) / ((0.5 * (b1 + b2)) - ((b1 * b2) / c));
	}
}


void KMClusteringQuality::ComputeNormalizedMutualInformationByClusters(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues) {

	/*
	Cf. description fonctionnelle d'MLClusters v8, paragraphe "Crit�re Normalized Mutual Information"

	Soit K = nombre de clusters et C le nombre de classes

	Soit un tableau a 2 dimensions, contenant les frequences des clusters pour chaque modalite cible (lignes = clusters, colonnes = classes).
	Les indices i et j designent respectivement les lignes (clusters) et les colonnes (modalites cibles)

	Soit N le nombre total d'instances de la base (somme des fr�quences de tous les clusters pour toutes les modalites)
	Soit Pij, la fr�quence pour le cluster i et la modalite j, divis�e par N
	Soit Pi+, total par ligne des Pij (un total par cluster, toutes classes confondues)
	Soit P+j, total par colonne des Pij (un total par modalite cible, tous clusters confondus)

	NormalizedMutualInformationByClusters (K, C) = A / B

	Avec :

		A = a1
		a1 = somme pour tous les i et les j, de a2
		a2 = Pij * log(Pij / (Pi+ * P+j))

		B = racine(b1 * b2)
		b1 = somme pour tous les clusters i de (Pi+ * log(Pi+))
		b2 = somme pour toutes les modalites cibles j de (P+j * log(P+j))

	*/

	assert(globalCluster != NULL);
	assert(globalCluster->GetFrequency() > 0);
	assert(clusters->GetSize() > 0);
	assert(oaTargetAttributeValues.GetSize() > 0);

	// creation de la structure devant contenir les valeurs Pij. (pas possible d'utiliser une KWFrequencyTable puisque valeurs continues, et non des frequences de type entier)
	ObjectArray* Pij = new ObjectArray;
	Pij->SetSize(clusters->GetSize());
	// pour chaque ligne (cluster), on instancie un ContinuousVector correspondant aux valeurs par modalit�s cibles
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
		ContinuousVector* cv = new ContinuousVector;
		cv->SetSize(oaTargetAttributeValues.GetSize());
		cv->Initialize();
		Pij->SetAt(idxCluster, cv);
	}

	// renseigner les valeurs Pij :
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));
		ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxCluster));

		for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) { // boucle sur les modalit�s de la variable cible
			if (cluster->GetFrequency() == 0)
				cv->SetAt(idxTargetvalue, 0);
			else {
				double d = ((double)cluster->GetFrequency()) * cluster->GetTargetProbs().GetAt(idxTargetvalue);
				d /= (double)globalCluster->GetFrequency();
				cv->SetAt(idxTargetvalue, d);
			}
		}
	}

	// boucle sur les clusters, pour calculer les valeurs Pi+ (cumul par cluster des valeurs Pij)
	// (NB. ces frequences ne comprennent que les individus ayant une modalite cible repertoriee lors de l'apprentissage, ce qui peut
	// etre different de la frequence totale d'un cluster, donnee par KMCluster::GetFrequency())   (si la base de test contient des modalites cibles inconnues)
	ContinuousVector P_i_plus;
	P_i_plus.SetSize(clusters->GetSize());
	P_i_plus.Initialize();
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
		ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxCluster));
		for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++)
			P_i_plus.SetAt(idxCluster, P_i_plus.GetAt(idxCluster) + cv->GetAt(idxTargetvalue));
	}

	// boucle sur les modalites cibles, pour initialiser les valeurs P+j (cumul par modalite cible des valeurs Pij)
	ContinuousVector P_plus_j;
	P_plus_j.SetSize(oaTargetAttributeValues.GetSize());
	P_plus_j.Initialize();
	for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) {
		for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
			ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxCluster));
			P_plus_j.SetAt(idxTargetvalue, P_plus_j.GetAt(idxTargetvalue) + cv->GetAt(idxTargetvalue));
		}
	}

	double a1 = 0;

	// calcul des parties du numerateur (a1 et a2)
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxCluster));

		for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) { // boucle sur les modalit�s de la variable cible
			if (P_i_plus.GetAt(idxCluster) != 0 and P_plus_j.GetAt(idxTargetvalue) != 0 and cv->GetAt(idxTargetvalue) != 0) {
				const double a2 = cv->GetAt(idxTargetvalue) * log(cv->GetAt(idxTargetvalue) / (P_i_plus.GetAt(idxCluster) * P_plus_j.GetAt(idxTargetvalue)));
				a1 = a1 + a2;
			}
		}
	}

	const double A = a1;

	// calcul de b1
	double b1 = 0;
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
		const Continuous c = P_i_plus.GetAt(idxCluster);
		if (c != 0)
			b1 = b1 + (c * log(c));
	}

	// calcul de b2
	double b2 = 0;
	for (int idxTargetvalue = 0; idxTargetvalue < oaTargetAttributeValues.GetSize(); idxTargetvalue++) {
		const Continuous c = P_plus_j.GetAt(idxTargetvalue);
		if (c > 0)
			b2 = b2 + (c * log(c));
	}

	const double B = sqrt(b1 * b2);

	dNormalizedMutualInformationByClusters = (B == 0 ? 0 : A / B);


	Pij->DeleteAll();
	delete Pij;

}

void KMClusteringQuality::ComputeNormalizedMutualInformationByClasses(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues, const KWFrequencyTable* kwctFrequencyByPredictedClass) {

	/*
	Cf. description fonctionnelle d'MLClusters v8, paragraphe "Crit�re Normalized Mutual Information"

	Soit K = nombre de clusters et C le nombre de classes

	Soit un tableau a 2 dimensions, contenant les frequences des classes predites versus celles des classes reelles (lignes = classes predites, colonnes = classes reelles).
	Les indices i et j designent respectivement les lignes et les colonnes

	Soit N le nombre total d'instances de la base (somme des fr�quences de tous les clusters pour toutes les modalites)
	Soit Pij, la fr�quence pour la classe predite i et la classe reelle j, divis�e par N
	Soit Pi+, total par ligne des Pij (un total par classe predite, toutes classes reelles confondues)
	Soit P+j, total par colonne des Pij (un total par classe reelle, toutes classes predites confondues)

	NormalizedMutualInformationByClusters (K, C) = A / B

	Avec :

	A = a1
	a1 = somme pour tous les i et les j, de a2
	a2 = Pij * log(Pij / (Pi+ * P+j))

	B = racine(b1 * b2)
	b1 = somme pour toutes les classes predites i de (Pi+ * log(Pi+))
	b2 = somme pour toutes les classes reelles j de (P+j * log(P+j))

	*/

	assert(globalCluster != NULL);
	assert(globalCluster->GetFrequency() > 0);
	assert(oaTargetAttributeValues.GetSize() > 0);
	assert(kwctFrequencyByPredictedClass != NULL);
	assert(kwctFrequencyByPredictedClass->GetFrequencyVectorSize() == oaTargetAttributeValues.GetSize());
	assert(kwctFrequencyByPredictedClass->GetFrequencyVectorNumber() == oaTargetAttributeValues.GetSize());

	// creation de la structure devant contenir les valeurs Pij
	ObjectArray* Pij = new ObjectArray;
	Pij->SetSize(oaTargetAttributeValues.GetSize());
	// pour chaque ligne (classe predite), on instancie un ContinuousVector correspondant aux frequences par classes reelles
	for (int idxPredictedTargetValue = 0; idxPredictedTargetValue < oaTargetAttributeValues.GetSize(); idxPredictedTargetValue++) {
		ContinuousVector* cv = new ContinuousVector;
		cv->SetSize(oaTargetAttributeValues.GetSize());
		cv->Initialize();
		Pij->SetAt(idxPredictedTargetValue, cv);
	}

	// renseigner les valeurs Pij :
	for (int idxPredictedTargetValue = 0; idxPredictedTargetValue < oaTargetAttributeValues.GetSize(); idxPredictedTargetValue++) {

		ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxPredictedTargetValue));

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, kwctFrequencyByPredictedClass->GetFrequencyVectorAt(idxPredictedTargetValue));

		for (int idxActualTargetValue = 0; idxActualTargetValue < oaTargetAttributeValues.GetSize(); idxActualTargetValue++) { // boucle sur les classes reelles

			const double dFrequency = fv->GetFrequencyVector()->GetAt(idxActualTargetValue);

			if (dFrequency == 0)
				cv->SetAt(idxActualTargetValue, 0);
			else {
				double d = dFrequency / (double)globalCluster->GetFrequency();
				cv->SetAt(idxActualTargetValue, d);
			}
		}
	}

	// boucle sur les classes predites, pour calculer les valeurs Pi+ (cumul par classes predites des valeurs Pij)
	// (NB. ces frequences ne comprennent que les individus ayant une modalite cible repertoriee lors de l'apprentissage
	ContinuousVector P_i_plus;
	P_i_plus.SetSize(oaTargetAttributeValues.GetSize());
	P_i_plus.Initialize();
	for (int idxPredictedTargetValue = 0; idxPredictedTargetValue < oaTargetAttributeValues.GetSize(); idxPredictedTargetValue++) {
		ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxPredictedTargetValue));
		for (int idxActualTargetValue = 0; idxActualTargetValue < oaTargetAttributeValues.GetSize(); idxActualTargetValue++) // boucle sur les classes reelles
			P_i_plus.SetAt(idxPredictedTargetValue, P_i_plus.GetAt(idxPredictedTargetValue) + cv->GetAt(idxActualTargetValue));
	}

	// boucle sur les classes reelles, pour initialiser les valeurs P+j (cumul par classes reelles, des valeurs Pij)
	ContinuousVector P_plus_j;
	P_plus_j.SetSize(oaTargetAttributeValues.GetSize());
	P_plus_j.Initialize();
	for (int idxActualTargetValue = 0; idxActualTargetValue < oaTargetAttributeValues.GetSize(); idxActualTargetValue++) { // boucle sur les classes reelles
		for (int idxPredictedTargetValue = 0; idxPredictedTargetValue < oaTargetAttributeValues.GetSize(); idxPredictedTargetValue++) {
			ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxPredictedTargetValue));
			P_plus_j.SetAt(idxActualTargetValue, P_plus_j.GetAt(idxActualTargetValue) + cv->GetAt(idxActualTargetValue));
		}
	}

	double a1 = 0;

	// calcul des parties du numerateur (a1 et a2)
	for (int idxPredictedTargetValue = 0; idxPredictedTargetValue < oaTargetAttributeValues.GetSize(); idxPredictedTargetValue++) {

		ContinuousVector* cv = cast(ContinuousVector*, Pij->GetAt(idxPredictedTargetValue));

		for (int idxActualTargetValue = 0; idxActualTargetValue < oaTargetAttributeValues.GetSize(); idxActualTargetValue++) {
			if (P_i_plus.GetAt(idxPredictedTargetValue) != 0 and P_plus_j.GetAt(idxActualTargetValue) != 0 and cv->GetAt(idxActualTargetValue) != 0) {
				const double a2 = cv->GetAt(idxActualTargetValue) * log(cv->GetAt(idxActualTargetValue) / (P_i_plus.GetAt(idxPredictedTargetValue) * P_plus_j.GetAt(idxActualTargetValue)));
				a1 = a1 + a2;
			}
		}
	}

	const double A = a1;

	// calcul de b1
	double b1 = 0;
	for (int idxPredictedTargetValue = 0; idxPredictedTargetValue < oaTargetAttributeValues.GetSize(); idxPredictedTargetValue++) {
		const Continuous c = P_i_plus.GetAt(idxPredictedTargetValue);
		if (c != 0)
			b1 = b1 + (c * log(c));
	}

	// calcul de b2
	double b2 = 0;
	for (int idxActualTargetValue = 0; idxActualTargetValue < oaTargetAttributeValues.GetSize(); idxActualTargetValue++) {
		const Continuous c = P_plus_j.GetAt(idxActualTargetValue);
		if (c != 0)
			b2 = b2 + (c * log(c));
	}

	const double B = sqrt(b1 * b2);

	dNormalizedMutualInformationByClasses = (B == 0 ? 0 : A / B);

	Pij->DeleteAll();
	delete Pij;

}

void KMClusteringQuality::ComputeCompactness(const ObjectArray& oaTargetAttributeValues, const KWAttribute* targetAttribute) {

	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));
		cluster->ComputeCompactness(oaTargetAttributeValues, targetAttribute);
	}
}

void KMClusteringQuality::ComputePredictiveClustering(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues,
	const KWAttribute* targetAttribute, const boolean useEvaluationCentroids) {

	Continuous sumRatios = 0;

	for (int idxClusterI = 0; idxClusterI < clusters->GetSize(); idxClusterI++) {

		KMCluster* clusterI = cast(KMCluster*, clusters->GetAt(idxClusterI));

		const ContinuousVector& clusterCentroidsI = (useEvaluationCentroids ? clusterI->GetEvaluationCentroidValues() : clusterI->GetModelingCentroidValues());

		const ALString majorityTargetValueI = clusterI->GetMajorityTargetValue();

		Continuous maxClusterRatio = 0;

		for (int idxClusterJ = 0; idxClusterJ < clusters->GetSize(); idxClusterJ++) {

			KMCluster* clusterJ = cast(KMCluster*, clusters->GetAt(idxClusterJ));

			if (clusterI != clusterJ) {

				const ContinuousVector& clusterCentroidsJ = (useEvaluationCentroids ? clusterJ->GetEvaluationCentroidValues() : clusterJ->GetModelingCentroidValues());
				const ALString majorityTargetValueJ = clusterJ->GetMajorityTargetValue();

				const Continuous similarity = KMClustering::GetSimilarityBetween(clusterCentroidsI, clusterCentroidsJ, majorityTargetValueI, majorityTargetValueJ, parameters);

				const Continuous ratio = (similarity == 0 ? 0 : (clusterI->GetCompactness() + clusterJ->GetCompactness()) / similarity);

				if (ratio > maxClusterRatio)
					maxClusterRatio = ratio;
			}
		}
		sumRatios += maxClusterRatio;
	}

	dPredictiveClustering = sumRatios / clusters->GetSize();
}

void KMClusteringQuality::ComputeVariationOfInformation(const KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues) {

	/*

	VIn = (2 * H(K,C)) / (H(K) + H(C)) - 1

	Et :

	N = nombre total d'individus
	K = nombre de clusters
	C = nombre de classes
	Pk = [ Nombre d�individus dans le cluster k ] / N
	H(K) =  - [ somme de k=1 jusqu'a K, de Pk * log(Pk)  ]  --> c'est l'entropie de la partition issue du clustering
	Pc = [ Nombre d�individus de la classe c ] / N
	H(C)=  - [ somme de c=1 jusqu'a C, de Pc * log(Pc) ] --> c'est l'entropie de la partition issue du clustering
	Pkc = [ Nombre d�individus dans le cluster k et de classe c ] / N
	H(K,C) =  - [ somme de k=1 jusqu'a K, de [ somme de c=1 jusqu'� C, de Pkc * log(Pkc)] ]

	*/

	assert(globalCluster != NULL);
	assert(globalCluster->GetFrequency() > 0);

	assert(clusters->GetSize() > 0);

	assert(oaTargetAttributeValues.GetSize() > 0);

	KWFrequencyTable* modalityFrequencyByCluster = new KWFrequencyTable; // frequences de modalites cibles, par cluster
	modalityFrequencyByCluster->SetFrequencyVectorNumber(clusters->GetSize());
	for (int i = 0; i < clusters->GetSize(); i++) {
		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(i));
		fv->GetFrequencyVector()->SetSize(oaTargetAttributeValues.GetSize());// nombre de colonnes pour chaque ligne
	}
	IntVector totalFrequencyByCluster; // frequences totales, par clusters
	totalFrequencyByCluster.SetSize(clusters->GetSize());
	totalFrequencyByCluster.Initialize();

	IntVector totalFrequencyByModality; // frequences totales, par modalite cible connue
	totalFrequencyByModality.SetSize(oaTargetAttributeValues.GetSize());
	totalFrequencyByModality.Initialize();

	// boucle sur les clusters de 1 � k, pour initialiser les valeurs de la table de contingence avec les effectifs des clusters pour chaque modalit� cible
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));

		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));

		for (int iTarget = 0; iTarget < oaTargetAttributeValues.GetSize(); iTarget++) { // boucle sur les modalit�s de la variable cible

			if (cluster->GetFrequency() == 0)
				fv->GetFrequencyVector()->SetAt(iTarget, 0);
			else
				fv->GetFrequencyVector()->SetAt(iTarget, ((double)cluster->GetFrequency()) * cluster->GetTargetProbs().GetAt(iTarget));
		}
	}

	// boucle sur les modalites cibles, pour initialiser les frequences totales par modalite cible
	for (int iTarget = 0; iTarget < oaTargetAttributeValues.GetSize(); iTarget++) {

		for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));
			totalFrequencyByModality.SetAt(iTarget, totalFrequencyByModality.GetAt(iTarget) + fv->GetFrequencyVector()->GetAt(iTarget));
		}
	}

	// boucle sur les clusters, pour initialiser les frequences totales par cluster
	// (NB. ces frequences ne comprennent que les individus ayant une modalite cible repertoriee lors de l'apprentissage, ce qui peut
	// etre different de la frequence totale d'un cluster, donnee par KMCluster::GetFrequency())   (si la base de test contient des modalites cibles inconnues)
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));
		for (int iTarget = 0; iTarget < oaTargetAttributeValues.GetSize(); iTarget++)
			totalFrequencyByCluster.SetAt(idxCluster, totalFrequencyByCluster.GetAt(idxCluster) + fv->GetFrequencyVector()->GetAt(iTarget));
	}

	// calcul de H(K)
	double Hk = 0.0;
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));
		double Pk = (double)cluster->GetFrequency() / (double)globalCluster->GetFrequency(); // question : la frequence totale du cluster doit-elle comprendre les individus ayant une modalite cible inconnue en test ?
		if (Pk > 0)
			Hk += (Pk * log(Pk));
	}
	Hk = -Hk;

	// calcul de H(C)
	double Hc = 0.0;
	for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {
		double Pc = (double)totalFrequencyByModality.GetAt(i) / (double)globalCluster->GetFrequency();
		if (Pc > 0)
			Hc += (Pc * log(Pc));
	}
	Hc = -Hc;

	// calcul de H(K,C)
	double Hkc = 0.0;

	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
		KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, modalityFrequencyByCluster->GetFrequencyVectorAt(idxCluster));
		for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {
			double Pkc = (double)fv->GetFrequencyVector()->GetAt(i) / (double)globalCluster->GetFrequency();
			if (Pkc > 0)
				Hkc += (Pkc * log(Pkc));
		}
	}
	Hkc = -Hkc;

	if ((Hk + Hc) == 0)
		dVariationOfInformation = 0;
	else
		dVariationOfInformation = ((2 * Hkc) / (Hk + Hc)) - 1;

	delete modalityFrequencyByCluster;

}

double KMClusteringQuality::ComputeARIFactorial(long int n, int k) {

	double d = KWStat::LnFactorial(n) - KWStat::LnFactorial(k) - KWStat::LnFactorial(n - k);
	return exp(d);
}

double KMClusteringQuality::ComputeEVA(const int K, KMCluster* globalCluster, const int nbTargetModalities) {

	/* formule de calcul :

	EVA(K) = sousPartie1 + sousPartie2 + sousPartie3

	avec :

	sousPartie1 = log(N) + logf(N+K-1) - logf(K) - logf(N-1)
	sousPartie2 = somme(k=1 � K) [logf(Nk + J - 1) - logf (J - 1) - logf(Nk)]
	sousPartie3 = somme(k=1 � K) [logf(Nk) - [somme(j=1 � J) logf(Nkj)] ]

	et :

	K = nombre de clusters
	N = nombre total d'instances de la base
	J = nombre de classes du mod�le (c'est � dire le nombre de modalit�s diff�rentes pour la variable cible). Attention, en test, les modalites peuvent etre differentes, et l'EVA vaudra alors 0
	Nk = nombre d'instances dans le cluster k
	Nkj = nombre d'instances dans le cluster k, qui sont de la classe j

	*/

	assert(K <= clusters->GetSize());
	assert(globalCluster != NULL);
	const int N = globalCluster->GetFrequency();
	assert(N > 0);
	const int J = nbTargetModalities;
	assert(J > 0);

	if (globalCluster->GetTargetProbs().GetSize() == 0)
		InitializeGlobalTargetProbs(globalCluster, nbTargetModalities);

	assert(globalCluster->GetTargetProbs().GetSize() == J);

	if (K == 1) {  // cas particulier, plus simple

		double result = log((double)N) +
			KWStat::LnFactorial(N) -
			KWStat::LnFactorial(N - 1) +
			KWStat::LnFactorial(N + J - 1) -
			KWStat::LnFactorial(J - 1);

		for (int j = 0; j < J; j++) {
			const int Nj = (int)(globalCluster->GetTargetProbs().GetAt(j) * N);
			result -= KWStat::LnFactorial(Nj);
		}

		return result;
	}

	// si K > 1 :

	// calcul de sousPartie1 :
	const double sousPartie1 = log((double)N) +
		KWStat::LnFactorial(N + K - 1) -
		KWStat::LnFactorial(K) -
		KWStat::LnFactorial(N - 1);

	// calcul de sousPartie2 :
	double sousPartie2 = 0.0;

	for (int i = 0; i < K; i++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(i));
		assert(cluster != NULL);

		if (cluster->GetFrequency() == 0)
			// cas d'un cluster devenu vide lors de l'evaluation de test
			continue;

		sousPartie2 += (KWStat::LnFactorial(cluster->GetFrequency() + J - 1) -
			KWStat::LnFactorial(J - 1) -
			KWStat::LnFactorial(cluster->GetFrequency()));

	}

	// calcul de sousPartie3 :
	double sousPartie3 = 0.0;

	for (int i = 0; i < K; i++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(i));
		assert(cluster != NULL);

		if (cluster->GetFrequency() == 0)
			// cas d'un cluster devenu vide lors de l'evaluation de test
			continue;

		double sumJ = 0.0;

		int instancesNumber = 0;

		for (int j = 0; j < J; j++) {
			const int NKj = (int)((cluster->GetTargetProbs().GetAt(j) * cluster->GetFrequency()) + 0.5); // le 0.5 sert a arrondir a l'entier le plus proche
			sumJ += KWStat::LnFactorial(NKj);
			instancesNumber += NKj;
		}

		if (instancesNumber != cluster->GetFrequency()) {
			// ne doit pas arriver, sauf s'il y a des valeurs de modalites cibles qui apparaissent en test, et qui etaient inconnues en train
			AddWarning("EVA computing on cluster " + ALString(IntToString(i)) +
				" : unreferenced target values have been detected. Setting EVA to zero.");
			return KWContinuous::GetMinValue();
		}

		sousPartie3 += KWStat::LnFactorial(cluster->GetFrequency());
		sousPartie3 -= sumJ;
	}

	return (sousPartie1 + sousPartie2 + sousPartie3);

}


double KMClusteringQuality::ComputeEVAFirstTerm(const int K, KWFrequencyTable* clustersFrequenciesByModalities) {

	// sousPartie1 = log(N) + logf(N+K-1) - logf(K) - logf(N-1)

	assert(clustersFrequenciesByModalities != NULL);
	assert(clustersFrequenciesByModalities->GetFrequencyVectorNumber() > 0);
	assert(clustersFrequenciesByModalities->GetFrequencyVectorSize() > 0);
	assert(K <= clustersFrequenciesByModalities->GetFrequencyVectorNumber());
	const int N = clustersFrequenciesByModalities->GetTotalFrequency();
	assert(N > 0);

	const double result = log((double)N) +
		KWStat::LnFactorial(N + K - 1) -
		KWStat::LnFactorial(K) -
		KWStat::LnFactorial(N - 1);

	return result;
}


double KMClusteringQuality::ComputeEVASecondTerm(const int K, KWFrequencyTable* clustersFrequenciesByModalities) {

	// sousPartie2 = somme(k = 1 � K)[logf(Nk + J - 1) - logf(J - 1) - logf(Nk)]

	assert(clustersFrequenciesByModalities != NULL);
	assert(clustersFrequenciesByModalities->GetFrequencyVectorNumber() > 0);
	assert(clustersFrequenciesByModalities->GetFrequencyVectorSize() > 0);
	assert(K <= clustersFrequenciesByModalities->GetFrequencyVectorNumber());
	assert(clustersFrequenciesByModalities->GetTotalFrequency() > 0);
	const int J = clustersFrequenciesByModalities->GetFrequencyVectorSize();
	assert(J > 0);

	double result = 0.0;

	if (K == 1) {
		result = KWStat::LnFactorial(clustersFrequenciesByModalities->GetTotalFrequency() + J - 1) -
			KWStat::LnFactorial(J - 1) -
			KWStat::LnFactorial(clustersFrequenciesByModalities->GetTotalFrequency());
	}
	else {

		for (int i = 0; i < clustersFrequenciesByModalities->GetFrequencyVectorNumber(); i++) {

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, clustersFrequenciesByModalities->GetFrequencyVectorAt(i));

			longint sourceFrequency = 0;
			for (int iTarget = 0; iTarget < fv->GetSize(); iTarget++) {
				sourceFrequency += fv->GetFrequencyVector()->GetAt(iTarget);
			}

			if (sourceFrequency == 0)
				continue;

			result += (KWStat::LnFactorial(sourceFrequency + J - 1) -
				KWStat::LnFactorial(J - 1) -
				KWStat::LnFactorial(sourceFrequency));

		}
	}

	return result;

}


double KMClusteringQuality::ComputeEVAThirdTerm(const int K, KWFrequencyTable* clustersFrequenciesByModalities) {

	// sousPartie3 = somme(k=1 � K) [logf(Nk) - [somme(j=1 � J) logf(Nkj)] ]
	assert(clustersFrequenciesByModalities != NULL);
	assert(clustersFrequenciesByModalities->GetFrequencyVectorNumber() > 0);
	assert(clustersFrequenciesByModalities->GetFrequencyVectorSize() > 0);
	assert(K <= clustersFrequenciesByModalities->GetFrequencyVectorNumber());
	const int J = clustersFrequenciesByModalities->GetFrequencyVectorSize();
	assert(J > 0);

	double result = 0.0;

	if (K == 1) {

		result = KWStat::LnFactorial(clustersFrequenciesByModalities->GetTotalFrequency());

		double sumJ = 0.0;

		for (int j = 0; j < J; j++) {

			longint targetFrequency = 0;

			for (int i = 0; i < clustersFrequenciesByModalities->GetFrequencyVectorNumber(); i++) {
				KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, clustersFrequenciesByModalities->GetFrequencyVectorAt(i));
				targetFrequency += fv->GetFrequencyVector()->GetAt(j);
			}
			const int NKj = targetFrequency;
			sumJ += KWStat::LnFactorial(NKj);
		}
		result -= sumJ;
	}
	else {

		for (int i = 0; i < clustersFrequenciesByModalities->GetFrequencyVectorNumber(); i++) {

			KWDenseFrequencyVector* fv = cast(KWDenseFrequencyVector*, clustersFrequenciesByModalities->GetFrequencyVectorAt(i));

			longint sourceFrequency = 0;
			for (int iTarget = 0; iTarget < fv->GetSize(); iTarget++) {
				sourceFrequency += fv->GetFrequencyVector()->GetAt(iTarget);
			}

			if (sourceFrequency == 0)
				continue;

			double sumJ = 0.0;

			int instancesNumber = 0;

			for (int j = 0; j < J; j++) {
				const int NKj = fv->GetFrequencyVector()->GetAt(j);
				sumJ += KWStat::LnFactorial(NKj);
				instancesNumber += NKj;
			}

			if (instancesNumber != sourceFrequency) {
				// ne doit pas arriver, sauf s'il y a des valeurs de modalites cibles qui apparaissent en test, et qui etaient inconnues en train
				AddWarning("EVA computing on cluster " + ALString(IntToString(i)) +
					" : unreferenced target values have been detected. Setting EVA to zero.");
				return KWContinuous::GetMinValue();
			}

			result += KWStat::LnFactorial(sourceFrequency);
			result -= sumJ;
		}
	}

	return result;

}

double KMClusteringQuality::ComputeLEVA(const int K, KMCluster* globalCluster, const ObjectArray& oaTargetAttributeValues) {

	/* formule de calcul :

	LEVA(K) = somme(k=1 � K) [logf(Nk) - [somme(j=1 � J) logf(Nkj)] ]

	avec :

	K = nombre de clusters
	N = nombre total d'instances de la base
	J = nombre de classes du mod�le (c'est � dire le nombre de modalit�s diff�rentes pour la variable cible). Attention, en test, les modalites peuvent etre differentes, et l'EVA vaudra alors 0
	Nk = nombre d'instances dans le cluster k
	Nkj = nombre d'instances dans le cluster k, qui sont de la classe j

	*/

	assert(K <= clusters->GetSize());
	assert(globalCluster != NULL);
	const int N = globalCluster->GetFrequency();
	assert(N > 0);
	const int J = oaTargetAttributeValues.GetSize();
	assert(J > 0);

	if (globalCluster->GetTargetProbs().GetSize() == 0)
		InitializeGlobalTargetProbs(globalCluster, oaTargetAttributeValues.GetSize());

	assert(globalCluster->GetTargetProbs().GetSize() == J);

	if (K == 1) {  // cas particulier, plus simple

		double result = KWStat::LnFactorial(N);

		for (int j = 0; j < J; j++) {
			const int Nj = (int)(globalCluster->GetTargetProbs().GetAt(j) * N);
			result -= KWStat::LnFactorial(Nj);
		}

		return result;
	}

	// si K > 1 :

	double result = 0.0;

	for (int i = 0; i < K; i++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(i));
		assert(cluster != NULL);

		if (cluster->GetFrequency() == 0)
			// cas d'un cluster devenu vide lors de l'evaluation de test
			continue;

		double sumJ = 0.0;

		int instancesNumber = 0;

		for (int j = 0; j < J; j++) {
			const int NKj = (int)((cluster->GetTargetProbs().GetAt(j) * cluster->GetFrequency()) + 0.5); // le 0.5 sert a arrondir a l'entier le plus proche
			sumJ += KWStat::LnFactorial(NKj);
			instancesNumber += NKj;
		}

		if (instancesNumber != cluster->GetFrequency()) {
			// ne doit pas arriver, sauf s'il y a des valeurs de modalites cibles qui apparaissent en test, et qui etaient inconnues en train
			AddWarning("LEVA computing on cluster " + ALString(IntToString(i)) +
				" : unreferenced target values have been detected. Setting LEVA to zero.");
			return KWContinuous::GetMinValue();
		}

		result += KWStat::LnFactorial(cluster->GetFrequency());
		result -= sumJ;
	}

	return (result);

}

void KMClusteringQuality::CopyFrom(const KMClusteringQuality* aSource)
{
	require(aSource != NULL);

	clusters = aSource->clusters;
	parameters = aSource->parameters;
	dEVA = aSource->dEVA;
	dLEVA = aSource->dLEVA;
	dVariationOfInformation = aSource->dVariationOfInformation;
	dPredictiveClustering = aSource->dPredictiveClustering;
	dARIByClusters = aSource->dARIByClusters;
	dNormalizedMutualInformationByClusters = aSource->dNormalizedMutualInformationByClusters;
	dNormalizedMutualInformationByClasses = aSource->dNormalizedMutualInformationByClasses;
	dARIByClasses = aSource->dARIByClasses;
	dDaviesBouldin = aSource->dDaviesBouldin;
	cvDaviesBouldin.CopyFrom(&aSource->cvDaviesBouldin);
}

void KMClusteringQuality::InitializeGlobalTargetProbs(KMCluster* globalCluster, const int nbTargetModalities) {

	// mettre a jour le cluster global avec les probas des modalites cibles
	// a partir de probas comprises entre 0 et 1, pour chaque modalit� et chaque cluster, on doit
	// d'abord reconstituer le nombre global d'instances pour chaque modalit�, puis traduire cette info en proba

	assert(globalCluster != NULL);
	assert(nbTargetModalities > 0);

	ContinuousVector globalTargetprobs;
	globalTargetprobs.SetSize(nbTargetModalities);
	globalTargetprobs.Initialize();

	for (int i = 0; i < clusters->GetSize(); i++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(i));

		for (int j = 0; j < cluster->GetTargetProbs().GetSize(); j++) {
			Continuous c = (cluster->GetFrequency() == 0 ? 0 : cluster->GetTargetProbs().GetAt(j) * cluster->GetFrequency());
			globalTargetprobs.SetAt(j, globalTargetprobs.GetAt(j) + c);
		}
	}

	// traduire le nombre d'instances en proba comprise entre 0 et 1
	for (int i = 0; i < globalTargetprobs.GetSize(); i++) {
		globalTargetprobs.SetAt(i, globalTargetprobs.GetAt(i) / globalCluster->GetFrequency());
	}

	globalCluster->SetTargetProbs(globalTargetprobs);
}

boolean KMClusteringQuality::CheckHuygensTheoremCorrectness(KMCluster* globalCluster) const {

	// verifier que l'inertie totale d'un clustering correspond bien a la somme des inerties intra et inter de chaque cluster

	if (clusters == NULL or clusters->GetSize() == 0)
		return false;

	assert(globalCluster != NULL);
	require(parameters->GetDistanceType() == KMParameters::L2Norm);

	ContinuousVector clustersInertiesIntra;
	clustersInertiesIntra.SetSize(clusters->GetSize());
	clustersInertiesIntra.Initialize();

	ContinuousVector clustersInertiesInter;
	clustersInertiesInter.SetSize(clusters->GetSize());
	clustersInertiesInter.Initialize();

	Continuous inertyTotal = 0;

	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

		KMCluster* cluster = cast(KMCluster*, clusters->GetAt(idxCluster));

		// cluster->ComputeIterationStatistics(); // necessaire si une reaffectation des instances a �t� faite, et que des instances ont change de cluster. A voir.

		// calcul de l'inertie intra du cluster
		Continuous inertyIntra = 0;

		NUMERIC key;
		Object* oCurrent;
		POSITION position = cluster->GetStartPosition();

		while (position != NULL) {

			cluster->GetNextAssoc(position, key, oCurrent);
			KWObject* currentInstance = static_cast<KWObject *>(oCurrent);

			if (currentInstance == NULL)
				continue;

			for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++) {

				const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
				if (not loadIndex.IsValid())
					continue;

				const Continuous distanceToCurentCluster = cluster->GetModelingCentroidValues().GetAt(i) - currentInstance->GetContinuousValueAt(loadIndex);
				inertyIntra += pow(distanceToCurentCluster, 2);

				const Continuous distanceToGlobalCluster = globalCluster->GetModelingCentroidValues().GetAt(i) - currentInstance->GetContinuousValueAt(loadIndex);
				inertyTotal += pow(distanceToGlobalCluster, 2);

			}
		}

		clustersInertiesIntra.SetAt(idxCluster, inertyIntra);

		// calcul de l'inertie inter du cluster
		Continuous inertyInter = 0.0;

		for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++) {
			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
			if (not loadIndex.IsValid())
				continue;
			const Continuous distance = cluster->GetModelingCentroidValues().GetAt(i) - globalCluster->GetModelingCentroidValues().GetAt(i);
			inertyInter += pow(distance, 2);
		}
		clustersInertiesInter.SetAt(idxCluster, cluster->GetFrequency() * inertyInter);
	}

	// calcul de l'inertie totale

	// verifier que l'inertie totale precedemment calculee, correspond bien a la somme des inerties intra et inter de chaque cluster, a 1% pres :
	Continuous sumInerties = 0;

	for (int i = 0; i < clusters->GetSize(); i++)
		sumInerties += (clustersInertiesInter.GetAt(i) + clustersInertiesIntra.GetAt(i));

	const Continuous dOnePerCent = sumInerties / 100;

	if (fabs(sumInerties - inertyTotal) > dOnePerCent) {
		if (parameters->GetVerboseMode()) {
			AddSimpleMessage(" ");
			ALString s = "Inerties sum = " + ALString(DoubleToString(sumInerties));
			s += ", total inerty = " + ALString(DoubleToString(inertyTotal));
			s += ". Difference between the 2 is " + ALString(DoubleToString(fabs(sumInerties - inertyTotal)));
			AddSimpleMessage(s);
		}
		return false;
	}
	else
		return true;
}


void KMClusteringQuality::SetClusters(const ObjectArray* o) {
	clusters = o;
}
void KMClusteringQuality::SetParameters(const KMParameters* p) {
	parameters = p;
}


