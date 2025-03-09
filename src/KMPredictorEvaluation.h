// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWPredictorEvaluation.h"

#include "KMParameters.h"
#include "KMCluster.h"
#include "KMClustering.h"
#include "KMPredictor.h"
#include "KMTrainedPredictor.h"

class KMPredictor;
class KMTrainedPredictor;
class KMPredictorEvaluationTask;

///////////////////////
/// Evaluation d'un predicteur KMeans
//


class KMPredictorEvaluation : public KWPredictorEvaluation
{
public:

	KMPredictorEvaluation();
	~KMPredictorEvaluation();

	/** implementation methode virtuelle ancetre */
	virtual void Evaluate(KWPredictor* predictor, KWDatabase* database);

	/** implementation methode virtuelle ancetre */
	virtual void WriteFullReport(ostream& ost,
		const ALString& sEvaluationLabel,
		ObjectArray* oaPredictorEvaluations);

	/** Ecriture JSON du contenu d'un rapport global */
	virtual void WriteJSONFullReportFields(JSONFile* fJSON,
		const ALString& sEvaluationLabel, ObjectArray* oaPredictorEvaluations);

	/** garder la valeur du nombre d'instances lors de l'evaluation */
	void SetInstanceEvaluationNumber(const longint);

	/** ecriture des centres de gravite globaux */
	static void WriteGlobalGravityCenters(ostream&, const KMClustering*);

	/** tableau des distances inter clusters (matrice), non normalisees */
	static void WriteClustersDistancesUnnormalized(ostream& ost, KMClustering* result);

	/** tableau des distances inter clusters (matrice), normalisees */
	static void WriteClustersDistancesNormalized(ostream& ost, KMClustering* result);

	/** deplacement des centroides obtenus, entre l'apprentissage et le deploiement */
	static void WriteTrainTestCentroidsShifting(ostream& ost, KMClustering* result);

	/** tableau des probas des attributs natifs */
	static void WriteNativeAttributesProbs(ostream& ost, const KMClustering*, const ObjectDictionary& groupedModalitiesFrequencyTables, const ObjectArray* oaAttributesList);

	/** tableau des probas des attributs natifs, cumulatif */
	static void WriteCumulativeNativeAttributesProbs(ostream& ost, const KMClustering*, const ObjectDictionary& groupedModalitiesFrequencyTables, const bool ascending, const ObjectArray* oaAttributesList);

	/** tableau des probas des attributs natifs, % par cluster et par modalite */
	static void WritePercentagePerLineNativeAttributesProbs(ostream& ost, const KMClustering*, const ObjectDictionary& groupedModalitiesFrequencyTables, const ObjectArray* oaAttributesList);

	/** tableau des moyennes de valeurs continues, par cluster, pour chaque attribut natif */
	static void WriteContinuousMeanValues(ostream& ost, const KMClustering* result, const ObjectArray* oaAttributesList);

	/** tableau des medianes de valeurs continues, par cluster, pour chaque attribut natif */
	static void WriteContinuousMedianValues(ostream& ost, const KMClustering* result, const ObjectArray* oaAttributesList, const longint iReadInstancesForMedianComputation, const longint lInstanceEvaluationNumber);

	/** tableau des % d'instances de clusters ayant une valeur de modalité donnée */
	static void WriteCategoricalModeValues(ostream& ost, const KMClustering* result, const ObjectDictionary& atomicModalitiesFrequencyTables, const ObjectArray* oaAttributesList, const KWClass*);

	/** tableau des probas des attributs natifs, % par cluster et par modalite */
	static void WritePercentagePerLineModeValues(ostream& ost, const KMClustering*, const ObjectDictionary& groupedModalitiesFrequencyTables, const ObjectArray* oaAttributesList);

	/** si memoire insuffisante : on calcule les medianes sur un sous-ensemble des donnees */
	static int ComputeReadPercentageForMedianComputation(const boolean bDetailedStatistics, const longint estimatedObjectsNumber, KWClass* kwc);

	static void CleanPredictorClass(KWClass* predictorClass);

	/** gestion du rapport JSON */

	/** tableau des moyennes de valeurs continues, par cluster, pour chaque attribut natif */
	static void WriteJSONContinuousMeanValues(JSONFile* fJSON, const KMClustering* result, const ObjectArray* oaAttributesList);

	/** tableau des medianes de valeurs continues, par cluster, pour chaque attribut natif */
	static void WriteJSONContinuousMedianValues(JSONFile* fJSON, const KMClustering* result, const ObjectArray* oaAttributesList, const longint iReadInstancesForMedianComputation, const longint lInstanceEvaluationNumber);

	/** tableau des probas des attributs natifs */
	static void WriteJSONNativeAttributesProbs(JSONFile* fJSON, const KMClustering*, const ObjectDictionary& groupedModalitiesFrequencyTables, const ObjectArray* oaAttributesList);

	/** tableau des probas des attributs natifs, % par cluster et par modalite */
	static void WriteJSONPercentagePerLineNativeAttributesProbs(JSONFile* fJSON, const KMClustering*, const ObjectDictionary& groupedModalitiesFrequencyTables, const ObjectArray* oaAttributesList);


protected:



	// Cree un objet de tache parallele pour la sous-traitance de l'evaluation
	virtual KWPredictorEvaluationTask* CreatePredictorEvaluationTask();

	/** ecriture du rapport d'evaluation */
	void WriteKMeanStatistics(ostream& ost);

	/** ecriture des centres de gravites des clusters */
	void WriteClustersGravityCenters(ostream&);

	// gestion du rapport JSON
	void WriteJSONKMeanStatistics(JSONFile*);
	void WriteJSONClustersGravityCenters(JSONFile*);

	// ----------------- variables membres ---------------------

	KMPredictorEvaluationTask* predictorEvaluationTask;

	KMTrainedPredictor* trainedPredictor;

	friend class KMPredictorEvaluationTask;
};

