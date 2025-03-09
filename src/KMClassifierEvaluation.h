// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWPredictorEvaluation.h"

#include "KMParameters.h"
#include "KMCluster.h"
#include "KMClustering.h"
#include "KMPredictor.h"
#include "KMTrainedClassifier.h"

class KMPredictor;
class KMTrainedClassifier;
class KMClassifierEvaluationTask;

///////////////////////////////////////////////////////////////////////////////
/// Evaluation d'un classifieur KMean

class KMClassifierEvaluation : public KWClassifierEvaluation
{
public:

	KMClassifierEvaluation();
	~KMClassifierEvaluation();

	/** Type de predicteur */
	int GetTargetType() const;

	/** redefinition methode ancetre */
	virtual void WriteFullReport(ostream& ost,
		const ALString& sEvaluationLabel,
		ObjectArray* oaPredictorEvaluations);

	// Ecriture JSON du contenu d'un rapport global
	virtual void WriteJSONFullReportFields(JSONFile* fJSON,
		const ALString& sEvaluationLabel, ObjectArray* oaPredictorEvaluations);

	/** implementation methode virtuelle */
	virtual void Evaluate(KWPredictor* predictor, KWDatabase* database);

	/** indiquer le nombre d'instances traitees, a des fins de reporting */
	void SetInstanceEvaluationNumber(const longint);

	KMClassifierEvaluationTask* GetClassifierEvaluationTask() const;

protected:

	// Cree un objet de tache parallele pour la sous-traitance de l'evaluation
	virtual KWPredictorEvaluationTask* CreatePredictorEvaluationTask();

	/** ecriture du rapport */
	void WriteKMeanStatistics(ostream& ost);

	/** ecriture des centres de gravite des clusters */
	void WriteClustersGravityCenters(ostream& ost);

	/** rapport JSON : stats KMean */
	void WriteJSONKMeanStatistics(JSONFile*);

	/** rapport JSON : centres de gravite des clusters */
	void WriteJSONClustersGravityCenters(JSONFile*);

	// variables membres
	KMTrainedClassifier* trainedPredictor;

	KMClassifierEvaluationTask* predictorEvaluationTask;

	friend class KMClassifierEvaluationTask;

};

inline int KMClassifierEvaluation::GetTargetType() const {
	return KWType::Symbol;
}

inline KMClassifierEvaluationTask* KMClassifierEvaluation::GetClassifierEvaluationTask() const {
	return predictorEvaluationTask;
}
