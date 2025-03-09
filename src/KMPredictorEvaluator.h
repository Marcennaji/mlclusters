// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWPredictorEvaluator.h"

#include "KMClassifierEvaluation.h"

////////////////////////////////////////////////////////////
// Classe KMPredictorEvaluator
///    Evaluation d'un predicteur KMeans a partir d'une base de donnes en parametre
class KMPredictorEvaluator : public KWPredictorEvaluator
{
public:

	/** redefinition methode ancetre */
	void EvaluatePredictorSpecs();

	/** redefinition methode ancetre */
	void FillEvaluatedPredictorSpecs();

	////////////////////////////////////////////////////////////////
	///// Implementation
protected:

	/** redefinition methode ancetre */
	void BuildEvaluatedTrainedPredictors(ObjectArray* oaEvaluatedTrainedPredictors);

	/** redefinition methode ancetre */
	void EvaluateTrainedPredictors(ObjectArray* oaEvaluatedTrainedPredictors, ObjectArray* oaOutputPredictorEvaluations);

	/** tri du tableau des predicteurs : le predicteur K-Means doit etre le premier de la liste, car c'est au
	* premier predicteur de la liste, que la methode KWPredictorEvaluator::EvaluatePredictors
	* va demander l'ecriture du rapport d'evaluation. */
	void SortPredictors(ObjectArray& oaPredictors);

};

/// classe predicteur externe, utilisee dans le cadre de l'evaluation d'un predicteur appris
class KMPredictorExternal : public KWPredictorExternal
{
public:

	/** redefinition methode ancetre */
	KWPredictorEvaluation* Evaluate(KWDatabase* database);

	virtual boolean IsTargetTypeManaged(int nType) const;

};

