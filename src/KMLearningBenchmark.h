// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWLearningBenchmark.h"
#include "KMPredictor.h"


////////////////////////////////////////////////////////////
/// Classe KMLearningBenchmark : benchmark pour KMeans

class KMLearningBenchmark : public KWLearningBenchmark
{
public:

	/* rredefinition methode ancetre : Filtrage des predicteurs specifiables pour l'evaluation  */
	const ALString GetPredictorFilter() const;

	virtual void EvaluateExperiment(int nBenchmark, int nPredictor,
		int nValidation, int nFold, IntVector* ivFoldIndexes);


protected:

	/** redefinition methode ancetre */
	virtual void CreateClassifierCriterions();

	/** redefinition methode ancetre */
	virtual void CollectAllClassifierResults(boolean bTrain,
		int nBenchmark, int nPredictor,
		int nExperiment, int nRun,
		KWPredictor* trainedPredictor,
		KWPredictorEvaluation* predictorEvaluation);
};

