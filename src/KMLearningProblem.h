// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWLearningProblem.h"

#include "KMModelingSpec.h"
#include "KMPredictorReport.h"
#include "KMLearningBenchmark.h"
#include "KMPredictorEvaluator.h"
#include "KMAnalysisResults.h"

////////////////////////////////////////////////////////////////
/// Classe KMLearningProblem :  Gestion de l'apprentissage avec kmean
class KMLearningProblem : public KWLearningProblem
{
public:
	/// Constructeur
	KMLearningProblem();
	~KMLearningProblem();

	/** redefinition de la methode afin de pouvoir generer des ConditionalInfo a partir d'un premier Computestats,
	puis de refaire un ComputeStats sur les ConditionalInfo obtenus. Ces derniers seront ensuite pretraites lors de l'apprentissage (centre-reduit ou
	normalisation), comme des attributs natifs */
	virtual void ComputeStats();

	/** Redefinition de la methode pour rechercher le predicteur kmean */
	virtual void CollectPredictors(KWClassStats* classStats, ObjectArray* oaPredictors);

	/** redefinition methode ancetre */
	boolean CheckTargetAttribute() const;

	/** Evaluateur de predicteur */
	KMPredictorEvaluator* GetPredictorEvaluator();

	/** Benchmark de classifier */
	KMLearningBenchmark* GetClassifierBenchmark();

	static void CleanClass(KWClass* kwc);

protected:

	KMLearningBenchmark* classifierBenchmark;

};



////////////////////////////////////////////////////////////////////////////////
/// Classe KMAnalysisSpec : Analysis parameters for K-Means
class KMAnalysisSpec : public KWAnalysisSpec
{
public:
	/// Constructeur
	KMAnalysisSpec();
	~KMAnalysisSpec();

};

