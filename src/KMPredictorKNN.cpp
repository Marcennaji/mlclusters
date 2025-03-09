// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictorKNN.h"
#include "KMLearningProject.h"


KMPredictorKNN::KMPredictorKNN()
{

}

KMPredictorKNN::~KMPredictorKNN()
{
}

boolean KMPredictorKNN::InternalTrain() {

	assert(parameters != NULL);

	parameters->SetClusteringType(KMParameters::KNN);

	//  un apprentissage KNN necessite obligatoirement les parametres ci-dessous :
	parameters->SetLearningNumberOfReplicates(1);
	parameters->SetMaxIterations(-1);
	parameters->SetReplicatePostOptimization(KMParameters::FastOptimization);
	parameters->SetCentroidType(KMParameters::CENTROID_REAL_INSTANCE_LABEL);

	if (parameters->GetClustersCentersInitializationMethod() == KMParameters::ClustersCentersInitMethod::ClustersCentersInitMethodAutomaticallyComputed)
		parameters->SetClustersCentersInitializationMethod(KMParameters::Random);

	// calcul de K :

	int K = 0;
	const int instancesNumber = GetDatabase()->GetSampleEstimatedObjectNumber();

	if (instancesNumber < 1000)
		K = instancesNumber;
	else {
		K = instancesNumber / log(instancesNumber);
		if (K < 1000)
			K = 1000;
	}
	// sauvegarder la valeur de K demandee par l'utilisateur via l'IHM, qui represente le nombre "plancher" de K (on ne descendra pas en dessous)
	parameters->SetMinKValuePostOptimization(parameters->GetKValue());
	parameters->SetKValue(K);

	return KMPredictor::InternalTrain();
}

KMPredictorKNN* KMPredictorKNN::Clone() const
{
	KMPredictorKNN* aClone;

	aClone = new KMPredictorKNN;
	aClone->CopyFrom(this);
	return aClone;
}

boolean KMPredictorKNN::IsTargetTypeManaged(int nType) const
{
	return (nType == KWType::Symbol);
}

KWPredictor* KMPredictorKNN::Create() const
{
	return new KMPredictorKNN;
}

const ALString KMPredictorKNN::GetPrefix() const
{
	return "KNN";
}

const ALString KMPredictorKNN::GetName() const
{
	return PREDICTOR_NAME;
}


const char* KMPredictorKNN::PREDICTOR_NAME = "KNN";

