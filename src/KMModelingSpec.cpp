// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMModelingSpec.h"


KMModelingSpec::KMModelingSpec()
{
	bIsKmeanActivated = true;
	bIsKNNActivated = false;
	clusteringPredictor = NULL;
	iKValue = KMParameters::K_DEFAULT_VALUE;
}


KMModelingSpec::~KMModelingSpec()
{
	if (clusteringPredictor != NULL)
		delete clusteringPredictor;
}
KMPredictor* KMModelingSpec::CreateClusteringPredictor() {

	KMParameters* parameters = NULL;

	if (clusteringPredictor != NULL) {
		// sauvegarder l'ancien parametrage afin de le transmettre au nouveau predicteur
		parameters = clusteringPredictor->GetKMParameters()->Clone();
		delete clusteringPredictor;
	}
	clusteringPredictor = NULL;

	if (bIsKNNActivated)
		clusteringPredictor = new KMPredictorKNN;
	else
		if (bIsKmeanActivated)
			clusteringPredictor = new KMPredictor;

	if (clusteringPredictor != NULL) {
		if (parameters != NULL)
			clusteringPredictor->GetKMParameters()->CopyFrom(parameters);
		else {
			clusteringPredictor->GetKMParameters()->SetKValue(iKValue);
		}
	}

	if (parameters != NULL)
		delete parameters;

	return clusteringPredictor;
}


KMModelingSpec* KMModelingSpec::Clone() const
{
	KMModelingSpec* aClone;

	aClone = new KMModelingSpec;

	return aClone;
}


const ALString KMModelingSpec::GetClassLabel() const
{
	return "Clustering specs";
}


//## Method implementation

const ALString KMModelingSpec::GetObjectLabel() const
{
	ALString sLabel;

	return sLabel;
}

void KMModelingSpec::SetKValue(const int i)
{
	iKValue = i;

	if (clusteringPredictor != NULL) {
		clusteringPredictor->GetKMParameters()->SetKValue(iKValue);
	}
}

void KMModelingSpec::SetKNNActivated(boolean bValue)
{
	bIsKNNActivated = bValue;
}

void KMModelingSpec::SetKmeanActivated(boolean bValue)
{
	bIsKmeanActivated = bValue;
}

//##
