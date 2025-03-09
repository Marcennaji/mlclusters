// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once


#include "KMPredictor.h"
#include "KMPredictorKNN.h"
#include "KWModelingSpec.h"


////////////////////////////////////////////////////////////
/// Classe KMModelingSpec :  specifications de modelisation KMeans, selection des predicteurs a utiliser

class KMModelingSpec : public KWModelingSpec
{
public:
	/// Constructeur
	KMModelingSpec();
	~KMModelingSpec();

	/// Duplication
	KMModelingSpec* Clone() const;

	////////////////////////////////////////////////////////
	// Acces aux attributs

	/** selection du predicteur KMeans */
	boolean IsKmeanActivated() const;

	/** selection du predicteur KMeans */
	void SetKmeanActivated(boolean bValue);

	/** selection du predicteur KNN */
	boolean IsKNNActivated() const;

	/** selection du predicteur KNN */
	void SetKNNActivated(boolean bValue);

	void SetKValue(int);
	int GetKValue() const;

	/// Libelles utilisateur
	const ALString GetClassLabel() const;
	const ALString GetObjectLabel() const;

	//## Custom declarations

	KMPredictor* GetClusteringPredictor();

	////////////////////////////////////////////////////////
	//// Implementation
protected:

	KMPredictor* CreateClusteringPredictor();

	boolean bIsKmeanActivated;
	boolean bIsKNNActivated;
	int iKValue;

	KMPredictor* clusteringPredictor;
};


////////////////////////////////////////////////////////////
// Implementations inline

inline boolean KMModelingSpec::IsKmeanActivated() const
{
	return bIsKmeanActivated;
}
inline boolean KMModelingSpec::IsKNNActivated() const
{
	return bIsKNNActivated;
}


inline int KMModelingSpec::GetKValue() const
{
	return iKValue;
}


inline KMPredictor* KMModelingSpec::GetClusteringPredictor()
{
	if (clusteringPredictor == NULL)
		clusteringPredictor = CreateClusteringPredictor();

	return clusteringPredictor;
}

