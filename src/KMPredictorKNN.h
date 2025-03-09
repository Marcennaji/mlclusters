// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KMPredictor.h"


/////////////////////////////////////////////////////////////////////
/// Classe predicteur KNN (K nearest neighbour)

class KMPredictorKNN : public KMPredictor
{
public:
	// Constructeur
	KMPredictorKNN();
	~KMPredictorKNN();

	// Copie et duplication
	KMPredictorKNN* Clone() const;

	boolean IsTargetTypeManaged(int nType) const;

	KWPredictor* Create() const;

	/** Nom du predicteur */
	const ALString GetName() const;

	/** Prefixe du predicteur, utilisable pour le nommage de la classe en deploiement */
	const ALString GetPrefix() const;

	static const char* PREDICTOR_NAME;

	///////////////////////////////////////////////////////
	//// Implementation
protected:

	/** Redefinition de la methode d'apprentissage */
	virtual boolean InternalTrain();


};
