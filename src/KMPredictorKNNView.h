// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "UserInterface.h"
#include "KWPredictorView.h"
#include "KMPredictorKNN.h"
#include "KMParametersView.h"
#include "KMModelingSpecView.h"


////////////////////////////////////////////////////////////////////////////////
/// Vue sur le parametrage specifique d'un classifieur KNN
class KMPredictorKNNView : public KWPredictorView
{
public:
	// Constructeur
	KMPredictorKNNView();
	~KMPredictorKNNView();

	// Constructeur generique
	KMPredictorKNNView* Create() const;

	// Mise a jour du classifieur specifique par les valeurs de l'interface
	void EventUpdate(Object* object);

	// Mise a jour des valeurs de l'interface par le predicteur specifique
	void EventRefresh(Object* object);

	void SetObject(Object* object);

	/** acces au predicteur KNN */
	KMPredictorKNN* GetPredictor();

};
