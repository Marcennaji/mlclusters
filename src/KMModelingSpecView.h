// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "UserInterface.h"
#include "KMModelingSpec.h"
#include "KWModelingSpecView.h"
#include "KMPredictorView.h"

////////////////////////////////////////////////////////////
/// vue sur les parametres de modele KMeans

class KMModelingSpecView : public KWModelingSpecView
{
public:

	/// Constructeur
	KMModelingSpecView();
	~KMModelingSpecView();

	////////////////////////////////////////////////////////
	// Redefinition des methodes a reimplementer obligatoirement

	/// Mise a jour de l'objet par les valeurs de l'interface
	void EventUpdate(Object* object);

	/// Mise a jour des valeurs de l'interface par l'objet
	void EventRefresh(Object* object);


	/// Libelles utilisateur
	const ALString GetClassLabel() const;

	//## Custom declarations

	/// Inspection des parametres avances
	void InspectAdvancedParameters();

	/// Parametrage de l'objet edite
	void SetObject(Object* object);

	//##
	////////////////////////////////////////////////////////
	//// Implementation
protected:

};

