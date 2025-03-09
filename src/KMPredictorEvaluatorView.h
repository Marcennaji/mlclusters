// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KWPredictorEvaluatorView.h"
#include "KMPredictorEvaluator.h"

///////////////////////////////////
/// vue sur l'evluation d'un predicteur KMeans
//

class KMPredictorEvaluatorView : public KWPredictorEvaluatorView
{
public:

	// Constructeur
	KMPredictorEvaluatorView();
	~KMPredictorEvaluatorView();

	////////////////////////////////////////////////////////
	// Redefinition des methodes a reimplementer obligatoirement

	/** Mise a jour de l'objet par les valeurs de l'interface */
	void EventUpdate(Object* object);

	/** Mise a jour des valeurs de l'interface par l'objet */
	void EventRefresh(Object* object);

	/** Reimplementation de la methode Open */
	void Open();

	/** Action d'evaluation des predicteurs (avec suivi de progression de la tache) */
	void EvaluatePredictors();

	/** Parametrage de l'objet edite */
	void SetObject(Object* object);

	/** Libelles utilisateur */
	const ALString GetClassLabel() const;

	////////////////////////////////////////////////////////
	//// Implementation
protected:
};

