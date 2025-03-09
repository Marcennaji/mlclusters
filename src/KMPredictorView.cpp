// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictorView.h"


KMPredictorView::KMPredictorView()
{
	// Nom de la vue (le meme que celui de l'objet edite)
	sName = KMPredictor::PREDICTOR_NAME;

	// Parametrage principal de l'interface
	SetIdentifier("Predictor");
	SetLabel(KMPredictor::PREDICTOR_NAME);

	GetFieldAt("TrainParameters")->SetVisible(false);

    // Ajout des sous-fiches
	AddCardField(KMParametersView::KMPARAMETERS_FIELD_NAME, KMParametersView::KMPARAMETERS_LABEL,
								new KMParametersView());
}


KMPredictorView::~KMPredictorView()
{
}


KMPredictorView* KMPredictorView::Create() const
{
	return new KMPredictorView();
}


void KMPredictorView::EventUpdate(Object* object)
{
    KMPredictor * editedObject;

    require(object != NULL);

    editedObject = cast(KMPredictor *, object);

}


void KMPredictorView::EventRefresh(Object* object)
{
    KMPredictor * editedObject;

    require(object != NULL);

    editedObject = cast(KMPredictor *, object);

}


void KMPredictorView::SetObject(Object* object)
{
    KMPredictor * predictor;

	require(object != NULL);

	// Acces a l'objet edite
	predictor = cast(KMPredictor *, object);

	// Appel de la methode ancetre
	KWPredictorView::SetObject(object);

	// Parametrage des sous-fiches
    cast(KMParametersView*, GetFieldAt(KMParametersView::KMPARAMETERS_FIELD_NAME))->
		SetObject(predictor->GetKMParameters());
}


KMPredictor * KMPredictorView::GetPredictor()
{
    return cast(KMPredictor *, objValue);
}
