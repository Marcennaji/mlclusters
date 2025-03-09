// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictorKNNView.h"


KMPredictorKNNView::KMPredictorKNNView()
{
	// Nom de la vue (le meme que celui de l'objet edite)
	sName = KMPredictorKNN::PREDICTOR_NAME;

	// Parametrage principal de l'interface
	SetIdentifier("PredictorKNN");
	SetLabel(KMPredictorKNN::PREDICTOR_NAME);

	GetFieldAt("TrainParameters")->SetVisible(false);

    // Ajout des sous-fiches
	AddCardField(KMParametersView::KMPARAMETERS_KNN_FIELD_NAME, KMParametersView::KMPARAMETERS_LABEL,
								new KMParametersView());
}


KMPredictorKNNView::~KMPredictorKNNView()
{
}


KMPredictorKNNView* KMPredictorKNNView::Create() const
{
	return new KMPredictorKNNView();
}


void KMPredictorKNNView::EventUpdate(Object* object)
{
    KMPredictorKNN * editedObject;

    require(object != NULL);

    editedObject = cast(KMPredictorKNN *, object);

}


void KMPredictorKNNView::EventRefresh(Object* object)
{
    KMPredictorKNN * editedObject;

    require(object != NULL);

    editedObject = cast(KMPredictorKNN *, object);

}


void KMPredictorKNNView::SetObject(Object* object)
{
    KMPredictorKNN * predictor;

	require(object != NULL);

	// Acces a l'objet edite
	predictor = cast(KMPredictorKNN *, object);

	// Appel de la methode ancetre
	KWPredictorView::SetObject(object);

	// Parametrage des sous-fiches
    cast(KMParametersView*, GetFieldAt(KMParametersView::KMPARAMETERS_KNN_FIELD_NAME))->
		SetObject(predictor->GetKMParameters());
}


KMPredictorKNN * KMPredictorKNNView::GetPredictor()
{
    return cast(KMPredictorKNN *, objValue);
}
