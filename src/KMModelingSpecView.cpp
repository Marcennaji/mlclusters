// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMModelingSpecView.h"


KMModelingSpecView::KMModelingSpecView()
{
	SetIdentifier("KMModelingSpec");
	SetLabel("KM Specifications");

	GetFieldAt("ConstructionSpec")->SetVisible(false);
	GetFieldAt("AdvancedSpec")->SetVisible(false);


	AddBooleanField("KMeanPredictor", "K-Means predictor", true);
	AddBooleanField("KNNPredictor", "K-nearest neighbor predictor (KNN)", false);

	AddIntField(KMParametersView::K_FIELD_NAME, KMParametersView::K_LABEL, KMParameters::K_DEFAULT_VALUE);
	GetFieldAt(KMParametersView::K_FIELD_NAME)->SetStyle("Spinner");

	cast(UIIntElement*, GetFieldAt(KMParametersView::K_FIELD_NAME))->SetMinValue(1);
	cast(UIIntElement*, GetFieldAt(KMParametersView::K_FIELD_NAME))->SetMaxValue(KMParameters::K_MAX_VALUE);

	// placement des nouveaux champs
	MoveFieldBefore(KMParametersView::K_FIELD_NAME, GetFieldAtIndex(0)->GetIdentifier());
	MoveFieldBefore("KNNPredictor", KMParametersView::K_FIELD_NAME);
	MoveFieldBefore("KMeanPredictor", "KNNPredictor");

	// Declaration des actions
	AddAction("InspectAdvancedParameters", "Advanced clustering parameters",
		(ActionMethod)(&KMModelingSpecView::InspectAdvancedParameters));
	GetActionAt("InspectAdvancedParameters")->SetStyle("Button");

}

void KMModelingSpecView::InspectAdvancedParameters()
{
	KMPredictorView predictorView;
	KMModelingSpec* modelingSpec;

	// Acces a l'objet edite
	modelingSpec = cast(KMModelingSpec*, GetObject());
	check(modelingSpec);

	if (modelingSpec->GetClusteringPredictor() == NULL) {
		AddWarning("No clustering predictor has been activated. Please select a clustering predictor, before inspecting clustering parameters");
		return;
	}

	// Ouverture de la sous-fiche
	predictorView.SetObject(modelingSpec->GetClusteringPredictor());
	predictorView.Open();

	// repercuter la valeur de K dans les specs de modelisation (cette valeur figure a 2 endroits dans l'IHM) :
	modelingSpec->SetKValue(modelingSpec->GetClusteringPredictor()->GetKMParameters()->GetKValue());
}
KMModelingSpecView::~KMModelingSpecView()
{

	//## Custom destructor

	//##
}


void KMModelingSpecView::EventUpdate(Object* object)
{
	KMModelingSpec* editedObject;

	require(object != NULL);

	KWModelingSpecView::EventUpdate(object);
	editedObject = cast(KMModelingSpec*, object);

	editedObject->SetKNNActivated(GetBooleanValueAt("KNNPredictor"));
	editedObject->SetKmeanActivated(GetBooleanValueAt("KMeanPredictor"));
	editedObject->SetKValue(GetIntValueAt(KMParametersView::K_FIELD_NAME));

}


void KMModelingSpecView::EventRefresh(Object* object)
{
	KMModelingSpec* editedObject;

	require(object != NULL);

	KWModelingSpecView::EventRefresh(object);
	editedObject = cast(KMModelingSpec*, object);

	SetBooleanValueAt("KMeanPredictor", editedObject->IsKmeanActivated());
	SetBooleanValueAt("KNNPredictor", editedObject->IsKNNActivated());
	SetIntValueAt(KMParametersView::K_FIELD_NAME, editedObject->GetKValue());

}


const ALString KMModelingSpecView::GetClassLabel() const
{
	return "Enneade modeling spec";
}



//## Method implementation

void KMModelingSpecView::SetObject(Object* object)
{
	KMModelingSpec* pModSpec;

	require(object != NULL);

	// Acces a l'objet edite
	pModSpec = cast(KMModelingSpec*, object);

	// Memorisation de l'objet pour la fiche courante
	UIObjectView::SetObject(object);
}

//##
