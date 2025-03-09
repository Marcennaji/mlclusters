// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMPredictorEvaluatorView.h"


KMPredictorEvaluatorView::KMPredictorEvaluatorView()
{
	// Ajout de l'action d'evaluation
    GetActionAt("EvaluatePredictors")->SetActionMethod(
        (ActionMethod)(&KMPredictorEvaluatorView::EvaluatePredictors));

}


KMPredictorEvaluatorView::~KMPredictorEvaluatorView()
{
}


void KMPredictorEvaluatorView::EventUpdate(Object* object)
{
    KMPredictorEvaluator* predictorEvaluator;

    require(object != NULL);

    predictorEvaluator = cast(KMPredictorEvaluator*, object);
    predictorEvaluator->SetEvaluationFileName(GetStringValueAt("EvaluationFileName"));
    predictorEvaluator->SetMainTargetModality(GetStringValueAt("MainTargetModality"));
}


void KMPredictorEvaluatorView::EventRefresh(Object* object)
{
    KMPredictorEvaluator* predictorEvaluator;

    require(object != NULL);

    predictorEvaluator = cast(KMPredictorEvaluator*, object);
    SetStringValueAt("EvaluationFileName", predictorEvaluator->GetEvaluationFileName());
    SetStringValueAt("MainTargetModality", predictorEvaluator->GetMainTargetModality());
}


void KMPredictorEvaluatorView::Open()
{
    KMPredictorEvaluator* predictorEvaluator;
	KWClassDomain* kwcdInitialClassesDomain;
	KWClassDomain* kwcdCurrentDomain;

	// Acces a l'objet edite
	predictorEvaluator = cast(KMPredictorEvaluator*, GetObject());
	check(predictorEvaluator);

	// Acces au domaines de classes courant et initiaux
	kwcdCurrentDomain = KWClassDomain::GetCurrentDomain();
	kwcdInitialClassesDomain = predictorEvaluator->GetInitialClassesDomain();

	// On positionne le domaine des classes initiales comme domaine courant
	// Cela permet ainsi le parameterage de la base d'evaluation par les
	// classes initiales des predicteurs
	if (kwcdInitialClassesDomain != NULL)
		KWClassDomain::SetCurrentDomain(kwcdInitialClassesDomain);

	// Appel de la methode ancetre pour l'ouverture
	UICard::Open();

	// Restitution du dimaine courant
	if (kwcdInitialClassesDomain != NULL)
		KWClassDomain::SetCurrentDomain(kwcdCurrentDomain);
}


void KMPredictorEvaluatorView::EvaluatePredictors()
{
    KMPredictorEvaluator* predictorEvaluator;
	ObjectArray oaEvaluatedPredictors;

	// Recherche de l'objet edite
	predictorEvaluator = cast(KMPredictorEvaluator*, GetObject());
	check(predictorEvaluator);

	// Evaluation
	predictorEvaluator->EvaluatePredictorSpecs();
}


void KMPredictorEvaluatorView::SetObject(Object* object)
{
    KMPredictorEvaluator* predictorEvaluator;

	require(object != NULL);

	// Acces a l'objet edite
	predictorEvaluator = cast(KMPredictorEvaluator*, object);

	// Parametrage des sous fenetres
    cast(KWDatabaseView*, GetFieldAt("EvaluationDatabase"))->
		SetObject(predictorEvaluator->GetEvaluationDatabase());
    cast(KWEvaluatedPredictorSpecArrayView*, GetFieldAt("EvaluatedPredictors"))->
		SetObjectArray(predictorEvaluator->GetEvaluatedPredictorSpecs());

	// Memorisation de l'objet pour la fiche courante
	UIObjectView::SetObject(object);
}


const ALString KMPredictorEvaluatorView::GetClassLabel() const
{
    return "Clustering predictor";
}
