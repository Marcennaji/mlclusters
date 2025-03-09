// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWTrainedPredictor.h"

#include "KMPredictor.h"
#include "KMCluster.h"

class KMParameters;
class KMClustering;

/////////////////////////////////////////////////////////////////////////////
/// Classifieur issu de l'apprentissage kmean (cas supervisé)
//

class KMTrainedClassifier : public KWTrainedClassifier
{
public:

	KMTrainedClassifier();
	~KMTrainedClassifier();

	/** reconstituer un resultat K-Means a partir du dico de modelisation */
	KMClustering* CreateModelingClustering();

	/** extraire les intervalles/modalites des attributs necessaires, a partir d'un dico */
	void ExtractPartitions(KWClass* aClass);

	/** acces au modele resultant d'un apprentissage KMean, reconstitué a partir d'un dico de modelisation */
	KMClustering* GetModelingClustering() const;

	/////////////////////////////////////////////////////////
	///// Implementation
protected:

	/** a partir d'un modele existant, l'extraire l'information necessaire a la reconstitution d'un clustering,
	sur un attribut de type :
	Continuous	CellIndexPSepalLength	 = CellIndex(PSepalLength, SepalLength)	;*/
	void ExtractSourceConditionalInfoContinuous(const KWAttribute* attribute, const KWAttribute* nativeAttribute);

	/** a partir d'un modele existant, l'extraire l'information necessaire a la reconstitution d'un clustering,
	sur un attribut de type :
	Continuous CellIndexVClass	 = CellIndex(VClass, Class) */
	void ExtractSourceConditionalInfoCategorical(const KWAttribute* attribute, const KWAttribute* nativeAttribute);

	/** retrouver les valeurs cibles, dans un resultat kmean reconstitué a partir d'un dictionnaire de modelisation */
	bool CreateTargetValues();

	/** retrouver les valeurs cibles et les probas de l'apprentissage, a partir d'un attribut classifieur de type global */
	void CreateTargetValuesAndTargetProbs(KWAttribute* classifierAttribute);

	/** modele reconstitué a partir d'un dico de modelisation, ou recuperé a partir d'un apprentissage effectué) */
	KMClustering* kmModelingClustering;

	/** parametres d'un traitement kmean */
	KMParameters* parameters;

};

inline KMClustering* KMTrainedClassifier::GetModelingClustering() const {
	return kmModelingClustering;
}
