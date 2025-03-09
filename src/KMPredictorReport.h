// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KWPredictorReport.h"
#include "KWLearningSpec.h"

#include "KMClustering.h"
#include "KMPredictor.h"

class KMPredictor;

///////////////////////////////////////////////////////////////////////////////
/// Rapport d'apprentissage pour predicteur KMean

class KMPredictorReport : public KWPredictorReport
{
public:
	// Constructeur
	KMPredictorReport();
	~KMPredictorReport();

	/** fournir le resultat d'un apprentissage kmean */
	void SetTrainedClustering(KMClustering*);

	/** Acces au resultat d'un apprentissage kmean */
	KMClustering* GetTrainedClustering() const;

	const KMPredictor* GetPredictor() const;
	void SetPredictor(const KMPredictor*);

	/** Ecriture d'un rapport detaille du predicteur */
	void WriteReport(ostream& ost);

	// Ecriture JSON du contenu d'un rapport global
	virtual void WriteJSONFullReportFields(JSONFile* fJSON,
		ObjectArray* oaTrainReports);

	/** json specifique KMean */
	void WriteJSONKMeanReport(JSONFile* fJSON);

	/////////////////////////////////////////////////////////
	///// Implementation
protected:

	/** ecriture des centroides du modele obtenu */
	void WriteCentroids(ostream& ost);

	/** ecriture des centroides initiaux */
	void WriteInitialCentroids(ostream& ost);

	/** ecriture des valeurs de la "vraie" instance (pas un centroide virtuel), avec ses valeurs pre-traitees  */
	void WriteCenterRealInstances(ostream& ost);

	/** ecriture des valeurs de la "vraie" instance (pas un centroide virtuel), avec ses valeurs natives (non pre-traitees)  */
	void WriteCenterRealNativeInstances(ostream& ost);

	/** ecriture des levels (mesure du caractere informatif ou non d'une variable) */
	void WriteLevels(ostream& ost);

	/** ecriture des valeurs Davies Bouldin du clustring */
	void WriteDaviesBouldin(ostream& ost);

	////////////////////////////////////////////////////////
	// Gestion d'un rapport JSON

	void WriteJSONCentroids(JSONFile*);
	void WriteJSONInitialCentroids(JSONFile*);
	void WriteJSONCenterRealInstances(JSONFile*);
	void WriteJSONCenterRealNativeInstances(JSONFile*);
	void WriteJSONLevels(JSONFile*);
	void WriteJSONDaviesBouldin(JSONFile*);

	/* resultat d'un apprentisssage kmean */
	KMClustering* kmTrainedClustering;

	const KMPredictor* predictor;
};

inline const KMPredictor* KMPredictorReport::GetPredictor() const {
	return predictor;
}

