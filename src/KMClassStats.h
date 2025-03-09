// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include <KWClassStats.h>
#include "KMParameters.h"

/// classe servant a la specialisation KMean de l'ecriture du rapport de preparation
class KMClassStats : public KWClassStats
{

public:

	KMClassStats();
	void SetKMParameters(KMParameters*);

	/** Ecriture d'un rapport, accessible uniquement si statistiques calculees */
	virtual void WriteReport(ostream& ost) override;

	/** indiquer le nombre de variables du clustering, a des fins de reporting */
	void SetClusteringVariablesNumber(int);

	/** le nombre de variables du clustering, a des fins de reporting */
	int GetClusteringVariablesNumber() const;

	// Ecriture du contenu d'un rapport JSON
	void WriteJSONFields(JSONFile* fJSON) override;


protected:

	KMParameters* parameters;
	int iClusteringVariablesNumber;
};

