// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#pragma once

#include "KMClustering.h"

////////////////////////
/// clustering K-Means selon l'algo des mini-batches
//

class KMClusteringMiniBatch : public KMClustering
{
public:

	KMClusteringMiniBatch(KMParameters*);
	~KMClusteringMiniBatch(void);

	/** calcul K-Means mini-batch : boucle principale d'un traitement de clustering */
	bool ComputeReplicate(KWDatabase* allInstances, const KWAttribute* targetAttribute, const int iMiniBatchesNumber,
		const int originalDatabaseSamplePercentage, const int miniBatchDatabaseSamplePercentage);

	/** calculer les statistiques globales 'a la volee', sur toutes les instances de la base */
	void ComputeGlobalClusterStatistics(KWDatabase* allInstances, const KWAttribute* targetAttribute);

protected:

	/** calcul des stats du cluster global, premiere passe de lecteure de la database */
	void ComputeGlobalClusterStatisticsFirstDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute);

	/** calcul des stats du cluster global, premiere passe de lecteure de la database */
	void ComputeGlobalClusterStatisticsSecondDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute);

	/** en apprentissage, mise a jour de la matrice de confusion "classes majoritaires / classes reelles" */
	void UpdateTrainingConfusionMatrix(const KWObject* kwoObject, const KMCluster* cluster, const KWAttribute* targetAttribute);

	/** finalisation du calcul d'un replicate */
	void FinalizeReplicateComputing(KWDatabase* allInstances, const KWAttribute* targetAttribute);

	/** finalisation du calcul d'un replicate, 1ere passe de lecture de la database */
	void FinalizeReplicateComputingFirstDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute);

	/** finalisation du calcul d'un replicate, 2eme passe de lecture de la database */
	void FinalizeReplicateComputingSecondDatabaseRead(KWDatabase* allInstances, const KWAttribute* targetAttribute);
};





