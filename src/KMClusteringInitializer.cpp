// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text of which is available
// at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#include "KMClusteringInitializer.h"
#include "KMClustering.h"
#include "KMClusteringQuality.h"
#include "KMRandomInitialisationTask.h"

KMClusteringInitializer::KMClusteringInitializer() {

	clustering = NULL;
	lInstancesWithMissingValues = 0;
}


KMClusteringInitializer::KMClusteringInitializer(KMClustering* _clustering) {

	assert(_clustering != NULL);
	clustering = _clustering;
	lInstancesWithMissingValues = 0;
}


KMClusteringInitializer::~KMClusteringInitializer() {}



boolean KMClusteringInitializer::InitializeClassDecompositionCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute) {

	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	boolean bOk = true;

	// creer les premiers clusters, correspondant aux modalites cibles
	CreateTargetModalitiesClusters(instances, targetAttribute);

	if (clusters->GetSize() == 0) {
		AddWarning("Class decomposition initialization : unable to create any cluster for the existing target modalities (too many missing values in the database ?)");
		bOk = false;
	}

	assert(parameters->GetKValue() >= clusters->GetSize());

	if (bOk) {

		const int nbClutersByTargetAttributeValue = parameters->GetKValue() / clusters->GetSize(); // nbre de clusters a creer pour chaque modalite cible

		if (nbClutersByTargetAttributeValue > 1) {

			ObjectArray* oaTargetModalitiesClusters = new ObjectArray();// copie de travail, car les clusters par modalites cibles seront remplaces
			oaTargetModalitiesClusters->CopyFrom(clusters);
			clusters->RemoveAll();

			ObjectArray* oaNewClusters = new ObjectArray();

			for (int i = 0; i < oaTargetModalitiesClusters->GetSize(); i++) {

				KMCluster* cluster = cast(KMCluster*, oaTargetModalitiesClusters->GetAt(i));
				ClassDecompositionCreateClustersFrom(cluster, nbClutersByTargetAttributeValue);

				// entre chaque iteration, sauvegarder les clusters venant d'etre crees, avant de renititialiser la liste
				// (necessaire pour faire un KMean++ correct sur le nouveau cluster, ne prenant pas en compte les clusters crees au cours des autres iterations)
				for (int iCluster = 0; iCluster < clusters->GetSize(); iCluster++)
					oaNewClusters->Add(clusters->GetAt(iCluster));

				clusters->RemoveAll();
			}

			oaTargetModalitiesClusters->DeleteAll();
			delete oaTargetModalitiesClusters;

			// ajout des clusters crees
			for (int i = 0; i < oaNewClusters->GetSize(); i++)
				clusters->Add(oaNewClusters->GetAt(i));

			delete oaNewClusters;
		}


		if (parameters->GetBisectingVerboseMode() and parameters->GetVerboseMode()) {
			AddSimpleMessage(" ");
			AddSimpleMessage("Regular clustering refinement after class decomposition initialization");
		}
	}
	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	return bOk;
}

boolean KMClusteringInitializer::ClassDecompositionCreateClustersFrom(const KMCluster* originCluster, const int nbClustersToCreate) {

	assert(clustering != NULL);
	boolean bOk = true;
	const KMParameters* parameters = clustering->GetParameters();

	// effectuer une convergence KMean++ a partir du cluster de modalite

	KMParameters bisectingParameters;
	bisectingParameters.CopyFrom(parameters);
	bisectingParameters.SetClustersCentersInitializationMethod(KMParameters::KMeanPlusPlus);
	bisectingParameters.SetReplicateChoice(KMParameters::Distance);
	bisectingParameters.SetMaxIterations(parameters->GetBisectingMaxIterations());
	bisectingParameters.SetVerboseMode(parameters->GetBisectingVerboseMode());
	bisectingParameters.SetKValue(nbClustersToCreate);

	bOk = DoClassDecomposition(bisectingParameters, originCluster);

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	return bOk;

}


boolean KMClusteringInitializer::InitializeVariancePartitioningCentroids(const ObjectArray* instances) {

	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();
	NumericKeyDictionary* instancesToClusters = clustering->GetInstancesToClusters();

	boolean bOk = true;

	// le premier centre est le centre de gravite global des donnees

	KMCluster* globalCluster = new KMCluster(parameters);

	for (int i = 0; i < instances->GetSize(); i++) {

		if (i % 100000 == 0) {
			if (TaskProgression::IsInterruptionRequested())
				break;
			TaskProgression::DisplayProgression((double)i / (double)instances->GetSize() * 100);
		}

		KWObject* instance = cast(KWObject*, instances->GetAt(i));
		if (parameters->HasMissingKMeanValue(instance))
			continue;
		globalCluster->AddInstance(instance);
	}
	globalCluster->ComputeIterationStatistics();
	globalCluster->ComputeInertyIntra(KMParameters::L2Norm);
	clusters->Add(globalCluster);

	boolean bContinue = clusters->GetSize() < parameters->GetKValue() ? true : false;

	while (bContinue) {

		TaskProgression::DisplayProgression((double)clusters->GetSize() / (double)parameters->GetKValue() * 100);
		TaskProgression::DisplayLabel("Clusters initialized : " + ALString(IntToString(clusters->GetSize())) + " on " + ALString(IntToString(parameters->GetKValue())));

		if (TaskProgression::IsInterruptionRequested())
			break;

		double dVarianceMax = 0;
		int idxClusterVarianceMax = 0;

		// trouver le cluster qui a la plus grande variance intra

		for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

			KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));

			const double dVarianceCluster = c->GetInertyIntra(KMParameters::L2Norm);

			if (dVarianceCluster > dVarianceMax) {
				dVarianceMax = dVarianceCluster;
				idxClusterVarianceMax = idxCluster;
			}
		}

		// trouver la variable de ce cluster qui a la plus grande variance

		KMCluster* clusterMaxVariance = cast(KMCluster*, clusters->GetAt(idxClusterVarianceMax));

		//cout << "cluster ayant la variance max (valeur = " << dVarianceMax << ") : " << idxClusterVarianceMax << ", nombre d'instances : " << clusterMaxVariance->GetFrequency() << endl << endl;

		double dAttributeVarianceMax = 0;
		KWLoadIndex loadIndexVarianceMax;

		for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++) {

			const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);

			if (loadIndex.IsValid()) {

				const double dAttributeVariance = clusterMaxVariance->ComputeInertyIntraForAttribute(i, KMParameters::L2Norm);

				if (dAttributeVariance > dAttributeVarianceMax) {
					dAttributeVarianceMax = dAttributeVariance;
					loadIndexVarianceMax = loadIndex;
				}
			}
		}

		//cout << "variable ayant la variance max : " << parameters->GetLoadedAttributeNameByRank(loadIndexVarianceMax) << ", valeur de variance : " << dAttributeVarianceMax << endl << endl;

		// couper le cluster en 2, en fonction de l'attribut de plus forte variance qui vient d'etre trouve.
		// on calcule la moyenne de cette variable, et on separe ensuite les instances en fonction de la valeur de leur variable (superieure ou inferieure a la moyenne)

		const double attributeMeanValue = clusterMaxVariance->ComputeMeanValueForAttribute(loadIndexVarianceMax, KMParameters::L2Norm);

		KMCluster* clusterSup = new KMCluster(parameters);
		KMCluster* clusterInf = new KMCluster(parameters);

		NUMERIC key;
		Object* oCurrent;
		POSITION position = clusterMaxVariance->GetStartPosition();

		while (position != NULL) {

			clusterMaxVariance->GetNextAssoc(position, key, oCurrent);
			KWObject* instance = static_cast<KWObject *>(oCurrent);

			if (instance->GetContinuousValueAt(loadIndexVarianceMax) > attributeMeanValue) {
				clusterSup->AddInstance(instance);
				instancesToClusters->SetAt(instance, clusterSup);
			}
			else {
				clusterInf->AddInstance(instance);
				instancesToClusters->SetAt(instance, clusterInf);
			}
		}

		if (clusterSup->GetCount() > 0 and clusterInf->GetCount() > 0) {

			// mise a jour centroides des nouveaux clusters
			clusterSup->ComputeIterationStatistics();
			clusterInf->ComputeIterationStatistics();

			// recalcul des inerties
			clusterSup->ComputeInertyIntra(KMParameters::L2Norm);
			clusterInf->ComputeInertyIntra(KMParameters::L2Norm);

			clusters->Add(clusterSup);
			clusters->Add(clusterInf);

			// supprimer le cluster qui a ete fractionne en 2, et qui est maintenant remplace par ces deux nouveaux clusters
			delete clusterMaxVariance;
			clusters->RemoveAt(idxClusterVarianceMax);
		}
		else {
			delete clusterSup;
			delete clusterInf;
			bContinue = false;
		}

		if (bContinue)
			bContinue = clusters->GetSize() < parameters->GetKValue() ? true : false;
	}

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	if (bOk) {

		if (clusters->GetSize() < parameters->GetKValue()) {
			AddWarning("Unable to initialize variance partitioning clusters with the requested value for K,  before instances re-assigment.");
			AddSimpleMessage("Found only " + ALString(IntToString(clusters->GetSize())) + " distinct centers.");
			AddSimpleMessage("Possible reasons : too many instances with missing values, or maybe too many instances have the same values.");
			AddSimpleMessage("Hint : decrease K value, or try changing preprocessing parameters.");
			bOk = false;
		}
	}

	return bOk;

}


KMClustering* KMClusteringInitializer::BisectingComputeAllReplicates(ObjectArray* instances, KMParameters& params, const KWAttribute* targetAttribute, const ALString sLabel) {

	assert(instances != NULL);
	assert(instances->GetSize() > 0);

	KMClustering* currentBestClustering = new KMClustering(&params);
	const int nbInstances = instances->GetSize();
	boolean bOk = true;

	if (params.GetKValue() > nbInstances) {
		params.SetKValue(nbInstances);
	}

	int bestExecutionNumber = 1;

	const bool bSelectReplicatesOnEVA = (params.GetReplicateChoice() == KMParameters::EVA ? true : false);
	const bool bSelectReplicatesOnARIByClusters = (params.GetReplicateChoice() == KMParameters::ARIByClusters ? true : false);
	const bool bSelectReplicatesOnNormalizedMutualInformationByClusters = (params.GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClusters ? true : false);
	const bool bSelectReplicatesOnNormalizedMutualInformationByClasses = (params.GetReplicateChoice() == KMParameters::NormalizedMutualInformationByClasses ? true : false);
	const bool bSelectReplicatesOnARIByClasses = (params.GetReplicateChoice() == KMParameters::ARIByClasses ? true : false);
	const bool bSelectReplicatesOnVariationOfInformation = (params.GetReplicateChoice() == KMParameters::VariationOfInformation ? true : false);
	const bool bSelectReplicatesOnLEVA = (params.GetReplicateChoice() == KMParameters::LEVA ? true : false);
	const bool bSelectReplicatesOnDaviesBouldin = (params.GetReplicateChoice() == KMParameters::DaviesBouldin ? true : false);
	const bool bSelectReplicatesOnPredictiveClustering = (params.GetReplicateChoice() == KMParameters::PredictiveClustering ? true : false);

	// on effectue plusieurs calculs kmean successifs (appel�s "replicates"), et on garde le meilleur resultat obtenu
	for (int iNumberOfReplicates = 0; iNumberOfReplicates < params.GetBisectingNumberOfReplicates(); iNumberOfReplicates++) {

		TaskProgression::DisplayProgression((double)iNumberOfReplicates / (double)params.GetBisectingNumberOfReplicates() * 100);

		KMClustering* currentClustering = new KMClustering(&params);

		// si ce n'est pas le premier replicate, recuperer les infos precedemment calculees, et dont ont est
		// sur qu'elles seront identiques lors des replicates suivants, afin de ne pas les recalculer inutilement
		if (iNumberOfReplicates > 0) {

			// recuperer les valeurs de modalit�s de la variable cible (mode supervis�)
			ObjectArray oaTargetAttributeValues;
			for (int i = 0; i < currentBestClustering->GetTargetAttributeValues().GetSize(); i++) {
				StringObject* value = new StringObject;
				value->SetString(cast(StringObject*, currentBestClustering->GetTargetAttributeValues().GetAt(i))->GetString());
				oaTargetAttributeValues.Add(value);
			}
			currentClustering->SetTargetAttributeValues(oaTargetAttributeValues);

			// recuperer les stats du cluster global
			assert(currentBestClustering->GetGlobalCluster() != NULL);
			currentClustering->SetGlobalCluster(currentBestClustering->GetGlobalCluster()->Clone());
		}

		if (params.GetBisectingNumberOfReplicates() > 1 and params.GetBisectingVerboseMode()) {
			AddSimpleMessage(" ");
			AddSimpleMessage(" ");
			AddSimpleMessage(sLabel + " replicate " + ALString(IntToString(iNumberOfReplicates + 1)));
		}

		// calcul kmean
		bOk = currentClustering->ComputeReplicate(instances, targetAttribute);

		if (TaskProgression::IsInterruptionRequested())
			bOk = false;

		if (bOk) {

			if (iNumberOfReplicates == 0) {
				// si c'est le premier replicate, garder en memoire le resultat obtenu
				currentBestClustering->CopyFrom(currentClustering);

			}
			else {

				// si plusieurs replicates ont deja ete effectues, comparer cette execution avec la meilleure conservee auparavant

				bool isBestExecution = false;

				if (currentClustering->GetClusters()->GetSize() == 2) {

					if (currentBestClustering->GetClusters()->GetSize() != 2)
						isBestExecution = true;// le tout premier replicate n'avait pas pu aboutir a 2 clusters. Si ce replicate y est arrive, alors il est forcement meilleur
					else
						// selection du meilleur replicate sur le critere de l'EVA max
						if (bSelectReplicatesOnEVA and currentClustering->GetClusteringQuality()->GetEVA() > currentBestClustering->GetClusteringQuality()->GetEVA())
							isBestExecution = true;
						else
							// selection du meilleur replicate sur le critere de l'ARI max par cluster
							if (bSelectReplicatesOnARIByClusters and currentClustering->GetClusteringQuality()->GetARIByClusters() > currentBestClustering->GetClusteringQuality()->GetARIByClusters())
								isBestExecution = true;
							else
								// selection du meilleur replicate sur le critere NMI par clusters
								if (bSelectReplicatesOnNormalizedMutualInformationByClusters and currentClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters() > currentBestClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())
									isBestExecution = true;
								else
									// selection du meilleur replicate sur le critere NMI par classes
									if (bSelectReplicatesOnNormalizedMutualInformationByClasses and currentClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses() > currentBestClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())
										isBestExecution = true;
									else
										if (bSelectReplicatesOnARIByClasses and currentClustering->GetClusteringQuality()->GetARIByClasses() > currentBestClustering->GetClusteringQuality()->GetARIByClasses())
											isBestExecution = true;
										else
											// selection du meilleur replicate sur le critere de la variation min de l'information
											if (bSelectReplicatesOnVariationOfInformation and currentClustering->GetClusteringQuality()->GetVariationOfInformation() < currentBestClustering->GetClusteringQuality()->GetVariationOfInformation())
												isBestExecution = true;
											else
												// selection du meilleur replicate sur le critere du LEVA max
												if (bSelectReplicatesOnLEVA and currentClustering->GetClusteringQuality()->GetLEVA() > currentBestClustering->GetClusteringQuality()->GetLEVA())
													isBestExecution = true;
												else
													// selection du meilleur replicate sur le critere du Davis Bouldin min
													if (bSelectReplicatesOnDaviesBouldin and currentClustering->GetClusteringQuality()->GetDaviesBouldin() < currentBestClustering->GetClusteringQuality()->GetDaviesBouldin())
														isBestExecution = true;
													else
														// selection du meilleur replicate sur le critere PCC
														if (bSelectReplicatesOnPredictiveClustering and currentClustering->GetClusteringQuality()->GetPredictiveClustering() < currentBestClustering->GetClusteringQuality()->GetPredictiveClustering())
															isBestExecution = true;
														else
															// selection du meilleur replicate sur le critere de la distance min, par defaut
															if (not bSelectReplicatesOnEVA and
																not bSelectReplicatesOnARIByClusters and
																not bSelectReplicatesOnARIByClasses and
																not bSelectReplicatesOnNormalizedMutualInformationByClusters and
																not bSelectReplicatesOnNormalizedMutualInformationByClasses and
																not bSelectReplicatesOnVariationOfInformation and
																not bSelectReplicatesOnLEVA and
																not bSelectReplicatesOnDaviesBouldin and
																not bSelectReplicatesOnPredictiveClustering) {

																if ((currentClustering->GetClustersDistanceSum(params.GetDistanceType()) < currentBestClustering->GetClustersDistanceSum(params.GetDistanceType())
																	or currentBestClustering->GetClustersDistanceSum(params.GetDistanceType()) == 0.0))

																	isBestExecution = true;
															}
				}

				if (isBestExecution) {

					bestExecutionNumber = iNumberOfReplicates + 1;
					currentBestClustering->CopyFrom(currentClustering);// ce resultat est le meilleur observ� jusqu'ici : le conserver

				}
			}
		}
		delete currentClustering;

		if (not bOk)
			break;
	}
	if (bOk and params.GetBisectingNumberOfReplicates() > 1 and params.GetBisectingVerboseMode()) {

		AddSimpleMessage(" ");

		AddSimpleMessage("Best " + sLabel + " replicate is number " + ALString(IntToString(bestExecutionNumber)) + ":");
		AddSimpleMessage("\t- Mean distance is " + ALString(DoubleToString(currentBestClustering->GetMeanDistance())));
		AddSimpleMessage("\t- Davies-Bouldin index is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetDaviesBouldin())));

		if (targetAttribute != NULL) {
			AddSimpleMessage("\t- ARI by clusters is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetARIByClusters())));
			if (bSelectReplicatesOnEVA)
				AddSimpleMessage("\t- EVA is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetEVA())));
			if (bSelectReplicatesOnLEVA)
				AddSimpleMessage("\t- LEVA is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetLEVA())));
			if (bSelectReplicatesOnARIByClasses)
				AddSimpleMessage("\t- ARI by classes is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetARIByClasses())));
			if (bSelectReplicatesOnVariationOfInformation)
				AddSimpleMessage("\t- Variation of information is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetVariationOfInformation())));
			if (bSelectReplicatesOnPredictiveClustering)
				AddSimpleMessage("\t- Predictive clustering value is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetPredictiveClustering())));
			if (bSelectReplicatesOnNormalizedMutualInformationByClusters)
				AddSimpleMessage("\t- NMI by clusters is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClusters())));
			if (bSelectReplicatesOnNormalizedMutualInformationByClasses)
				AddSimpleMessage("\t- NMI by classes is " + ALString(DoubleToString(currentBestClustering->GetClusteringQuality()->GetNormalizedMutualInformationByClasses())));
		}
		AddSimpleMessage(" ");
	}

	if (bOk)
		currentBestClustering->AddInstancesToClusters(instances);// car lors d'un clonage de clusters, les instances sont perdues, on ne conserve que les centroides

	return currentBestClustering;
}


boolean KMClusteringInitializer::DoBisecting(KMParameters& bisectingParameters, const KWAttribute* targetAttribute) {

	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	// tant qu'on n'a pas obtenu le nombre de clusters voulu : creer un nouveau cluster, en
	// partageant en 2 le cluster qui a l'inertie la plus grande, a l'aide d'une convergence KMean classique
	// Si le cluster a partager ne contient qu'un seule classe, on utilise l'initialisation KMean++R pour la convergence. Sinon, on
	// utilise KMean++.

	int currentClustersNumber = clusters->GetSize();
	bool headerDisplayed = false;

	while (currentClustersNumber < parameters->GetKValue()) {

		TaskProgression::DisplayProgression((double)currentClustersNumber / (double)parameters->GetKValue() * 100);
		TaskProgression::DisplayLabel("Clusters initialized : " + ALString(IntToString(currentClustersNumber)) + " on " + ALString(IntToString(parameters->GetKValue())));

		if (TaskProgression::IsInterruptionRequested())
			break;

		if (bisectingParameters.GetVerboseMode() and not headerDisplayed) {
			AddSimpleMessage(" ");
			AddSimpleMessage("--------------------------------------");
			AddSimpleMessage("      Bisecting Initialization");
			AddSimpleMessage("--------------------------------------");
			headerDisplayed = true;
		}

		double maxInertyIntra = 0;
		KMCluster* clusterMaxInertyIntra = NULL;
		int idxClusterMaxInertyIntra = -1;

		if (bisectingParameters.GetVerboseMode()) {
			AddSimpleMessage(" ");
			AddSimpleMessage("Starting cluster(s) :");
		}

		// recherche du cluster de plus grande inertie intra
		for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

			KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));
			assert(c->GetFrequency() > 0);

			if (c->GetInertyIntra(parameters->GetDistanceType()) >= maxInertyIntra) { // attention, on peut avoir des clusters avec inertie intra = 0 (si peu d'elements dans le cluster)
				clusterMaxInertyIntra = c;
				maxInertyIntra = c->GetInertyIntra(parameters->GetDistanceType());
				idxClusterMaxInertyIntra = idxCluster;
			}
			if (bisectingParameters.GetVerboseMode())
				AddSimpleMessage("\tCluster " + c->GetLabel() + " : inerty intra is "
					+ ALString(DoubleToString(c->GetInertyIntra(parameters->GetDistanceType())))
					+ ", instances number is " + ALString(IntToString(c->GetFrequency()))
				);
		}

		assert(clusterMaxInertyIntra != NULL);

		// separer le cluster de plus grande inertie en 2 nouveaux clusters, en effectuant un 2-mean : on se sert du cluster a partager comme dataset initial, et on
		// effectue une convergence dessus, avec K=2. Les 2 clusters obtenus remplaceront le cluster initial)

		ObjectArray oaTargetAttributeValues;
		ObjectArray oaNewDataset;

		NUMERIC key;
		Object* oCurrent;
		POSITION position = clusterMaxInertyIntra->GetStartPosition();

		while (position != NULL) {

			clusterMaxInertyIntra->GetNextAssoc(position, key, oCurrent);
			KWObject* instance = static_cast<KWObject *>(oCurrent);
			oaNewDataset.Add(instance);

			if (targetAttribute != NULL and oaTargetAttributeValues.GetSize() <= 1) {
				// Au passage, si on est en mode supervise, on determine si l'ensemble de donnees contient une ou plusieurs classes, afin d'adapter notre methode d'initialisation

				const ALString sInstanceTargetValue = instance->GetSymbolValueAt(targetAttribute->GetLoadIndex()).GetValue();

				bool found = false;
				for (int i = 0; i < oaTargetAttributeValues.GetSize(); i++) {
					if (cast(StringObject*, oaTargetAttributeValues.GetAt(i))->GetString() == sInstanceTargetValue)
						found = true;
				}
				if (not found) {
					StringObject* value = new StringObject;
					value->SetString(sInstanceTargetValue);
					oaTargetAttributeValues.Add(value);
				}
			}
		}

		if (targetAttribute != NULL) {
			if (oaTargetAttributeValues.GetSize() == 1) {
				bisectingParameters.SetClustersCentersInitializationMethod(KMParameters::KMeanPlusPlus);
			}
			else {
				bisectingParameters.SetClustersCentersInitializationMethod(KMParameters::KMeanPlusPlusR);
			}
		}

		oaTargetAttributeValues.DeleteAll();

		if (bisectingParameters.GetVerboseMode()) {
			AddSimpleMessage(" ");
			AddSimpleMessage("Centroids initialization : computing bisecting replicates on cluster " + clusterMaxInertyIntra->GetLabel() +
				" (" + ALString(IntToString(clusterMaxInertyIntra->GetFrequency())) + " instances)");
			AddSimpleMessage(" ");
			AddSimpleMessage("Bisecting parameters:");
			AddSimpleMessage("K = " + ALString(IntToString(bisectingParameters.GetKValue())));
			AddSimpleMessage("Distance norm: " + ALString(parameters->GetDistanceTypeLabel()));
			AddSimpleMessage("Clusters initialization: " + ALString(bisectingParameters.GetClustersCentersInitializationMethodLabel()));
			AddSimpleMessage("Number of replicates: " + ALString(IntToString(bisectingParameters.GetBisectingNumberOfReplicates())));
			AddSimpleMessage("Best bisecting replicate is based on " + ALString(bisectingParameters.GetReplicateChoiceLabel()));
			AddSimpleMessage("Max iterations number: " + ALString(IntToString(bisectingParameters.GetMaxIterations())));
			AddSimpleMessage("Centroids type: " + ALString(bisectingParameters.GetCentroidTypeLabel()));
			AddSimpleMessage("Continuous preprocessing: " + ALString(bisectingParameters.GetContinuousPreprocessingTypeLabel(true)));
			AddSimpleMessage("Categorical preprocessing: " + ALString(bisectingParameters.GetCategoricalPreprocessingTypeLabel(true)));
		}

		KMClustering* bestClustering = BisectingComputeAllReplicates(&oaNewDataset, bisectingParameters, targetAttribute, "bisecting");

		if (bestClustering->GetClusters()->GetSize() != 2) {
			AddWarning("Bisecting initialization : unable to split cluster " + ALString(IntToString(idxClusterMaxInertyIntra + 1)) + ", won't try to split next clusters.");
			delete bestClustering;
			break;
		}

		KMCluster* result1 = cast(KMCluster*, bestClustering->GetClusters()->GetAt(0));
		KMCluster* result2 = cast(KMCluster*, bestClustering->GetClusters()->GetAt(1));

		result1->ComputeIterationStatistics();
		result2->ComputeIterationStatistics();

		result1->SetLabel(clusterMaxInertyIntra->GetLabel() + "_1");
		result2->SetLabel(clusterMaxInertyIntra->GetLabel() + "_2");

		// remplacer l'ancien cluster par les 2 nouveaux
		delete clusterMaxInertyIntra;
		clusters->SetAt(idxClusterMaxInertyIntra, result1->Clone());
		clusters->Add(result2->Clone());

		KMCluster* cluster1 = cast(KMCluster*, clusters->GetAt(idxClusterMaxInertyIntra));
		KMCluster* cluster2 = cast(KMCluster*, clusters->GetAt(clusters->GetSize() - 1));

		// recuperer les instances (perdues lors du clonage)
		cluster1->CopyInstancesFrom(result1);
		cluster2->CopyInstancesFrom(result2);

		// l'ajout de ces instances ne necessite pas de recalculer les stats (qui restent identiques aux clusters clones), donc on force le statut des stats a "deja calcule"
		cluster1->SetStatisticsUpToDate(true);
		cluster2->SetStatisticsUpToDate(true);

		// calculer les interties intra pour les 2 nouveaux clusters obtenus
		cluster1->ComputeInertyIntra(parameters->GetDistanceType());
		cluster2->ComputeInertyIntra(parameters->GetDistanceType());

		delete bestClustering;

		currentClustersNumber++;

		if (bisectingParameters.GetVerboseMode()) {
			AddSimpleMessage("--------------------------------------");
		}
	}
	if (TaskProgression::IsInterruptionRequested())
		return false;
	else
		return true;

}

boolean KMClusteringInitializer::DoClassDecomposition(KMParameters& bisectingParameters, const KMCluster* modalityCluster) {

	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	// effectuer une convergence a partir du cluster de modalite recu en parametre

	ObjectArray oaNewDataset;

	NUMERIC key;
	Object* oCurrent;
	POSITION position = modalityCluster->GetStartPosition();

	while (position != NULL) {

		modalityCluster->GetNextAssoc(position, key, oCurrent);
		KWObject* instance = static_cast<KWObject *>(oCurrent);
		oaNewDataset.Add(instance);
	}

	if (bisectingParameters.GetVerboseMode()) {
		AddSimpleMessage(" ");
		AddSimpleMessage("Centroids initialization : computing class decomposition replicates on cluster " + modalityCluster->GetLabel() +
			" (" + ALString(IntToString(modalityCluster->GetFrequency())) + " instances)");
		AddSimpleMessage(" ");
		AddSimpleMessage("Class decomposition parameters:");
		AddSimpleMessage("K = " + ALString(IntToString(bisectingParameters.GetKValue())));
		AddSimpleMessage("Distance norm: " + ALString(parameters->GetDistanceTypeLabel()));
		AddSimpleMessage("Clusters initialization: " + ALString(bisectingParameters.GetClustersCentersInitializationMethodLabel()));
		AddSimpleMessage("Number of replicates: " + ALString(IntToString(bisectingParameters.GetBisectingNumberOfReplicates())));
		AddSimpleMessage("Best class decomposition replicate is based on " + ALString(bisectingParameters.GetReplicateChoiceLabel()));
		AddSimpleMessage("Max iterations number: " + ALString(IntToString(bisectingParameters.GetMaxIterations())));
		AddSimpleMessage("Centroids type: " + ALString(bisectingParameters.GetCentroidTypeLabel()));
		AddSimpleMessage("Continuous preprocessing: " + ALString(bisectingParameters.GetContinuousPreprocessingTypeLabel(true)));
		AddSimpleMessage("Categorical preprocessing: " + ALString(bisectingParameters.GetCategoricalPreprocessingTypeLabel(true)));
	}

	KMClustering* bestClustering = BisectingComputeAllReplicates(&oaNewDataset, bisectingParameters, NULL, "class decomposition");

	for (int i = 0; i < bestClustering->GetClusters()->GetSize(); i++) {

		KMCluster* result = cast(KMCluster*, bestClustering->GetClusters()->GetAt(i));
		result->SetParameters(modalityCluster->GetParameters());
		result->ComputeIterationStatistics();
		result->SetLabel(modalityCluster->GetLabel() + "_" + ALString(IntToString(i + 1)));
		clusters->Add(result->Clone());
	}

	delete bestClustering;

	if (bisectingParameters.GetVerboseMode()) {
		AddSimpleMessage("--------------------------------------");
	}

	if (TaskProgression::IsInterruptionRequested())
		return false;
	else
		return true;
}

boolean KMClusteringInitializer::InitializeBisectingCentroidsUnsupervised(const ObjectArray* instances)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);
	const KMParameters* parameters = clustering->GetParameters();
	ObjectArray* clusters = clustering->GetClusters();
	assert(clusters->GetSize() == 0);

	boolean bOk = true;

	KMParameters bisectingParameters;
	bisectingParameters.CopyFrom(parameters);
	bisectingParameters.SetClustersCentersInitializationMethod(KMParameters::KMeanPlusPlus);
	bisectingParameters.SetReplicateChoice(KMParameters::Distance);
	bisectingParameters.SetMaxIterations(parameters->GetBisectingMaxIterations());
	bisectingParameters.SetVerboseMode(parameters->GetBisectingVerboseMode());
	bisectingParameters.SetKValue(2);

	// commencer a partir du cluster global :
	KMCluster* globalCluster = new KMCluster(&bisectingParameters);

	for (int i = 0; i < instances->GetSize(); i++) {

		if (i % 100000 == 0) {
			if (TaskProgression::IsInterruptionRequested())
				break;
			TaskProgression::DisplayProgression((double)i / (double)instances->GetSize() * 100);
		}

		KWObject* instance = cast(KWObject*, instances->GetAt(i));
		if (bisectingParameters.HasMissingKMeanValue(instance))
			continue;
		globalCluster->AddInstance(instance);
	}

	globalCluster->ComputeIterationStatistics();
	globalCluster->ComputeInertyIntra(parameters->GetDistanceType());
	globalCluster->SetLabel("global");

	clusters->Add(globalCluster);

	bOk = DoBisecting(bisectingParameters, NULL);

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	return bOk;

}
boolean KMClusteringInitializer::InitializeBisectingCentroidsSupervised(const ObjectArray* instances, const KWAttribute* targetAttribute)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	boolean bOk = true;

	// creer les premiers clusters, correspondant aux modalites cibles
	CreateTargetModalitiesClusters(instances, targetAttribute);

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	if (clusters->GetSize() == 0) {
		AddWarning("Bisecting initialization : unable to create any cluster for the existing target modalities (too many missing values in the database ?)");
		bOk = false;
	}

	if (bOk) {

		for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

			KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));
			assert(c->GetFrequency() > 0);
			c->ComputeInertyIntra(parameters->GetDistanceType());
		}

		KMParameters bisectingParameters;
		bisectingParameters.CopyFrom(parameters);
		bisectingParameters.SetClustersCentersInitializationMethod(KMParameters::KMeanPlusPlus);
		bisectingParameters.SetReplicateChoice(KMParameters::Distance);
		bisectingParameters.SetMaxIterations(parameters->GetBisectingMaxIterations());
		bisectingParameters.SetVerboseMode(parameters->GetBisectingVerboseMode());
		bisectingParameters.SetKValue(2);

		bOk = DoBisecting(bisectingParameters, targetAttribute);

		if (TaskProgression::IsInterruptionRequested())
			bOk = false;
	}

	return bOk;

}
boolean KMClusteringInitializer::InitializeBisectingCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	boolean bOk = true;

	if (targetAttribute == NULL)
		bOk = InitializeBisectingCentroidsUnsupervised(instances);
	else
		bOk = InitializeBisectingCentroidsSupervised(instances, targetAttribute);

	// retablir le parametrage initial, dans les clusters issus du bisecting :
	for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
		KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));
		c->SetParameters(parameters);
	}

	if (bOk) {
		if (parameters->GetBisectingVerboseMode() and parameters->GetVerboseMode()) {
			AddSimpleMessage(" ");
			AddSimpleMessage("Regular clustering refinement after bisecting initialization");
		}
	}
	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	return bOk;
}


boolean KMClusteringInitializer::InitializeRandomCentroids(const ObjectArray* instances)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	const KMParameters* parameters = clustering->GetParameters();
	boolean bOk = true;

	if (parameters->GetParallelMode())
		bOk = InitializeRandomCentroidsParallelized(instances);
	else
		bOk = InitializeRandomCentroidsNotParallelized(instances);

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	if (bOk)
		TaskProgression::DisplayLabel("Clusters initialization done.");

	return bOk;
}


boolean KMClusteringInitializer::InitializeRandomCentroidsNotParallelized(const ObjectArray* instances) {

	const KMParameters* parameters = clustering->GetParameters();
	ObjectArray* clusters = clustering->GetClusters();
	KWLoadIndex loadIndex;
	bool bIsDuplicate;
	boolean bOk = true;
	ObjectArray existingCenters;
	ObjectArray existingCentersKMeanValues;

	assert(clusters != NULL);
	assert(clusters->GetSize() == 0);

	const int nbKMeanAttributes = parameters->GetKMeanAttributesLoadIndexes().GetSize();

	KWObject* kwoCurrentInstance = NULL;

	// on lit sequentiellement les instances (auparavant deja melangees) pour tenter de les utiliser comme centres
	for (int jInstance = 0; jInstance < instances->GetSize(); jInstance++) {

		kwoCurrentInstance = cast(KWObject*, instances->GetAt(jInstance));

		// verifier si cette instance a une valeur manquante
		if (parameters->HasMissingKMeanValue(kwoCurrentInstance))
			kwoCurrentInstance = NULL;
		else {

			ContinuousVector* currentInstanceValues = new ContinuousVector;
			currentInstanceValues->SetSize(nbKMeanAttributes);
			currentInstanceValues->Initialize();

			for (int i = 0; i < nbKMeanAttributes; i++) {
				loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
				if (loadIndex.IsValid())
					currentInstanceValues->SetAt(i, kwoCurrentInstance->GetContinuousValueAt(loadIndex));
			}

			bIsDuplicate = false;

			// verifier que cette instance ne fait pas doublon avec l'un des centres deja choisis
			// pour ce faire, on calcule la distance entre l'instance et les centres deja repertories. Si on trouve une distance nulle, alors c'est un doublon
			for (int iExistingCenter = 0; iExistingCenter < existingCentersKMeanValues.GetSize(); iExistingCenter++) {

				ContinuousVector* existingCenterValues = cast(ContinuousVector*, existingCentersKMeanValues.GetAt(iExistingCenter));
				const Continuous distance = KMClustering::GetDistanceBetween(*existingCenterValues, *currentInstanceValues, parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());
				if (distance == 0) {
					bIsDuplicate = true;
					break;
				}
			}
			if (bIsDuplicate) {
				kwoCurrentInstance = NULL;// il y a deja un centre repertorie pour ces valeurs, le nouveau centre n'est pas utilisable
				delete currentInstanceValues;
			}
			else {
				// repertorier le nouveau centre et memoriser la somme de ses valeurs
				existingCenters.Add(kwoCurrentInstance);
				existingCentersKMeanValues.Add(currentInstanceValues);
				if (existingCenters.GetSize() >= parameters->GetKValue())
					break;// on a suffisamment de centres
			}
		}
	}

	if (parameters->GetClusteringType() == KMParameters::KMeans and parameters->GetKValue() > existingCenters.GetSize())
		bOk = false;
	else
		if (parameters->GetClusteringType() == KMParameters::KNN and parameters->GetMinKValuePostOptimization() > existingCenters.GetSize())
			bOk = false;


	// creation des clusters proprement dit, a partir des centres trouves
	if (bOk) {
		for (int iExistingCenter = 0; iExistingCenter < existingCenters.GetSize(); iExistingCenter++) {
			KWObject* existingCenter = cast(KWObject*, existingCenters.GetAt(iExistingCenter));
			KMCluster* cluster = new KMCluster(parameters);
			cluster->InitializeModelingCentroidValues(existingCenter); // initialiser le centroide de cluster a partir de l'instance trouv�e
			clusters->Add(cluster);
		}
	}
	else {
		const int requestedKValue = (parameters->GetClusteringType() == KMParameters::KMeans ? parameters->GetKValue() : parameters->GetMinKValuePostOptimization());
		AddWarning("Unable to initialize clustering with the requested value for K (" + ALString(IntToString(requestedKValue)) + "),  before instances re-assigment.");
		AddSimpleMessage("Found only " + ALString(IntToString(existingCenters.GetSize())) + " distinct centers.");
		AddSimpleMessage("Possible reasons : too many instances with missing values, or maybe too many instances have the same values.");
		AddSimpleMessage("Hint : decrease K value, or try changing preprocessing parameters.");
	}

	existingCentersKMeanValues.DeleteAll();

	return bOk;
}

boolean KMClusteringInitializer::InitializeRandomCentroidsParallelized(const ObjectArray* instances) {

	boolean bOk = true;
	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	KMRandomInitialisationTask* initialisationTask = new KMRandomInitialisationTask;
	initialisationTask->SetParameters(parameters);

	//  a partir de la liste des instances recues en parametre, et prealablement melangees (Shuffle), on recree une base de travail qui
	// sera transmise a la tache d'initialisation random des clusters
	KWDatabase* database = CreateDatabaseFromInstances(instances);
	if (not database)
		bOk = false;

	if (bOk) {

		initialisationTask->FindCenters(database);

		if (parameters->GetClusteringType() == KMParameters::KMeans and parameters->GetKValue() > initialisationTask->GetCenters().GetSize())
			bOk = false;
		else
			if (parameters->GetClusteringType() == KMParameters::KNN and parameters->GetMinKValuePostOptimization() > initialisationTask->GetCenters().GetSize())
				bOk = false;

		if (bOk) {
			for (int i = 0; i < initialisationTask->GetCenters().GetSize(); i++) {
				ContinuousVector* cvExistingCenter = cast(ContinuousVector*, initialisationTask->GetCenters().GetAt(i));
				KMCluster* cluster = new KMCluster(parameters);
				cluster->SetModelingCentroidValues(*cvExistingCenter);
				clusters->Add(cluster);
			}
		}
		else {
			const int requestedKValue = (parameters->GetClusteringType() == KMParameters::KMeans ? parameters->GetKValue() : parameters->GetMinKValuePostOptimization());
			AddWarning("Unable to initialize clustering with the requested value for K (" + ALString(IntToString(requestedKValue)) + "),  before instances re-assigment.");
			AddSimpleMessage("Found only " + ALString(IntToString(initialisationTask->GetCenters().GetSize())) + " distinct centers.");
			AddSimpleMessage("Possible reasons : too many instances with missing values, or maybe too many instances have the same values.");
			AddSimpleMessage("Hint : decrease K value, or try changing preprocessing parameters.");
		}

		// nettoyage
		ALString sDatabaseClassName = database->GetClassName();
		database->DeleteAll();
		delete database;
		KWClassDomain::GetCurrentDomain()->DeleteClass(sDatabaseClassName);
		delete initialisationTask;
	}

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	return bOk;

}

KWDatabase* KMClusteringInitializer::CreateDatabaseFromInstances(const ObjectArray* instances) {

	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);
	boolean bOk = true;
	const KMParameters* parameters = clustering->GetParameters();
	KWObject* kwo = cast(KWObject*, instances->GetAt(0));

	KWSTDatabaseTextFile* database = new KWSTDatabaseTextFile;
	database->SetClassName(kwo->GetClass()->GetName());
	const ALString databaseFileName = FileService::CreateTmpFile("KhiopsEnneade_randomDatabase.txt", this);
	database->SetDatabaseName(databaseFileName);

	if (not database->OpenForWrite()) {
		AddError("Can't create database '" + databaseFileName + "'");
		bOk = false;
	}

	if (bOk) {

		for (int i = 0; i < instances->GetSize(); i++) {
			kwo = cast(KWObject*, instances->GetAt(i));
			if (not parameters->HasMissingKMeanValue(kwo))
				database->Write(kwo);
		}
		database->Close();

		// construire une nouvelle classe pour la base que l'on vient d'ecrire sur disque (sans attributs derives)
		ALString sClassName = FileService::GetFilePrefix(database->GetDatabaseName());
		if (sClassName == "")
			sClassName = FileService::GetFileSuffix(database->GetDatabaseName());

		// On recherche un nom de classe nouveau
		sClassName = KWClassDomain::GetCurrentDomain()->BuildClassName(sClassName);

		// Construction effective de la classe
		database->SetClassName(sClassName);
		KWClass* kwc = database->ComputeClass();
		if (kwc == NULL)
			bOk = false;
	}

	if (bOk)
		bOk = database->ReadAll();

	if (not bOk)
		delete database;

	return (bOk ? database : NULL);
}

boolean KMClusteringInitializer::InitializeMinMaxCentroids(const ObjectArray* instances, const boolean isDeterministic) {


	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	boolean bOk = true;

	if (not isDeterministic) {

		// premier centre : tir� au hasard

		KWObject* center = NULL;

		while (center == NULL or parameters->HasMissingKMeanValue(center)) {

			const int randomCenter = RandomInt(instances->GetSize() - 1);
			center = cast(KWObject*, instances->GetAt(randomCenter));
		}
		assert(center != NULL);
		KMCluster* cluster = new KMCluster(parameters);
		cluster->InitializeModelingCentroidValues(center); // initialiser le centroide de cluster a partir de l'instance trouv�e
		clusters->Add(cluster);
	}
	else {

		// premier centre pour la methode deterministe : est le centre de gravite global des donnees

		KMCluster* globalCluster = new KMCluster(parameters);
		for (int i = 0; i < instances->GetSize(); i++) {

			if (i % 100000 == 0) {
				if (TaskProgression::IsInterruptionRequested())
					break;
				TaskProgression::DisplayProgression((double)i / (double)instances->GetSize() * 100);
			}

			KWObject* instance = cast(KWObject*, instances->GetAt(i));
			if (parameters->HasMissingKMeanValue(instance))
				continue;
			globalCluster->AddInstance(instance);
		}
		globalCluster->ComputeIterationStatistics();
		clusters->Add(globalCluster);
	}

	InitializeMinMaxNextCenters(instances);

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	return bOk;

}

void KMClusteringInitializer::InitializeMinMaxNextCenters(const ObjectArray* instances)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	ContinuousVector distances; // contient les distances au centre existant le plus proche, pour chaque instance
	distances.SetSize(instances->GetSize());

	clustering->ComputeClustersCentersDistances(); // calculer la matrice des distances entre centres de clusters

	// calcul des centres suivants, a partir d'un ou plusieurs centres deja calcules

	boolean bContinue = clusters->GetSize() < parameters->GetKValue() ? true : false;

	while (bContinue) {

		if (TaskProgression::IsInterruptionRequested())
			break;

		TaskProgression::DisplayProgression((double)clusters->GetSize() / (double)parameters->GetKValue() * 100);
		TaskProgression::DisplayLabel("Clusters initialized : " + ALString(IntToString(clusters->GetSize())) + " on " + ALString(IntToString(parameters->GetKValue())));

		distances.Initialize(); // remise a zero des distances calculees

		for (int idxInstance = 0; idxInstance < instances->GetSize(); idxInstance++) {

			KWObject* instance = cast(KWObject*, instances->GetAt(idxInstance));

			if (parameters->HasMissingKMeanValue(instance))
				continue;

			double dDistanceMin = -1;

			// trouver la plus petite distance entre cette instance et les centres deja connus
			KMCluster* nearestCluster = clustering->FindNearestCluster(instance);

			double d = nearestCluster->FindDistanceFromCentroid(instance, nearestCluster->GetModelingCentroidValues(), parameters->GetDistanceType());

			if (d != KWContinuous::GetMaxValue()) { // tenir compte du probleme des eventuelles valeurs manquantes

				if (dDistanceMin == -1 or d < dDistanceMin) {
					distances.SetAt(idxInstance, d);
					dDistanceMin = d;
				}
			}
		}

		// choix du centre suivant : c'est l'instance dont la distance a son plus proche centre, est la plus grande parmi toutes les instances

		int idxNewCenter = 0;
		double dDistanceMax = 0;

		for (int idxInstance = 0; idxInstance < distances.GetSize(); idxInstance++) {

			if (distances.GetAt(idxInstance) > dDistanceMax) {
				idxNewCenter = idxInstance;
				dDistanceMax = distances.GetAt(idxInstance);
			}
		}

		KMCluster* cluster = new KMCluster(parameters);
		KWObject* center = cast(KWObject*, instances->GetAt(idxNewCenter));
		assert(center != NULL);
		cluster->InitializeModelingCentroidValues(center); // initialiser le centroide de cluster a partir de l'instance trouv�e
		clusters->Add(cluster);
		clustering->ComputeClustersCentersDistances(); // apres ajout du nouveau cluster, recalculer la matrice des distances entre centres de clusters

		if (bContinue)
			bContinue = clusters->GetSize() < parameters->GetKValue() ? true : false;
	}
}

boolean KMClusteringInitializer::InitializeKMeanPlusPlusCentroids(const ObjectArray* instances)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	assert(clusters->GetSize() == 0);

	boolean bOk = true;

	KWObject* center = NULL;

	while (center == NULL or parameters->HasMissingKMeanValue(center)) {
		// premier centre : tir� au hasard
		const int randomCenter = RandomInt(instances->GetSize() - 1);
		center = cast(KWObject*, instances->GetAt(randomCenter));
	}
	assert(center != NULL);
	KMCluster* cluster = new KMCluster(parameters);
	cluster->InitializeModelingCentroidValues(center); // initialiser le centroide de cluster a partir de l'instance trouv�e
	clusters->Add(cluster);

	InitializeKMeanPlusPlusNextCenters(instances, parameters->GetKValue() - clusters->GetSize());

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	if (clusters->GetSize() < parameters->GetKValue()) {
		AddWarning("Unable to initialize KMean++ clustering with the requested value for K (" + ALString(IntToString(parameters->GetKValue())) + "),  before instances re-assigment.");
		AddSimpleMessage("Found only " + ALString(IntToString(clusters->GetSize())) + " distinct centers.");
		AddSimpleMessage("Possible reasons : too many instances with missing values, or maybe too many instances have the same values.");
		AddSimpleMessage("Hint : decrease K value, or try changing preprocessing parameters.");
		bOk = false;
	}

	return bOk;
}

boolean KMClusteringInitializer::InitializeKMeanPlusPlusRCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();

	boolean bOk = true;

	// creer les premiers clusters, correspondant aux modalites cibles
	CreateTargetModalitiesClusters(instances, targetAttribute);

	if (clusters->GetSize() == 0) {
		AddWarning("KMean++R initialization : unable to create any cluster for the existing target modalities (too many missing values in the database ?)");
		bOk = false;
	}

	if (bOk) {

		// supprimer les instances, ne garder que les centroides
		for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {
			KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));
			c->RemoveAll();
		}

		InitializeKMeanPlusPlusNextCenters(instances, parameters->GetKValue() - clusters->GetSize());
	}
	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	if (clusters->GetSize() < parameters->GetKValue()) {
		AddWarning("Unable to initialize KMean++R clustering with the requested value for K,  before instances re-assigment.");
		AddSimpleMessage("Found only " + ALString(IntToString(clusters->GetSize())) + " distinct centers.");
		AddSimpleMessage("Possible reasons : too many instances with missing values, or maybe too many instances have the same values.");
		AddSimpleMessage("Hint : decrease K value, or try changing preprocessing parameters.");
		bOk = false;
	}

	return bOk;
}

boolean KMClusteringInitializer::InitializeRocchioThenSplitCentroids(const ObjectArray* instances, const KWAttribute* targetAttribute)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();
	NumericKeyDictionary* instancesToClusters = clustering->GetInstancesToClusters();

	boolean bOk = true;

	// creer les premiers clusters, correspondant aux modalites cibles
	CreateTargetModalitiesClusters(instances, targetAttribute);

	if (clusters->GetSize() == 0) {
		AddWarning("Rocchio-Split initialization : unable to create any cluster for the existing target modalities (too many missing values in the database ?)");
		bOk = false;
	}

	if (bOk) {

		// tant qu'on n'a pas obtenu le nombre de clusters voulu : creer un nouveau cluster, en
		// partageant le cluster qui a l'inertie la plus grande
		int currentClustersNumber = clusters->GetSize();

		while (currentClustersNumber < parameters->GetKValue()) {

			if (TaskProgression::IsInterruptionRequested())
				break;

			TaskProgression::DisplayProgression((double)currentClustersNumber / (double)parameters->GetKValue() * 100);
			TaskProgression::DisplayLabel("Clusters initialized : " + ALString(IntToString(currentClustersNumber)) + " on " + ALString(IntToString(parameters->GetKValue())));

			double maxInertyIntra = 0;
			KMCluster* clusterMaxInertyIntra = NULL;
			int idxClusterMaxInertyIntra = -1;

			// calcul des inerties intra
			for (int idxCluster = 0; idxCluster < clusters->GetSize(); idxCluster++) {

				KMCluster* c = cast(KMCluster*, clusters->GetAt(idxCluster));
				assert(c->GetCount() > 0);

				c->ComputeIterationStatistics();

				c->ComputeInertyIntra(parameters->GetDistanceType());

				if (c->GetInertyIntra(parameters->GetDistanceType()) >= maxInertyIntra) { // attention, on peut avoir des clusters avec inertie intra = 0 (si peu d'elements dans le cluster)
					clusterMaxInertyIntra = c;
					maxInertyIntra = c->GetInertyIntra(parameters->GetDistanceType());
					idxClusterMaxInertyIntra = idxCluster;
				}
			}

			assert(clusterMaxInertyIntra != NULL);

			clusterMaxInertyIntra->ComputeInstanceFurthestToCentroid(parameters->GetDistanceType());

			const KMClusterInstance* furthestInstance = clusterMaxInertyIntra->GetInstanceFurthestToCentroid();

			const double distanceMax = clusterMaxInertyIntra->FindDistanceFromCentroid(furthestInstance,
				clusterMaxInertyIntra->GetModelingCentroidValues(),
				parameters->GetDistanceType());

			// stocker les valeurs kmean de l'instance la plus eloignee
			const int nbAttr = furthestInstance->GetLoadedAttributes().GetSize();
			ContinuousVector furthestInstanceValues;
			furthestInstanceValues.SetSize(nbAttr);
			furthestInstanceValues.Initialize();
			for (int i = 0; i < parameters->GetKMeanAttributesLoadIndexes().GetSize(); i++) {
				const KWLoadIndex loadIndex = parameters->GetKMeanAttributesLoadIndexes().GetAt(i);
				if (loadIndex.IsValid()) {
					furthestInstanceValues.SetAt(i, furthestInstance->GetContinuousValueAt(loadIndex));
				}
			}

			// separer le cluster de plus grande inertie en 2 nouveaux clusters : le premier contient les instances ayant
			// une distance � furthestInstance qui est > distanceMax, et le second contient les instances restantes

			KMCluster* clusterSup = new KMCluster(parameters);
			KMCluster* clusterInf = new KMCluster(parameters);

			NUMERIC key;
			Object* oCurrent;
			POSITION position = clusterMaxInertyIntra->GetStartPosition();

			while (position != NULL) {

				clusterMaxInertyIntra->GetNextAssoc(position, key, oCurrent);
				KWObject* instance = static_cast<KWObject *>(oCurrent);

				const double distance = clusterMaxInertyIntra->FindDistanceFromCentroid(instance, furthestInstanceValues, parameters->GetDistanceType());

				if (distance > distanceMax) {
					clusterSup->AddInstance(instance);
					instancesToClusters->SetAt(instance, clusterSup);
				}
				else {
					clusterInf->AddInstance(instance);
					instancesToClusters->SetAt(instance, clusterInf);
				}
			}

			// mise a jour centroides des nouveaux clusters
			clusterSup->ComputeIterationStatistics();
			clusterInf->ComputeIterationStatistics();

			if (clusterSup->GetFrequency() == 0) {
				delete clusterSup;
				currentClustersNumber = parameters->GetKValue(); // aucun fractionnement n'est desormais possible, sortir de la boucle
			}
			else
				clusters->Add(clusterSup);

			if (clusterInf->GetFrequency() == 0) {
				delete clusterInf;
				currentClustersNumber = parameters->GetKValue();
			}
			else
				clusters->Add(clusterInf);

			// supprimer le cluster qui a ete fractionne en 2, et qui est maintenant remplace par ces deux nouveaux clusters
			delete clusterMaxInertyIntra;
			clusters->RemoveAt(idxClusterMaxInertyIntra);

			currentClustersNumber++;
		}
	}

	if (TaskProgression::IsInterruptionRequested())
		bOk = false;

	return bOk;
}

void KMClusteringInitializer::InitializeKMeanPlusPlusNextCenters(const ObjectArray* instances, const int nbCentersToCreate)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();
	KWLoadIndex loadIndex;
	boolean bIsDuplicate;
	assert(clusters->GetSize() > 0); // a ce stade, on a : soit des clusters correspondant a tout ou partie des modalites cibles (KMean++R), soit 1 seul cluster dont le centre a ete tire au hasard
	const int initiallyCreatedClusters = clusters->GetSize();

	ContinuousVector distances; // contient les plus petites distances, pour chaque instance
	distances.SetSize(instances->GetSize());
	distances.Initialize();

	ContinuousVector normalizedDistances; // contient les plus petites distances normalisees, pour chaque instance
	normalizedDistances.SetSize(instances->GetSize());
	normalizedDistances.Initialize();

	clustering->ComputeClustersCentersDistances(); // calculer la matrice des distances entre centres de clusters

	int nbCreatedCenters = 0;

	// calcul des centres suivants, a partir de centres KMean++ ou KMean++R deja calcules

	boolean bContinue = nbCreatedCenters < nbCentersToCreate ? true : false;

	while (bContinue) {

		if (TaskProgression::IsInterruptionRequested())
			break;

		TaskProgression::DisplayProgression((double)nbCreatedCenters / (double)nbCentersToCreate * 100);
		TaskProgression::DisplayLabel("Clusters initialized : " + ALString(IntToString(nbCreatedCenters)) + " on " + ALString(IntToString(nbCentersToCreate + initiallyCreatedClusters)));

		for (int idxInstance = 0; idxInstance < instances->GetSize(); idxInstance++) {

			KWObject* instance = cast(KWObject*, instances->GetAt(idxInstance));

			if (parameters->HasMissingKMeanValue(instance))
				continue;

			double dDistanceMin = -1;

			// trouver la plus petite distance entre cette instance et les centres deja connus
			KMCluster* nearestCluster = clustering->FindNearestCluster(instance);

			double d = nearestCluster->FindDistanceFromCentroid(instance, nearestCluster->GetModelingCentroidValues(), parameters->GetDistanceType());

			if (d != KWContinuous::GetMaxValue()) { // tenir compte du probleme des eventuelles valeurs manquantes

				if (dDistanceMin == -1 or d < dDistanceMin) {
					distances.SetAt(idxInstance, d);
					dDistanceMin = d;
				}
			}
		}

		// normalisation du vecteur des distances
		double distancesSum = 0.0;
		for (int i = 0; i < instances->GetSize(); i++) {
			distancesSum += distances.GetAt(i);
		}

		if (distancesSum > 0.0) {
			for (int i = 0; i < distances.GetSize(); i++) {
				normalizedDistances.SetAt(i, distances.GetAt(i) / distancesSum);
			}
		}
		else
			bContinue = false;

		// tirage nombre aleatoire entre 0 et 1
		double rand = (double)RandomInt(instances->GetSize()) / (double)instances->GetSize();

		// choix du centre suivant
		double sum = 0.0;

		for (int idxInstance = 0; idxInstance < normalizedDistances.GetSize(); idxInstance++) {

			sum += normalizedDistances.GetAt(idxInstance);

			if (sum > rand) {

				KWObject* center = cast(KWObject*, instances->GetAt(idxInstance));
				assert(center != NULL);
				KMCluster* newCluster = new KMCluster(parameters);
				newCluster->InitializeModelingCentroidValues(center); // initialiser le centroide de cluster a partir de l'instance trouv�e

				// detecter si ce nouveau centre potentiel n'a pas deja ete utilise
				bIsDuplicate = false;
				for (int i = 0; i < clusters->GetSize(); i++)
				{
					KMCluster* existingCenter = cast(KMCluster*, clusters->GetAt(i));
					const Continuous distance = KMClustering::GetDistanceBetween(existingCenter->GetModelingCentroidValues(),
						newCluster->GetModelingCentroidValues(), parameters->GetDistanceType(), parameters->GetKMeanAttributesLoadIndexes());
					if (distance == 0) {
						bIsDuplicate = true;
						break;
					}
				}

				if (bIsDuplicate) {
					delete newCluster;
				}
				else {
					clusters->Add(newCluster);
					nbCreatedCenters++;
					clustering->ComputeClustersCentersDistances();// on a ajoute un cluster : recalculer la matrice des distances entre centres de clusters
					break;
				}
			}
		}

		if (bContinue)
			bContinue = nbCreatedCenters < nbCentersToCreate ? true : false;
	}
}

// creer les clusters correspondant aux modalites cibles
void KMClusteringInitializer::CreateTargetModalitiesClusters(const ObjectArray* instances, const KWAttribute* targetAttribute)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);

	assert(targetAttribute != NULL);
	assert(targetAttribute->GetLoadIndex().IsValid());
	assert(clustering != NULL);

	const KMParameters* parameters = clustering->GetParameters();
	const ObjectArray* oaTargetAttributeValues = &clustering->GetTargetAttributeValues();

	assert(oaTargetAttributeValues->GetSize() > 0);

	if (oaTargetAttributeValues->GetSize() > parameters->GetKValue()) {

		// s'il y a plus de modalites cibles que de clusters a creer, alors creer les clusters en priorite a partir des modalites cibles de plus fort prior.

		ObjectArray* targetModalitiesCount = ComputeTargetModalitiesCount(instances, targetAttribute); // retourne un tableau d'objets TargetModalityCount, tries par frequences de modalites decroissantes

		for (int i = 0; i < parameters->GetKValue(); i++) {

			TaskProgression::DisplayProgression((double)i / (double)parameters->GetKValue() * 100);

			TargetModalityCount* count = cast(TargetModalityCount*, targetModalitiesCount->GetAt(i));
			CreateClusterForTargetModality(count->sModality.GetValue(), instances, targetAttribute);
		}

		targetModalitiesCount->DeleteAll();
		delete targetModalitiesCount;
	}
	else {

		// cas standard (K >= C) :

		// pour chaque modalit� cible, creer et remplir le cluster correspondant
		for (int i = 0; i < oaTargetAttributeValues->GetSize(); i++) {

			TaskProgression::DisplayProgression((double)i / (double)oaTargetAttributeValues->GetSize() * 100);

			const StringObject* modalityValue = cast(StringObject*, oaTargetAttributeValues->GetAt(i));
			CreateClusterForTargetModality(modalityValue->GetString(), instances, targetAttribute);

			if (TaskProgression::IsInterruptionRequested())
				break;
		}
	}
}

void KMClusteringInitializer::CreateClusterForTargetModality(const ALString modalityValue, const ObjectArray* instances, const KWAttribute* targetAttribute) {

	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(clustering != NULL);

	ObjectArray* clusters = clustering->GetClusters();
	const KMParameters* parameters = clustering->GetParameters();
	NumericKeyDictionary* instancesToClusters = clustering->GetInstancesToClusters();

	assert(targetAttribute != NULL);
	assert(targetAttribute->GetLoadIndex().IsValid());

	KMCluster* cluster = new KMCluster(parameters);

	for (int i = 0; i < instances->GetSize(); i++)
	{
		if (i % 100000 == 0) {
			if (TaskProgression::IsInterruptionRequested())
				break;
			TaskProgression::DisplayProgression((double)i / (double)instances->GetSize() * 100);
		}

		KWObject* instance = cast(KWObject*, instances->GetAt(i));

		if (parameters->HasMissingKMeanValue(instance))
			continue;

		if (ALString(instance->GetSymbolValueAt(targetAttribute->GetLoadIndex())) == modalityValue) {
			// ajout de l'instance au cluster
			cluster->AddInstance(instance);
			instancesToClusters->SetAt(instance, cluster);
		}
	}

	if (cluster->GetCount() == 0) {
		// aucune instance ayant cette modalit� cible, et n'ayant aucune valeur manquante
		delete cluster;
	}
	else {

		cluster->SetLabel(modalityValue);
		clusters->Add(cluster);

		// calcul du centroide, a partir des instances ajoutees
		cluster->ComputeIterationStatistics();
	}
}

ObjectArray* KMClusteringInitializer::ComputeTargetModalitiesCount(const ObjectArray* instances, const KWAttribute* targetAttribute)
{
	assert(instances != NULL);
	assert(instances->GetSize() > 0);
	assert(targetAttribute != NULL);
	assert(targetAttribute->GetLoadIndex().IsValid());
	assert(clustering != NULL);

	const KMParameters* parameters = clustering->GetParameters();
	const ObjectArray* oaTargetAttributeValues = &clustering->GetTargetAttributeValues();
	assert(oaTargetAttributeValues->GetSize() > 0);

	NumericKeyDictionary targetModalitiesCount;

	for (int i = 0; i < instances->GetSize(); i++)
	{
		if (i % 100000 == 0) {
			TaskProgression::DisplayProgression((double)i / (double)instances->GetSize() * 100);
			if (TaskProgression::IsInterruptionRequested())
				break;
		}

		KWObject* instance = cast(KWObject*, instances->GetAt(i));

		if (parameters->HasMissingKMeanValue(instance))
			continue;

		const Symbol sInstanceModality = instance->GetSymbolValueAt(targetAttribute->GetLoadIndex());

		for (int j = 0; j < oaTargetAttributeValues->GetSize(); j++) {

			const StringObject* modalityValue = cast(StringObject*, oaTargetAttributeValues->GetAt(j));

			if (ALString(sInstanceModality) == modalityValue->GetString()) {

				// incrementer le compte de la modalit� cible

				Object* count = targetModalitiesCount.Lookup(sInstanceModality.GetNumericKey());

				if (count == NULL) {

					TargetModalityCount* modalityCount = new TargetModalityCount;
					modalityCount->sModality = sInstanceModality;
					modalityCount->iCount = 1;
					targetModalitiesCount.SetAt(sInstanceModality.GetNumericKey(), modalityCount);
				}
				else {
					TargetModalityCount* modalityCount = cast(TargetModalityCount*, count);
					modalityCount->iCount++;
				}

				break; // passer a l'instance suivante
			}
		}
	}

	ObjectArray* result = new ObjectArray();
	targetModalitiesCount.ExportObjectArray(result);

	result->SetCompareFunction(KMClusteringTargetCountCompare);
	result->Sort();

	return result;

}

void KMClusteringInitializer::CopyFrom(const KMClusteringInitializer* aSource)
{
	require(aSource != NULL);

	lInstancesWithMissingValues = aSource->lInstancesWithMissingValues;
	clustering = aSource->clustering;
}

// fonction de comparaison pour tri de tableau
int KMClusteringTargetCountCompare(const void* elem1, const void* elem2) {

	KMClusteringInitializer::TargetModalityCount* i1 = (KMClusteringInitializer::TargetModalityCount*) * (Object**)elem1;
	KMClusteringInitializer::TargetModalityCount* i2 = (KMClusteringInitializer::TargetModalityCount*) * (Object**)elem2;

	if (i1->iCount < i2->iCount)
		return 1;
	else if (i1->iCount > i2->iCount)
		return -1;
	else
		return 0;

}
