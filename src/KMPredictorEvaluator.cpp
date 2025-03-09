// Copyright (c) 2023-2025 Orange. All rights reserved.
// This software is distributed under the BSD 3-Clause-clear License, the text
// of which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or
// see the "LICENSE" file for more details.

#include "KMPredictorEvaluator.h"
#include "KMPredictorKNN.h"

// modification de la methode ancetre, afin d'ajouter des objets
// KMTrainedClassifier ou KMTrainedPredictor au tableau
// oaEvaluatedTrainedPredictors
void KMPredictorEvaluator::BuildEvaluatedTrainedPredictors(
    ObjectArray *oaEvaluatedTrainedPredictors) {
  int i;
  KWClass *kwcPredictorClass;
  KWTrainedPredictor *trainedPredictor;
  boolean bIsPredictor;
  KWEvaluatedPredictorSpec *evaluatedPredictorSpec;

  require(oaEvaluatedTrainedPredictors != NULL);
  require(oaEvaluatedPredictorSpecs.GetSize() == 0 or
          GetInitialClassesDomain() != NULL);
  require(kwcdInitialCurrentDomain != NULL);

  // Nettoyage initial du tableau resultat
  oaEvaluatedTrainedPredictors->RemoveAll();

  // Parcours des specifications de predicteurs a evaluer
  for (i = 0; i < oaEvaluatedPredictorSpecs.GetSize(); i++) {
    evaluatedPredictorSpec =
        cast(KWEvaluatedPredictorSpec *, oaEvaluatedPredictorSpecs.GetAt(i));

    // Construction du predicteur si evaluation demandee
    if (evaluatedPredictorSpec->GetEvaluated()) {
      // Recherche de la classe correspondante
      kwcPredictorClass = kwcdInitialCurrentDomain->LookupClass(
          evaluatedPredictorSpec->GetClassName());
      assert(kwcPredictorClass != NULL);

      // Recherche du predicteur correspondant
      trainedPredictor = NULL;
      // cas ou c'est un predicteur de type KMean
      if (evaluatedPredictorSpec->GetPredictorName() ==
              KMPredictor::PREDICTOR_NAME or
          evaluatedPredictorSpec->GetPredictorName() ==
              KMPredictorKNN::PREDICTOR_NAME) {
        if (evaluatedPredictorSpec->GetPredictorType() ==
            KWType::GetPredictorLabel(KWType::Symbol))
          trainedPredictor = new KMTrainedClassifier;
        else if (evaluatedPredictorSpec->GetPredictorType() ==
                 KWType::GetPredictorLabel(KWType::None))
          trainedPredictor = new KMTrainedPredictor;

        // cas ou ce n'est pas un predicteur KMean
      } else {
        if (evaluatedPredictorSpec->GetPredictorType() ==
            KWType::GetPredictorLabel(KWType::Symbol))
          trainedPredictor = new KWTrainedClassifier;
        else if (evaluatedPredictorSpec->GetPredictorType() ==
                 KWType::GetPredictorLabel(KWType::Continuous))
          trainedPredictor = new KWTrainedRegressor;
      }
      bIsPredictor = trainedPredictor->ImportPredictorClass(kwcPredictorClass);
      assert(bIsPredictor);
      oaEvaluatedTrainedPredictors->Add(trainedPredictor);
      check(trainedPredictor);
    }
  }
}

void KMPredictorEvaluator::EvaluatePredictorSpecs() {
  boolean bOk = true;
  ObjectArray oaEvaluatedTrainedPredictors;
  ObjectArray oaOutputPredictorEvaluations;
  ALString sOutputPathName;

  // Recherche des predicteurs a evaluer
  BuildEvaluatedTrainedPredictors(&oaEvaluatedTrainedPredictors);

  // Test de coherence des predicteurs
  bOk = CheckEvaluatedTrainedPredictors(&oaEvaluatedTrainedPredictors);

  // On tente de cree le repertoire cible du rapport d'evaluation
  // (c'est le meme pour le rapport JSON)
  if (bOk) {
    sOutputPathName = FileService::GetPathName(GetEvaluationFilePathName());
    if (sOutputPathName != "" and
        not FileService::FileExists(sOutputPathName)) {
      bOk = FileService::MakeDirectories(sOutputPathName);
      if (not bOk)
        AddError("Unable to create output directory (" + sOutputPathName +
                 ") for evaluation file");
    }
  }

  // Evaluation des predicteurs s'ils sont coherents
  if (bOk) {
    // Demarage du suivi de la tache
    TaskProgression::SetTitle("Evaluate predictors");
    TaskProgression::SetDisplayedLevelNumber(2);
    TaskProgression::Start();

    // Evaluation des predicteurs
    EvaluateTrainedPredictors(&oaEvaluatedTrainedPredictors,
                              &oaOutputPredictorEvaluations);
    oaOutputPredictorEvaluations.DeleteAll();

    // Fin du suivi de la tache
    TaskProgression::Stop();
  }

  // Destruction des predicteurs (qui n'ont ici ete utiles que pour identifier
  // leurs specifications)
  oaEvaluatedTrainedPredictors.DeleteAll();
}

void KMPredictorEvaluator::EvaluateTrainedPredictors(
    ObjectArray *oaEvaluatedTrainedPredictors,
    ObjectArray *oaOutputPredictorEvaluations) {
  boolean bOk = true;
  KWLearningSpec learningSpec;
  KWClass *learningSpecClass;
  KWTrainedPredictor *trainedPredictor;
  int i;
  ObjectArray oaPredictors;
  KMPredictorExternal *predictorExternal;
  KWClassDomain *kwcdCurrentDomain;
  ObjectArray oaEvaluationDatabaseFileSpecs;
  int nRef;
  FileSpec *specRef;
  FileSpec specEvaluationReportFile;
  FileSpec specJSONReportFile;

  require(oaEvaluatedTrainedPredictors != NULL);
  require(oaEvaluatedPredictorSpecs.GetSize() == 0 or
          GetInitialClassesDomain() != NULL);
  require(kwcdInitialCurrentDomain != NULL);
  require(oaOutputPredictorEvaluations != NULL);
  require(oaOutputPredictorEvaluations->GetSize() == 0);

  // On positionne le domaine des classes initiales comme domaine courant
  // Cela permet ainsi le parametrage de la base d'evaluation par les
  // classes initiales des predicteurs
  kwcdCurrentDomain = KWClassDomain::GetCurrentDomain();
  if (kwcdInitialClassesDomain != NULL)
    KWClassDomain::SetCurrentDomain(kwcdInitialClassesDomain);

  // Verification de la coherence des predicteurs
  if (bOk)
    bOk = CheckEvaluatedTrainedPredictors(oaEvaluatedTrainedPredictors);

  // Le nom du rapport d'evaluation doit etre renseigne
  if (bOk and GetEvaluationFileName() == "") {
    bOk = false;
    AddError("Missing evaluation report name");
  }

  // Le nom de la base d'evaluation doit etre renseigne
  if (bOk and evaluationDatabase->GetDatabaseName() == "") {
    bOk = false;
    AddError("Missing evaluation database name");
  }

  // Verification de la validite des specifications de la base d'evaluation
  bOk = bOk and evaluationDatabase->Check();

  // Le parametrage de selection doit etre valide
  // Les messages d'erreurs sont emis par la methode appelee
  if (bOk and not evaluationDatabase->CheckSelectionValue(
                  evaluationDatabase->GetSelectionValue()))
    bOk = false;

  // Le nom du rapport d'evaluation doit etre different du ou des fichiers de la
  // base source
  if (bOk) {
    specEvaluationReportFile.SetLabel("evaluation report");
    specEvaluationReportFile.SetFilePathName(GetEvaluationFilePathName());
    evaluationDatabase->ExportUsedFileSpecs(&oaEvaluationDatabaseFileSpecs);
    for (nRef = 0; nRef < oaEvaluationDatabaseFileSpecs.GetSize(); nRef++) {
      specRef = cast(FileSpec *, oaEvaluationDatabaseFileSpecs.GetAt(nRef));
      specRef->SetLabel("evaluation " + specRef->GetLabel());
      bOk = bOk and specEvaluationReportFile.CheckReferenceFileSpec(specRef);
      if (not bOk)
        break;
    }
    oaEvaluationDatabaseFileSpecs.DeleteAll();
    if (not bOk)
      AddError("The evaluation report file name should differ from that of the "
               "evaluation database");

    // Le nom du rapport doit etre different du ou des fichiers de la base
    // source
    if (bOk and GetEvaluationFilePathName() != "") {
      specJSONReportFile.SetLabel("JSON report");
      specJSONReportFile.SetFilePathName(GetEvaluationFilePathName());
      for (nRef = 0; nRef < oaEvaluationDatabaseFileSpecs.GetSize(); nRef++) {
        specRef = cast(FileSpec *, oaEvaluationDatabaseFileSpecs.GetAt(nRef));
        specRef->SetLabel("evaluation " + specRef->GetLabel());
        bOk = bOk and specJSONReportFile.CheckReferenceFileSpec(specRef);
        if (not bOk)
          break;
      }
      if (not bOk)
        AddError("The JSON report file name should differ from that of the "
                 "evaluation database");

      // Et il doit etre different du rapport d'evaluation
      if (bOk)
        bOk = specJSONReportFile.CheckReferenceFileSpec(
            &specEvaluationReportFile);
    }
  }

  // Il doit y avoir au moins un predicteur a evaluer
  if (bOk and oaEvaluatedTrainedPredictors->GetSize() == 0) {
    bOk = false;
    AddWarning("No requested predictor evaluation");
  }

  // Evaluation des predicteurs
  if (bOk) {
    // Destruction du domaine initial (a reconstruire), en faisant attention au
    // domaine courant
    check(kwcdInitialClassesDomain);
    if (kwcdCurrentDomain == kwcdInitialClassesDomain)
      kwcdCurrentDomain = NULL;
    delete kwcdInitialClassesDomain;

    // On reconstruit le domaine initial a partir du premier predicteur a
    // evaluer En effet, ce domaine initial de reference peut dependre de la
    // selection en cours des predicteurs a evaluer
    trainedPredictor =
        cast(KWTrainedPredictor *, oaEvaluatedTrainedPredictors->GetAt(0));
    kwcdInitialClassesDomain = BuildInitialDomainPredictor(trainedPredictor);
    KWClassDomain::SetCurrentDomain(kwcdInitialClassesDomain);
    assert(
        DomainCheckClassesInitialNames(trainedPredictor->GetPredictorDomain()));

    // Si le domaine courant correspondait avec le domaine initial, on maintient
    // cette correspondance
    if (kwcdCurrentDomain == NULL)
      kwcdCurrentDomain = kwcdInitialClassesDomain;

    // Parametrage des specifications d'apprentissage a partir du premier
    // predicteur a evaluer
    trainedPredictor =
        cast(KWTrainedPredictor *, oaEvaluatedTrainedPredictors->GetAt(0));
    learningSpec.SetDatabase(GetEvaluationDatabase());
    learningSpec.SetTargetAttributeName(
        trainedPredictor->GetTargetAttribute()->GetName());
    learningSpec.SetMainTargetModality((const char *)GetMainTargetModality());
    learningSpecClass = kwcdInitialClassesDomain->LookupClass(
        GetEvaluationDatabase()->GetClassName());
    learningSpec.SetClass(learningSpecClass);
    assert(learningSpecClass != NULL);
    assert(learningSpecClass->GetName() ==
           KWTrainedPredictor::GetMetaDataInitialClassName(
               trainedPredictor->GetPredictorClass()));
    assert(learningSpec.Check());

    // Reconstruction de predicteurs
    for (i = 0; i < oaEvaluatedTrainedPredictors->GetSize(); i++) {
      trainedPredictor =
          cast(KWTrainedPredictor *, oaEvaluatedTrainedPredictors->GetAt(i));

      // On restitue les noms initiaux des classes du predicteurs afin de
      // pouvoiur utiliser la  base d'evaluation, qui est parametree par ces
      // classes initiales valide pour tous les predicteurs
      DomainRenameClassesWithInitialNames(
          trainedPredictor->GetPredictorDomain());

      // Construction du predicteur
      predictorExternal = new KMPredictorExternal;
      predictorExternal->SetLearningSpec(&learningSpec);
      predictorExternal->SetExternalTrainedPredictor(trainedPredictor);
      evaluationDatabase->SetVerboseMode(false);
      predictorExternal->Train();
      evaluationDatabase->SetVerboseMode(true);
      assert(predictorExternal->IsTrained());

      // Memorisation
      oaPredictors.Add(predictorExternal);
    }

    // tri des predicteurs : le predicteur K-Means doit etre le premier, car
    // c'est au premier predicteur de la liste, que la methode
    // KWPredictorEvaluator::EvaluatePredictors  va demander l'ecriture du
    // rapport d'evaluation.
    SortPredictors(oaPredictors);

    // Evaluation des predicteurs
    EvaluatePredictors(&oaPredictors, GetEvaluationDatabase(), "Predictor",
                       oaOutputPredictorEvaluations);

    // Ecriture du rapport d'evaluation
    WriteEvaluationReport(GetEvaluationFilePathName(), "Predictor",
                          oaOutputPredictorEvaluations);

    // Ecriture du rapport JSON
    if (GetEvaluationFilePathName() != "")
      WriteJSONEvaluationReport(GetEvaluationFilePathName(), "Predictor",
                      oaOutputPredictorEvaluations);

    // Nettoyage du tableau de predicteurs, en dereferencant prealablement
    // leur predicteur appris (pour eviter une double destruction)
    for (i = 0; i < oaPredictors.GetSize(); i++) {
      predictorExternal = cast(KMPredictorExternal *, oaPredictors.GetAt(i));
      predictorExternal->UnreferenceTrainedPredictor();
    }
    oaPredictors.DeleteAll();
  }

  // Restitution du domaine courant
  if (kwcdInitialClassesDomain != NULL)
    KWClassDomain::SetCurrentDomain(kwcdCurrentDomain);
}

void KMPredictorEvaluator::FillEvaluatedPredictorSpecs() {
  int i;
  ALString sInitialClassName;
  KWClass *kwcClass;
  ObjectArray oaClasses;
  KWTrainedPredictor *trainedPredictor;
  ObjectArray oaTrainedPredictors;
  ObjectArray oaEvaluatedPredictors;
  boolean bIsPredictor;
  ALString sReferenceTargetAttribute;
  KWEvaluatedPredictorSpec *evaluatedPredictorSpec;
  KWEvaluatedPredictorSpec *previousEvaluatedPredictorSpec;
  ObjectDictionary odEvaluatedPredictorSpecs;
  ALString sTmp;

  // Nettoyage du domaine des classes initiales
  kwcdInitialCurrentDomain = KWClassDomain::GetCurrentDomain();
  if (kwcdInitialClassesDomain != NULL) {
    assert(KWClassDomain::GetCurrentDomain() != kwcdInitialClassesDomain);
    assert(KWClassDomain::LookupDomain(kwcdInitialClassesDomain->GetName()) ==
           NULL);
    delete kwcdInitialClassesDomain;
  }
  kwcdInitialClassesDomain = NULL;

  // Recherche des predicteurs compatibles
  for (i = 0; i < KWClassDomain::GetCurrentDomain()->GetClassNumber(); i++) {
    kwcClass = KWClassDomain::GetCurrentDomain()->GetClassAt(i);
    trainedPredictor = NULL;

    // On determine si elle correspond a un classifieur
    if (KWTrainedPredictor::GetMetaDataPredictorType(kwcClass) ==
        KWType::Symbol) {
      trainedPredictor = new KWTrainedClassifier;
      bIsPredictor = trainedPredictor->ImportPredictorClass(kwcClass);
      if (not bIsPredictor) {
        delete trainedPredictor;
        trainedPredictor = NULL;
      }
    }

    // On determine si elle correspond a un regresseur
    else if (KWTrainedPredictor::GetMetaDataPredictorType(kwcClass) ==
             KWType::Continuous) {
      trainedPredictor = new KWTrainedRegressor;
      bIsPredictor = trainedPredictor->ImportPredictorClass(kwcClass);
      if (not bIsPredictor) {
        delete trainedPredictor;
        trainedPredictor = NULL;
      }
    }

    // Memorisation du predicteur construit
    if (trainedPredictor != NULL) {
      // Construction si necessaire du domaine de classe initial, a partir de
      // celui du premier predicteur valide
      if (kwcdInitialClassesDomain == NULL) {
        assert(oaTrainedPredictors.GetSize() == 0);

        // Construction d'un domaine initial a partir des specification d'un
        // predicteur
        kwcdInitialClassesDomain =
            BuildInitialDomainPredictor(trainedPredictor);
        assert(DomainCheckClassesInitialNames(
            trainedPredictor->GetPredictorDomain()));

        // Memorisation du nom de classe initial du predicteur
        sInitialClassName = KWTrainedPredictor::GetMetaDataInitialClassName(
            trainedPredictor->GetPredictorClass());
      }

      // Memorisation si nouveau predicteur compatible avec le domaine de classe
      // initial
      if (oaTrainedPredictors.GetSize() == 0 or
          DomainCheckClassesInitialNames(
              trainedPredictor->GetPredictorDomain()))
        oaTrainedPredictors.Add(trainedPredictor);
      // Destruction sinon
      else {
        AddWarning("Predictor " +
                   trainedPredictor->GetPredictorClass()->GetName() +
                   " is ignored because the native variables of its dictionary "
                   "are not consistent with the other predictors");
        delete trainedPredictor;
        trainedPredictor = NULL;
      }
    }
  }
  assert(kwcdInitialClassesDomain == NULL or oaTrainedPredictors.GetSize() > 0);
  assert(kwcdInitialClassesDomain == NULL or sInitialClassName != "");

  // Transfert des specifications precedentes dans un dictionnaire, pour
  // memoriser leur etat de selection
  for (i = 0; i < oaEvaluatedPredictorSpecs.GetSize(); i++) {
    previousEvaluatedPredictorSpec =
        cast(KWEvaluatedPredictorSpec *, oaEvaluatedPredictorSpecs.GetAt(i));
    odEvaluatedPredictorSpecs.SetAt(
        previousEvaluatedPredictorSpec->GetClassName(),
        previousEvaluatedPredictorSpec);
  }
  oaEvaluatedPredictorSpecs.RemoveAll();

  // Exploitation de tous les predicteurs a evaluer
  for (i = 0; i < oaTrainedPredictors.GetSize(); i++) {
    trainedPredictor = cast(KWTrainedPredictor *, oaTrainedPredictors.GetAt(i));
    assert(trainedPredictor->GetTargetAttribute() != NULL);

    // Creation d'une specification d'�valuation
    evaluatedPredictorSpec = new KWEvaluatedPredictorSpec;
    evaluatedPredictorSpec->SetEvaluated(true);
    evaluatedPredictorSpec->SetPredictorType(
        KWType::GetPredictorLabel(trainedPredictor->GetTargetType()));
    evaluatedPredictorSpec->SetPredictorName(trainedPredictor->GetName());
    evaluatedPredictorSpec->SetClassName(
        trainedPredictor->GetPredictorClass()->GetName());
    evaluatedPredictorSpec->SetTargetAttributeName(
        trainedPredictor->GetTargetAttribute()->GetName());
    oaEvaluatedPredictorSpecs.Add(evaluatedPredictorSpec);

    // Mise a jour de la selection en fonction de la selection precedente
    previousEvaluatedPredictorSpec =
        cast(KWEvaluatedPredictorSpec *,
             odEvaluatedPredictorSpecs.Lookup(
                 evaluatedPredictorSpec->GetClassName()));
    if (previousEvaluatedPredictorSpec != NULL)
      evaluatedPredictorSpec->SetEvaluated(
          previousEvaluatedPredictorSpec->GetEvaluated());
  }

  // Nettoyage des specifications precedentes
  odEvaluatedPredictorSpecs.DeleteAll();

  // Destruction des predicteurs (qui n'ont ici ete utiles que pour identifier
  // leurs specifications)
  oaTrainedPredictors.DeleteAll();

  // Initialisation de la classe associee a la base d'evaluation
  GetEvaluationDatabase()->SetClassName(sInitialClassName);

  // Warning s'il n'y a pas de dictionnaire
  if (KWClassDomain::GetCurrentDomain()->GetClassNumber() == 0)
    AddWarning("No available dictionary");
  // Warning s'il n'y a pas de predicteurs parmi les dictionnaire
  else if (oaEvaluatedPredictorSpecs.GetSize() == 0)
    AddWarning("No available predictor among the dictionaries");
}

void KMPredictorEvaluator::SortPredictors(ObjectArray &oaPredictors) {

  // tri du tableau des predicteurs : le predicteur K-Means ou KNN doit etre le
  // premier de la liste, car c'est au premier predicteur de la liste, que la
  // methode KWPredictorEvaluator::EvaluatePredictors va demander l'ecriture du
  // rapport d'evaluation. l'ordre des autres predicteurs eventuels est
  // indifferent

  assert(oaPredictors.GetSize() > 0);

  KWPredictor *predictor;

  for (int i = 0; i < oaPredictors.GetSize(); i++) {
    predictor = cast(KWPredictor *, oaPredictors.GetAt(i));

    if (predictor->GetName() == KMPredictor::PREDICTOR_NAME or
        predictor->GetName() == KMPredictorKNN::PREDICTOR_NAME) {

      if (i > 0) {
        // inverser les positions entre le premier predicteur et le predicteur
        // kmean
        KWPredictor *firstPredictor =
            cast(KWPredictor *, oaPredictors.GetAt(0));
        oaPredictors.SetAt(0, predictor);
        oaPredictors.SetAt(i, firstPredictor);
      }
      break;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Classe KMPredictorExternal

// obligation de redefinir cette methode, afin de pouvoir utiliser les methodes
// Evaluate (specifiques) des objets KMClassifierEvaluation et
// KMPredictorEvaluation
KWPredictorEvaluation *KMPredictorExternal::Evaluate(KWDatabase *database) {
  require(IsTrained());
  require(database != NULL);

  Global::SetSilentMode(false);

  // si ce n'est pas un predicteur de type KMean, appeler simplement la methode
  // ancetre
  if (GetTrainedPredictor()->GetName() != KMPredictor::PREDICTOR_NAME and
      GetTrainedPredictor()->GetName() != KMPredictorKNN::PREDICTOR_NAME)
    return KWPredictorExternal::Evaluate(database);

  // Creation des resultats d'evaluation selon le type de predicteur KMean
  if (GetTargetAttributeType() == KWType::Symbol) {
    // mode supervis�
    KMClassifierEvaluation *classifierEvaluation = new KMClassifierEvaluation;
    classifierEvaluation->Evaluate(this, database);
    return classifierEvaluation;
  } else {
    // mode non supervis�
    KMPredictorEvaluation *predictorEvaluation = new KMPredictorEvaluation;
    predictorEvaluation->Evaluate(this, database);
    return predictorEvaluation;
  }
}

boolean KMPredictorExternal::IsTargetTypeManaged(int nType) const {
  return (nType == KWType::Symbol or nType == KWType::None);
}
