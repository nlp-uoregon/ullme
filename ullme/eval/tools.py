import json
import os
import pathlib
from typing import List
import mteb
import numpy as np
import pandas as pd

from src.eval.constants import LANG_TO_CODES, MTEB_DS_TO_PROMPT, MULTILINGUAL_DS_TO_PROMPT, QUICK_EVAL


def get_eval_mteb_dataset(
        mteb_model_meta: mteb.ModelMeta,
        dataset_name: str,
        langs: List[str],
        output_folder: str='results',
        aggregation=np.mean,
):  
    print(f"Getting results for {dataset_name} in {langs}")
    
    task = mteb.get_task(task_name=dataset_name, languages=langs)
    task_name = task.metadata.name
    langs = task.languages

    model_path_name = mteb_model_meta.model_name_as_path()
    model_revision = mteb_model_meta.revision
    results_path = pathlib.Path(output_folder) / model_path_name / model_revision / f'{task_name}.json'
    if results_path.exists():
        print(f"Loading results from {results_path}")
        results = mteb.MTEBResults.from_disk(results_path)
    else:
        results = None
    all_results = {}
    for lang in langs:
        try:
            test_result_lang = results.get_score(
                languages=[lang],
                aggregation=aggregation,
            )
        except Exception as e:
            print(f"Error in getting results for {lang} in {dataset_name}")
            test_result_lang = 0.0
        all_results[lang] = test_result_lang

    print(f"Results for {dataset_name}: {all_results}")
    return all_results


def get_eval_mteb(
        mteb_model_meta: mteb.ModelMeta,
        output_folder: str='results',
        is_quick_run: bool=False,
):  
    langs = ['eng']
    results = {
        'avg': {}
    }
    for task in MTEB_DS_TO_PROMPT.keys():
        dataset_in_task = MTEB_DS_TO_PROMPT[task]
        all_task_results = []
        for dataset in dataset_in_task.keys():
            if is_quick_run and dataset not in QUICK_EVAL:
                continue
            main_metric = get_eval_mteb_dataset(
                dataset_name=dataset,
                langs=langs,
                output_folder=f"{output_folder}/{langs[0]}",
                mteb_model_meta=mteb_model_meta,
            )
            results[dataset] = main_metric['eng']
            all_task_results.append(main_metric['eng'])
        
        if len(all_task_results) != 0:
            avg_task_result = sum(all_task_results) / len(all_task_results)
            results['avg'][task] = avg_task_result
    
    all_results = [r for r in results.values() if isinstance(r, float)]
    avg_result = sum(all_results) / len(all_results)
    results['avg']['all_tasks'] = avg_result
    print(f"Results for all tasks: {results}")

    with open(os.path.join(output_folder, 'mteb_all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    return results


def get_eval_multilingual(
        mteb_model_meta: mteb.ModelMeta,
        langs: List[str] = 'all',
        output_folder: str='results',
        is_quick_run: bool=False,
):  
    if isinstance(langs, list):
        name = '_'.join(langs)
    if langs == 'all':
        langs = LANG_TO_CODES.keys()
        name = 'all'
    
    all_results = {
        'avg': {},
    }
    _all_results = []
    for lang in langs:
        datasets_in_lang = MULTILINGUAL_DS_TO_PROMPT[lang]
        lang_codes = LANG_TO_CODES[lang]
        all_results[lang] = {}
        for dataset in datasets_in_lang.keys():
            if is_quick_run and dataset not in QUICK_EVAL:
                continue
            main_metric = get_eval_mteb_dataset(
                dataset_name=dataset,
                langs=lang_codes,
                mteb_model_meta=mteb_model_meta,
                output_folder=f"{output_folder}/{lang_codes[0]}",
            )
            lang_results = [main_metric[lang_code] for lang_code in lang_codes if lang_code in main_metric]
            assert len(lang_results) > 0, f"No results for {dataset} in {lang_codes}"
            lang_results = sum(lang_results) / len(lang_results)
            all_results[lang][dataset] = lang_results
            _all_results.append(lang_results)
        if len(all_results[lang]) != 0:
            lang_avg_result = sum([x for x in all_results[lang].values()]) / len(all_results[lang])
            all_results['avg'][lang] = lang_avg_result
    
    all_results['avg']['all_tasks'] = sum(_all_results) / len(_all_results)
    print(f"Results for all languages: {all_results}")

    with open(os.path.join(output_folder, f'{name}_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    return all_results


def read_results(result_dir: str):
    with open(pathlib.Path(result_dir) / "mteb_all_results.json", "r") as f:
        mteb_results = json.load(f)

    with open(pathlib.Path(result_dir) / "all_results.json", "r") as f:
        all_results = json.load(f)

    en_datas = [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
        "ArxivClusteringP2P",
        "ArxivClusteringS2S",
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        "RedditClustering",
        "RedditClusteringP2P",
        "StackExchangeClustering",
        "StackExchangeClusteringP2P",
        "TwentyNewsgroupsClustering",
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
        "MIRACLReranking",
        "ArguAna",
        "ClimateFEVER",
        "CQADupstackTexRetrieval",
        "DBPedia",
        "FEVER",
        "FiQA2018",
        "HotpotQA",
        "MSMARCO",
        "NFCorpus",
        "NQ",
        "QuoraRetrieval",
        "SCIDOCS",
        "SciFact",
        "Touche2020",
        "TRECCOVID",
        "MIRACLRetrieval",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STS17",
        "STS22",
        "BIOSSES",
        "SICK-R",
        "STSBenchmark",
        "SummEval"
    ]
    en_results = []
    for en_data in en_datas:
        if en_data not in ['MIRACLReranking', 'MIRACLRetrieval']:
            en_results.append([en_data, mteb_results.get(en_data, 0.0) * 100])
        else:
            en_results.append([en_data, all_results['en'].get(en_data, 0.0) * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(en_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "en_results.xlsx")

    zh_datas = [
        "AmazonReviewsClassification",
        "JDReview",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MultilingualSentiment",
        "OnlineShopping",
        "Waimai",
        "Ocnli",
        "PawsXPairClassification",
        "CLSClusteringP2P",
        "CLSClusteringS2S",
        "ThuNewsClusteringP2P",
        "ThuNewsClusteringS2S",
        "MIRACLReranking",
        "MMarcoReranking",
        "T2Reranking",
        "CMedQAv2-reranking",
        "MLQARetrieval",
        "MIRACLRetrieval",
        "CmedqaRetrieval",
        "CovidRetrieval",
        "DuRetrieval",
        "EcomRetrieval",
        "MedicalRetrieval",
        "MMarcoRetrieval",
        "T2Retrieval",
        "VideoRetrieval",
        "AFQMC",
        "ATEC",
        "BQ",
        "LCQMC",
        "PAWSX",
        "STSB",
        "STS22"
    ]
    zh_results = []
    for zh_data in zh_datas:
        zh_results.append([zh_data, all_results['zh'][zh_data] * 100]) 
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(zh_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "zh_results.xlsx")

    es_datas = [
        "AmazonReviewsClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPIntentClassification",
        "MultilingualSentimentClassification",
        "TweetSentimentClassification",
        "SpanishNewsClassification",
        "PawsXPairClassification",
        "XNLI",
        "SpanishNewsClusteringP2P",
        "MLSUMClusteringP2P.v2",
        "MLSUMClusteringS2S.v2",
        "SIB200ClusteringS2S",
        "MultiEURLEXMultilabelClassification",
        "MIRACLReranking",
        "BelebeleRetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "XPQARetrieval",
        "XQuADRetrieval",
        "STS17",
        "STS22",
        "STSBenchmarkMultilingualSTS"
    ]
    es_results = []
    for es_data in es_datas:
        es_results.append([es_data, all_results['es'][es_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(es_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "es_results.xlsx")

    de_datas = [
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "MTOPIntentClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MultilingualSentimentClassification",
        "TweetSentimentClassification",
        "RTE3",
        "XNLI",
        "TenKGnadClusteringP2P.v2",
        "TenKGnadClusteringS2S.v2",
        "BlurbsClusteringP2P",
        "BlurbsClusteringS2S",
        "MultiEURLEXMultilabelClassification",
        "WikipediaRerankingMultilingual",
        "MIRACLReranking",
        "GermanDPR",
        "GermanGovServiceRetrieval",
        "GermanQuAD-Retrieval",
        "BelebeleRetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "WikipediaRetrievalMultilingual",
        "XPQARetrieval",
        "XQuADRetrieval",
        "STS17",
        "STS22",
        "STSBenchmarkMultilingualSTS"
    ]
    de_results = []
    for de_data in de_datas:
        try:
            res = all_results['de'][de_data]
        except:
            res = all_results['de'][de_data+'.v2']
        de_results.append([de_data,  res* 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(de_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "de_results.xlsx")

    ru_datas = [
        "GeoreviewClassification",
        "HeadlineClassification",
        "InappropriatenessClassification",
        "KinopoiskClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "RuReviewsClassification",
        "RuSciBenchGRNTIClassification",
        "RuSciBenchOECDClassification",
        "GeoreviewClusteringP2P",
        "RuSciBenchGRNTIClusteringP2P",
        "RuSciBenchOECDClusteringP2P",
        "TERRa",
        "MIRACLReranking",
        "NeuCLIR2023Retrieval",
        "RiaNewsRetrieval",
        "RuBQRetrieval",
        "MIRACLRetrieval",
        "RuSTSBenchmarkSTS",
        "STS22"
    ]
    ru_results = []
    for ru_data in ru_datas:
        ru_results.append([ru_data, all_results['ru'][ru_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(ru_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "ru_results.xlsx")

    fr_datas = [
        "AmazonReviewsClassification",
        "MTOPIntentClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "TweetSentimentClassification",
        "SIB200Classification",
        "FrenchBookReviews",
        "PawsXPairClassification",
        "RTE3",
        "XNLI",
        "MasakhaNEWSClusteringP2P",
        "MasakhaNEWSClusteringS2S",
        "MLSUMClusteringP2P.v2",
        "MLSUMClusteringS2S.v2",
        "AlloProfClusteringP2P",
        "AlloProfClusteringS2S",
        "HALClusteringS2S",
        "SIB200ClusteringS2S",
        "MultiEURLEXMultilabelClassification",
        "AlloprofReranking",
        "SyntecReranking",
        "MIRACLReranking",
        "AlloprofRetrieval",
        "FQuADRetrieval",
        "BelebeleRetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "PublicHealthQA",
        "XPQARetrieval",
        "OpusparcusPC",
        "STS17",
        "SICKFr",
        "STS22",
        "STSBenchmarkMultilingualSTS",
        "SummEvalFr"
    ]
    fr_results = []
    for fr_data in fr_datas:
        try:
            res = all_results['fr'][fr_data]
        except:
            res = all_results['fr'][fr_data+'.v2']
        fr_results.append([fr_data, res * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(fr_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "fr_results.xlsx")

    ja_datas = [
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "WRIMEClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "SIB200Classification",
        "PawsXPairClassification",
        "LivedoorNewsClustering",
        "MewsC16JaClustering",
        "SIB200ClusteringS2S",
        "VoyageMMarcoReranking",
        "MIRACLReranking",
        "JaGovFaqsRetrieval",
        "JaQuADRetrieval",
        "NLPJournalAbsIntroRetrieval",
        "NLPJournalTitleAbsRetrieval",
        "BelebeleRetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "XPQARetrieval",
        "JSICK",
        "JSTS"
    ]
    ja_results = []
    for ja_data in ja_datas:
        try:
            res = all_results['ja'][ja_data]
        except:
            res = all_results['ja'][ja_data+'.v2']  
        ja_results.append([ja_data, res * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(ja_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "ja_results.xlsx")

    vi_datas = [
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MultilingualSentimentClassification",
        "SIB200Classification",
        "VieStudentFeedbackClassification",
        "XNLI",
        "SIB200ClusteringS2S",
        "BelebeleRetrieval",
        "MLQARetrieval",
        "PublicHealthQA",
        "XQuADRetrieval",
        "VieQuADRetrieval"
    ]
    vi_results = []
    for vi_data in vi_datas:
        vi_results.append([vi_data, all_results['vi'][vi_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(vi_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "vi_results.xlsx")

    fa_datas = [
        "MassiveScenarioClassification",
        "MassiveIntentClassification",
        "MultilingualSentimentClassification",
        "FarsTail",
        "MIRACLReranking",
        "WikipediaRerankingMultilingual",
        "MIRACLRetrieval",
        "NeuCLIR2023Retrieval",
        "WikipediaRetrievalMultilingual"
    ]
    fa_results = []
    for fa_data in fa_datas:
        fa_results.append([fa_data, all_results['fa'][fa_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(fa_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "fa_results.xlsx")

    id_datas = [
        "IndonesianMongabayConservationClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "SIB200Classification",
        "indonli",
        "SIB200ClusteringS2S",
        "MIRACLReranking",
        "BelebeleRetrieval",
        "MIRACLRetrieval",
        "SemRel24STS"
    ]
    id_results = []
    for id_data in id_datas:
        id_results.append([id_data, all_results['id'][id_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(id_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "id_results.xlsx")

    ar_datas = [
        "TweetEmotionClassification",
        "ArEntail",
        "XNLI",
        "MIRACLReranking",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "MLQARetrieval",
        "XPQARetrieval",
        "STS17",
        "STS22"
    ]
    ar_results = []
    for ar_data in ar_datas:
        try:
            res = all_results['ar'][ar_data]
        except:
            res = all_results['ar'][ar_data+'.v2']
        ar_results.append([ar_data, res * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(ar_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "ar_results.xlsx")

    fi_datas = [
        "FinToxicityClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MultilingualSentimentClassification",
        "SIB200Classification",
        "MIRACLReranking",
        "WikipediaRerankingMultilingual",
        "BelebeleRetrieval",
        "MIRACLRetrieval",
        "WikipediaRetrievalMultilingual",
        "OpusparcusPC",
        "FinParaSTS"
    ]
    fi_results = []
    for fi_data in fi_datas:
        fi_results.append([fi_data, all_results['fi'][fi_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(fi_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "fi_results.xlsx")

    ko_datas = [
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "KorSarcasmClassification",
        "SIB200Classification",
        "KorHateSpeechMLClassification",
        "PawsXPairClassification",
        "KLUE-TC",
        "SIB200ClusteringS2S",
        "MIRACLReranking",
        "Ko-StrategyQA",
        "BelebeleRetrieval",
        "MIRACLRetrieval",
        "PublicHealthQA",
        "XPQARetrieval",
        "KLUE-STS",
        "KorSTS",
        "STS17"
    ]
    ko_results = []
    for ko_data in ko_datas:
        ko_results.append([ko_data, all_results['ko'][ko_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(ko_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "ko_results.xlsx")

    hi_datas = [
        "MTOPIntentClassification",
        "SentimentAnalysisHindi",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "SIB200Classification",
        "TweetSentimentClassification",
        "XNLI",
        "IndicReviewsClusteringP2P",
        "SIB200ClusteringS2S",
        "MIRACLReranking",
        "WikipediaRerankingMultilingual",
        "BelebeleRetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "MLQARetrieval",
        "WikipediaRetrievalMultilingual",
        "XPQARetrieval",
        "XQuADRetrieval",
        "IndicCrosslingualSTS",
        "SemRel24STS"
    ]
    hi_results = []
    for hi_data in hi_datas:
        hi_results.append([hi_data, all_results['hi'][hi_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(hi_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "hi_results.xlsx")

    bn_datas = [
        "BengaliDocumentClassification",
        "BengaliHateSpeechClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "XNLIV2",
        "IndicReviewsClusteringP2P",
        "SIB200ClusteringS2S",
        "MIRACLReranking",
        "WikipediaRerankingMultilingual",
        "BelebeleRetrieval",
        "IndicQARetrieval",
        "MIRACLRetrieval",
        "WikipediaRetrievalMultilingual",
        "IndicCrosslingualSTS"
    ]
    bn_results = []
    for bn_data in bn_datas:
        bn_results.append([bn_data, all_results['bn'][bn_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(bn_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "bn_results.xlsx")

    te_datas = [
        "IndicNLPNewsClassification",
        "IndicSentimentClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "SIB200Classification",
        "TeluguAndhraJyotiNewsClassification",
        "IndicReviewsClusteringP2P",
        "SIB200ClusteringS2S",
        "MIRACLReranking",
        "BelebeleRetrieval",
        "IndicQARetrieval",
        "MIRACLRetrieval",
        "IndicCrosslingualSTS",
        "SemRel24STS"
    ]
    te_results = []
    for te_data in te_datas:
        te_results.append([te_data, all_results['te'][te_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(te_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "te_results.xlsx")

    sw_datas = [
        "AfriSentiClassification",
        "MasakhaNEWSClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "SwahiliNewsClassification",
        "XNLI",
        "MasakhaNEWSClusteringP2P",
        "MasakhaNEWSClusteringS2S",
        "MIRACLReranking",
        "MIRACLRetrieval"
    ]
    sw_results = []
    for sw_data in sw_datas:
        sw_results.append([sw_data, all_results['sw'][sw_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(sw_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "sw_results.xlsx")

    yo_datas = [
        "AfriSentiClassification",
        "MasakhaNEWSClassification",
        "NaijaSenti",
        "SIB200Classification",
        "MasakhaNEWSClusteringP2P",
        "MasakhaNEWSClusteringS2S",
        "SIB200ClusteringS2S",
        "MIRACLReranking",
        "BelebeleRetrieval",
        "MIRACLRetrieval"
    ]
    yo_results = []
    for yo_data in yo_datas:
        yo_results.append([yo_data, all_results['yo'][yo_data] * 100])
    # Save to a xlsx file with each row as a dataset
    df = pd.DataFrame.from_records(yo_results, columns=["Dataset", "Accuracy"])
    df.to_excel(pathlib.Path(result_dir) / "yo_results.xlsx")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Path to the result directory"
    )

    args = parser.parse_args()
    read_results(args.result_dir)

