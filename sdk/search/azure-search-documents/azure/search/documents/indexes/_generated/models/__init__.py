# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator (autorest: 3.10.2, generator: @autorest/python@6.19.0)
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._models_py3 import AIServicesVisionParameters
from ._models_py3 import AIServicesVisionVectorizer
from ._models_py3 import AnalyzeRequest
from ._models_py3 import AnalyzeResult
from ._models_py3 import AnalyzedTokenInfo
from ._models_py3 import AsciiFoldingTokenFilter
from ._models_py3 import AzureActiveDirectoryApplicationCredentials
from ._models_py3 import AzureMachineLearningParameters
from ._models_py3 import AzureMachineLearningSkill
from ._models_py3 import AzureMachineLearningVectorizer
from ._models_py3 import AzureOpenAIEmbeddingSkill
from ._models_py3 import AzureOpenAITokenizerParameters
from ._models_py3 import AzureOpenAIVectorizer
from ._models_py3 import AzureOpenAIVectorizerParameters
from ._models_py3 import BM25SimilarityAlgorithm
from ._models_py3 import BinaryQuantizationCompression
from ._models_py3 import CharFilter
from ._models_py3 import CjkBigramTokenFilter
from ._models_py3 import ClassicSimilarityAlgorithm
from ._models_py3 import ClassicTokenizer
from ._models_py3 import CognitiveServicesAccount
from ._models_py3 import CognitiveServicesAccountKey
from ._models_py3 import CommonGramTokenFilter
from ._models_py3 import ConditionalSkill
from ._models_py3 import CorsOptions
from ._models_py3 import CustomAnalyzer
from ._models_py3 import CustomEntity
from ._models_py3 import CustomEntityAlias
from ._models_py3 import CustomEntityLookupSkill
from ._models_py3 import CustomNormalizer
from ._models_py3 import DataChangeDetectionPolicy
from ._models_py3 import DataDeletionDetectionPolicy
from ._models_py3 import DataSourceCredentials
from ._models_py3 import DefaultCognitiveServicesAccount
from ._models_py3 import DictionaryDecompounderTokenFilter
from ._models_py3 import DistanceScoringFunction
from ._models_py3 import DistanceScoringParameters
from ._models_py3 import DocumentExtractionSkill
from ._models_py3 import DocumentKeysOrIds
from ._models_py3 import EdgeNGramTokenFilter
from ._models_py3 import EdgeNGramTokenFilterV2
from ._models_py3 import EdgeNGramTokenizer
from ._models_py3 import ElisionTokenFilter
from ._models_py3 import EntityLinkingSkill
from ._models_py3 import EntityRecognitionSkill
from ._models_py3 import EntityRecognitionSkillV3
from ._models_py3 import ErrorAdditionalInfo
from ._models_py3 import ErrorDetail
from ._models_py3 import ErrorResponse
from ._models_py3 import ExhaustiveKnnAlgorithmConfiguration
from ._models_py3 import ExhaustiveKnnParameters
from ._models_py3 import FieldMapping
from ._models_py3 import FieldMappingFunction
from ._models_py3 import FreshnessScoringFunction
from ._models_py3 import FreshnessScoringParameters
from ._models_py3 import GetIndexStatisticsResult
from ._models_py3 import HighWaterMarkChangeDetectionPolicy
from ._models_py3 import HnswAlgorithmConfiguration
from ._models_py3 import HnswParameters
from ._models_py3 import ImageAnalysisSkill
from ._models_py3 import IndexerCurrentState
from ._models_py3 import IndexerExecutionResult
from ._models_py3 import IndexingParameters
from ._models_py3 import IndexingParametersConfiguration
from ._models_py3 import IndexingSchedule
from ._models_py3 import InputFieldMappingEntry
from ._models_py3 import KeepTokenFilter
from ._models_py3 import KeyPhraseExtractionSkill
from ._models_py3 import KeywordMarkerTokenFilter
from ._models_py3 import KeywordTokenizer
from ._models_py3 import KeywordTokenizerV2
from ._models_py3 import LanguageDetectionSkill
from ._models_py3 import LengthTokenFilter
from ._models_py3 import LexicalAnalyzer
from ._models_py3 import LexicalNormalizer
from ._models_py3 import LexicalTokenizer
from ._models_py3 import LimitTokenFilter
from ._models_py3 import ListAliasesResult
from ._models_py3 import ListDataSourcesResult
from ._models_py3 import ListIndexersResult
from ._models_py3 import ListIndexesResult
from ._models_py3 import ListSkillsetsResult
from ._models_py3 import ListSynonymMapsResult
from ._models_py3 import LuceneStandardAnalyzer
from ._models_py3 import LuceneStandardTokenizer
from ._models_py3 import LuceneStandardTokenizerV2
from ._models_py3 import MagnitudeScoringFunction
from ._models_py3 import MagnitudeScoringParameters
from ._models_py3 import MappingCharFilter
from ._models_py3 import MergeSkill
from ._models_py3 import MicrosoftLanguageStemmingTokenizer
from ._models_py3 import MicrosoftLanguageTokenizer
from ._models_py3 import NGramTokenFilter
from ._models_py3 import NGramTokenFilterV2
from ._models_py3 import NGramTokenizer
from ._models_py3 import NativeBlobSoftDeleteDeletionDetectionPolicy
from ._models_py3 import OcrSkill
from ._models_py3 import OutputFieldMappingEntry
from ._models_py3 import PIIDetectionSkill
from ._models_py3 import PathHierarchyTokenizerV2
from ._models_py3 import PatternAnalyzer
from ._models_py3 import PatternCaptureTokenFilter
from ._models_py3 import PatternReplaceCharFilter
from ._models_py3 import PatternReplaceTokenFilter
from ._models_py3 import PatternTokenizer
from ._models_py3 import PhoneticTokenFilter
from ._models_py3 import RequestOptions
from ._models_py3 import ResourceCounter
from ._models_py3 import ScalarQuantizationCompression
from ._models_py3 import ScalarQuantizationParameters
from ._models_py3 import ScoringFunction
from ._models_py3 import ScoringProfile
from ._models_py3 import SearchAlias
from ._models_py3 import SearchField
from ._models_py3 import SearchIndex
from ._models_py3 import SearchIndexer
from ._models_py3 import SearchIndexerCache
from ._models_py3 import SearchIndexerDataContainer
from ._models_py3 import SearchIndexerDataIdentity
from ._models_py3 import SearchIndexerDataNoneIdentity
from ._models_py3 import SearchIndexerDataSource
from ._models_py3 import SearchIndexerDataUserAssignedIdentity
from ._models_py3 import SearchIndexerError
from ._models_py3 import SearchIndexerIndexProjection
from ._models_py3 import SearchIndexerIndexProjectionSelector
from ._models_py3 import SearchIndexerIndexProjectionsParameters
from ._models_py3 import SearchIndexerKnowledgeStore
from ._models_py3 import SearchIndexerKnowledgeStoreBlobProjectionSelector
from ._models_py3 import SearchIndexerKnowledgeStoreFileProjectionSelector
from ._models_py3 import SearchIndexerKnowledgeStoreObjectProjectionSelector
from ._models_py3 import SearchIndexerKnowledgeStoreParameters
from ._models_py3 import SearchIndexerKnowledgeStoreProjection
from ._models_py3 import SearchIndexerKnowledgeStoreProjectionSelector
from ._models_py3 import SearchIndexerKnowledgeStoreTableProjectionSelector
from ._models_py3 import SearchIndexerLimits
from ._models_py3 import SearchIndexerSkill
from ._models_py3 import SearchIndexerSkillset
from ._models_py3 import SearchIndexerStatus
from ._models_py3 import SearchIndexerWarning
from ._models_py3 import SearchResourceEncryptionKey
from ._models_py3 import SearchServiceCounters
from ._models_py3 import SearchServiceLimits
from ._models_py3 import SearchServiceStatistics
from ._models_py3 import SearchSuggester
from ._models_py3 import SemanticConfiguration
from ._models_py3 import SemanticField
from ._models_py3 import SemanticPrioritizedFields
from ._models_py3 import SemanticSearch
from ._models_py3 import SentimentSkill
from ._models_py3 import SentimentSkillV3
from ._models_py3 import ShaperSkill
from ._models_py3 import ShingleTokenFilter
from ._models_py3 import SimilarityAlgorithm
from ._models_py3 import SkillNames
from ._models_py3 import SnowballTokenFilter
from ._models_py3 import SoftDeleteColumnDeletionDetectionPolicy
from ._models_py3 import SplitSkill
from ._models_py3 import SqlIntegratedChangeTrackingPolicy
from ._models_py3 import StemmerOverrideTokenFilter
from ._models_py3 import StemmerTokenFilter
from ._models_py3 import StopAnalyzer
from ._models_py3 import StopwordsTokenFilter
from ._models_py3 import SynonymMap
from ._models_py3 import SynonymTokenFilter
from ._models_py3 import TagScoringFunction
from ._models_py3 import TagScoringParameters
from ._models_py3 import TextTranslationSkill
from ._models_py3 import TextWeights
from ._models_py3 import TokenFilter
from ._models_py3 import TruncateTokenFilter
from ._models_py3 import UaxUrlEmailTokenizer
from ._models_py3 import UniqueTokenFilter
from ._models_py3 import VectorSearch
from ._models_py3 import VectorSearchAlgorithmConfiguration
from ._models_py3 import VectorSearchCompression
from ._models_py3 import VectorSearchProfile
from ._models_py3 import VectorSearchVectorizer
from ._models_py3 import VisionVectorizeSkill
from ._models_py3 import WebApiSkill
from ._models_py3 import WebApiVectorizer
from ._models_py3 import WebApiVectorizerParameters
from ._models_py3 import WordDelimiterTokenFilter

from ._search_service_client_enums import AIStudioModelCatalogName
from ._search_service_client_enums import AzureOpenAIModelName
from ._search_service_client_enums import BlobIndexerDataToExtract
from ._search_service_client_enums import BlobIndexerImageAction
from ._search_service_client_enums import BlobIndexerPDFTextRotationAlgorithm
from ._search_service_client_enums import BlobIndexerParsingMode
from ._search_service_client_enums import CharFilterName
from ._search_service_client_enums import CjkBigramTokenFilterScripts
from ._search_service_client_enums import CustomEntityLookupSkillLanguage
from ._search_service_client_enums import EdgeNGramTokenFilterSide
from ._search_service_client_enums import EntityCategory
from ._search_service_client_enums import EntityRecognitionSkillLanguage
from ._search_service_client_enums import Enum0
from ._search_service_client_enums import ImageAnalysisSkillLanguage
from ._search_service_client_enums import ImageDetail
from ._search_service_client_enums import IndexProjectionMode
from ._search_service_client_enums import IndexerExecutionEnvironment
from ._search_service_client_enums import IndexerExecutionStatus
from ._search_service_client_enums import IndexerExecutionStatusDetail
from ._search_service_client_enums import IndexerStatus
from ._search_service_client_enums import IndexingMode
from ._search_service_client_enums import KeyPhraseExtractionSkillLanguage
from ._search_service_client_enums import LexicalAnalyzerName
from ._search_service_client_enums import LexicalNormalizerName
from ._search_service_client_enums import LexicalTokenizerName
from ._search_service_client_enums import MicrosoftStemmingTokenizerLanguage
from ._search_service_client_enums import MicrosoftTokenizerLanguage
from ._search_service_client_enums import OcrLineEnding
from ._search_service_client_enums import OcrSkillLanguage
from ._search_service_client_enums import PIIDetectionSkillMaskingMode
from ._search_service_client_enums import PhoneticEncoder
from ._search_service_client_enums import RegexFlags
from ._search_service_client_enums import ScoringFunctionAggregation
from ._search_service_client_enums import ScoringFunctionInterpolation
from ._search_service_client_enums import SearchFieldDataType
from ._search_service_client_enums import SearchIndexerDataSourceType
from ._search_service_client_enums import SentimentSkillLanguage
from ._search_service_client_enums import SnowballTokenFilterLanguage
from ._search_service_client_enums import SplitSkillEncoderModelName
from ._search_service_client_enums import SplitSkillLanguage
from ._search_service_client_enums import SplitSkillUnit
from ._search_service_client_enums import StemmerTokenFilterLanguage
from ._search_service_client_enums import StopwordsList
from ._search_service_client_enums import TextSplitMode
from ._search_service_client_enums import TextTranslationSkillLanguage
from ._search_service_client_enums import TokenCharacterKind
from ._search_service_client_enums import TokenFilterName
from ._search_service_client_enums import VectorEncodingFormat
from ._search_service_client_enums import VectorSearchAlgorithmKind
from ._search_service_client_enums import VectorSearchAlgorithmMetric
from ._search_service_client_enums import VectorSearchCompressionKind
from ._search_service_client_enums import VectorSearchCompressionTarget
from ._search_service_client_enums import VectorSearchVectorizerKind
from ._search_service_client_enums import VisualFeature
from ._patch import __all__ as _patch_all
from ._patch import *  # pylint: disable=unused-wildcard-import
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "AIServicesVisionParameters",
    "AIServicesVisionVectorizer",
    "AnalyzeRequest",
    "AnalyzeResult",
    "AnalyzedTokenInfo",
    "AsciiFoldingTokenFilter",
    "AzureActiveDirectoryApplicationCredentials",
    "AzureMachineLearningParameters",
    "AzureMachineLearningSkill",
    "AzureMachineLearningVectorizer",
    "AzureOpenAIEmbeddingSkill",
    "AzureOpenAITokenizerParameters",
    "AzureOpenAIVectorizer",
    "AzureOpenAIVectorizerParameters",
    "BM25SimilarityAlgorithm",
    "BinaryQuantizationCompression",
    "CharFilter",
    "CjkBigramTokenFilter",
    "ClassicSimilarityAlgorithm",
    "ClassicTokenizer",
    "CognitiveServicesAccount",
    "CognitiveServicesAccountKey",
    "CommonGramTokenFilter",
    "ConditionalSkill",
    "CorsOptions",
    "CustomAnalyzer",
    "CustomEntity",
    "CustomEntityAlias",
    "CustomEntityLookupSkill",
    "CustomNormalizer",
    "DataChangeDetectionPolicy",
    "DataDeletionDetectionPolicy",
    "DataSourceCredentials",
    "DefaultCognitiveServicesAccount",
    "DictionaryDecompounderTokenFilter",
    "DistanceScoringFunction",
    "DistanceScoringParameters",
    "DocumentExtractionSkill",
    "DocumentKeysOrIds",
    "EdgeNGramTokenFilter",
    "EdgeNGramTokenFilterV2",
    "EdgeNGramTokenizer",
    "ElisionTokenFilter",
    "EntityLinkingSkill",
    "EntityRecognitionSkill",
    "EntityRecognitionSkillV3",
    "ErrorAdditionalInfo",
    "ErrorDetail",
    "ErrorResponse",
    "ExhaustiveKnnAlgorithmConfiguration",
    "ExhaustiveKnnParameters",
    "FieldMapping",
    "FieldMappingFunction",
    "FreshnessScoringFunction",
    "FreshnessScoringParameters",
    "GetIndexStatisticsResult",
    "HighWaterMarkChangeDetectionPolicy",
    "HnswAlgorithmConfiguration",
    "HnswParameters",
    "ImageAnalysisSkill",
    "IndexerCurrentState",
    "IndexerExecutionResult",
    "IndexingParameters",
    "IndexingParametersConfiguration",
    "IndexingSchedule",
    "InputFieldMappingEntry",
    "KeepTokenFilter",
    "KeyPhraseExtractionSkill",
    "KeywordMarkerTokenFilter",
    "KeywordTokenizer",
    "KeywordTokenizerV2",
    "LanguageDetectionSkill",
    "LengthTokenFilter",
    "LexicalAnalyzer",
    "LexicalNormalizer",
    "LexicalTokenizer",
    "LimitTokenFilter",
    "ListAliasesResult",
    "ListDataSourcesResult",
    "ListIndexersResult",
    "ListIndexesResult",
    "ListSkillsetsResult",
    "ListSynonymMapsResult",
    "LuceneStandardAnalyzer",
    "LuceneStandardTokenizer",
    "LuceneStandardTokenizerV2",
    "MagnitudeScoringFunction",
    "MagnitudeScoringParameters",
    "MappingCharFilter",
    "MergeSkill",
    "MicrosoftLanguageStemmingTokenizer",
    "MicrosoftLanguageTokenizer",
    "NGramTokenFilter",
    "NGramTokenFilterV2",
    "NGramTokenizer",
    "NativeBlobSoftDeleteDeletionDetectionPolicy",
    "OcrSkill",
    "OutputFieldMappingEntry",
    "PIIDetectionSkill",
    "PathHierarchyTokenizerV2",
    "PatternAnalyzer",
    "PatternCaptureTokenFilter",
    "PatternReplaceCharFilter",
    "PatternReplaceTokenFilter",
    "PatternTokenizer",
    "PhoneticTokenFilter",
    "RequestOptions",
    "ResourceCounter",
    "ScalarQuantizationCompression",
    "ScalarQuantizationParameters",
    "ScoringFunction",
    "ScoringProfile",
    "SearchAlias",
    "SearchField",
    "SearchIndex",
    "SearchIndexer",
    "SearchIndexerCache",
    "SearchIndexerDataContainer",
    "SearchIndexerDataIdentity",
    "SearchIndexerDataNoneIdentity",
    "SearchIndexerDataSource",
    "SearchIndexerDataUserAssignedIdentity",
    "SearchIndexerError",
    "SearchIndexerIndexProjection",
    "SearchIndexerIndexProjectionSelector",
    "SearchIndexerIndexProjectionsParameters",
    "SearchIndexerKnowledgeStore",
    "SearchIndexerKnowledgeStoreBlobProjectionSelector",
    "SearchIndexerKnowledgeStoreFileProjectionSelector",
    "SearchIndexerKnowledgeStoreObjectProjectionSelector",
    "SearchIndexerKnowledgeStoreParameters",
    "SearchIndexerKnowledgeStoreProjection",
    "SearchIndexerKnowledgeStoreProjectionSelector",
    "SearchIndexerKnowledgeStoreTableProjectionSelector",
    "SearchIndexerLimits",
    "SearchIndexerSkill",
    "SearchIndexerSkillset",
    "SearchIndexerStatus",
    "SearchIndexerWarning",
    "SearchResourceEncryptionKey",
    "SearchServiceCounters",
    "SearchServiceLimits",
    "SearchServiceStatistics",
    "SearchSuggester",
    "SemanticConfiguration",
    "SemanticField",
    "SemanticPrioritizedFields",
    "SemanticSearch",
    "SentimentSkill",
    "SentimentSkillV3",
    "ShaperSkill",
    "ShingleTokenFilter",
    "SimilarityAlgorithm",
    "SkillNames",
    "SnowballTokenFilter",
    "SoftDeleteColumnDeletionDetectionPolicy",
    "SplitSkill",
    "SqlIntegratedChangeTrackingPolicy",
    "StemmerOverrideTokenFilter",
    "StemmerTokenFilter",
    "StopAnalyzer",
    "StopwordsTokenFilter",
    "SynonymMap",
    "SynonymTokenFilter",
    "TagScoringFunction",
    "TagScoringParameters",
    "TextTranslationSkill",
    "TextWeights",
    "TokenFilter",
    "TruncateTokenFilter",
    "UaxUrlEmailTokenizer",
    "UniqueTokenFilter",
    "VectorSearch",
    "VectorSearchAlgorithmConfiguration",
    "VectorSearchCompression",
    "VectorSearchProfile",
    "VectorSearchVectorizer",
    "VisionVectorizeSkill",
    "WebApiSkill",
    "WebApiVectorizer",
    "WebApiVectorizerParameters",
    "WordDelimiterTokenFilter",
    "AIStudioModelCatalogName",
    "AzureOpenAIModelName",
    "BlobIndexerDataToExtract",
    "BlobIndexerImageAction",
    "BlobIndexerPDFTextRotationAlgorithm",
    "BlobIndexerParsingMode",
    "CharFilterName",
    "CjkBigramTokenFilterScripts",
    "CustomEntityLookupSkillLanguage",
    "EdgeNGramTokenFilterSide",
    "EntityCategory",
    "EntityRecognitionSkillLanguage",
    "Enum0",
    "ImageAnalysisSkillLanguage",
    "ImageDetail",
    "IndexProjectionMode",
    "IndexerExecutionEnvironment",
    "IndexerExecutionStatus",
    "IndexerExecutionStatusDetail",
    "IndexerStatus",
    "IndexingMode",
    "KeyPhraseExtractionSkillLanguage",
    "LexicalAnalyzerName",
    "LexicalNormalizerName",
    "LexicalTokenizerName",
    "MicrosoftStemmingTokenizerLanguage",
    "MicrosoftTokenizerLanguage",
    "OcrLineEnding",
    "OcrSkillLanguage",
    "PIIDetectionSkillMaskingMode",
    "PhoneticEncoder",
    "RegexFlags",
    "ScoringFunctionAggregation",
    "ScoringFunctionInterpolation",
    "SearchFieldDataType",
    "SearchIndexerDataSourceType",
    "SentimentSkillLanguage",
    "SnowballTokenFilterLanguage",
    "SplitSkillEncoderModelName",
    "SplitSkillLanguage",
    "SplitSkillUnit",
    "StemmerTokenFilterLanguage",
    "StopwordsList",
    "TextSplitMode",
    "TextTranslationSkillLanguage",
    "TokenCharacterKind",
    "TokenFilterName",
    "VectorEncodingFormat",
    "VectorSearchAlgorithmKind",
    "VectorSearchAlgorithmMetric",
    "VectorSearchCompressionKind",
    "VectorSearchCompressionTarget",
    "VectorSearchVectorizerKind",
    "VisualFeature",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()
