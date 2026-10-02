from __future__ import annotations


ENTITY_CATALOGUE = [
    # Model families
    {"entity_id": "family:llama", "canonical_name": "Llama", "entity_type": "MODEL_FAMILY", "aliases": ["Llama"]},
    {"entity_id": "family:mistral", "canonical_name": "Mistral", "entity_type": "MODEL_FAMILY", "aliases": ["Mistral"]},
    {"entity_id": "family:gemma", "canonical_name": "Gemma", "entity_type": "MODEL_FAMILY", "aliases": ["Gemma"]},
    {"entity_id": "family:qwen", "canonical_name": "Qwen", "entity_type": "MODEL_FAMILY", "aliases": ["Qwen"]},
    {"entity_id": "family:deepseek", "canonical_name": "DeepSeek", "entity_type": "MODEL_FAMILY", "aliases": ["DeepSeek models", "DeepSeek model"]},
    {"entity_id": "family:falcon", "canonical_name": "Falcon", "entity_type": "MODEL_FAMILY", "aliases": ["Falcon models", "Falcon LLM"]},
    {"entity_id": "family:bloom", "canonical_name": "BLOOM", "entity_type": "MODEL_FAMILY", "aliases": ["BLOOM"]},
    {"entity_id": "family:olmo", "canonical_name": "OLMo", "entity_type": "MODEL_FAMILY", "aliases": ["OLMo"]},
    {"entity_id": "family:phi", "canonical_name": "Phi", "entity_type": "MODEL_FAMILY", "aliases": ["Phi models", "Phi model"]},
    {"entity_id": "family:smollm", "canonical_name": "SmolLM", "entity_type": "MODEL_FAMILY", "aliases": ["SmolLM"]},
    {"entity_id": "family:apertus", "canonical_name": "Apertus", "entity_type": "MODEL_FAMILY", "aliases": ["Apertus"]},
    # Named model releases
    {"entity_id": "model:llama_2", "canonical_name": "Llama 2", "entity_type": "MODEL", "aliases": ["Llama 2", "Llama-2"], "parent_family_id": "family:llama"},
    {"entity_id": "model:llama_3", "canonical_name": "Llama 3", "entity_type": "MODEL", "aliases": ["Llama 3", "Llama-3"], "parent_family_id": "family:llama"},
    {"entity_id": "model:llama_3_1", "canonical_name": "Llama 3.1", "entity_type": "MODEL", "aliases": ["Llama 3.1", "Llama-3.1"], "parent_family_id": "family:llama"},
    {"entity_id": "model:llama_3_2", "canonical_name": "Llama 3.2", "entity_type": "MODEL", "aliases": ["Llama 3.2", "Llama-3.2"], "parent_family_id": "family:llama"},
    {"entity_id": "model:llama_3_3", "canonical_name": "Llama 3.3", "entity_type": "MODEL", "aliases": ["Llama 3.3", "Llama-3.3"], "parent_family_id": "family:llama"},
    {"entity_id": "model:mistral_7b", "canonical_name": "Mistral 7B", "entity_type": "MODEL", "aliases": ["Mistral 7B", "Mistral-7B"], "parent_family_id": "family:mistral"},
    {"entity_id": "model:mixtral_8x7b", "canonical_name": "Mixtral 8x7B", "entity_type": "MODEL", "aliases": ["Mixtral 8x7B", "Mixtral-8x7B"], "parent_family_id": "family:mistral"},
    {"entity_id": "model:mixtral_8x22b", "canonical_name": "Mixtral 8x22B", "entity_type": "MODEL", "aliases": ["Mixtral 8x22B", "Mixtral-8x22B"], "parent_family_id": "family:mistral"},
    {"entity_id": "model:gemma_2", "canonical_name": "Gemma 2", "entity_type": "MODEL", "aliases": ["Gemma 2", "Gemma-2"], "parent_family_id": "family:gemma"},
    {"entity_id": "model:gemma_3", "canonical_name": "Gemma 3", "entity_type": "MODEL", "aliases": ["Gemma 3", "Gemma-3"], "parent_family_id": "family:gemma"},
    {"entity_id": "model:gemma_4", "canonical_name": "Gemma 4", "entity_type": "MODEL", "aliases": ["Gemma 4", "Gemma-4"], "parent_family_id": "family:gemma"},
    {"entity_id": "model:qwen_2", "canonical_name": "Qwen2", "entity_type": "MODEL", "aliases": ["Qwen2", "Qwen 2"], "parent_family_id": "family:qwen"},
    {"entity_id": "model:qwen_2_5", "canonical_name": "Qwen2.5", "entity_type": "MODEL", "aliases": ["Qwen2.5", "Qwen 2.5"], "parent_family_id": "family:qwen"},
    {"entity_id": "model:qwen_3", "canonical_name": "Qwen3", "entity_type": "MODEL", "aliases": ["Qwen3", "Qwen 3"], "parent_family_id": "family:qwen"},
    {"entity_id": "model:deepseek_v2", "canonical_name": "DeepSeek-V2", "entity_type": "MODEL", "aliases": ["DeepSeek-V2", "DeepSeek V2"], "parent_family_id": "family:deepseek"},
    {"entity_id": "model:deepseek_v3", "canonical_name": "DeepSeek-V3", "entity_type": "MODEL", "aliases": ["DeepSeek-V3", "DeepSeek V3"], "parent_family_id": "family:deepseek"},
    {"entity_id": "model:deepseek_r1", "canonical_name": "DeepSeek-R1", "entity_type": "MODEL", "aliases": ["DeepSeek-R1", "DeepSeek R1"], "parent_family_id": "family:deepseek"},
    {"entity_id": "model:phi_3", "canonical_name": "Phi-3", "entity_type": "MODEL", "aliases": ["Phi-3", "Phi 3"], "parent_family_id": "family:phi"},
    {"entity_id": "model:phi_4", "canonical_name": "Phi-4", "entity_type": "MODEL", "aliases": ["Phi-4", "Phi 4"], "parent_family_id": "family:phi"},
    # Organisations
    {"entity_id": "org:meta", "canonical_name": "Meta", "entity_type": "ORGANIZATION", "aliases": ["Meta", "Meta AI"]},
    {"entity_id": "org:mistral_ai", "canonical_name": "Mistral AI", "entity_type": "ORGANIZATION", "aliases": ["Mistral AI"]},
    {"entity_id": "org:google", "canonical_name": "Google", "entity_type": "ORGANIZATION", "aliases": ["Google", "Google DeepMind", "DeepMind"]},
    {"entity_id": "org:alibaba", "canonical_name": "Alibaba", "entity_type": "ORGANIZATION", "aliases": ["Alibaba", "Alibaba Cloud"]},
    {"entity_id": "org:deepseek", "canonical_name": "DeepSeek", "entity_type": "ORGANIZATION", "aliases": ["DeepSeek AI", "DeepSeek company"]},
    {"entity_id": "org:hugging_face", "canonical_name": "Hugging Face", "entity_type": "ORGANIZATION", "aliases": ["Hugging Face", "HuggingFace"]},
    {"entity_id": "org:microsoft", "canonical_name": "Microsoft", "entity_type": "ORGANIZATION", "aliases": ["Microsoft"]},
    {"entity_id": "org:technology_innovation_institute", "canonical_name": "Technology Innovation Institute", "entity_type": "ORGANIZATION", "aliases": ["Technology Innovation Institute", "TII"]},
    {"entity_id": "org:allen_ai", "canonical_name": "Allen Institute for AI", "entity_type": "ORGANIZATION", "aliases": ["Allen Institute for AI", "AI2"]},
    {"entity_id": "org:bigscience", "canonical_name": "BigScience", "entity_type": "ORGANIZATION", "aliases": ["BigScience"]},
    # Licences
    {"entity_id": "license:apache_2_0", "canonical_name": "Apache License 2.0", "entity_type": "LICENSE", "aliases": ["Apache 2.0", "Apache License 2.0", "Apache-2.0"]},
    {"entity_id": "license:mit", "canonical_name": "MIT License", "entity_type": "LICENSE", "aliases": ["MIT License", "MIT licence"]},
    {"entity_id": "license:openrail", "canonical_name": "OpenRAIL", "entity_type": "LICENSE", "aliases": ["OpenRAIL", "Open RAIL"]},
    {"entity_id": "license:llama_community", "canonical_name": "Llama Community License", "entity_type": "LICENSE", "aliases": ["Llama Community License", "Llama license", "Llama licence"]},
    {"entity_id": "license:gemma_terms", "canonical_name": "Gemma Terms of Use", "entity_type": "LICENSE", "aliases": ["Gemma Terms of Use", "Gemma license", "Gemma licence"]},
    # Architectures
    {"entity_id": "architecture:transformer", "canonical_name": "Transformer", "entity_type": "ARCHITECTURE", "aliases": ["Transformer architecture", "transformer-based"]},
    {"entity_id": "architecture:moe", "canonical_name": "Mixture of Experts", "entity_type": "ARCHITECTURE", "aliases": ["mixture of experts", "mixture-of-experts", "MoE architecture"]},
    {"entity_id": "architecture:decoder_only", "canonical_name": "Decoder-only Transformer", "entity_type": "ARCHITECTURE", "aliases": ["decoder-only", "decoder only architecture"]},
    {"entity_id": "architecture:multimodal", "canonical_name": "Multimodal Architecture", "entity_type": "ARCHITECTURE", "aliases": ["multimodal architecture", "multimodal model"]},
    # Capabilities
    {"entity_id": "capability:reasoning", "canonical_name": "Reasoning", "entity_type": "CAPABILITY", "aliases": ["reasoning capability", "reasoning tasks", "reasoning model"]},
    {"entity_id": "capability:code_generation", "canonical_name": "Code Generation", "entity_type": "CAPABILITY", "aliases": ["code generation", "coding tasks", "code completion"]},
    {"entity_id": "capability:multilingual", "canonical_name": "Multilingual Processing", "entity_type": "CAPABILITY", "aliases": ["multilingual capability", "multilingual support", "multiple languages"]},
    {"entity_id": "capability:vision", "canonical_name": "Vision", "entity_type": "CAPABILITY", "aliases": ["vision capability", "image understanding", "visual understanding"]},
    {"entity_id": "capability:tool_use", "canonical_name": "Tool Use", "entity_type": "CAPABILITY", "aliases": ["tool use", "tool calling", "function calling"]},
    {"entity_id": "capability:long_context", "canonical_name": "Long Context", "entity_type": "CAPABILITY", "aliases": ["long context", "long-context", "context window"]},
]


CATALOGUE_BY_ID = {row["entity_id"]: row for row in ENTITY_CATALOGUE}


RELATION_SIGNATURES = {
    "DEVELOPED_BY": {("MODEL", "ORGANIZATION"), ("MODEL_FAMILY", "ORGANIZATION")},
    "RELEASED_BY": {("MODEL", "ORGANIZATION"), ("MODEL_FAMILY", "ORGANIZATION")},
    "BELONGS_TO_FAMILY": {("MODEL", "MODEL_FAMILY")},
    "SUCCESSOR_OF": {("MODEL", "MODEL")},
    "DERIVED_FROM": {("MODEL", "MODEL"), ("MODEL", "MODEL_FAMILY")},
    "LICENSED_UNDER": {("MODEL", "LICENSE"), ("MODEL_FAMILY", "LICENSE")},
    "USES_ARCHITECTURE": {("MODEL", "ARCHITECTURE"), ("MODEL_FAMILY", "ARCHITECTURE")},
    "HAS_CAPABILITY": {("MODEL", "CAPABILITY"), ("MODEL_FAMILY", "CAPABILITY")},
}


OPEN_MODEL_TAGS = {
    "DeepSeek",
    "HuggingFace",
    "Llama",
    "Mistral",
    "Gemma",
    "Mixtral",
    "Olmo",
    "BLOOM",
    "Apertus",
    "SmolLM",
}


OPEN_MODEL_TEXT_PATTERN = (
    r"\b(?:open[ -]source (?:ai |language |foundation |large language )?models?"
    r"|open[ -]weight(?:ed)? (?:ai |language |foundation |large language )?models?"
    r"|model weights|llama(?:[ -]?[23](?:\.\d)?)?|mixtral|mistral(?:[ -]?7b)?"
    r"|gemma(?:[ -]?[234])?|qwen(?:[ -]?(?:2(?:\.5)?|3))?"
    r"|deepseek(?:[ -]?(?:v[23]|r1))?|falcon llm|\bbloom\b|\bolmo\b"
    r"|phi[ -]?[34]|smollm|apertus)\b"
)

