"""Unreal C++ semantic parser tests covering AST and macro edge cases."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ue_rag.crawler.engine import EngineFileRecord, EngineFileType
from ue_rag.jsonl import load_jsonl, save_jsonl
from ue_rag.parser.cpp import (
    CPPParserConfig,
    CPPSymbolType,
    UnrealCPPParser,
    load_cpp_parser_config,
)
from ue_rag.schema import SourceScope, SourceType, UEDocument


SEMANTIC_MACROS = {
    "UCLASS",
    "USTRUCT",
    "UENUM",
    "UINTERFACE",
    "UFUNCTION",
    "UPROPERTY",
    "UDELEGATE",
}
MASK_ONLY_MACROS = {
    "GENERATED_BODY",
    "GENERATED_UCLASS_BODY",
    "GENERATED_USTRUCT_BODY",
    "GENERATED_IINTERFACE_BODY",
    "UMETA",
    "UPARAM",
    "UE_DEPRECATED",
}


def make_config(tmp_path: Path) -> CPPParserConfig:
    return CPPParserConfig(
        engine_version="5.8",
        engine_root=tmp_path,
        inventory_path=tmp_path / "files.jsonl",
        output_path=tmp_path / "documents.jsonl",
        issues_path=tmp_path / "issues.jsonl",
        include_file_types={
            EngineFileType.HEADER,
            EngineFileType.CPP,
            EngineFileType.INL,
        },
        semantic_macros=SEMANTIC_MACROS,
        mask_only_macros=MASK_ONLY_MACROS,
        mask_only_macro_patterns=[r"UE_DEPRECATED[A-Z0-9_]*"],
        api_macro_pattern=r"\b[A-Z][A-Z0-9_]*_API\b",
    )


def make_record(
    source: bytes,
    *,
    relative_path: str = "Engine/Source/Runtime/Test/Public/Test.h",
    module: str = "Test",
    plugin: str | None = None,
    file_type: EngineFileType = EngineFileType.HEADER,
    engine_version: str = "5.8",
) -> EngineFileRecord:
    return EngineFileRecord(
        relative_path=relative_path,
        module=module,
        plugin=plugin,
        file_type=file_type,
        engine_version=engine_version,
        size=len(source),
        sha256=hashlib.sha256(source).hexdigest(),
    )


def parse(tmp_path: Path, text: str) -> tuple[list[UEDocument], bool]:
    source = text.encode("utf-8")
    parser = UnrealCPPParser(make_config(tmp_path))
    return parser.parse_source(source, make_record(source))


def get_symbol(documents: list[UEDocument], symbol: str) -> UEDocument:
    return next(document for document in documents if document.symbol == symbol)


def macro_names(document: UEDocument) -> list[str]:
    return [macro["name"] for macro in document.metadata["ue_macros"]]


def test_uclass_is_attached_to_class(tmp_path: Path) -> None:
    documents, has_error = parse(
        tmp_path, "UCLASS(Blueprintable)\nclass TEST_API AHero : public AActor {};"
    )
    document = get_symbol(documents, "AHero")

    assert document.symbol_type == CPPSymbolType.CLASS.value
    assert macro_names(document) == ["UCLASS"]
    assert document.metadata["ue_macros"][0]["arguments"] == "Blueprintable"
    assert document.content.startswith("UCLASS(Blueprintable)")
    assert has_error is False


def test_ustruct_is_attached_to_struct(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path, "USTRUCT(BlueprintType)\nstruct FItem { GENERATED_BODY() int32 Count; };"
    )
    document = get_symbol(documents, "FItem")

    assert document.symbol_type == CPPSymbolType.STRUCT.value
    assert macro_names(document) == ["USTRUCT"]


def test_uenum_is_attached_to_enum(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path,
        "UENUM(BlueprintType)\nenum class EMode : uint8 { Walk UMETA(DisplayName=\"Walk\"), Run };",
    )
    document = get_symbol(documents, "EMode")

    assert document.symbol_type == CPPSymbolType.ENUM.value
    assert macro_names(document) == ["UENUM"]
    assert "UMETA" in document.content


def test_uinterface_is_attached_to_class(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path, "UINTERFACE(MinimalAPI)\nclass UInteractable : public UInterface {};"
    )

    assert macro_names(get_symbol(documents, "UInteractable")) == ["UINTERFACE"]


def test_uproperty_is_attached_to_field(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path,
        "class AHero { UPROPERTY(EditAnywhere, BlueprintReadWrite) float Speed; };",
    )
    document = get_symbol(documents, "AHero::Speed")

    assert document.symbol_type == CPPSymbolType.FIELD.value
    assert macro_names(document) == ["UPROPERTY"]
    assert document.metadata["ue_macros"][0]["arguments"] == (
        "EditAnywhere, BlueprintReadWrite"
    )


def test_ufunction_is_attached_to_method(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path,
        "class AHero { UFUNCTION(BlueprintCallable) void Jump(float Height); };",
    )
    document = get_symbol(documents, "AHero::Jump")

    assert document.symbol_type == CPPSymbolType.METHOD.value
    assert document.function_name == "Jump"
    assert macro_names(document) == ["UFUNCTION"]


def test_constructor_declaration_is_identified(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "class AHero { AHero(); }; ")
    document = get_symbol(documents, "AHero::AHero")

    assert document.symbol_type == CPPSymbolType.CONSTRUCTOR.value
    assert document.function_name == "AHero"


def test_constructor_definition_is_identified(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "AHero::AHero() {}")
    document = get_symbol(documents, "AHero::AHero")

    assert document.symbol_type == CPPSymbolType.CONSTRUCTOR.value


def test_destructor_is_a_method(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "class FThing { ~FThing(); }; ")
    document = get_symbol(documents, "FThing::~FThing")

    assert document.symbol_type == CPPSymbolType.METHOD.value


def test_regular_method_declaration_is_identified(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "class FThing { virtual int GetValue() const; }; ")

    assert get_symbol(documents, "FThing::GetValue").symbol_type == "method"


def test_qualified_method_definition_is_identified(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "void UMovement::Tick(float Delta) { Update(); }")
    document = get_symbol(documents, "UMovement::Tick")

    assert document.class_name == "UMovement"
    assert document.symbol_type == "method"


def test_free_function_declaration_is_identified(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "int CalculateScore(int Value);")
    document = get_symbol(documents, "CalculateScore")

    assert document.symbol_type == "function"
    assert document.class_name is None


def test_free_function_definition_is_identified(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "int CalculateScore(int Value) { return Value * 2; }")

    assert get_symbol(documents, "CalculateScore").content.endswith("}")


def test_namespace_function_keeps_qualified_symbol(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "namespace UE { void Helper() {} }")
    document = get_symbol(documents, "UE::Helper")

    assert document.symbol_type == "function"
    assert document.class_name is None


def test_multiple_fields_emit_individual_documents(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "class FPair { int32 X, Y; }; ")
    symbols = {document.symbol for document in documents}

    assert {"FPair::X", "FPair::Y"} <= symbols


def test_pointer_and_array_fields_are_identified(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "class FData { UObject* Owner; int32 Values[4]; }; ")
    symbols = {document.symbol for document in documents}

    assert {"FData::Owner", "FData::Values"} <= symbols


def test_nested_enum_keeps_class_context(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "class FState { enum class EMode { A, B }; }; ")
    document = get_symbol(documents, "FState::EMode")

    assert document.class_name == "FState"


def test_template_class_and_field_are_identified(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path, "template <typename T> class TBox { T Value; };"
    )
    symbols = {document.symbol for document in documents}

    assert {"TBox", "TBox::Value"} <= symbols


def test_multiple_inheritance_is_preserved(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path, "class FChild : public FBase, private IInterface {};"
    )

    assert get_symbol(documents, "FChild").metadata["inheritance"] == [
        "FBase",
        "IInterface",
    ]


def test_nested_macro_arguments_are_captured_whole(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path,
        'class AHero { UPROPERTY(meta=(ClampMin="0", EditCondition="CanMove()")) float Speed; };',
    )
    arguments = get_symbol(documents, "AHero::Speed").metadata["ue_macros"][0][
        "arguments"
    ]

    assert arguments == 'meta=(ClampMin="0", EditCondition="CanMove()")'


def test_parenthesis_inside_macro_string_does_not_end_macro(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path,
        'UCLASS(meta=(DisplayName="Hero (Player)"))\nclass AHero {};',
    )

    assert "Hero (Player)" in get_symbol(documents, "AHero").metadata["ue_macros"][0][
        "arguments"
    ]


def test_utf8_before_macro_preserves_byte_and_line_offsets(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path,
        "// 角色类型\n\nUCLASS()\nclass AHero {};",
    )
    document = get_symbol(documents, "AHero")

    assert document.metadata["line_start"] == 3
    assert document.metadata["line_end"] == 4
    assert document.content == "UCLASS()\nclass AHero {}"


def test_comment_between_macro_and_symbol_is_allowed_trivia(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path, "UCLASS()\n// reflected actor\nclass AHero {};"
    )

    assert macro_names(get_symbol(documents, "AHero")) == ["UCLASS"]


def test_nontrivia_between_macro_and_symbol_prevents_attachment(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path, "UCLASS()\nint Other;\nclass AHero {};"
    )

    assert macro_names(get_symbol(documents, "AHero")) == []


def test_api_export_macro_is_masked_and_recorded(tmp_path: Path) -> None:
    documents, has_error = parse(tmp_path, "class ENGINE_API AHero {};")
    document = get_symbol(documents, "AHero")

    assert document.metadata["api_macro"] == "ENGINE_API"
    assert has_error is False


def test_uparam_is_masked_without_losing_method(tmp_path: Path) -> None:
    documents, has_error = parse(
        tmp_path,
        "class AHero { void SetTarget(UPARAM(ref) UObject*& Target); };",
    )

    assert get_symbol(documents, "AHero::SetTarget")
    assert has_error is False


def test_deprecation_macro_variants_do_not_break_class_ast(tmp_path: Path) -> None:
    documents, has_error = parse(
        tmp_path,
        'class AHero { UE_DEPRECATED_FORGAME(5.0, "Use NewSpeed") '
        'UPROPERTY() float OldSpeed; UPROPERTY() float MaxWalkSpeed; };',
    )

    assert get_symbol(documents, "AHero::OldSpeed")
    assert macro_names(get_symbol(documents, "AHero::MaxWalkSpeed")) == ["UPROPERTY"]
    assert has_error is False


def test_operator_method_is_identified(tmp_path: Path) -> None:
    documents, _ = parse(
        tmp_path, "class FVectorLike { FVectorLike operator+(const FVectorLike&) const; };"
    )

    assert get_symbol(documents, "FVectorLike::operator+").symbol_type == "method"


def test_recoverable_syntax_error_still_emits_valid_symbol(tmp_path: Path) -> None:
    documents, has_error = parse(
        tmp_path, "this is invalid !!!;\nvoid ValidFunction() {}"
    )

    assert has_error is True
    assert get_symbol(documents, "ValidFunction").metadata["ast_has_error"] is False
    assert get_symbol(documents, "ValidFunction").metadata["file_ast_has_error"] is True


def test_deep_global_initializer_is_not_walked_as_semantic_source(tmp_path: Path) -> None:
    nested_initializer = "{" * 1500 + "0" + "}" * 1500
    documents, _ = parse(tmp_path, f"static int Lookup[] = {nested_initializer};")

    assert documents == []


def test_deep_recovery_comma_expression_is_not_walked(tmp_path: Path) -> None:
    malformed_table = ",".join(str(index) for index in range(2000))
    documents, has_error = parse(tmp_path, malformed_table)

    assert documents == []
    assert has_error is True


def test_document_has_unified_source_provenance(tmp_path: Path) -> None:
    source = b"class AHero {};"
    record = make_record(
        source,
        relative_path="Engine/Plugins/GameFeatures/Hero/Source/Hero/Hero.h",
        module="Hero",
        plugin="GameFeatures",
    )
    documents, _ = UnrealCPPParser(make_config(tmp_path)).parse_source(source, record)
    document = get_symbol(documents, "AHero")

    assert document.source_scope is SourceScope.GLOBAL
    assert document.source_type is SourceType.ENGINE_SOURCE
    assert document.engine_version == "5.8"
    assert document.module == "Hero"
    assert document.plugin == "GameFeatures"
    assert document.file_path == record.relative_path
    assert document.metadata["file_sha256"] == record.sha256


def test_symbol_ids_are_deterministic_and_overloads_are_distinct(tmp_path: Path) -> None:
    text = "class FMathLike { int Add(int A); float Add(float A); };"
    first, _ = parse(tmp_path, text)
    second, _ = parse(tmp_path, text)
    overloads = [document for document in first if document.symbol == "FMathLike::Add"]

    assert [document.id for document in first] == [document.id for document in second]
    assert len(overloads) == 2
    assert len({document.id for document in overloads}) == 2


def test_duplicate_ast_visits_do_not_emit_duplicate_documents(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "typedef struct Foo Foo;")

    identities = [
        (
            document.metadata["line_start"],
            document.metadata["line_end"],
            document.symbol_type,
            document.symbol,
            document.content,
        )
        for document in documents
    ]
    assert len(identities) == len(set(identities))
    assert len({document.id for document in documents}) == len(documents)


def test_anonymous_enum_is_not_emitted_with_empty_symbol(tmp_path: Path) -> None:
    documents, _ = parse(tmp_path, "enum : int { First = 1, Second = 2 };")

    assert all(document.symbol for document in documents)
    assert not any(document.symbol_type == "enum" for document in documents)


def test_engine_version_mismatch_is_rejected(tmp_path: Path) -> None:
    source = b"class AHero {};"
    parser = UnrealCPPParser(make_config(tmp_path))

    with pytest.raises(ValueError, match="does not match configured version"):
        parser.parse_source(source, make_record(source, engine_version="5.7"))


def test_inventory_pipeline_skips_build_cs_and_filters_module(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    header_source = b"UCLASS()\nclass AHero {};"
    other_source = b"class FOther {};"
    build_source = b"public class Test : ModuleRules {}"
    records = [
        make_record(header_source),
        make_record(
            other_source,
            relative_path="Engine/Source/Runtime/Other/Other.h",
            module="Other",
        ),
        make_record(
            build_source,
            relative_path="Engine/Source/Runtime/Test/Test.Build.cs",
            file_type=EngineFileType.BUILD_CS,
        ),
    ]
    for record, source in zip(records, (header_source, other_source, build_source)):
        path = tmp_path / Path(record.relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
    save_jsonl(config.inventory_path, records)
    output = tmp_path / "filtered.jsonl"

    summary = UnrealCPPParser(config).parse_inventory(
        modules={"Test"}, output_path=output
    )
    documents = load_jsonl(output, UEDocument)

    assert summary.inventory_files == 3
    assert summary.selected_files == summary.parsed_files == 1
    assert summary.documents == 1
    assert summary.issues == 0
    assert [document.symbol for document in documents] == ["AHero"]
    assert summary.by_symbol_type[CPPSymbolType.CLASS] == 1


def test_filtered_inventory_run_requires_positive_limit(tmp_path: Path) -> None:
    parser = UnrealCPPParser(make_config(tmp_path))

    with pytest.raises(ValueError, match="greater than zero"):
        parser.parse_inventory(limit_files=0)


def test_workspace_cpp_config_uses_inventory_and_ue_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UE_ROOT", str(tmp_path / "UE_5.8"))
    config = load_cpp_parser_config()

    assert config.engine_version == "5.8"
    assert config.inventory_path.name == "files.jsonl"
    assert config.output_path.name == "documents.jsonl"
    assert EngineFileType.BUILD_CS not in config.include_file_types
