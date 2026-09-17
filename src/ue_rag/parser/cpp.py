"""Tree-sitter based Unreal C++ semantic symbol parser."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import tree_sitter_cpp
import yaml
from pydantic import BaseModel, ConfigDict, Field
from tree_sitter import Language, Node, Parser

from ue_rag.crawler.engine import EngineFileRecord, EngineFileType
from ue_rag.schema import SourceScope, SourceType, UEDocument


PARSER_VERSION = "1"


class CPPSymbolType(str, Enum):
    """Semantic C++ symbols emitted as individual documents."""

    CLASS = "class"
    STRUCT = "struct"
    ENUM = "enum"
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    FIELD = "field"


class UEMacro(BaseModel):
    """One Unreal reflection macro captured from the original source."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str
    raw: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)


@dataclass(frozen=True)
class _MacroSpan:
    macro: UEMacro
    start_byte: int
    end_byte: int


class CPPParserConfig(BaseModel):
    """Resolved input/output paths and UE preprocessing rules."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    engine_root: Path
    inventory_path: Path
    output_path: Path
    issues_path: Path
    include_file_types: set[EngineFileType] = Field(min_length=1)
    semantic_macros: set[str]
    mask_only_macros: set[str]
    mask_only_macro_patterns: list[str]
    api_macro_pattern: str


def load_cpp_parser_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    parser_config_path: str | Path = "config/cpp_parser.yaml",
) -> CPPParserConfig:
    """Load engine/data paths from UE config and parser rules separately."""

    ue_path = Path(ue_config_path).resolve()
    parser_path = Path(parser_config_path).resolve()
    with ue_path.open(encoding="utf-8") as config_file:
        ue_config = yaml.safe_load(config_file)
    with parser_path.open(encoding="utf-8") as config_file:
        parser_config = yaml.safe_load(config_file)

    project_root = ue_path.parent.parent
    configured_root = os.environ.get("UE_ROOT", "").strip()
    if not configured_root:
        configured_root = str(ue_config["engine"]["root"]).strip()
    if not configured_root:
        raise ValueError(
            f"Unreal Engine root is not configured; set UE_ROOT or engine.root in {ue_path}"
        )
    engine_root = Path(configured_root).expanduser()
    if not engine_root.is_absolute():
        engine_root = project_root / engine_root
    engine_root = engine_root.resolve()
    parsed_dir = Path(ue_config["data"]["parsed"])
    if not parsed_dir.is_absolute():
        parsed_dir = project_root / parsed_dir
    output_dir = parsed_dir / "engine"

    return CPPParserConfig(
        engine_version=str(ue_config["engine"]["version"]),
        engine_root=engine_root,
        inventory_path=output_dir / "files.jsonl",
        output_path=output_dir / "documents.jsonl",
        issues_path=output_dir / "parse_issues.jsonl",
        **parser_config,
    )


class CPPParseIssue(BaseModel):
    """Operational problem for a specific inventory file."""

    model_config = ConfigDict(extra="forbid")

    relative_path: str
    error: str


@dataclass(frozen=True)
class CPPParseSummary:
    """Aggregate parse counts returned to the CLI."""

    inventory_files: int
    selected_files: int
    parsed_files: int
    syntax_error_files: int
    documents: int
    issues: int
    by_symbol_type: dict[CPPSymbolType, int]
    output_path: Path
    issues_path: Path


@dataclass(frozen=True)
class _ExtractedSymbol:
    symbol_type: CPPSymbolType
    name: str
    symbol: str
    class_name: str | None
    function_name: str | None
    node: Node
    inheritance: list[str] = field(default_factory=list)


class UnrealCPPParser:
    """Parse C++ ASTs after length-preserving Unreal macro preprocessing."""

    def __init__(self, config: CPPParserConfig) -> None:
        self.config = config
        self.language = Language(tree_sitter_cpp.language())
        self.parser = Parser(self.language)
        macro_names = sorted(
            config.semantic_macros | config.mask_only_macros,
            key=len,
            reverse=True,
        )
        macro_patterns = [re.escape(name.encode("ascii")) for name in macro_names]
        macro_patterns.extend(
            f"(?:{pattern})".encode("ascii") for pattern in config.mask_only_macro_patterns
        )
        self._macro_start = re.compile(
            rb"\b(" + b"|".join(macro_patterns) + rb")\s*\("
        )
        self._api_macro = re.compile(config.api_macro_pattern.encode("ascii"))

    def parse_source(
        self,
        source: bytes,
        file_record: EngineFileRecord,
        *,
        source_scope: SourceScope = SourceScope.GLOBAL,
        source_type: SourceType = SourceType.ENGINE_SOURCE,
    ) -> tuple[list[UEDocument], bool]:
        """Parse one source buffer and return symbol documents plus AST error state."""

        if file_record.engine_version != self.config.engine_version:
            raise ValueError(
                f"File engine version {file_record.engine_version!r} does not match "
                f"configured version {self.config.engine_version!r}: "
                f"{file_record.relative_path}"
            )
        newline_offsets = [index for index, value in enumerate(source) if value == 10]
        macro_spans, mask_spans = self._scan_macros(source, newline_offsets)
        macro_end_offsets = [span.end_byte for span in macro_spans]
        masked_source = _mask_spans(source, mask_spans)
        tree = self.parser.parse(masked_source)
        symbols: list[_ExtractedSymbol] = []
        seen_symbols: set[tuple[int, int, str, str]] = set()
        for symbol in self._extract_symbols(tree.root_node, source):
            if not symbol.symbol:
                continue
            identity = (
                symbol.node.start_byte,
                symbol.node.end_byte,
                symbol.symbol_type.value,
                symbol.symbol,
            )
            if identity in seen_symbols:
                continue
            seen_symbols.add(identity)
            symbols.append(symbol)
        documents: list[UEDocument] = []
        seen_document_ids: set[str] = set()
        for symbol in symbols:
            document = self._to_document(
                symbol,
                source,
                newline_offsets,
                macro_spans,
                macro_end_offsets,
                file_record,
                tree.root_node.has_error,
                source_scope,
                source_type,
            )
            if document.id in seen_document_ids:
                continue
            seen_document_ids.add(document.id)
            documents.append(document)
        return documents, tree.root_node.has_error

    def parse_inventory(
        self,
        *,
        modules: set[str] | None = None,
        plugins: set[str] | None = None,
        path_contains: str | None = None,
        limit_files: int | None = None,
        output_path: str | Path | None = None,
    ) -> CPPParseSummary:
        """Stream the inventory into semantic document and issue JSONL files."""

        if limit_files is not None and limit_files <= 0:
            raise ValueError("limit_files must be greater than zero")
        output = Path(output_path) if output_path else self.config.output_path
        issues_path = output.with_name(f"{output.stem}_issues.jsonl") if output_path else self.config.issues_path
        output.parent.mkdir(parents=True, exist_ok=True)
        issues_path.parent.mkdir(parents=True, exist_ok=True)
        output_temp = output.with_suffix(output.suffix + ".tmp")
        issues_temp = issues_path.with_suffix(issues_path.suffix + ".tmp")

        inventory_files = selected_files = parsed_files = syntax_error_files = 0
        document_count = issue_count = 0
        by_symbol_type = {symbol_type: 0 for symbol_type in CPPSymbolType}
        selected_modules = {value.casefold() for value in modules} if modules else None
        selected_plugins = {value.casefold() for value in plugins} if plugins else None
        path_filter = path_contains.casefold() if path_contains else None

        try:
            with self.config.inventory_path.open(encoding="utf-8") as inventory, output_temp.open(
                "w", encoding="utf-8", newline="\n"
            ) as document_file, issues_temp.open("w", encoding="utf-8", newline="\n") as issue_file:
                for line_number, line in enumerate(inventory, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = EngineFileRecord.model_validate_json(line)
                    except Exception as error:
                        raise ValueError(
                            f"Invalid engine inventory at line {line_number}: "
                            f"{self.config.inventory_path}"
                        ) from error
                    inventory_files += 1
                    if record.file_type not in self.config.include_file_types:
                        continue
                    if selected_modules and (record.module or "").casefold() not in selected_modules:
                        continue
                    if selected_plugins and (record.plugin or "").casefold() not in selected_plugins:
                        continue
                    if path_filter and path_filter not in record.relative_path.casefold():
                        continue
                    if limit_files is not None and selected_files >= limit_files:
                        continue
                    selected_files += 1
                    source_path = self.config.engine_root / Path(record.relative_path)
                    try:
                        source = source_path.read_bytes()
                        documents, has_error = self.parse_source(source, record)
                    except Exception as error:
                        issue = CPPParseIssue(
                            relative_path=record.relative_path,
                            error=f"{type(error).__name__}: {error}",
                        )
                        issue_file.write(issue.model_dump_json() + "\n")
                        issue_count += 1
                        continue
                    parsed_files += 1
                    syntax_error_files += int(has_error)
                    for document in documents:
                        document_file.write(document.model_dump_json() + "\n")
                        document_count += 1
                        by_symbol_type[CPPSymbolType(document.symbol_type)] += 1

            os.replace(output_temp, output)
            os.replace(issues_temp, issues_path)
        except BaseException:
            output_temp.unlink(missing_ok=True)
            issues_temp.unlink(missing_ok=True)
            raise

        return CPPParseSummary(
            inventory_files=inventory_files,
            selected_files=selected_files,
            parsed_files=parsed_files,
            syntax_error_files=syntax_error_files,
            documents=document_count,
            issues=issue_count,
            by_symbol_type=by_symbol_type,
            output_path=output,
            issues_path=issues_path,
        )

    def _scan_macros(
        self, source: bytes, newline_offsets: list[int]
    ) -> tuple[list[_MacroSpan], list[tuple[int, int]]]:
        semantic_spans: list[_MacroSpan] = []
        mask_spans: list[tuple[int, int]] = []
        for match in self._macro_start.finditer(source):
            name = match.group(1).decode("ascii")
            open_paren = source.find(b"(", match.start(), match.end())
            end_byte = _balanced_call_end(source, open_paren)
            if end_byte is None:
                line_end = source.find(b"\n", match.end())
                end_byte = len(source) if line_end < 0 else line_end
            mask_spans.append((match.start(), end_byte))
            if name not in self.config.semantic_macros:
                continue
            raw = source[match.start() : end_byte].decode("utf-8", errors="replace")
            arguments = source[open_paren + 1 : max(open_paren + 1, end_byte - 1)].decode(
                "utf-8", errors="replace"
            )
            semantic_spans.append(
                _MacroSpan(
                    macro=UEMacro(
                        name=name,
                        arguments=arguments.strip(),
                        raw=raw,
                        line_start=_line_number(newline_offsets, match.start()),
                        line_end=_line_number(newline_offsets, max(match.start(), end_byte - 1)),
                    ),
                    start_byte=match.start(),
                    end_byte=end_byte,
                )
            )
        mask_spans.extend(match.span() for match in self._api_macro.finditer(source))
        return semantic_spans, sorted(set(mask_spans))

    def _extract_symbols(
        self,
        root: Node,
        source: bytes,
    ) -> Iterator[_ExtractedSymbol]:
        yield from self._walk_symbols(root, source, class_stack=[], namespace_stack=[])

    def _walk_symbols(
        self,
        node: Node,
        source: bytes,
        *,
        class_stack: list[str],
        namespace_stack: list[str],
    ) -> Iterator[_ExtractedSymbol]:
        if node.type in {"class_specifier", "struct_specifier"}:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node, source)
                if not name:
                    return
                symbol_type = (
                    CPPSymbolType.CLASS
                    if node.type == "class_specifier"
                    else CPPSymbolType.STRUCT
                )
                qualified = "::".join([*namespace_stack, *class_stack, name])
                yield _ExtractedSymbol(
                    symbol_type=symbol_type,
                    name=name,
                    symbol=qualified,
                    class_name=name if not class_stack else "::".join([*class_stack, name]),
                    function_name=None,
                    node=node,
                    inheritance=_inheritance(node, source),
                )
                body = node.child_by_field_name("body")
                if body:
                    for child in body.named_children:
                        yield from self._walk_symbols(
                            child,
                            source,
                            class_stack=[*class_stack, name],
                            namespace_stack=namespace_stack,
                        )
            return

        if node.type == "enum_specifier":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node, source)
                if not name:
                    return
                qualified = "::".join([*namespace_stack, *class_stack, name])
                yield _ExtractedSymbol(
                    symbol_type=CPPSymbolType.ENUM,
                    name=name,
                    symbol=qualified,
                    class_name="::".join(class_stack) or None,
                    function_name=None,
                    node=node,
                )
            return

        if node.type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            namespace = _node_text(name_node, source) if name_node else ""
            body = node.child_by_field_name("body")
            if body:
                for child in body.named_children:
                    yield from self._walk_symbols(
                        child,
                        source,
                        class_stack=class_stack,
                        namespace_stack=[*namespace_stack, namespace] if namespace else namespace_stack,
                    )
            return

        if node.type == "function_definition":
            extracted = _function_symbol(
                node, source, class_stack=class_stack, namespace_stack=namespace_stack
            )
            if extracted:
                yield extracted
            return

        if node.type in {"field_declaration", "declaration"}:
            function_declarator = _first_descendant(node, {"function_declarator"})
            if function_declarator:
                extracted = _function_symbol(
                    node,
                    source,
                    class_stack=class_stack,
                    namespace_stack=namespace_stack,
                    function_declarator=function_declarator,
                )
                if extracted:
                    yield extracted
                return
            if node.type == "field_declaration" and class_stack:
                field_names = _field_names(node, source)
                for field_name in field_names:
                    class_name = "::".join(class_stack)
                    yield _ExtractedSymbol(
                        symbol_type=CPPSymbolType.FIELD,
                        name=field_name,
                        symbol=f"{class_name}::{field_name}",
                        class_name=class_name,
                        function_name=None,
                        node=node,
                    )
                if field_names:
                    return
            for child in node.named_children:
                if child.type in {
                    "class_specifier",
                    "struct_specifier",
                    "enum_specifier",
                    "template_declaration",
                }:
                    yield from self._walk_symbols(
                        child,
                        source,
                        class_stack=class_stack,
                        namespace_stack=namespace_stack,
                    )
            return

        if node.type == "expression_statement" or node.type.endswith("_expression"):
            return

        for child in node.named_children:
            yield from self._walk_symbols(
                child,
                source,
                class_stack=class_stack,
                namespace_stack=namespace_stack,
            )

    def _to_document(
        self,
        symbol: _ExtractedSymbol,
        source: bytes,
        newline_offsets: list[int],
        macro_spans: list[_MacroSpan],
        macro_end_offsets: list[int],
        file_record: EngineFileRecord,
        ast_has_error: bool,
        source_scope: SourceScope,
        source_type: SourceType,
    ) -> UEDocument:
        attached_macros = _attached_macros(
            symbol.node.start_byte, source, macro_spans, macro_end_offsets
        )
        start_byte = attached_macros[0].start_byte if attached_macros else symbol.node.start_byte
        line_start = _line_number(newline_offsets, start_byte)
        line_end = _line_number(newline_offsets, max(start_byte, symbol.node.end_byte - 1))
        content = source[start_byte : symbol.node.end_byte].decode("utf-8", errors="replace").strip()
        identifier = _document_id(
            file_record.relative_path,
            symbol.symbol_type.value,
            symbol.symbol,
            line_start,
            line_end,
            content,
        )
        api_macro_match = self._api_macro.search(
            source[symbol.node.start_byte : min(symbol.node.end_byte, symbol.node.start_byte + 256)]
        )
        return UEDocument(
            id=identifier,
            engine_version=file_record.engine_version,
            source_scope=source_scope,
            source_type=source_type,
            content=content,
            title=symbol.symbol,
            module=file_record.module,
            plugin=file_record.plugin,
            file_path=file_record.relative_path,
            symbol=symbol.symbol,
            symbol_type=symbol.symbol_type.value,
            class_name=symbol.class_name,
            function_name=symbol.function_name,
            metadata={
                "line_start": line_start,
                "line_end": line_end,
                "ue_macros": [item.macro.model_dump(mode="json") for item in attached_macros],
                "api_macro": (
                    api_macro_match.group(0).decode("ascii") if api_macro_match else None
                ),
                "inheritance": symbol.inheritance,
                "file_sha256": file_record.sha256,
                "file_type": file_record.file_type.value,
                "ast_has_error": symbol.node.has_error,
                "file_ast_has_error": ast_has_error,
                "parser_version": PARSER_VERSION,
            },
        )


def _balanced_call_end(source: bytes, open_paren: int) -> int | None:
    """Find a macro call end while ignoring parentheses inside strings/comments."""

    if open_paren < 0:
        return None
    depth = 0
    index = open_paren
    quote: int | None = None
    escaped = False
    line_comment = False
    block_comment = False
    while index < len(source):
        value = source[index]
        next_value = source[index + 1] if index + 1 < len(source) else None
        if line_comment:
            if value == 10:
                line_comment = False
            index += 1
            continue
        if block_comment:
            if value == 42 and next_value == 47:
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == quote:
                quote = None
            index += 1
            continue
        if value == 47 and next_value == 47:
            line_comment = True
            index += 2
            continue
        if value == 47 and next_value == 42:
            block_comment = True
            index += 2
            continue
        if value in {34, 39}:
            quote = value
            index += 1
            continue
        if value == 40:
            depth += 1
        elif value == 41:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _mask_spans(source: bytes, spans: list[tuple[int, int]]) -> bytes:
    masked = bytearray(source)
    for start, end in spans:
        for index in range(start, min(end, len(masked))):
            if masked[index] not in {10, 13}:
                masked[index] = 32
    return bytes(masked)


def _line_number(newline_offsets: list[int], byte_offset: int) -> int:
    return bisect.bisect_right(newline_offsets, byte_offset) + 1


def _node_text(node: Node | None, source: bytes) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()


def _first_descendant(node: Node, types: set[str]) -> Node | None:
    stack = list(reversed(node.named_children))
    while stack:
        current = stack.pop()
        if current.type in types:
            return current
        stack.extend(reversed(current.named_children))
    return None


def _declarator_name(function_declarator: Node, source: bytes) -> str:
    declarator = function_declarator.child_by_field_name("declarator")
    if declarator is None:
        declarator = _first_descendant(
            function_declarator,
            {"qualified_identifier", "field_identifier", "identifier", "operator_name", "destructor_name"},
        )
    while declarator and declarator.type in {
        "function_declarator",
        "pointer_declarator",
        "reference_declarator",
        "parenthesized_declarator",
    }:
        nested = declarator.child_by_field_name("declarator")
        if nested is None:
            break
        declarator = nested
    return _node_text(declarator, source)


def _function_symbol(
    node: Node,
    source: bytes,
    *,
    class_stack: list[str],
    namespace_stack: list[str],
    function_declarator: Node | None = None,
) -> _ExtractedSymbol | None:
    declarator = function_declarator or _first_descendant(node, {"function_declarator"})
    if declarator is None:
        return None
    qualified_name = _declarator_name(declarator, source)
    if not qualified_name:
        return None
    normalized_name = re.sub(r"\s+", "", qualified_name)
    parts = normalized_name.split("::")
    function_name = parts[-1]
    explicit_scope = parts[:-1]
    class_name = "::".join(class_stack) or None

    if explicit_scope:
        candidate = explicit_scope[-1]
        if len(candidate.lstrip("~")) > 2 and re.match(r"^[AUFSI][A-Za-z0-9_]*$", candidate):
            class_name = "::".join(explicit_scope)

    comparison_name = function_name.lstrip("~")
    class_leaf = class_name.split("::")[-1] if class_name else None
    is_constructor = bool(class_leaf and comparison_name == class_leaf and not function_name.startswith("~"))
    if is_constructor:
        symbol_type = CPPSymbolType.CONSTRUCTOR
    elif class_name:
        symbol_type = CPPSymbolType.METHOD
    else:
        symbol_type = CPPSymbolType.FUNCTION

    if explicit_scope:
        symbol_name = normalized_name
    elif class_name:
        symbol_name = f"{class_name}::{function_name}"
    elif namespace_stack:
        symbol_name = "::".join([*namespace_stack, function_name])
    else:
        symbol_name = function_name
    return _ExtractedSymbol(
        symbol_type=symbol_type,
        name=function_name,
        symbol=symbol_name,
        class_name=class_name,
        function_name=function_name,
        node=node,
    )


def _field_names(node: Node, source: bytes) -> list[str]:
    names: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "function_declarator":
            continue
        if current.type == "field_identifier":
            name = _node_text(current, source)
            if name and name not in names:
                names.append(name)
            continue
        stack.extend(reversed(current.named_children))
    return names


def _inheritance(node: Node, source: bytes) -> list[str]:
    clause = node.child_by_field_name("bases") or next(
        (child for child in node.named_children if child.type == "base_class_clause"),
        None,
    )
    if clause is None:
        return []
    bases: list[str] = []
    for child in clause.named_children:
        if child.type == "access_specifier":
            continue
        text = _node_text(child, source)
        if text and text not in bases:
            bases.append(text)
    return bases


def _attached_macros(
    symbol_start: int,
    source: bytes,
    macro_spans: list[_MacroSpan],
    macro_end_offsets: list[int],
) -> list[_MacroSpan]:
    candidate_count = bisect.bisect_right(macro_end_offsets, symbol_start)
    if candidate_count == 0:
        return []
    attached: list[_MacroSpan] = []
    cursor = symbol_start
    for index in range(candidate_count - 1, -1, -1):
        span = macro_spans[index]
        between = source[span.end_byte:cursor]
        if not _is_cpp_trivia(between):
            break
        attached.append(span)
        cursor = span.start_byte
    return list(reversed(attached))


def _is_cpp_trivia(content: bytes) -> bool:
    text = content.decode("utf-8", errors="replace")
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"^[ \t]*#[^\n]*", "", text, flags=re.MULTILINE)
    return not text.strip()


def _document_id(
    relative_path: str,
    symbol_type: str,
    symbol: str,
    line_start: int,
    line_end: int,
    content: str,
) -> str:
    digest = hashlib.sha256(
        f"{relative_path}:{symbol_type}:{symbol}:{line_start}:{line_end}:{content}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"cpp-{digest}"
