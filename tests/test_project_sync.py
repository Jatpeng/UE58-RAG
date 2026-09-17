from __future__ import annotations

import shutil
from pathlib import Path

from ue_rag.project_sync import ProjectSyncService
from ue_rag.retrieval import LexicalIndex


CONFIGS = (
    "ue58.yaml",
    "project_scanner.yaml",
    "cpp_parser.yaml",
    "cpp_chunker.yaml",
    "lexical.yaml",
)


def _workspace(tmp_path: Path) -> Path:
    source_config = Path(__file__).parents[1] / "config"
    target_config = tmp_path / "workspace" / "config"
    target_config.mkdir(parents=True)
    for name in CONFIGS:
        shutil.copyfile(source_config / name, target_config / name)
    return target_config.parent


def _write_header(path: Path, name: str) -> None:
    path.write_text(
        f"""UCLASS()
class {name}
{{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere)
    int32 Health;
}};
""",
        encoding="utf-8",
    )


def test_project_sync_add_modify_delete_removes_stale_chunks(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    project = tmp_path / "MyGame"
    source = project / "Source" / "Game"
    source.mkdir(parents=True)
    header = source / "Player.h"
    _write_header(header, "AOldPlayer")
    index_path = workspace / "data" / "index" / "lexical.sqlite3"
    service = ProjectSyncService(
        workspace,
        state_dir=workspace / "data" / "state",
        lexical_index_path=index_path,
    )

    first = service.preview(project)
    assert first["counts"]["added"] == 1
    first_result = service.apply(first["plan_id"], sync_vectors=False)
    assert first_result["changed_files"] == 1
    assert first_result["chunks"] >= 1
    assert first_result["data_source"]["name"] == "MyGame"
    assert first_result["data_source"]["chunks"] >= 1
    assert list((workspace / "data" / "state").glob("*.meta.json"))
    with LexicalIndex(index_path) as index:
        assert index.search_symbol("AOldPlayer")

    _write_header(header, "ANewPlayer")
    second = service.preview(project)
    assert second["counts"]["modified"] == 1
    second_result = service.apply(second["plan_id"], sync_vectors=False)
    assert second_result["removed_chunks"] >= 1
    with LexicalIndex(index_path) as index:
        assert index.search_symbol("AOldPlayer") == []
        assert index.search_symbol("ANewPlayer")

    header.unlink()
    third = service.preview(project)
    assert third["counts"]["deleted"] == 1
    third_result = service.apply(third["plan_id"], sync_vectors=False)
    assert third_result["deleted_files"] == 1
    with LexicalIndex(index_path) as index:
        assert index.search_symbol("ANewPlayer") == []
