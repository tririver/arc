from arc_paper import service
from arc_paper.cache import write_json


def _write_valid_run(tmp_path, *, claim_status="verified", consensus_status="all_agree"):
    run_dir = tmp_path / "calculate" / "run-1"
    source_path = tmp_path / "parsed-source.json"
    write_json(source_path, {"paper_id": "lecture-9", "parser_version": 7, "source_hash": "hash", "toc": [], "sections": [], "equations": []})
    write_json(
        run_dir / "note-check-triage.json",
        {
            "notes": [{"source_id": "lecture-9", "parsed_source_path": str(source_path)}],
            "claims_to_check": [
                {
                    "id": "claim-1",
                    "equation_id": "eq_00001",
                    "status": claim_status,
                    "consensus_step_id": "step-1",
                }
            ],
        },
    )
    write_json(run_dir / "plan.json", {"schema_version": "arc.plan.v1"})
    write_json(run_dir / "foundation" / "latest.json", {"schema_version": "arc.foundation.v1"})
    write_json(run_dir / "consensus" / "config.json", {"steps": [{"step_id": "step-1"}]})
    write_json(run_dir / "consensus" / "results.json", {"steps": [{"step_id": "step-1", "status": consensus_status}]})
    return run_dir


def test_validate_note_check_accepts_complete_consensus_run(tmp_path):
    run_dir = _write_valid_run(tmp_path)

    result = service.validate_note_check(run_dir)

    assert result["ok"] is True
    assert result["data"]["claims_checked"] == 1
    assert result["data"]["status_counts"] == {"verified": 1}


def test_validate_note_check_rejects_verified_without_all_agree(tmp_path):
    run_dir = _write_valid_run(tmp_path, consensus_status="disagree")

    result = service.validate_note_check(run_dir)

    assert result["ok"] is False
    assert result["error"]["code"] == "note_check_validation_failed"
    assert "claim-1" in result["violations"][0]


def test_validate_note_check_accepts_human_resolved_without_consensus_result(tmp_path):
    run_dir = _write_valid_run(tmp_path)
    source_path = tmp_path / "parsed-source.json"
    write_json(
        run_dir / "note-check-triage.json",
        {
            "notes": [{"source_id": "lecture-9", "parsed_source_path": str(source_path)}],
            "claims_to_check": [
                {
                    "id": "claim-1",
                    "equation_id": "eq_00001",
                    "status": "human_resolved",
                    "consensus_step_id": "step-1",
                    "resolution": {
                        "resolved_by": "user",
                        "resolved_at": "2026-05-28T12:00:00+00:00",
                        "type": "corrected_formula",
                        "corrected_latex": "x=1",
                        "rationale": "User supplied the corrected formula.",
                    },
                }
            ],
        },
    )
    write_json(run_dir / "consensus" / "results.json", {"steps": []})

    result = service.validate_note_check(run_dir)

    assert result["ok"] is True
    assert result["data"]["status_counts"] == {"human_resolved": 1}


def test_validate_note_check_rejects_human_resolved_without_resolution(tmp_path):
    run_dir = _write_valid_run(tmp_path)
    source_path = tmp_path / "parsed-source.json"
    write_json(
        run_dir / "note-check-triage.json",
        {
            "notes": [{"source_id": "lecture-9", "parsed_source_path": str(source_path)}],
            "claims_to_check": [
                {
                    "id": "claim-1",
                    "equation_id": "eq_00001",
                    "status": "human_resolved",
                    "consensus_step_id": "step-1",
                }
            ],
        },
    )

    result = service.validate_note_check(run_dir)

    assert result["ok"] is False
    assert "claim-1: human_resolved requires resolution object" in result["violations"]


def test_validate_note_check_reports_missing_required_artifacts(tmp_path):
    run_dir = tmp_path / "calculate" / "run-1"
    run_dir.mkdir(parents=True)

    result = service.validate_note_check(run_dir)

    assert result["ok"] is False
    assert "note-check-triage.json" in result["missing"]
    assert "plan.json" in result["missing"]


def test_validate_note_check_requires_parsed_source_path(tmp_path):
    run_dir = _write_valid_run(tmp_path)
    legacy_path = tmp_path / "legacy-parsed.json"
    write_json(legacy_path, {"paper_id": "lecture-9"})
    write_json(
        run_dir / "note-check-triage.json",
        {
            "notes": [{"source_id": "lecture-9", "parsed_" + "note_path": str(legacy_path)}],
            "claims_to_check": [],
        },
    )

    result = service.validate_note_check(run_dir)

    assert result["ok"] is False
    assert "parsed source JSON" in result["missing"]
