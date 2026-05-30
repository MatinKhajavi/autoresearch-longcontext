from aso.local_sandbox import LocalSandbox


def _mk(tmp_path):
    docs, out, ws = tmp_path / "d", tmp_path / "o", tmp_path / "w"
    for p in (docs, out, ws):
        p.mkdir()
    return docs, out, ws


def test_exec_returns_execresult_and_runs_in_workspace(tmp_path):
    docs, out, ws = _mk(tmp_path)
    (docs / "hello.txt").write_text("hi")
    sb = LocalSandbox(documents_dir=docs, output_dir=out, workspace_dir=ws, default_timeout=10)
    sb.start()
    res = sb.exec("pwd")
    sb.stop()
    # ExecResult shape (reused from sandbox.sandbox)
    assert hasattr(res, "stdout") and hasattr(res, "returncode") and hasattr(res, "timed_out")
    assert res.returncode == 0 and res.timed_out is False
    # cwd is the workspace, not documents
    assert str(ws) in res.stdout


def test_documents_and_output_are_reachable_as_relative_paths(tmp_path):
    docs, out, ws = _mk(tmp_path)
    (docs / "hello.txt").write_text("hi")
    sb = LocalSandbox(documents_dir=docs, output_dir=out, workspace_dir=ws, default_timeout=10)
    sb.start()
    # the agent expects `documents/` and `output/` under the workspace
    listing = sb.exec("ls").stdout
    assert "documents" in listing and "output" in listing
    # relative read through the documents symlink works
    cat = sb.exec("cat documents/hello.txt")
    assert cat.stdout.strip() == "hi"
    # writes to output land in the real output_dir
    sb.exec("echo deliverable > output/result.md")
    sb.stop()
    assert (out / "result.md").read_text().strip() == "deliverable"


def test_file_ops_map_sandbox_paths_to_host(tmp_path):
    """exists/read_file/write_file must resolve /workspace[/documents|/output] paths."""
    from sandbox.sandbox import DOCUMENTS_PATH, OUTPUT_PATH, WORKSPACE_PATH

    docs, out, ws = _mk(tmp_path)
    (docs / "deal.txt").write_text("EBITDA $17.1M")
    sb = LocalSandbox(documents_dir=docs, output_dir=out, workspace_dir=ws)
    sb.start()
    # read a document by sandbox path
    assert sb.exists(f"{DOCUMENTS_PATH}/deal.txt") is True
    assert sb.read_file(f"{DOCUMENTS_PATH}/deal.txt") == b"EBITDA $17.1M"
    # write a deliverable by sandbox path -> lands in the real output_dir
    sb.write_file(f"{OUTPUT_PATH}/memo.md", "analysis")
    assert (out / "memo.md").read_text() == "analysis"
    # workspace-relative scratch file
    sb.write_file(f"{WORKSPACE_PATH}/notes.md", b"scratch")
    assert (ws / "notes.md").read_bytes() == b"scratch"
    assert sb.exists(f"{DOCUMENTS_PATH}/missing.txt") is False
    sb.stop()


def test_timeout_sets_timed_out(tmp_path):
    docs, out, ws = _mk(tmp_path)
    sb = LocalSandbox(documents_dir=docs, output_dir=out, workspace_dir=ws, default_timeout=1)
    sb.start()
    res = sb.exec("sleep 5", timeout=1)
    sb.stop()
    assert res.timed_out is True and res.returncode is None


def test_exec_before_start_raises(tmp_path):
    docs, out, ws = _mk(tmp_path)
    sb = LocalSandbox(documents_dir=docs, output_dir=out, workspace_dir=ws)
    try:
        sb.exec("pwd")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
