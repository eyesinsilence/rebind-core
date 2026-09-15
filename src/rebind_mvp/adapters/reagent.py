"""Upstream availability check. Never relabel a replacement prompt as ReAgent."""
import pathlib,sys
from ..audit import execute,Blocked

def check(root):
    root=pathlib.Path(root)
    first=execute([sys.executable,'-c',"import runpy,sys;sys.path.insert(0,'upstream/ReAgent');runpy.run_path('upstream/ReAgent/main.py',run_name='rebind_import_audit')"],root,'reagent_native_import_smoke',timeout=30)
    if first['exit_code']:
        # Try official package namespace explicitly; preserve original source.
        code="import sys,types; sys.path.insert(0,'upstream/ReAgent'); api=types.ModuleType('backend.api'); api.api_call=lambda *a,**k: None; api.api_call_completion=api.api_call; sys.modules['backend.api']=api; from Agent.agent import BaseAgent; from Agent.moderator2 import Moderator2"
        second=execute([sys.executable,'-c',code],root,'reagent_explicit_import_retry',timeout=30)
        syntax="import ast,pathlib,sys; failures=[]\nfor p in pathlib.Path('upstream/ReAgent').rglob('*.py'):\n try: ast.parse(p.read_text())\n except SyntaxError as e: failures.append((str(p),e.lineno,e.msg))\nprint(failures);sys.exit(bool(failures))"
        execute([sys.executable,'-c',syntax],root,'reagent_official_syntax_audit',timeout=30)
        raise Blocked('Official ReAgent entrypoint/package import fails; see '+first['log']+' and '+second['log']+'. No external QA result is claimed.')
    return first
