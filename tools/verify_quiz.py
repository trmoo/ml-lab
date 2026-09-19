# -*- coding: utf-8 -*-
# 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
# 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요.
"""문제 은행의 코드를 진짜 파이썬으로 돌려 본다 — python tools/verify_quiz.py [열쇠 …] [--show]

회차마다 setup → 1번 문제 → 2번 문제 … 를 한 이름 공간에서 차례로 실행한다.
  · blank  — 빈칸을 정답으로 채운 코드
  · bug    — 잘못된 줄을 fix 로 바꾼 코드
  · choice — code 가 있으면 그 코드
  · order  — 실행할 코드가 없다
그리고 문제마다 적어 둔 check 식이 모두 True 인지 본다.
지문·보기에 적은 숫자가 데이터와 어긋나면 여기서 잡힌다.

그래프(seaborn·matplotlib)와 keras 는 가짜 모듈로 바꿔 끼워, 설치되어 있지 않아도 돈다.
그래프는 그리지 않고 넘어가고, keras 문제("run": false)는 문법만 본다.

pandas 의 FutureWarning 은 오류로 다룬다 — 곧 사라질 문법이면 코랩과 새 판에서 결과가 갈린다.
"""
import ast
import contextlib
import importlib.abc
import importlib.machinery
import io
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUIZ = ROOT / 'quiz'
DATA = ROOT / 'data'
MARK = re.compile(r'⟦([^⟦⟧]*)⟧')
ORDER = ['exam', 'crop', 'car', 'ad', 'pay', 'yield', 'home', 'churn', 'loan', 'quality']


# ── 가짜 모듈: 무엇을 부르든 조용히 받아 준다 ─────────────────
class Stub:
    def __init__(self, name='stub'):
        self._name = name

    def __getattr__(self, key):
        if key.startswith('__'):
            raise AttributeError(key)
        return Stub(f'{self._name}.{key}')

    def __call__(self, *a, **k):
        return Stub(f'{self._name}()')

    def __iter__(self):
        return iter([Stub(), Stub()])

    def __getitem__(self, key):
        return Stub()

    def __repr__(self):
        return f'<가짜 {self._name}>'


class StubModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, key):
        if key.startswith('__'):
            raise AttributeError(key)
        return Stub(f'{self.__name__}.{key}')


FAKE = ('seaborn', 'matplotlib', 'tensorflow', 'keras')


class FakeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in FAKE:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return StubModule(spec.name)

    def exec_module(self, module):
        pass


def code_of(p):
    if p.get('type') == 'blank':
        return MARK.sub(lambda m: m.group(1), p['code'])
    if p.get('type') == 'bug':
        lines = p['code'].split('\n')
        lines[p['answer'] - 1] = p['fix']
        return '\n'.join(lines)
    if p.get('type') == 'choice':
        return p.get('code', '')
    return ''


def run_block(code, ns, label, show):
    """마지막 줄이 식이면 주피터처럼 그 값을 보여 준다(--show 일 때)"""
    buf = io.StringIO()
    tree = ast.parse(code, filename=label)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, label, 'exec'), ns)
        if last is not None:
            v = eval(compile(last, label, 'eval'), ns)
            if v is not None and not isinstance(v, Stub):
                print(repr(v))
    if show and buf.getvalue().strip():
        print('    ┌ 출력')
        for line in buf.getvalue().rstrip().split('\n')[:40]:
            print('    │ ' + line)


def verify(key, show=False):
    path = QUIZ / f'{key}.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    fails = []
    ns = {'__name__': '__quiz__'}
    try:
        run_block(data.get('setup', ''), ns, f'{key}/setup', show)
    except Exception as e:  # noqa: BLE001
        return [f'{key}/setup: {type(e).__name__}: {e}']

    for i, p in enumerate(data['problems'], 1):
        pid = p.get('id', f'{key}-{i:02d}')
        code = code_of(p)
        if show:
            print(f'  [{pid}] {p.get("title", "")}')
        try:
            if p.get('run') is False:
                ast.parse(code or 'pass')
                continue
            target = dict(ns) if p.get('sandbox') else ns
            if code.strip():
                run_block(code, target, pid, show)
            for expr in p.get('check', []):
                try:
                    ok = eval(expr, target)
                except Exception as e:  # noqa: BLE001
                    fails.append(f'{pid}: check 식이 오류 — {expr}  ({type(e).__name__}: {e})')
                    continue
                # numpy 의 True(np.True_)도 받는다. 다만 표·목록처럼 「참 같은 값」은 받지 않는다
                if not (type(ok).__name__ in ('bool', 'bool_') and bool(ok)):
                    fails.append(f'{pid}: check 가 True 가 아님 — {expr}  (결과 {ok!r})')
        except Exception as e:  # noqa: BLE001
            fails.append(f'{pid}: 실행 오류 — {type(e).__name__}: {e}')
    return fails


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    show = '--show' in sys.argv
    keys = args or ORDER
    sys.meta_path.insert(0, FakeFinder())
    warnings.simplefilter('error', FutureWarning)
    warnings.simplefilter('ignore', UserWarning)
    os.chdir(DATA)   # read_csv('exam_result.csv') 처럼 파일 이름만 적어도 열리게

    total_fail = 0
    for key in keys:
        if show:
            print(f'\n== {key} ==')
        fails = verify(key, show)
        if fails is None:
            print(f'  {key:8s} (문제 파일 없음)')
            continue
        n = len(json.loads((QUIZ / f'{key}.json').read_text(encoding='utf-8'))['problems'])
        mark = '✓' if not fails else '✗'
        print(f'  {mark} {key:8s} 문제 {n}개' + (f' — 실패 {len(fails)}건' if fails else ''))
        for f in fails:
            print(f'      · {f}')
        total_fail += len(fails)
    print('\n모두 통과' if not total_fail else f'\n실패 {total_fail}건')
    sys.exit(1 if total_fail else 0)


if __name__ == '__main__':
    main()
