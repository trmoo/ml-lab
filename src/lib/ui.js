/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 화면 만들기 도우미 — 라이브러리 없이 DOM 을 직접 다룬다. */

/** h('div', {class:'x'}, '글', 자식요소) 처럼 쓴다 */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(props || {}).forEach(([k, v]) => {
    if (v === null || v === undefined || v === false) return;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (k === 'value') node.value = v;
    else if (k === 'checked' || k === 'disabled' || k === 'selected') node[k] = !!v;
    else node.setAttribute(k, v);
  });
  children.flat(3).forEach((c) => {
    if (c === null || c === undefined || c === false) return;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  });
  return node;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

/* ── 다시 그려도 보던 자리를 지킨다 ─────────────────────────────
 * 이 앱은 옵션을 하나 누를 때마다 탭 화면을 통째로 다시 그린다.
 * 그런데 clear() 로 내용을 비우는 순간 문서가 짧아져서
 * 브라우저가 스크롤을 맨 위로 끌어올려 버린다.
 * 그래서 지우기 전에 「스크롤 위치」와 「설명 상자를 펼쳐 뒀는지」를 적어 두고,
 * 다 그린 뒤에 그대로 되돌려 놓는다.
 *
 * 설명 상자까지 챙기는 까닭 — 펼쳐 둔 상자가 닫히면 그만큼 화면이 짧아져서
 * 스크롤 위치만 되돌려 봐야 엉뚱한 곳을 보게 되기 때문이다.
 */
export function redraw(root, draw) {
  const y = window.scrollY;
  const folds = foldState(root);
  clear(root);
  try {
    draw();
  } finally {
    foldEntries(root).forEach(({ node, key }) => {
      if (folds.has(key)) node.open = folds.get(key);
    });
    window.scrollTo(0, y);
  }
}

/* 설명 상자를 구별할 이름. 제목이 같은 상자가 여럿이라 뒤에 순번을 붙인다 */
function foldEntries(root) {
  const seen = new Map();
  return [...root.querySelectorAll('details.fold')].map((node) => {
    const title = node.querySelector('summary')?.textContent || '';
    const n = seen.get(title) || 0;
    seen.set(title, n + 1);
    return { node, key: `${title}#${n}` };
  });
}

function foldState(root) {
  return new Map(foldEntries(root).map(({ node, key }) => [key, node.open]));
}

/** 제목이 붙은 상자 */
export function card(title, ...body) {
  return h('section', { class: 'card' },
    title ? h('h3', { class: 'card-h' }, title) : null,
    ...body);
}

/** 접었다 펼 수 있는 상자 — 설명이 길 때 쓴다 */
export function foldout(title, bodyHtml, open = false) {
  const d = h('details', { class: 'fold' }, h('summary', {}, title));
  if (open) d.open = true;
  d.append(h('div', { class: 'fold-body', html: bodyHtml }));
  return d;
}

/** 라벨 + 드롭다운 */
export function selectRow(label, options, value, onChange, hint = '') {
  const sel = h('select', {
    class: 'sel',
    onchange: (e) => onChange(e.target.value),
  }, options.map((o) => h('option', {
    value: o.value, selected: String(o.value) === String(value),
  }, o.label)));
  return h('label', { class: 'ctrl' },
    h('span', { class: 'ctrl-l' }, label),
    sel,
    hint ? h('span', { class: 'ctrl-hint' }, hint) : null);
}

/** 라벨 + 슬라이더 + 현재값
 *
 * onInput  — 끄는 동안 계속 불린다. 값만 적어 두는 가벼운 일에 쓴다.
 * onChange — 손을 뗐을 때 한 번 불린다. **화면을 다시 그리는 일은 여기에 넣는다.**
 *            끄는 도중에 다시 그리면 잡고 있던 슬라이더가 사라져 드래그가 끊긴다.
 */
export function sliderRow(label, {
  min, max, step = 1, value, format = (v) => v, onInput, onChange, hint = '',
}) {
  const out = h('output', { class: 'ctrl-v' }, format(value));
  const input = h('input', {
    type: 'range', min, max, step, value, class: 'rng',
    oninput: (e) => {
      const v = Number(e.target.value);
      out.textContent = format(v);          // 숫자는 끄는 동안 바로바로 바뀐다
      if (onInput) onInput(v);
    },
    onchange: onChange ? (e) => onChange(Number(e.target.value)) : null,
  });
  return h('label', { class: 'ctrl' },
    h('span', { class: 'ctrl-l' }, label),
    h('span', { class: 'ctrl-rngwrap' }, input, out),
    hint ? h('span', { class: 'ctrl-hint' }, hint) : null);
}

/** 여러 개 중 하나를 고르는 알약 버튼들 */
export function pills(options, value, onChange, { small = false } = {}) {
  const wrap = h('div', { class: `pills${small ? ' pills-s' : ''}` });
  options.forEach((o) => {
    wrap.append(h('button', {
      type: 'button',
      class: `pill${String(o.value) === String(value) ? ' on' : ''}`,
      title: o.title || '',
      onclick: () => onChange(o.value),
    }, o.label));
  });
  return wrap;
}

export function checkRow(label, checked, onChange, hint = '') {
  return h('label', { class: 'ctrl ctrl-chk' },
    h('input', { type: 'checkbox', checked, onchange: (e) => onChange(e.target.checked) }),
    h('span', { class: 'ctrl-l' }, label),
    hint ? h('span', { class: 'ctrl-hint' }, hint) : null);
}

/** 표 만들기. cells 는 문자열이나 DOM 노드 모두 된다 */
export function table(headers, rows, { className = '', rowClass = null } = {}) {
  const thead = h('thead', {}, h('tr', {}, headers.map((x) =>
    h('th', { class: x?.align === 'left' ? 'ta-l' : '' }, x?.label ?? x))));
  const tbody = h('tbody', {}, rows.map((r, i) => h('tr', {
    class: rowClass ? rowClass(r, i) : '',
  }, (r.cells || r).map((c) => h('td', {}, c)))));
  return h('div', { class: 'tw' }, h('table', { class: `tbl ${className}` }, thead, tbody));
}

/** SVG 문자열을 담는 상자 */
export function figure(svg, caption = '') {
  return h('figure', { class: 'fig' },
    h('div', { class: 'fig-svg', html: svg }),
    caption ? h('figcaption', {}, caption) : null);
}

export function badge(text, kind = '') {
  return h('span', { class: `badge ${kind}` }, text);
}

/** 진행률 막대 */
export function progress(label) {
  const bar = h('div', { class: 'pg-bar' });
  const txt = h('span', { class: 'pg-txt' }, label);
  const node = h('div', { class: 'pg' }, h('div', { class: 'pg-track' }, bar), txt);
  return {
    node,
    set(ratio, text) {
      bar.style.width = `${Math.round(ratio * 100)}%`;
      if (text) txt.textContent = text;
    },
  };
}

/** 숫자를 화면용으로 다듬는다 */
export function num(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return v.toLocaleString('ko-KR');
  const a = Math.abs(v);
  // MSE 처럼 아주 큰 값은 지수 표기가 오히려 읽기 어렵다. 자릿점을 찍어 준다
  if (a >= 1e12) return v.toExponential(3);
  if (a >= 1000) return v.toLocaleString('ko-KR', { maximumFractionDigits: 1 });
  return v.toFixed(a < 0.001 ? 6 : digits);
}

/** 파이썬 코드 상자 */
export function codeBlock(code, { title = '' } = {}) {
  return h('div', { class: 'codebox' },
    title ? h('div', { class: 'codebox-h' }, title) : null,
    h('pre', { class: 'code' }, h('code', {}, code)));
}

/** 아주 간단한 마크다운 → HTML (문제 지문·해설을 보여 주는 데 쓴다) */
export function md(text) {
  if (!text) return '';
  const escaped = String(text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const lines = escaped.split('\n');
  const out = [];
  let inList = false;
  let inCode = false;
  const inline = (s) => s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  lines.forEach((raw) => {
    const line = raw.replace(/\s+$/, '');
    if (/^\s*```/.test(line)) {
      out.push(inCode ? '</pre>' : '<pre class="code">');
      inCode = !inCode; return;
    }
    if (inCode) { out.push(inline(line)); return; }
    const li = line.match(/^\s*[*-]\s+(.*)$/);
    const nested = /^\s{2,}[*-]\s+/.test(line);
    if (li) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li${nested ? ' class="sub"' : ''}>${inline(li[1])}</li>`);
      return;
    }
    if (inList) { out.push('</ul>'); inList = false; }
    if (!line.trim()) return;
    const hd = line.match(/^(#{1,6})\s+(.*)$/);
    if (hd) { out.push(`<h4>${inline(hd[2])}</h4>`); return; }
    if (/^\s*-{3,}\s*$/.test(line)) { out.push('<hr/>'); return; }
    out.push(`<p>${inline(line)}</p>`);
  });
  if (inList) out.push('</ul>');
  if (inCode) out.push('</pre>');
  return out.join('');
}
