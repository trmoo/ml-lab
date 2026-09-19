import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

/* 번들 맨 앞에 남길 저작권 배너.
 *
 * `/*!` 로 시작해야 압축(minify)해도 지워지지 않는다. esbuild 는 이렇게 시작하는 주석과
 * @license·@preserve 가 든 주석만 「법적 주석」으로 보고 살려 두는데, 그러려면
 * esbuild.legalComments 를 'inline' 으로 켜 줘야 한다.
 *   - 'none' 으로 두면 `/*!` 배너까지 지워진다. (실제로 한 번 그렇게 날렸다)
 *   - 소스 파일마다 붙여 둔 머리 주석은 보통 주석이라 어차피 걷혀 나가므로,
 *     'inline' 이어도 번들에는 이 배너 하나만 남는다.
 * 아래 finishBuild 플러그인이 배너가 살아남았는지 빌드마다 확인한다. */
const BANNER = '/*! 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유\n'
  + ' * 무단 배포 및 상업적 이용을 금합니다. 학교 수업 목적으로만 이용해 주세요.\n'
  + ' * 오픈소스 라이선스를 붙이지 않았습니다. 자세한 내용은 LICENSE 파일을 보세요. */';

/* 빌드 결과를 dist/index.html 한 파일로 묶는다.
 * 실습 데이터(csv 11개)와 문제 은행(quiz/*.json)도 함께 안에 들어가므로,
 * 교실 PC 에서 인터넷도 서버도 없이 더블클릭만으로 열린다.
 *
 * 이때 두 가지를 함께 맞춰 준다.
 *   ① 번들 형식을 iife 로 — ES 모듈은 file:// 에서 브라우저가 막을 수 있다.
 *   ② <script type="module"> 을 그냥 <script> 로 바꿔 준다.
 * 개발 서버(npm start)는 그대로 ES 모듈로 돌아가므로 작업하기에도 편하다.
 */
const finishBuild = {
  name: 'plain-script-tag',
  enforce: 'post',
  async closeBundle() {
    const { readFile, writeFile } = await import('node:fs/promises');
    const file = new URL('./dist/index.html', import.meta.url);
    let html = await readFile(file, 'utf8');

    html = html
      .replace(/<script\s+type="module"\s+crossorigin\s*>/g, '<script>')
      .replace(/<script\s+type="module"\s*>/g, '<script>')
      .replace(/<link\s+rel="modulepreload"[^>]*>\s*/g, '');

    /* 정규식이 (Vite 가 태그 모양을 바꾸는 등의 이유로) 하나도 맞지 않으면
     * html 이 그대로 남는다. 조용히 성공하지 않도록 반드시 확인한다. */
    if (/type="module"/.test(html)) {
      throw new Error(
        '빌드 결과에 type="module" 이 남아 있습니다. 더블클릭(file://) 실행이 막힐 수 있습니다.\n'
        + 'vite.config.js 의 정규식이 <script> 태그 모양과 맞는지 확인하세요.');
    }

    /* singlefile 플러그인은 스크립트를 <head> 안에 넣는다.
     * 위에서 type="module" 을 떼어 냈으므로 이 스크립트는 본문보다 먼저 실행된다.
     * src/main.js 가 DOMContentLoaded 를 기다리지 않으면 화면이 텅 빈 채로 뜨는데,
     * 개발 서버(모듈은 지연 실행)에서는 멀쩡해 보여서 알아채기 어렵다. 그래서 여기서 막는다. */
    const scriptInHead = html.indexOf('<script') < html.indexOf('<body');
    if (scriptInHead && !html.includes('DOMContentLoaded')) {
      throw new Error(
        '스크립트가 <head> 에 있는데 DOMContentLoaded 를 기다리는 코드가 없습니다.\n'
        + '이대로 배포하면 화면이 비어 있게 됩니다. src/main.js 의 start() 대기 코드를 확인하세요.');
    }

    /* 저작권 표시가 빌드 결과 안에 세 곳 남아 있어야 한다 —
     *   ① 맨 위 HTML 주석  ② 스크립트의 /*! 배너  ③ 화면 아래 푸터
     * 압축 설정을 잘못 건드리면 ②가 조용히 사라진다. 그래서 여기서 센다.
     * (파워셸 Get-Content 로 세면 안 된다. BOM 없는 UTF-8 을 CP949 로 읽어
     *  한글이 깨져 「0회」라는 거짓 결과가 나온다.) */
    const marks = (html.match(/티쳐무/g) || []).length;
    if (marks < 3) {
      throw new Error(
        `빌드 결과에 저작권 표시가 ${marks}곳뿐입니다 (세 곳이어야 합니다).\n`
        + '  ① 맨 위 HTML 주석  ② 스크립트의 /*! 배너  ③ 화면 아래 푸터\n'
        + "빠진 것이 ② 라면 vite.config.js 의 esbuild.legalComments 가 'inline' 인지,\n"
        + 'BANNER 가 /*! 로 시작하는지 확인하세요.');
    }

    await writeFile(file, html, 'utf8');
  },
};

export default defineConfig({
  base: './',            // GitHub Pages 하위 경로에서도 자원 경로가 맞도록
  plugins: [viteSingleFile({ useRecommendedBuildConfig: false }), finishBuild],
  esbuild: { legalComments: 'inline' },   // /*! 배너를 압축에서 살려 둔다 (위 설명 참고)
  build: {
    outDir: 'dist',
    assetsInlineLimit: 100000000,
    cssCodeSplit: false,
    target: 'es2020',
    modulePreload: false,
    chunkSizeWarningLimit: 4000,   // 실습 데이터를 함께 넣으므로 번들이 크다
    rollupOptions: {
      output: {
        format: 'iife',
        inlineDynamicImports: true,
        entryFileNames: 'app.js',
        assetFileNames: 'app.[ext]',
        banner: BANNER,
      },
    },
  },
  server: { port: 5187, open: false },
  preview: { port: 5188 },
});
