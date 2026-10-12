import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import katex from 'katex';
import 'katex/dist/katex.min.css';

/** The backend writes LaTeX as \( ... \) and \[ ... \]; remark-math expects $ ... $ and $$ ... $$. */
export function normalizeMath(text = '') {
  return String(text)
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n\n$$\n${m.trim()}\n$$\n\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$${m.trim()}$`);
}

function renderKatex(tex, display) {
  try {
    return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore', output: 'html' });
  } catch {
    return null;
  }
}

/** Standalone formula (used for the "equations" list). */
export function MathBlock({ tex }) {
  const html = renderKatex(tex, true);
  if (!html) return <code className="text-emerald-300">{tex}</code>;
  return <div className="overflow-x-auto py-1 text-slate-100" dangerouslySetInnerHTML={{ __html: html }} />;
}

const components = {
  // remark-math emits <code class="language-math ..."> - render those with KaTeX
  pre: ({ children }) => <>{children}</>,
  code({ className = '', children }) {
    const text = String(children ?? '').replace(/\n$/, '');
    if (className.includes('language-math')) {
      const display = className.includes('math-display');
      const html = renderKatex(text.trim(), display);
      if (!html) return <code>{text}</code>;
      return (
        <span
          className={display ? 'block my-3 overflow-x-auto text-center text-slate-100' : 'text-slate-100'}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    }
    if (className || text.includes('\n')) {
      return (
        <pre className="my-3 overflow-x-auto rounded-lg bg-slate-950 border border-slate-800 p-3 text-xs text-slate-200">
          <code>{text}</code>
        </pre>
      );
    }
    return <code className="rounded bg-slate-800 px-1.5 py-0.5 text-[0.85em] text-sky-300">{text}</code>;
  },
  // Tailwind's preflight strips default table styling, so tables need explicit classes
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-slate-700/70">
      <table className="min-w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-slate-800/80 text-sky-300">{children}</thead>,
  th: ({ children }) => <th className="px-3 py-2 text-left font-semibold whitespace-nowrap border-b border-slate-700">{children}</th>,
  td: ({ children }) => <td className="px-3 py-1.5 text-slate-300 border-b border-slate-800/80 align-top">{children}</td>,
  tr: ({ children }) => <tr className="even:bg-slate-900/40 hover:bg-slate-800/40">{children}</tr>,
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="my-2 list-disc pl-5 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 list-decimal pl-5 space-y-1">{children}</ol>,
  strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-sky-400 underline underline-offset-2">
      {children}
    </a>
  ),
};

export default function Markdown({ children }) {
  return (
    <div className="text-slate-300 text-sm">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} components={components}>
        {normalizeMath(children)}
      </ReactMarkdown>
    </div>
  );
}
