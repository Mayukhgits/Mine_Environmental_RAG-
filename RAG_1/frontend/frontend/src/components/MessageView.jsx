import React, { useState } from 'react';
import CitationViewer from './CitationViewer';
import Markdown, { MathBlock } from './Markdown';
import { assetUrl } from '../config';
import { Image as ImageIcon, Sparkles, CheckCircle2, User, FunctionSquare, AlertTriangle, Clock } from 'lucide-react';

function Figure({ fig }) {
  const [failed, setFailed] = useState(false);
  // backend sends a web path in image_url; image_path is a server-side file path and cannot be loaded by a browser
  const src = assetUrl(fig.image_url || fig.figure_path);
  if (!src) return null;

  return (
    <figure className="border border-slate-800 rounded-xl p-3 bg-slate-950/50 text-center space-y-2 group hover:border-slate-700 transition-colors">
      {failed ? (
        <div className="py-10 text-xs text-slate-500">Image could not be loaded</div>
      ) : (
        <a href={src} target="_blank" rel="noopener noreferrer" title="Open full size">
          <img
            src={src}
            alt={fig.caption || `Figure on page ${fig.page}`}
            loading="lazy"
            onError={() => setFailed(true)}
            className="max-h-64 mx-auto object-contain rounded-lg shadow-md bg-white/95 group-hover:scale-[1.01] transition-transform"
          />
        </a>
      )}
      <figcaption className="text-xs text-slate-400 font-medium">
        {fig.caption} {fig.page ? <span className="text-sky-400 font-mono">(p. {fig.page})</span> : null}
      </figcaption>
    </figure>
  );
}

function ErrorBubble({ text }) {
  return (
    <div className="flex items-start gap-4 mb-8 max-w-4xl mx-auto">
      <div className="w-9 h-9 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 shrink-0 mt-1">
        <AlertTriangle className="w-4 h-4" />
      </div>
      <div className="flex-1 bg-red-950/30 border border-red-500/20 rounded-2xl px-5 py-4 text-sm text-red-200">{text}</div>
    </div>
  );
}

export default function MessageView({ message }) {
  if (message.role === 'user') {
    return (
      <div className="flex items-start gap-3 justify-end mb-8">
        <div className="bg-linear-to-r from-sky-500 to-blue-600 text-white px-5 py-3.5 rounded-2xl rounded-tr-sm max-w-xl shadow-lg shadow-sky-500/10 border border-sky-400/20 text-sm leading-relaxed whitespace-pre-wrap wrap-break-word">
          {message.content}
        </div>
        <div className="w-9 h-9 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0 shadow">
          <User className="w-4 h-4" />
        </div>
      </div>
    );
  }

  if (message.error || !message.data) {
    return <ErrorBubble text={message.error || 'The server returned an empty answer.'} />;
  }

  const ans = message.data;
  const meta = message.meta;
  const figures = (ans.figures ?? []).filter((f) => f.image_url || f.figure_path);

  return (
    <div className="flex items-start gap-4 mb-10 max-w-4xl mx-auto">
      <div className="w-9 h-9 rounded-xl bg-linear-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white shrink-0 shadow-lg shadow-sky-500/20 mt-1">
        <Sparkles className="w-4 h-4" />
      </div>

      <div className="flex-1 min-w-0 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6">
        <div className="border-b border-slate-800 pb-4">
          <h2 className="text-xl font-bold bg-linear-to-r from-white via-slate-100 to-slate-300 bg-clip-text text-transparent">
            {ans.title}
          </h2>
          {ans.summary && (
            <div className="mt-2 bg-slate-800/30 p-3 rounded-xl border border-slate-800">
              <Markdown>{ans.summary}</Markdown>
            </div>
          )}
        </div>

        {ans.sections?.map((sec, idx) => (
          <div key={idx} className="space-y-2">
            <h3 className="text-sm font-bold text-sky-400 tracking-wide uppercase">{sec.heading}</h3>
            <Markdown>{sec.content}</Markdown>
          </div>
        ))}

        {ans.equations?.length > 0 && (
          <div className="p-4 bg-slate-950/60 border border-indigo-500/20 rounded-xl space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
              <FunctionSquare className="w-4 h-4" /> Formulas & Mathematical Models
            </div>
            <div className="space-y-2">
              {ans.equations.map((eq, i) => (
                <div key={i} className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800 text-center">
                  <MathBlock tex={eq} />
                </div>
              ))}
            </div>
          </div>
        )}

        {figures.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <ImageIcon className="w-4 h-4 text-sky-400" /> Context Visualizations
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {figures.map((fig, idx) => (
                <Figure key={idx} fig={fig} />
              ))}
            </div>
          </div>
        )}

        {ans.key_takeaways?.length > 0 && (
          <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 space-y-2">
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Key Takeaways</h4>
            <ul className="space-y-2">
              {ans.key_takeaways.map((point, idx) => (
                <li key={idx} className="flex items-start gap-2.5 text-sm text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span className="leading-relaxed min-w-0">
                    <Markdown>{point}</Markdown>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <CitationViewer citations={ans.citations} />

        {meta && (
          <div className="flex items-center gap-3 pt-1 text-[10px] text-slate-500 font-mono">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" /> {(meta.latency_ms / 1000).toFixed(1)}s
            </span>
            {meta.grounded === false && <span className="text-amber-400">no supporting passages found</span>}
          </div>
        )}
      </div>
    </div>
  );
}
