import React, { useMemo, useState } from 'react';
import { FileText, Bookmark, Image as ImageIcon, ChevronDown } from 'lucide-react';

function CitationCard({ cite }) {
  const [open, setOpen] = useState(false);
  const isFigure = !!cite.figure_path;

  return (
    <button
      type="button"
      onClick={() => setOpen((o) => !o)}
      className="text-left p-3 bg-slate-800/40 hover:bg-slate-800/80 border border-slate-700/50 hover:border-sky-500/40 rounded-xl transition-all shadow-sm group w-full"
      aria-expanded={open}
    >
      <div className="flex items-center justify-between gap-2 text-xs font-semibold mb-1.5">
        <span className="flex items-center gap-1.5 text-slate-200 truncate min-w-0" title={cite.document}>
          <FileText className="w-3.5 h-3.5 text-sky-400 shrink-0" />
          <span className="truncate">{cite.document}</span>
        </span>
        <span className="flex items-center gap-1.5 shrink-0">
          {isFigure && <ImageIcon className="w-3.5 h-3.5 text-indigo-300" aria-label="Figure" />}
          <span className="bg-sky-500/10 text-sky-300 border border-sky-500/20 text-[10px] px-2 py-0.5 rounded-full font-mono">
            Pg. {cite.page}
          </span>
        </span>
      </div>

      {cite.section && cite.section !== 'General' && (
        <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-1.5 truncate">{cite.section}</p>
      )}

      <p className={`text-xs text-slate-400 italic leading-relaxed pl-2 border-l-2 border-sky-500/40 group-hover:border-sky-400 ${open ? '' : 'line-clamp-2'}`}>
        "{cite.snippet}"
      </p>
      <ChevronDown className={`w-3 h-3 text-slate-600 mt-1 ml-auto transition-transform ${open ? 'rotate-180' : ''}`} />
    </button>
  );
}

export default function CitationViewer({ citations }) {
  // the model may cite the same chunk twice: show each distinct reference once
  const unique = useMemo(() => {
    const seen = new Set();
    return (citations ?? []).filter((c) => {
      const key = `${c.document}|${c.page}|${c.snippet}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [citations]);

  if (unique.length === 0) return null;

  return (
    <div className="mt-6 pt-5 border-t border-slate-800/80">
      <div className="flex items-center gap-2 mb-3">
        <Bookmark className="w-4 h-4 text-sky-400" />
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
          Page-Level PDF References ({unique.length})
        </h4>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {unique.map((cite, idx) => (
          <CitationCard key={`${cite.document}-${cite.page}-${idx}`} cite={cite} />
        ))}
      </div>
    </div>
  );
}
