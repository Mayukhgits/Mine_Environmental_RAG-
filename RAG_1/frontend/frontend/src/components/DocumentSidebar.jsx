import React, { useRef, useState } from 'react';
import { Upload, FileText, Loader2, BookOpen, Layers, Trash2, AlertCircle, Check } from 'lucide-react';
import { uploadPDF, deleteDocument, getErrorMessage } from '../api';

export default function DocumentSidebar({ documents, deselected, onToggle, onDocumentsChanged, health }) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(null); // 0-100 while sending, null once the server is indexing
  const [fileLabel, setFileLabel] = useState('');
  const [notice, setNotice] = useState(null); // { type: 'ok' | 'error', text }
  const [dragging, setDragging] = useState(false);
  const [deleting, setDeleting] = useState(null);
  const inputRef = useRef(null);

  const processFiles = async (fileList) => {
    const files = Array.from(fileList || []).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (files.length === 0) {
      setNotice({ type: 'error', text: 'Please choose PDF files.' });
      return;
    }
    setUploading(true);
    setNotice(null);
    let lastHash = null;
    const problems = [];
    let indexed = 0;
    let existed = 0;

    for (const file of files) {
      setFileLabel(file.name);
      setProgress(0);
      try {
        const result = await uploadPDF(file, setProgress);
        lastHash = result.doc_hash;
        if (result.status === 'already_exists') existed += 1;
        else indexed += 1;
      } catch (err) {
        problems.push(`${file.name}: ${getErrorMessage(err, 'Failed to process this PDF.')}`);
      }
    }

    setUploading(false);
    setProgress(null);
    if (indexed || existed) await onDocumentsChanged(lastHash);

    if (problems.length) setNotice({ type: 'error', text: problems.join('\n') });
    else
      setNotice({
        type: 'ok',
        text: [indexed && `${indexed} indexed`, existed && `${existed} already in the corpus`].filter(Boolean).join(', '),
      });
    if (inputRef.current) inputRef.current.value = ''; // allows re-selecting the same file
  };

  const handleDelete = async (doc) => {
    if (!window.confirm(`Remove "${doc.document_name}" from the knowledge base?`)) return;
    setDeleting(doc.doc_hash);
    try {
      await deleteDocument(doc.doc_hash);
      await onDocumentsChanged();
    } catch (err) {
      setNotice({ type: 'error', text: getErrorMessage(err, 'Could not delete the document.') });
    } finally {
      setDeleting(null);
    }
  };

  const activeCount = documents.filter((d) => !deselected.has(d.doc_hash)).length;
  const statusText = !health.online
    ? 'Backend offline'
    : !health.llm
      ? 'Backend online - Gemini key missing'
      : 'Qdrant + Gemini Online';
  const statusColor = !health.online ? 'bg-red-500' : !health.llm ? 'bg-amber-500' : 'bg-emerald-500';

  return (
    <aside className="w-80 shrink-0 bg-slate-900/90 backdrop-blur-xl border-r border-slate-800/80 flex flex-col h-screen text-slate-100 select-none shadow-2xl z-20">
      <div className="p-6 border-b border-slate-800/60 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-linear-to-tr from-sky-500 via-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight bg-linear-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              Academic RAG
            </h1>
            <p className="text-[10px] text-sky-400 font-medium tracking-widest uppercase">Multimodal AI Engine</p>
          </div>
        </div>
      </div>

      <div className="p-5 space-y-3">
        <label
          onDragOver={(e) => {
            e.preventDefault();
            if (!uploading) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (!uploading) processFiles(e.dataTransfer.files);
          }}
          className={`group relative flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-2xl transition-all duration-300 shadow-inner overflow-hidden ${
            uploading ? 'cursor-wait' : 'cursor-pointer'
          } ${dragging ? 'border-sky-400 bg-sky-500/10' : 'border-slate-700/80 hover:border-sky-500/80 bg-slate-800/30 hover:bg-sky-500/5'}`}
        >
          {uploading ? (
            <div className="flex flex-col items-center gap-2 z-10 px-4 w-full">
              <Loader2 className="w-7 h-7 animate-spin text-sky-400" />
              <span className="text-xs font-semibold text-slate-300 truncate max-w-full">{fileLabel}</span>
              <span className="text-[10px] text-slate-400">
                {progress !== null && progress < 100 ? `Uploading ${progress}%` : 'Extracting text, tables & figures...'}
              </span>
              {progress !== null && progress < 100 && (
                <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-sky-400 transition-all" style={{ width: `${progress}%` }} />
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 text-center p-3 z-10">
              <div className="w-10 h-10 rounded-full bg-slate-800/80 group-hover:bg-sky-500/20 flex items-center justify-center text-slate-400 group-hover:text-sky-400 transition-colors shadow-sm">
                <Upload className="w-5 h-5" />
              </div>
              <span className="text-xs font-medium text-slate-300 group-hover:text-white transition-colors">
                Drop or click to add PDFs
              </span>
              <span className="text-[10px] text-slate-500">Tables, figures and formulas are indexed</span>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            multiple
            className="hidden"
            onChange={(e) => processFiles(e.target.files)}
            disabled={uploading}
          />
        </label>

        {notice && (
          <div
            className={`flex items-start gap-2 text-[11px] rounded-lg px-3 py-2 border whitespace-pre-line ${
              notice.type === 'ok'
                ? 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20'
                : 'text-red-300 bg-red-500/10 border-red-500/20'
            }`}
            role="status"
          >
            {notice.type === 'ok' ? <Check className="w-3.5 h-3.5 mt-0.5 shrink-0" /> : <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />}
            <span className="min-w-0 wrap-break-word">{notice.text}</span>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-2 space-y-2">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1 px-1">
          <span className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-sky-400" /> Active Corpus
          </span>
          <span className="bg-slate-800 text-sky-400 text-[10px] px-2 py-0.5 rounded-full font-mono">
            {activeCount}/{documents.length}
          </span>
        </div>
        {documents.length > 0 && <p className="text-[10px] text-slate-500 px-1 pb-1">Questions search only the checked documents.</p>}

        {documents.length === 0 ? (
          <div className="text-center py-8 px-4 border border-slate-800/40 rounded-xl bg-slate-800/10">
            <p className="text-xs text-slate-500">No PDFs indexed yet.</p>
          </div>
        ) : (
          documents.map((doc) => {
            const active = !deselected.has(doc.doc_hash);
            return (
              <div
                key={doc.doc_hash}
                className={`group relative flex items-center gap-3 p-3 rounded-xl border transition-all duration-200 shadow-sm ${
                  active
                    ? 'bg-slate-800/60 border-sky-500/30'
                    : 'bg-slate-800/20 border-slate-700/40 opacity-60'
                }`}
              >
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onToggle(doc.doc_hash)}
                  className="accent-sky-500 w-3.5 h-3.5 shrink-0 cursor-pointer"
                  aria-label={`Include ${doc.document_name} in answers`}
                />
                <FileText className="w-4 h-4 text-sky-400 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-slate-200 font-medium truncate" title={doc.document_name}>
                    {doc.document_name}
                  </p>
                  {doc.chunks ? <p className="text-[10px] text-slate-500 font-mono">{doc.chunks} passages</p> : null}
                </div>
                <button
                  type="button"
                  onClick={() => handleDelete(doc)}
                  disabled={deleting === doc.doc_hash}
                  className="shrink-0 p-1 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 focus:opacity-100 transition disabled:opacity-100"
                  title="Remove document"
                  aria-label={`Remove ${doc.document_name}`}
                >
                  {deleting === doc.doc_hash ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                </button>
              </div>
            );
          })
        )}
      </div>

      <div className="p-4 border-t border-slate-800/60 bg-slate-950/40">
        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <span className="relative flex h-2 w-2">
            {health.online && health.llm && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />}
            <span className={`relative inline-flex rounded-full h-2 w-2 ${statusColor}`} />
          </span>
          <span className="font-mono text-slate-300">{statusText}</span>
        </div>
      </div>
    </aside>
  );
}
