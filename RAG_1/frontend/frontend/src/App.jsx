import React, { useCallback, useEffect, useRef, useState } from 'react';
import DocumentSidebar from './components/DocumentSidebar';
import MessageView from './components/MessageView';
import { fetchDocuments, fetchHealth, getErrorMessage, sendChatMessage } from './api';
import { Send, Loader2, Square } from 'lucide-react';

/** Short plain-text version of an answer, sent back to the server as conversation history. */
const toHistoryText = (m) => {
  if (m.role === 'user') return m.content;
  if (!m.data) return '';
  return [m.data.title, m.data.summary].filter(Boolean).join('. ');
};

export default function App() {
  const [documents, setDocuments] = useState([]); // [{ document_name, doc_hash, chunks }]
  const [deselected, setDeselected] = useState(() => new Set()); // everything is searched unless unchecked
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState({ online: true, llm: true });

  const bottomRef = useRef(null);
  const abortRef = useRef(null);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await fetchDocuments());
    } catch (err) {
      console.error('Could not fetch indexed documents', err);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      const h = await fetchHealth();
      if (alive) setHealth(h);
    };
    check();
    const id = setInterval(check, 30_000);
    return () => {
      alive = false;
      clearInterval(id);
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, loading]);

  const toggleDocument = (hash) =>
    setDeselected((prev) => {
      const next = new Set(prev);
      next.has(hash) ? next.delete(hash) : next.add(hash);
      return next;
    });

  const handleDocumentsChanged = async () => {
    await refreshDocuments();
    fetchHealth().then(setHealth);
  };

  const selectedHashes = documents.map((d) => d.doc_hash).filter((h) => !deselected.has(h));
  const noDocuments = documents.length === 0;
  const nothingSelected = !noDocuments && selectedHashes.length === 0;
  const canSend = !!input.trim() && !loading && !noDocuments && !nothingSelected;

  const handleSend = async (e) => {
    e.preventDefault();
    if (!canSend) return;

    const userQuery = input.trim();
    const history = messages
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: toHistoryText(m) }))
      .filter((m) => m.content);

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userQuery }]);
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await sendChatMessage(userQuery, history, selectedHashes, controller.signal);
      setMessages((prev) => [...prev, { role: 'assistant', data: res.response, meta: res.meta, sources: res.sources }]);
    } catch (err) {
      if (controller.signal.aborted) {
        setMessages((prev) => [...prev, { role: 'assistant', error: 'Stopped.' }]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', error: getErrorMessage(err, 'Failed to get an answer.') }]);
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  };

  const placeholder = noDocuments
    ? 'Upload a PDF to start asking questions...'
    : nothingSelected
      ? 'Check at least one document in the sidebar...'
      : 'Ask a question about your textbook...';

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-950 font-sans">
      <DocumentSidebar
        documents={documents}
        deselected={deselected}
        onToggle={toggleDocument}
        onDocumentsChanged={handleDocumentsChanged}
        health={health}
      />

      <div className="flex-1 min-w-0 flex flex-col h-screen overflow-hidden">
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 text-center">
              <p className="text-lg font-medium mb-1">Multimodal Academic PDF Chatbot</p>
              <p className="text-sm">
                {noDocuments
                  ? 'Add a PDF from the sidebar to get started.'
                  : `${documents.length} document(s) indexed and ready - ask about text, tables, figures or formulas.`}
              </p>
            </div>
          ) : (
            messages.map((msg, i) => <MessageView key={i} message={msg} />)
          )}
          {loading && (
            <div className="flex items-center justify-center gap-3 p-4 text-sm text-slate-400">
              <Loader2 className="w-5 h-5 animate-spin text-sky-500" />
              Searching documents and writing the answer...
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={handleSend} className="p-4 bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 flex gap-2">
          <input
            type="text"
            className="flex-1 bg-slate-100 dark:bg-slate-800 border-0 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-sky-500 outline-none dark:text-white disabled:opacity-60"
            placeholder={placeholder}
            value={input}
            maxLength={4000}
            disabled={noDocuments || nothingSelected}
            onChange={(e) => setInput(e.target.value)}
          />
          {loading ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="bg-slate-600 hover:bg-slate-500 text-white rounded-xl px-5 py-3 flex items-center justify-center transition-colors"
              title="Stop"
              aria-label="Stop generating"
            >
              <Square className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSend}
              className="bg-sky-500 hover:bg-sky-600 text-white rounded-xl px-5 py-3 flex items-center justify-center transition-colors disabled:opacity-50"
              aria-label="Send"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
