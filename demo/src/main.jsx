import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { checkAnswer } from './verify.js';
import './style.css';

const START = { numbers: [2, 3, 5, 10], target: 33 };
const PUZZLES = [
  START,
  { numbers: [1, 3, 4, 6], target: 24 },
  { numbers: [1, 2, 3, 4], target: 10 },
  { numbers: [2, 4, 6, 8], target: 20 },
];

function App() {
  const [puzzle, setPuzzle] = useState(START);
  const [draft, setDraft] = useState({ numbers: START.numbers.map(String), target: String(START.target) });
  const [phase, setPhase] = useState('idle');
  const [progress, setProgress] = useState(null);
  const [status, setStatus] = useState('');
  const [backend, setBackend] = useState('');
  const [error, setError] = useState('');
  const [attempts, setAttempts] = useState([]);
  const worker = useRef(null);
  const pending = useRef(null);
  const idCounter = useRef(0);

  useEffect(() => {
    const instance = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
    worker.current = instance;
    instance.onmessage = ({ data }) => {
      if (data.type === 'progress') setProgress(data);
      if (data.type === 'status') setStatus(data.label);
      if (data.type === 'ready') { setPhase('ready'); setBackend(data.backend); setStatus(''); setError(''); }
      if (data.type === 'backend') setBackend(data.backend);
      if (data.type === 'error') { setPhase(data.id ? 'ready' : 'error'); setError(data.error); }
      if (data.type === 'started') setPhase('generating');
      if (data.type === 'token') {
        setAttempts((items) => items.map((item) => item.id === data.id ? { ...item, output: item.output + data.text } : item));
      }
      if (data.type === 'finished') {
        setAttempts((items) => items.map((item) => item.id === data.id ? {
          ...item,
          done: true,
          result: checkAnswer(item.output, pending.current.numbers, pending.current.target),
        } : item));
        setPhase('ready');
      }
    };
    return () => instance.terminate();
  }, []);

  function loadModel() {
    setError('');
    setPhase('loading');
    worker.current?.postMessage({ type: 'load' });
  }

  function updateNumber(index, value) {
    setDraft((current) => ({ ...current, numbers: current.numbers.map((n, i) => i === index ? value : n) }));
  }

  function choosePuzzle() {
    const other = PUZZLES.filter((item) => item.target !== Number(draft.target));
    const next = other[Math.floor(Math.random() * other.length)];
    setDraft({ numbers: next.numbers.map(String), target: String(next.target) });
    setAttempts([]);
  }

  function run(event) {
    event.preventDefault();
    const numbers = draft.numbers.map(Number);
    const target = Number(draft.target);
    if (draft.numbers.some((item) => !/^\d+$/.test(item)) || !/^\d+$/.test(draft.target) || numbers.some((n) => n < 1 || n > 100) || target < 1 || target > 999) {
      setError('Enter four whole numbers from 1–100 and a target from 1–999.');
      return;
    }
    if (phase !== 'ready') return;
    const next = { numbers, target };
    const id = ++idCounter.current;
    pending.current = next;
    setPuzzle(next);
    setAttempts((items) => [{ id, output: '', done: false }, ...items].slice(0, 10));
    setError('');
    worker.current?.postMessage({ type: 'generate', id, ...next });
  }

  const latest = attempts[0];
  const percent = progress?.total ? Math.min(100, Math.round(progress.loaded / progress.total * 100)) : 0;

  return <div className="page">
    <main>
      <p className="back"><a href="/">← Home</a></p>
      <h1>Countdown</h1>
      <p>Give a small model four numbers and a target. It runs in your browser and tries to find an expression using each number once.</p>
      <form onSubmit={run} aria-label="Countdown puzzle">
          <label className="field-label">Numbers</label>
          <div className="numbers">{draft.numbers.map((value, index) => <input key={index} aria-label={`Number ${index + 1}`} inputMode="numeric" value={value} onChange={(e) => updateNumber(index, e.target.value)} maxLength={3} />)}</div>
          <label className="field-label" htmlFor="target">Target</label>
          <input className="target" id="target" inputMode="numeric" value={draft.target} onChange={(e) => setDraft((current) => ({ ...current, target: e.target.value }))} maxLength={3} />
          <div className="actions"><button type="button" onClick={choosePuzzle}>New puzzle</button>{phase === 'idle' || phase === 'error' || phase === 'loading' ? <button type="button" disabled={phase === 'loading'} onClick={loadModel}>{phase === 'loading' ? `Loading ${percent}%` : 'Load model'}</button> : <button type="submit" disabled={phase === 'generating'}>{phase === 'generating' ? 'Generating…' : 'Try the model'}</button>}</div>
          <p className="helper">{phase === 'loading' || (phase === 'generating' && status.includes('retrying')) ? status : phase === 'ready' || phase === 'generating' ? `Running locally on ${backend === 'cpu' ? 'CPU' : 'GPU'}. Your puzzle stays in your browser.` : 'First visit downloads about 505 MB. Firefox may use the CPU if its GPU path fails a check.'}</p>
          {phase === 'loading' && <progress value={percent} max="100" aria-label="Model download progress" />}
          {error && <p className="error" role="alert">{error}</p>}
      </form>
      <h2>Model output</h2>
      <pre className="output" aria-live="polite">{latest ? latest.output || 'Generating…' : 'Load the model, then try a puzzle.'}</pre>
      {latest?.done && <p className={latest.result.valid ? 'success' : 'error'}>{latest.result.valid ? 'Solved.' : 'Not solved.'} {latest.result.reason}</p>}
      <p className="footnote">This is the 100,000-step Qwen3.5-0.8B RL checkpoint, quantized for the browser. It solved 56 of 252 held-out four-number puzzles in one attempt. Failed attempts are shown as they are.</p>
      <p><a href="https://github.com/IlyaasK/countdown-rl">Training and source</a> · <a href="https://huggingface.co/IlyaasK/countdown-qwen3.5-0.8b-grpo">Model weights</a></p>
      <hr />
      <p><a href="/">Home</a> | <a href="/blog/">Writing</a> | <a href="mailto:kapadiiy@mail.uc.edu">Email</a></p>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
