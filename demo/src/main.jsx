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
      if (data.type === 'ready') { setPhase('ready'); setError(''); }
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
    <header className="masthead"><a href="/" className="back">← ilya.as</a><span className="masthead-end">A BROWSER-NATIVE RL EXPERIMENT <span className="star">✳</span></span></header>
    <main>
      <section className="intro"><h1>Count<span>down</span><i>.</i></h1><p>Four numbers. One target. A small model trained with reinforcement learning to find the arithmetic in between.</p></section>
      <section className="workspace" aria-label="Countdown demo">
        <form className="challenge" onSubmit={run}>
          <div className="section-head"><span>01 / THE CHALLENGE</span><button type="button" onClick={choosePuzzle} className="text-button">New puzzle ↗</button></div>
          <label className="field-label">YOUR NUMBERS <small>Use each exactly once</small></label>
          <div className="numbers">{draft.numbers.map((value, index) => <input key={index} aria-label={`Number ${index + 1}`} inputMode="numeric" value={value} onChange={(e) => updateNumber(index, e.target.value)} maxLength={3} />)}</div>
          <label className="field-label" htmlFor="target">TARGET NUMBER</label>
          <div className="target-line"><span className="equals">=</span><input id="target" inputMode="numeric" value={draft.target} onChange={(e) => setDraft((current) => ({ ...current, target: e.target.value }))} maxLength={3} /></div>
          {phase === 'idle' || phase === 'error' || phase === 'loading' ? <button className="primary" type="button" disabled={phase === 'loading'} onClick={loadModel}>{phase === 'loading' ? `Loading model ${percent}%` : 'Load model'} <span>↗</span></button> : <button className="primary" type="submit" disabled={phase === 'generating'}>{phase === 'generating' ? 'Thinking…' : 'Try the model'} <span>↗</span></button>}
          <p className="helper">{phase === 'ready' || phase === 'generating' ? 'Runs locally on your GPU. Your puzzle stays in your browser.' : 'First visit downloads ~505 MB. A WebGPU-capable browser and enough memory are required.'}</p>
          {phase === 'loading' && <div className="progress" role="progressbar" aria-valuenow={percent} aria-valuemin="0" aria-valuemax="100"><span style={{ width: `${percent}%` }} /></div>}
          {error && <p className="error" role="alert">{error}</p>}
        </form>
        <div className="result">
          <div className="section-head"><span>02 / THE MODEL</span><span className="live"><span className={phase === 'ready' || phase === 'generating' ? 'on' : ''} /> {phase === 'ready' ? 'READY' : phase === 'generating' ? 'GENERATING' : phase === 'loading' ? 'LOADING' : 'NOT LOADED'}</span></div>
          <div className="terminal"><div className="terminal-top"><span className="dots"><i/><i/><i/></span><span>QWEN3.5 · 0.8B · GRPO</span></div><div className="terminal-body"><span className="terminal-prefix">&gt; model output</span><pre>{latest ? latest.output || 'Generating…' : 'Load the model, then try a puzzle. The output shown here will be generated by your browser—not a scripted answer.'}</pre>{latest?.done && <div className={latest.result.valid ? 'verdict success' : 'verdict miss'}>{latest.result.valid ? '✓ VERIFIED' : '× NOT SOLVED'} <span>{latest.result.reason}</span></div>}</div></div>
          <div className="result-foot"><span>Actual model output · exact arithmetic verification</span><span>{attempts.length ? `ATTEMPT ${attempts.length}` : 'AWAITING INPUT'}</span></div>
        </div>
      </section>
      <div className="footnote"><p>This is the latest 100,000-step checkpoint, quantized for the browser. On a held-out set it solved 56 of 252 four-number puzzles in one attempt. It may miss; failed attempts are shown honestly.</p><a href="https://huggingface.co/IlyaasK/countdown-qwen3.5-0.8b-grpo" target="_blank" rel="noreferrer">Model & weights ↗</a></div>
    </main><footer><span>ILYAAS KAPADIA / 2026</span><span>BUILT WITH RL, RUN IN YOUR BROWSER</span></footer>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
