import { Wllama, LoggerWithoutDebug } from '@wllama/wllama';

const modelUrl = import.meta.env.DEV
  ? `${import.meta.env.BASE_URL}shards/countdown-Q4_K_M-00001-of-00005.gguf`
  : 'https://huggingface.co/IlyaasK/countdown-qwen3.5-0.8b-grpo/resolve/main/countdown-Q4_K_M-00001-of-00005.gguf';
const wasmUrl = new URL(`${import.meta.env.BASE_URL}wllama.wasm`, self.location.origin).href;
const firefox = /Firefox\//.test(navigator.userAgent);
const hasJSPI = !!WebAssembly.Suspending;
let model;
let ready = false;
let backend = 'gpu';

function message(type, detail = {}) {
  self.postMessage({ type, ...detail });
}

function promptFor(numbers, target) {
  const user = `Use each number in [${numbers.join(', ')}] exactly once to make ${target}.\nUse only +, -, *, /, and parentheses. Do not concatenate numbers or add new ones.\nRespond with only <answer>one arithmetic expression</answer>. No explanation.`;
  return `<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

async function openModel(useGPU) {
  model = new Wllama({ default: wasmUrl }, { logger: LoggerWithoutDebug });
  await model.loadModelFromUrl(modelUrl, {
      n_ctx: 512,
      n_gpu_layers: useGPU ? 99 : 0,
      n_threads: 1,
      reasoning: false,
      useCache: true,
      progressCallback: ({ loaded, total }) => message('progress', { loaded, total }),
  });
  backend = useGPU ? 'gpu' : 'cpu';
}

async function gpuSmokeTest() {
  const result = await model.createCompletion({
    prompt: promptFor([2, 3, 5, 10], 33),
    max_tokens: 48,
    temperature: 0.9,
    top_p: 0.95,
    stop: ['<|im_end|>', '<|endoftext|>'],
  });
  return /^\s*<answer>[^<>]+<\/answer>\s*$/.test(result.choices?.[0]?.text || '');
}

async function load() {
  if (ready) return message('ready', { backend });
  try {
    const canTryGPU = !!navigator.gpu && (!firefox || hasJSPI);
    if (!navigator.gpu && !firefox) throw new Error('WebGPU is unavailable in this browser.');
    message('status', { label: 'Downloading model' });
    if (canTryGPU) {
      try {
        await openModel(true);
        if (firefox) {
          message('status', { label: 'Checking Firefox GPU output' });
          if (!(await gpuSmokeTest())) throw new Error('Firefox GPU produced invalid model output');
        }
      } catch (error) {
        if (!firefox) throw error;
        await model?.exit().catch(() => {});
        model = null;
        message('status', { label: 'Switching to CPU for reliable output' });
        await openModel(false);
      }
    } else {
      message('status', { label: 'Loading Firefox CPU mode' });
      await openModel(false);
    }
    ready = true;
    message('ready', { backend });
  } catch (error) {
    message('error', { error: error?.message || String(error) });
  }
}

async function generate({ id, numbers, target }) {
  if (!ready) return message('error', { error: 'Load the model first.' });
  try {
    message('started', { id });
    const prompt = promptFor(numbers, target);
    const runCompletion = async (bufferOutput) => {
      let output = '';
      await model.createCompletion({
        prompt,
        max_tokens: 64,
        temperature: 0.9,
        top_p: 0.95,
        stop: ['<|im_end|>', '<|endoftext|>'],
        stream: true,
        onData: (chunk) => {
          const text = chunk.choices?.[0]?.text || '';
          output += text;
          if (text && !bufferOutput) message('token', { id, text });
        },
      });
      return output;
    };
    if (firefox && backend === 'gpu') {
      let output = '';
      try { output = await runCompletion(true); } catch { /* Retry below on CPU. */ }
      if (/^\s*<answer>[^<>]+<\/answer>\s*$/.test(output)) {
        message('token', { id, text: output });
      } else {
        message('status', { label: 'Firefox GPU output was invalid; retrying on CPU' });
        await model.exit().catch(() => {});
        await openModel(false);
        message('backend', { backend });
        await runCompletion(false);
      }
    } else {
      await runCompletion(false);
    }
    message('finished', { id });
  } catch (error) {
    message('error', { id, error: error?.message || String(error) });
  }
}

self.onmessage = ({ data }) => {
  if (data.type === 'load') load();
  if (data.type === 'generate') generate(data);
};
