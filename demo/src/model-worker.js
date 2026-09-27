import { Wllama, LoggerWithoutDebug } from '@wllama/wllama';

const modelUrl = import.meta.env.DEV
  ? `${import.meta.env.BASE_URL}shards/countdown-Q4_K_M-00001-of-00005.gguf`
  : 'https://huggingface.co/IlyaasK/countdown-qwen3.5-0.8b-grpo/resolve/main/countdown-Q4_K_M-00001-of-00005.gguf';
const wasmUrl = new URL(`${import.meta.env.BASE_URL}wllama.wasm`, self.location.origin).href;
let model;
let ready = false;

function message(type, detail = {}) {
  self.postMessage({ type, ...detail });
}

async function load() {
  if (ready) return message('ready');
  try {
    model = new Wllama({ default: wasmUrl }, { logger: LoggerWithoutDebug });
    if (!model.isSupportWebGPU()) throw new Error('WebGPU is unavailable in this browser or device. Try recent Chrome or Edge with a supported GPU.');
    message('status', { label: 'Downloading model' });
    await model.loadModelFromUrl(modelUrl, {
      n_ctx: 512,
      n_gpu_layers: 99,
      n_threads: 1,
      reasoning: false,
      useCache: true,
      progressCallback: ({ loaded, total }) => message('progress', { loaded, total }),
    });
    ready = true;
    message('ready');
  } catch (error) {
    message('error', { error: error?.message || String(error) });
  }
}

async function generate({ id, numbers, target }) {
  if (!ready) return message('error', { error: 'Load the model first.' });
  const user = `Use each number in [${numbers.join(', ')}] exactly once to make ${target}.\nUse only +, -, *, /, and parentheses. Do not concatenate numbers or add new ones.\nRespond with only <answer>one arithmetic expression</answer>. No explanation.`;
  // Match the Qwen chat template used for direct-answer RL training.
  const prompt = `<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
  try {
    message('started', { id });
    await model.createCompletion({
      prompt,
      max_tokens: 64,
      temperature: 0.9,
      top_p: 0.95,
      stop: ['<|im_end|>', '<|endoftext|>'],
      stream: true,
      onData: (chunk) => {
        const text = chunk.choices?.[0]?.text || '';
        if (text) message('token', { id, text });
      },
    });
    message('finished', { id });
  } catch (error) {
    message('error', { id, error: error?.message || String(error) });
  }
}

self.onmessage = ({ data }) => {
  if (data.type === 'load') load();
  if (data.type === 'generate') generate(data);
};
