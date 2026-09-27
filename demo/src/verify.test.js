import test from 'node:test';
import assert from 'node:assert/strict';
import { checkAnswer } from './verify.js';

test('accepts an exact, all-number answer', () => {
  assert.equal(checkAnswer('<answer>10 * 3 + 5 - 2</answer>', [2, 3, 5, 10], 33).valid, true);
});

test('uses exact fractions', () => {
  assert.equal(checkAnswer('<answer>8 / (3 - 8 / 3)</answer>', [8, 3, 8, 3], 24).valid, true);
});

test('rejects omitted or invented numbers', () => {
  assert.equal(checkAnswer('<answer>33</answer>', [2, 3, 5, 10], 33).valid, false);
  assert.equal(checkAnswer('<answer>10 * 3 + 5 - 1</answer>', [2, 3, 5, 10], 34).valid, false);
});

test('rejects extra text, unsafe syntax and division by zero', () => {
  assert.equal(checkAnswer('I think <answer>10 * 3 + 5 - 2</answer>', [2, 3, 5, 10], 33).valid, false);
  assert.equal(checkAnswer('<answer>10 ** 2</answer>', [10, 2], 100).valid, false);
  assert.equal(checkAnswer('<answer>3 / (2 - 2)</answer>', [3, 2, 2], 3).valid, false);
});
