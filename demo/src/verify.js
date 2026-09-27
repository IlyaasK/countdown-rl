const ANSWER = /^\s*<answer>\s*([^<>]+?)\s*<\/answer>\s*$/s;

function gcd(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) [a, b] = [b, a % b];
  return a || 1n;
}

function fraction(n, d = 1n) {
  if (d === 0n) throw new Error('Division by zero');
  if (d < 0n) [n, d] = [-n, -d];
  const divisor = gcd(n, d);
  return [n / divisor, d / divisor];
}

function operate(left, right, operator) {
  const [a, b] = left;
  const [c, d] = right;
  switch (operator) {
    case '+': return fraction(a * d + c * b, b * d);
    case '-': return fraction(a * d - c * b, b * d);
    case '*': return fraction(a * c, b * d);
    case '/': return fraction(a * d, b * c);
    default: throw new Error('Unsupported operator');
  }
}

export function checkAnswer(output, numbers, target) {
  const visible = output.replace(/(?:<\|im_end\|>|<\|endoftext\|>)\s*$/g, '').trim();
  const tagged = ANSWER.exec(visible);
  if (!tagged) return { valid: false, reason: 'Reply must contain only one <answer> expression.' };
  const expression = tagged[1].trim();
  if (expression.length > 256 || !/^[\d\s()+*/-]+$/.test(expression)) {
    return { valid: false, expression, reason: 'Only integers, +, −, ×, ÷ and parentheses are allowed.' };
  }

  try {
    const tokens = expression.match(/\d+|[()+*/-]/g) || [];
    let position = 0;
    const used = [];
    function atom() {
      const token = tokens[position++];
      if (token === '(') {
        const result = sum();
        if (tokens[position++] !== ')') throw new Error('Unbalanced parentheses');
        return result;
      }
      if (!/^\d+$/.test(token ?? '')) throw new Error('Expected a number');
      // Match Python's AST reward: 01 is not a valid integer literal.
      if (token.length > 1 && token.startsWith('0')) throw new Error('Invalid integer literal');
      used.push(BigInt(token));
      return fraction(BigInt(token));
    }
    function product() {
      let result = atom();
      while (tokens[position] === '*' || tokens[position] === '/') {
        const op = tokens[position++];
        result = operate(result, atom(), op);
      }
      return result;
    }
    function sum() {
      let result = product();
      while (tokens[position] === '+' || tokens[position] === '-') {
        const op = tokens[position++];
        result = operate(result, product(), op);
      }
      return result;
    }
    const result = sum();
    if (position !== tokens.length) throw new Error('Unexpected text in expression');
    const expected = numbers.map(BigInt).sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
    used.sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
    if (used.length !== expected.length || used.some((value, index) => value !== expected[index])) {
      throw new Error('Use each supplied number exactly once');
    }
    const valid = result[0] === BigInt(target) * result[1];
    return {
      valid,
      expression,
      value: result[1] === 1n ? String(result[0]) : `${result[0]}/${result[1]}`,
      reason: valid ? 'Exact match. Every number was used once.' : 'Valid expression, but it misses the target.',
    };
  } catch (error) {
    return { valid: false, expression, reason: error.message };
  }
}
