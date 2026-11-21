import { createInterface } from 'node:readline';
const lines = createInterface({ input: process.stdin });
let clientId;
lines.on('line', line => {
  const m = JSON.parse(line);
  if (m.method === 'test') { clientId = m.id; process.stdout.write(JSON.stringify({ id: 'ask-1', method: 'ask', params: {} }) + '\n'); }
  else if (m.id === 'ask-1') process.stdout.write(JSON.stringify({ id: clientId, result: { received: m.result?.approved === false } }) + '\n');
});
