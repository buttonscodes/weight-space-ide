import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { AuthFlow, safeLoginUrl, claudeLoginUrl, claudeReadiness, codexReadiness } from '../auth.mjs';
const claudeUrl = 'https://claude.com/cai/oauth/authorize?code=true&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth%2Fcode%2Fcallback&state=example';
const codexUrl = 'https://auth.openai.com/oauth/authorize?state=example';
test('only official complete provider URLs are exposed', () => {
  assert.equal(safeLoginUrl('claude', claudeUrl), claudeUrl);
  assert.equal(claudeLoginUrl('\x1b[32m'+claudeUrl), null);
  assert.equal(claudeLoginUrl('\x1b[32m'+claudeUrl+'\x1b[0m\n'), claudeUrl);
  for (const url of [codexUrl.replace('auth.openai.com','auth.openai.com.evil.test'), codexUrl.replace('https:', 'http:'), codexUrl.replace('auth.openai.com','user@auth.openai.com'), codexUrl.replace('authorize','other'), codexUrl+'#fragment']) assert.equal(safeLoginUrl('codex', url), null);
  assert.equal(safeLoginUrl('claude', claudeUrl.replace('platform.claude.com','localhost')), null);
});
test('readiness honors configured providers without exposing identity', () => {
  assert.equal(codexReadiness({requiresOpenaiAuth:false,account:null}).status,'ready');
  assert.equal(codexReadiness({requiresOpenaiAuth:true,account:null}).status,'signed_out');
  assert.deepEqual(claudeReadiness({loggedIn:true,authMethod:'api_key',apiKey:'secret'}),{status:'ready',method:'api_key'});
  assert.throws(()=>claudeReadiness({loggedIn:'yes'}));
});
function fakeRpc({ready=false, early=false, fail=false}={}) {
  return class {
    static instance;
    constructor(exe,args,options,notify) { this.notify=notify; this.calls=[]; this.ready=ready; this.constructor.instance=this; }
    write() {}
    close() {this.closed=true;}
    async request(method, params) {
      this.calls.push([method,params]);
      if(method==='account/read') return {requiresOpenaiAuth:true,account:this.ready?{type:'chatgpt',email:'private'}:null};
      if(method==='account/login/start') {
        if(early) {this.ready=!fail; this.notify('account/login/completed',{loginId:'login-1',success:!fail,error:'secret error'});}
        return {loginId:'login-1',authUrl:codexUrl};
      }
      return {};
    }
  };
}
test('Codex login verifies completion and uses no threads or inference', async () => {
  const Rpc=fakeRpc({early:true}), events=[];
  await new AuthFlow({provider:'codex',operation:'login'}, e=>events.push(e),{Rpc}).run();
  assert.equal(events.at(-1).status,'ready');
  assert.equal(Rpc.instance.calls.filter(x=>x[0]==='account/read').length,2);
  assert.ok(Rpc.instance.calls.every(x=>/^(initialize|account\/)/.test(x[0])));
  assert.ok(!JSON.stringify(events).includes('private'));
  assert.equal(Rpc.instance.closed,true);
});
test('existing ready account is never replaced and errors are sanitized', async () => {
  for(const opts of [{ready:true},{early:true,fail:true}]) {
    const Rpc=fakeRpc(opts), events=[];
    await new AuthFlow({provider:'codex',operation:'login'},e=>events.push(e),{Rpc}).run();
    if(opts.ready) assert.ok(!Rpc.instance.calls.some(x=>x[0]==='account/login/start'));
    else assert.equal(events.at(-1).status,'error');
    assert.ok(!JSON.stringify(events).includes('secret error'));
  }
});
test('Codex cancellation routes the login id and retires transport', async () => {
  const Rpc=fakeRpc(), events=[];
  const flow=new AuthFlow({provider:'codex',operation:'login'},e=>events.push(e),{Rpc});
  const run=flow.run();
  await new Promise(r=>setImmediate(r));
  await flow.cancel(); await run;
  assert.deepEqual(Rpc.instance.calls.find(x=>x[0]==='account/login/cancel')[1],{loginId:'login-1'});
  assert.equal(Rpc.instance.closed,true);
  assert.ok(!events.some(e=>e.status==='ready'));
});
test('Claude parses signed-out JSON even with exit1, accepts code once and verifies', async () => {
  let statuses=0, loginChild; const events=[], writes=[];
  const spawnImpl=(exe,args,options)=> {
    const child=new EventEmitter(); child.stdout=new PassThrough(); child.stderr=new PassThrough(); child.stdin=new PassThrough(); child.kill=()=>{};
    if(args[1]==='status') setImmediate(()=>{child.stdout.write(JSON.stringify({loggedIn:++statuses>1,authMethod:'claude.ai'})); child.emit('close',statuses===1?1:0);});
    else {loginChild=child; assert.ok(options.env.BROWSER.endsWith('no-browser.sh')); child.stdin.on('data', chunk=>{writes.push(chunk.toString()); child.emit('close',0);}); setImmediate(()=>child.stdout.write(claudeUrl+'\n'));}
    return child;
  };
  const flow=new AuthFlow({provider:'claude',operation:'login'},e=>events.push(e),{spawnImpl});
  const run=flow.run();
  while(!flow.needsCode) await new Promise(r=>setImmediate(r));
  assert.throws(()=>flow.code('bad\ncode'));
  flow.code('secret-code#state'); assert.throws(()=>flow.code('again'));
  await run;
  assert.equal(events.at(-1).status,'ready'); assert.equal(statuses,2);
  assert.deepEqual(writes,['secret-code#state\n']); assert.ok(!JSON.stringify(events).includes('secret-code'));
});
