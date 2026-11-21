// Model-free Probe tab: each view's request, rendering, bounds, waiting state and errors.
import test from 'node:test';
import assert from 'node:assert/strict';
import { probeView, onProbeSteeringEvent, rowTools } from '../web/js/probe.js';
const tick=()=>new Promise(r=>setImmediate(r));
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  click(){this.events.click?.({target:this});}
  focus(){}
}
global.document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text}),documentElement:{dataset:{theme:'light'}}};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const button=(root,label)=>walk(root).find(n=>n.tag==='button'&&text(n)===label);
const labelled=(root,label)=>walk(root).find(n=>n['aria-label']===label);
const type=(root,label,value)=>labelled(root,label).events.input({target:{value}});
const response=(data,ok=true)=>({ok,statusText:'Bad Request',headers:{get:()=> 'application/json'},json:async()=>data});
function app(view){
  const a={sel:'model',tab:'Probe',variant:'current',keep:{},state:{open:{name:'fixture',kind:'wsp'},status:{can_run:true,version:0}},selected:[],rerender(){},select(addr,tab){a.selected.push([addr,tab]);}};
  if(view)a.keep.probe={view,reads:{text:'',neurons:''},fires:{text:'',k:20,sign:'both'},next:{prompt:'',source:'panel',typed:'',top:10},contrast:{a:'',b:'',k:20}};
  return a;
}
// Each request is held until the test answers it.
function server(){
  const held={requests:[]};
  global.fetch=async(url,o)=>{
    const r={url,body:JSON.parse(o.body)};held.requests.push(r);
    return new Promise(resolve=>{held.answer=(data,ok=true)=>resolve(response(data,ok));});
  };
  return held;
}
async function view(state){const root=new Element('root');await probeView(root,state);return root;}
const settle=async()=>{await tick();await tick();};

test('a model that cannot run says so and offers no probe',async()=>{
  const a=app();a.state.status={can_run:false,run_error:'its weights are still on the Hub'};
  const root=await view(a);
  assert.match(text(root),/Probes run the model, and this model cannot be run here: its weights are still on the Hub/);
  assert.equal(button(root,'Find'),undefined);
});

test('reads: tokens shaded per neuron, the base row only where it differs, exact values on hover',async()=>{
  const a=app('reads'),held=server();
  let root=await view(a);
  assert.match(text(root),/Up to 256 tokens and 16 neurons/);
  assert.match(text(root),/0 of 20,000 characters/);
  type(root,'reads text','abc');
  assert.match(text(root),/3 of 20,000 characters/);
  type(root,'reads neurons','L1 N3, blocks.0.mlp.neurons.1-2');
  button(root,'Read').click();
  assert.match(text(root),/running… If a reply is being generated, this waits until it finishes\./);
  assert.deepEqual(held.requests[0],{url:'/api/probe/reads',body:{prompt:'abc',neurons:['L1N3','blocks.0.mlp.neurons.1-2'],variants:['base','current']}});
  held.answer({addrs:['L1N3','L0N1'],token_strs:['a','b','c'],steered:false,values:{current:[[1,-2,0.5],[4,0,0]],base:[[1,-2,0.5],[3,0,0]]}});
  await settle();
  root=await view(a);
  const rows=walk(root).filter(n=>n.className==='toks');
  assert.equal(rows.length,3);                                    // L1N3 edited; L0N1 edited and base
  assert.match(text(root),/L1N3 · edited/);
  assert.match(text(root),/base: the same/);
  assert.match(text(root),/L0N1 · base/);
  const tok=rows[0].children[1];
  assert.equal(tok.title,'L1N3 edited, token 1 "b": -2');
  assert.match(text(root),/scaled to the largest magnitude in this table \(4\)/);
  assert.notEqual(rows[0].children[0].style.background,rows[0].children[2].style.background);
});

test('reads: problems are shown under the button before anything is sent; server errors too',async()=>{
  const a=app('reads'),held=server();
  let root=await view(a);
  button(root,'Read').click();
  assert.equal(held.requests.length,0);
  assert.match(text(root),/Enter a text\. Enter one or more neurons/);
  type(root,'reads text','abc');type(root,'reads neurons','L3H1, nonsense');
  button(root,'Read').click();
  assert.match(text(root),/L3H1 is not a neuron\. unknown target "nonsense"/);
  type(root,'reads neurons',Array.from({length:17},(_,i)=>`L0N${i}`).join(','));
  button(root,'Read').click();
  assert.match(text(root),/At most 16 neurons at once\./);
  assert.equal(held.requests.length,0);
  type(root,'reads neurons','L0N40');
  button(root,'Read').click();
  held.answer({error:'ValueError: neuron 40 does not exist (layer 0 has 40)'},false);
  await settle();
  root=await view(a);
  const alert=walk(root).find(n=>n.role==='alert');
  assert.match(text(alert),/neuron 40 does not exist/);
});

test('reads: the edited row can carry the Run panel steering',async()=>{
  const a=app('reads'),held=server();
  a.keep.steer={on:true,recipe:'',p50:'',entries:[{addr:'L1N3',mode:'add',dose:'0.75',unit:'resid',on:true}]};
  const root=await view(a);
  assert.match(text(root),/steer the edited row \(L1N3 \+0\.75\)/);
  type(root,'reads text','abc');type(root,'reads neurons','L1N3');
  labelled(root,'reads with steering').events.change({target:{checked:true}});
  button(root,'Read').click();
  assert.deepEqual(held.requests[0].body.steer,[{addr:'L1N3',mode:'add',dose:0.75,unit:'resid'}]);
  held.answer({addrs:['L1N3'],token_strs:['a'],steered:true,values:{current:[[2]],base:[[1]]}});
  await settle();
  assert.match(text(await view(a)),/L1N3 · edited, steered/);
});

test('fires: clickable tokens, raw-activation ranking, inspect and add to steering',async()=>{
  const a=app('fires'),held=server();
  let root=await view(a);
  type(root,'fires text','abcab');
  button(root,'Find').click();
  assert.deepEqual(held.requests[0].body,{prompt:'abcab',position:-1,k:20,sign:'both'});
  const answer={position:4,token:1,token_str:'b',token_strs:['a','b','c','a','b'],ranking:'raw activation at this position',
    neurons:[{layer:1,neuron:3,addr:'L1N3',value:2.5},{layer:0,neuron:7,addr:'L0N7',value:-1.25}]};
  held.answer(answer);await settle();
  root=await view(a);
  assert.match(text(root),/At token 4 "b", ranked by raw activation: a large value is not the same as a large effect on the output\./);
  const cells=walk(walk(root).filter(n=>n.tag==='tr')[1]).filter(n=>n.tag==='td').map(text);
  assert.deepEqual(cells.slice(0,4),['L1N3','1','3','2.5']);
  // Another token: the text that was sent, at that position (even if the box changed since).
  type(root,'fires text','something else');
  const toks=walk(root).find(n=>n.className==='toks');
  toks.children[2].click();
  assert.deepEqual(held.requests[1].body,{prompt:'abcab',position:2,k:20,sign:'both'});
  held.answer({...answer,position:2,token_str:'c'});await settle();
  root=await view(a);
  assert.match(text(root),/At token 2 "c"/);
  button(root,'inspect').click();
  assert.deepEqual(a.selected,[['blocks.1.mlp.neuron.3','Weights']]);
  button(root,'+ steer').click();
  assert.deepEqual(a.keep.steer.entries,[{addr:'L1N3',mode:'add',dose:'0.5',unit:'resid',on:true}]);
  assert.match(text(await view(a)),/L1N3 \+0\.5 added to the Run tab's steering panel\./);
});

test('fires: bounds are checked before sending',async()=>{
  const a=app('fires'),held=server();
  const root=await view(a);
  assert.match(text(root),/Up to 256 tokens; 1 to 200 neurons/);
  type(root,'fires text','abc');type(root,'fires count','201');
  button(root,'Find').click();
  assert.match(text(root),/Show between 1 and 200 neurons\./);
  type(root,'fires text','x'.repeat(20001));type(root,'fires count','5');
  button(root,'Find').click();
  assert.match(text(root),/longer than 20000 characters/);
  assert.equal(held.requests.length,0);
});

test('next token: the panel entries or one typed entry, KL, tops and movers',async()=>{
  const a=app('next token'),held=server();
  let root=await view(a);
  type(root,'next token prompt','abc');
  button(root,'Check the next token').click();
  assert.match(text(root),/steering panel has no active entries/);
  assert.equal(held.requests.length,0);
  a.keep.probe.next.source='typed';
  root=await view(a);
  type(root,'next token entry','L1 N3 +0.75, L2N1 +1');
  button(root,'Check the next token').click();
  assert.match(text(root),/Type one entry/);
  type(root,'next token entry','L11 N3+');
  button(root,'Check the next token').click();
  assert.match(text(root),/L11 N3\+: a sign without a dose/);
  type(root,'next token entry','L1 N3 +0.75');
  button(root,'Check the next token').click();
  assert.deepEqual(held.requests[0].body,{prompt:'abc',top:10,steer:[{addr:'L1N3',mode:'add',dose:0.75,unit:'resid'}]});
  onProbeSteeringEvent({type:'steering',workspace:'fixture',state:'calibrating',tokens:90},a);
  assert.match(text(root),/measuring the base model's residual norms: one forward pass over 90 tokens/);
  held.answer({kl:0.1234,top_unsteered:[{id:1,token:'a',p:0.5}],top_steered:[{id:2,token:'b',p:0.6}],
    rose:[{id:2,token:'b',p:0.1,q:0.6,delta_logp:1.79}],fell:[],calibration:{ran:true}});
  await settle();
  root=await view(a);
  assert.match(text(root),/steered: L1N3 \+0\.75\. KL\(unsteered ‖ steered\) = 0\.1234 nats\./);
  assert.match(text(root),/without.*"a".*50\.0%/);
  assert.match(text(root),/with the push.*"b".*60\.0%/);
  assert.match(text(root),/rose most.*"b".*10\.0%.*60\.0%.*1\.79/);
  assert.match(text(root),/fell mostnone/);
  // With the panel's entries (and its given p50).
  a.keep.steer={on:true,recipe:'',p50:'1: 2',entries:[{addr:'L0H1',mode:'mul',dose:'0',unit:'resid',on:true}]};
  a.keep.probe.next.source='panel';
  root=await view(a);
  assert.match(text(root),/the Run panel's entries \(L0H1 ×0\)/);
  button(root,'Check the next token').click();
  assert.deepEqual(held.requests[1].body,{prompt:'abc',top:10,steer:[{addr:'L0H1',mode:'mul',dose:0}],p50:{'1':2}});
  held.answer({kl:null,top_unsteered:[],top_steered:[],rose:[],fell:[]});await settle();
  assert.match(text(await view(a)),/is infinite/);
});

test('contrast: one text per line, both lists, rows like Fires, errors where clicked',async()=>{
  const a=app('contrast'),held=server();
  let root=await view(a);
  assert.match(text(root),/1 to 16 texts per set, each up to 256 tokens/);
  type(root,'contrast set A','aaab\n\nabab\n');
  button(root,'Contrast').click();
  assert.match(text(root),/Set B needs 1 to 16 texts, one per line\./);
  type(root,'contrast set B',Array.from({length:17},()=>'z').join('\n'));
  button(root,'Contrast').click();
  assert.match(text(root),/Set B needs 1 to 16 texts/);
  assert.equal(held.requests.length,0);
  type(root,'contrast set B','zzzy');
  button(root,'Contrast').click();
  assert.deepEqual(held.requests[0].body,{a:['aaab','abab'],b:['zzzy'],k:20});
  held.answer({a:2,b:1,measure:'mean over texts of max activation (position 0 excluded); A minus B',
    up:[{layer:1,neuron:3,addr:'L1N3',diff:2,a:2.5,b:0.5}],down:[]});
  await settle();
  root=await view(a);
  assert.match(text(root),/2 texts in A, 1 in B\. mean over texts of max activation/);
  assert.match(text(root),/fire on A, not B.*L1N3/);
  assert.match(text(root),/fire on B, not Anone/);
  button(root,'inspect').click();
  assert.deepEqual(a.selected,[['blocks.1.mlp.neuron.3','Weights']]);
  button(root,'Contrast').click();
  held.answer({error:'ProbeError: a text of set A has 300 tokens; at most 256'},false);
  await settle();
  root=await view(a);
  assert.match(text(walk(root).find(n=>n.role==='alert')),/at most 256/);
});

test('later row tools (save as selection) are shown under the Fires and Contrast tables',async()=>{
  const seen=[];
  rowTools.push((rows)=>{seen.push(rows.map(r=>r.addr));const b=new Element('button');b.textContent='save as selection';return b;});
  try {
    const a=app('fires'),held=server();
    let root=await view(a);
    type(root,'fires text','ab');button(root,'Find').click();
    held.answer({position:1,token:1,token_str:'b',token_strs:['a','b'],neurons:[{layer:0,neuron:1,addr:'L0N1',value:1}]});
    await settle();
    root=await view(a);
    assert.ok(button(root,'save as selection'));
    assert.deepEqual(seen.at(-1),['L0N1']);
  } finally { rowTools.length=0; }
});
