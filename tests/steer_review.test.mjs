// Model-free checks for the steering review fixes: the recipe parser's refusals (ranges, dashes, a sign apart
// from its number, a decimal comma), its length cap and linear time, the per-layer p50, the panel header, the
// labels on chat turns and run results, the Probe tab's "as a chat message" option, its bounds and visible
// escapes, and the surviving mutations FM1, FM2, FM7, FM8, FM9, FM12 and FM16 of the review.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRecipe, parseP50, parseTarget, entryLabel, headerText, steerPanel, MAX_RECIPE } from '../web/js/steer.js';
import { runView } from '../web/js/run.js';
import { probeView } from '../web/js/probe.js';
import { diverge, visible, generationStarted, generationSettled } from '../web/js/util.js';
const tick=()=>new Promise(r=>setImmediate(r));
const settle=async()=>{await tick();await tick();await tick();};
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
const runtime={generation:{default_output:16,output_limit:64,context_limit:128}};
const generation={requested_output:8,effective_output:8,prompt_tokens:2,context:128};
function app(tab='Run'){return {sel:'model',tab,variant:'current',keep:{},selected:[],state:{open:{name:'fixture',kind:'wsp'},status:{can_run:true,version:0}},rerender(){},select(){}};}
const entries=(...list)=>list.map(([addr,mode,dose,unit='resid',on=true])=>({addr,mode,dose:String(dose),unit,on}));
// GETs answer at once (the runtime, the chat-template check); the POST to `path` is held until the test answers it.
function server(path,{chat={available:false,reason:'this built-in model has no chat template'}}={}){
  const held={bodies:[],gets:[]};
  global.fetch=async(url,o={})=>{
    if(!o.body){held.gets.push(url);return response(url==='/api/probe/chat'?chat:runtime);}
    if(url==='/api/generation/stop')return response({ok:true,found:true});
    if(path&&url!==path)return response(runtime);
    held.body=JSON.parse(o.body);held.bodies.push({url,body:held.body});
    return new Promise(resolve=>{held.finish=(data,ok=true)=>resolve(response(data,ok));});
  };
  return held;
}
async function run(state){const root=new Element('root');await runView(root,state);return root;}
async function probe(state){const root=new Element('root');await probeView(root,state);return root;}
function probeApp(view){const a=app('Probe');a.keep.probe={view,reads:{text:'',neurons:''},fires:{text:'',k:20,sign:'both'},next:{prompt:'',source:'panel',typed:'',top:10},contrast:{a:'',b:'',k:20}};return a;}

// ---------------------------------------------------------------- the recipe parser: ranges and dashes are refused
const AMBIGUOUS=[
  ['L12 N5-7',/ambiguous: a range or a push\?/],          // a range in neuron-explorer notes
  ['L12 N5–7',/ambiguous: a range or a push\?/],          // an en dash
  ['L12 N5—7',/ambiguous: a range or a push\?/],          // an em dash
  ['L12 N5 - 7',/ambiguous: a range or a push\?.*sign stands apart/],
  ['L12 N5 + 7',/ambiguous: a range or a push\?.*sign stands apart/],
  ['L19 N7839−',/ambiguous: a range or a push\?/],        // a polarity note with a typographic minus
  ['L19 N7839-',/ambiguous: a range or a push\?/],
  ['L12 N5-',/ambiguous: a range or a push\?/],
  ['L5N9 –0.5',/ambiguous: a range or a push\?.*not read as a minus/],
  ['L5N9 —0.5',/ambiguous: a range or a push\?.*not read as a minus/],
  ['L5N9 ‐0.5',/ambiguous: a range or a push\?.*not read as a minus/],
  ['L5N9 : – 0.5',/ambiguous: a range or a push\?/],
  ['L5N9+0.25',/the dose must stand apart from the target/],
  ['L5N9 -',/a sign without a dose/],
  ['L5N9 +',/a sign without a dose/],
  ['L5N9 - 0.5 raw',/sign stands apart/],
];
for (const [input,reason] of AMBIGUOUS) test(`the recipe ${JSON.stringify(input)} is refused, never read as a push`,()=>{
  const r=parseRecipe(input);
  assert.deepEqual(r.entries,[]);
  assert.equal(r.problems.length,1);
  assert.equal(r.problems[0].item,input);
  assert.match(r.problems[0].reason,reason);
});

test('a dose apart from the target, with its sign against its number, is still read',()=>{
  assert.deepEqual(parseRecipe('L12N5 -7').entries,[{addr:'L12N5',mode:'add',dose:-7,unit:'resid'}]);
  assert.deepEqual(parseRecipe('L12 N5: -7').entries,[{addr:'L12N5',mode:'add',dose:-7,unit:'resid'}]);
  assert.deepEqual(parseRecipe('L5N9 −0.5').entries,[{addr:'L5N9',mode:'add',dose:-0.5,unit:'resid'}]);   // U+2212 is a minus
  assert.deepEqual(parseRecipe('L5N9 x −2').entries,[{addr:'L5N9',mode:'mul',dose:-2}]);
  assert.deepEqual(parseRecipe('L5N9×2').entries,[{addr:'L5N9',mode:'mul',dose:2}]);                    // a multiply mark may touch it
  assert.deepEqual(parseRecipe('blocks.3.mlp.neurons.1-4 -0.5').entries,[{addr:'blocks.3.mlp.neurons.1-4',mode:'add',dose:-0.5,unit:'resid'}]);
});

test('a decimal comma is reported and neither half is kept',()=>{
  for (const [input,item,reason] of [['L5N9 +0,75','L5N9 +0,75','decimal comma? write +0.75'],['L5N9 -0,75','L5N9 -0,75','decimal comma? write -0.75'],
    ['L5N9 x2,5','L5N9 x2,5','decimal comma? write 2.5']]) {
    assert.deepEqual(parseRecipe(input),{entries:[],problems:[{item,reason}]});
  }
  const r=parseRecipe('L5N9 -0,75, L6N1 +1; L7N2 x2,5');
  assert.deepEqual(r.entries,[{addr:'L6N1',mode:'add',dose:1,unit:'resid'}]);
  assert.deepEqual(r.problems.map(p=>p.reason),['decimal comma? write -0.75','decimal comma? write 2.5']);
  // A digit right after the comma: "0,75 raw" is one dose, not ×0 and a stray "75 raw".
  assert.deepEqual(parseRecipe('L5N9 x0,75 raw'),{entries:[],problems:[{item:'L5N9 x0,75 raw',reason:'decimal comma? write 0.75'}]});
  // A list separated by commas is still a list.
  assert.deepEqual(parseRecipe('L5N9 +1, L6N1 +2').entries.map(e=>e.dose),[1,2]);
});

test('no recipe parses to a sign or a mode other than the one written',()=>{
  const targets=['L12 N5','L5N9','L3H4','blocks.2.mlp','group:fox'];
  const joins=['',' ',':',': ','  ',' : ','\t'];
  const ops=['+','-','−','–','—','‐','―','x','×','*','X','add ','mul ','add:','+ ','- ','x ','',' +','+-','-+'];
  const nums=['7','0.5','.5','1e-3','0,75','2e+2'];
  const tails=['',' raw',' resid','-','+','-7','–3',' -1'];
  let read=0;
  for (const t of targets) for (const j of joins) for (const op of ops) for (const n of nums) for (const tail of tails) {
    const input=t+j+op+n+tail;
    const {entries:got,problems}=parseRecipe(input);
    assert.ok(got.length+problems.length>=1,input);
    if (!got.length) continue;
    read++;
    assert.equal(got.length,1,input);
    const e=got[0],o=op.trim().toLowerCase();
    if (e.addr!==parseTarget(t).addr) {   // the dose slot joined the target (L12 N57, group:fox-7): only the tail's own push may be read
      assert.ok(j===''&&tail===' -1'&&e.mode==='add'&&e.dose===-1,`${input}: ${JSON.stringify(e)}`);
      continue;
    }
    assert.notEqual(o,'',`${input}: a dose without a sign was read`);
    const mul=['x','×','*','mul'].includes(o);
    assert.equal(e.mode,mul?'mul':'add',input);
    if (!mul) {
      assert.ok(['+','-','−','add','add:'].includes(o),`${input}: read as add without a written sign`);
      assert.ok(/^[\s:]/.test(j+op),`${input}: a signed dose against the target was read`);
      assert.equal(Math.sign(e.dose)<0,o==='-'||o==='−',`${input}: the sign read is not the one written`);
    } else assert.ok(e.dose>=0,`${input}: a multiplier's sign was invented`);
    assert.ok(!/[‐‑‒–—―]/.test(input),`${input}: a dash was read`);
    assert.ok(!/,\d/.test(n),`${input}: a decimal comma was read`);
  }
  assert.ok(read>300,`the sweep read only ${read} inputs`);
});

test('a recipe is capped before it is parsed and parsing is linear',()=>{
  const big=parseRecipe(' '.repeat(50000));
  assert.deepEqual(big.entries,[]);
  assert.match(big.problems[0].reason,new RegExp(`at most ${MAX_RECIPE.toLocaleString('en-US')} characters; nothing was read`));
  const sp=' '.repeat(MAX_RECIPE-20);
  for (const input of ['L5N9 add'+sp+'x','L5N9 x'+sp+'q','L5N9 mul:'+sp+':','L1'+sp+'N','L5N9 +1'+sp+'raw q','- '+sp+'L1N2']) {
    assert.ok(input.length<=MAX_RECIPE);
    const t=performance.now();parseRecipe(input);const ms=performance.now()-t;
    assert.ok(ms<100,`${JSON.stringify(input.slice(0,12))}…: ${ms.toFixed(0)} ms`);   // quadratic took ~200-300 ms here
  }
  assert.ok(parseRecipe('L5N9 add'+sp+'x').problems[0].reason.length<120);         // a long dose is quoted short
});

test('the per-layer p50 refuses a layer given twice, as text or JSON',()=>{
  assert.equal(parseP50('12: 3, 12: 4').error,'layer 12 is given twice');
  assert.equal(parseP50('12: 3, L012: 4').error,'layer 12 is given twice');
  assert.equal(parseP50('{"12": 3, "L12": 4}').error,'layer 12 is given twice');
  assert.equal(parseP50('{"12": 3, "12": 4}').error,'layer 12 is given twice');   // JSON.parse alone keeps the last
  assert.deepEqual(parseP50('12: 3, 13: 4'),{map:{'12':3,'13':4}});
});

// ---------------------------------------------------------------- the panel header comes from the request
test('the panel header says not steered while an entry or the p50 needs fixing, and never shows a bad dose as +0',()=>{
  const st=(e,p50='',on=true)=>({on,recipe:'',p50,entries:e});
  assert.equal(headerText(st(entries(['L1N2','add','abc']))),'not steered (fix 1 entry)');
  assert.equal(headerText(st(entries(['L1N2','add','abc'],['L3H4','add','1'],['L2N1','add','0.5']))),'not steered (fix 2 entries)');
  assert.equal(headerText(st(entries(['L1N2','add','0.5']),'1: -2')),'not steered (fix the per-layer p50)');
  assert.equal(headerText(st(entries(['L1N2','add',''],['L2N2','add','1']),'1: 2, 1: 3')),'not steered (fix 1 entry and the per-layer p50)');
  assert.equal(headerText(st(entries(['L1N2','add','0.5'],['L2N2','mul','2','resid',false]))),'the edited model is steered: L1N2 +0.5');
  assert.equal(headerText(st(entries(['L1N2','add','0.5']),'',false)),'off: the edited model runs unsteered');
  assert.equal(entryLabel({addr:'L1N2',mode:'add',dose:'abc'}),'L1N2 (no valid dose)');
  assert.equal(entryLabel({addr:'L1N2',mode:'add',dose:''}),'L1N2 (no valid dose)');
  const a=app();a.keep.steer=st(entries(['L1N2','add','abc']));
  const root=steerPanel(a);
  assert.equal(text(labelled(root,'steering header')),'not steered (fix 1 entry)');
  assert.doesNotMatch(text(root),/\+0\b/);
});

// ---------------------------------------------------------------- Run and Chat labels
test('FM12: the live card shows only what is sent, not every entry of the panel',async()=>{
  const state=app(),held=server('/api/run');
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','0.75'],['L0H1','mul','0','resid',false])};
  button(await run(state),'Run  (Ctrl+Enter)').click();await tick();
  assert.deepEqual(held.body.steer,[{addr:'L1N2',mode:'add',dose:0.75,unit:'resid'}]);
  const live=text(await run(state));
  assert.match(live,/steered: L1N2 \+0\.75/);
  assert.doesNotMatch(live,/L0H1/);
  held.finish({prompt:'ab',join:'',tokens:['a'],generation,current:{text:'z',status:'complete',next:[]},base:{text:'z',status:'complete',next:[]}});await settle();
  // The panel off: nothing is steered, so the live card says nothing of steering.
  state.keep.steer.on=false;
  button(await run(state),'Run  (Ctrl+Enter)').click();await tick();
  assert.equal(held.body.steer,undefined);
  assert.doesNotMatch(text(await run(state)),/steered:/);
});

test('FM2, FM16: a run result is labelled from what the server reports, and unsteered results have the plain title',async()=>{
  const state=app(),held=server('/api/run');
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','0.75'])};
  button(await run(state),'Run  (Ctrl+Enter)').click();await tick();
  // The server reports what it applied (here a different dose than the panel's, as a changed panel would).
  held.finish({prompt:'ab',join:'',tokens:['a','b'],generation,steering:{variant:'current',resolved:true,entries:[{addr:'L1N2',mode:'add',dose:0.25,unit:'resid'}]},
    current:{text:'zz',status:'complete',next:[['z',0.9,3]]},base:{text:'cd',status:'complete',next:[['c',0.8,2]]}});
  await settle();
  // A live record still present when the result is drawn (a later request's) must not label it.
  state.keep.run.live={completion:{id:'later',steer:'L9N9 +5',current:{text:'',state:'waiting'},base:{text:'',state:'waiting'}}};
  let root=await run(state);
  const cards=()=>walk(root).filter(n=>n.className==='card'&&walk(n).some(c=>c.tag==='h3'));
  const edited=cards().find(c=>text(c).startsWith('edited'));
  assert.match(text(edited),/steered: L1N2 \+0\.25/);
  assert.doesNotMatch(text(edited),/L9N9|\+0\.75/);
  assert.doesNotMatch(text(root),/L9N9/);
  assert.match(text(edited),/next token \(steered\)/);
  // Unsteered: the edited card's title is the plain one.
  state.keep.run.live=null;state.keep.steer.on=false;
  button(root,'Run  (Ctrl+Enter)').click();await tick();
  held.finish({prompt:'ab',join:'',tokens:['a','b'],generation,current:{text:'zz',status:'complete',next:[['z',0.9,3]]},base:{text:'cd',status:'complete',next:[['c',0.8,2]]}});
  await settle();
  root=await run(state);
  assert.doesNotMatch(text(root),/next token \(steered\)/);
  assert.doesNotMatch(text(walk(root).find(n=>n.className==='split')),/unsteered|steered:/);
});

test('FM1: a chat turn records the steering the server reports, not the panel\'s',async()=>{
  const state=app(),held=server('/api/chat');
  state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','-0.5','raw'])};
  const root=await run(state);
  walk(root).find(n=>n.placeholder==='message').value='hi';button(root,'Send').click();await tick();
  held.finish({current:'yo',base:'hey',chat_template:false,generation,variants:{current:'complete',base:'complete'},
    steering:{variant:'current',resolved:true,entries:[{addr:'L1N2',mode:'add',dose:-0.25,unit:'raw'}]}});
  await settle();
  assert.equal(state.keep.run.chat[1].steer,'L1N2 -0.25 raw');
});

test('the base column answers its own conversation and no longer says it answers the edited one, in the log and live',async()=>{
  const state=app(),held=server('/api/chat');
  state.keep.run={mode:'chat',tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'},
    chat:[{role:'user',content:'hi'},{role:'assistant',content:'yo'},{role:'user',content:'more'},{role:'assistant',content:'ok'}],
    baseChat:[{role:'user',content:'hi'},{role:'assistant',content:'hey'},{role:'user',content:'more'},{role:'assistant',content:'fine'}]};
  let root=await run(state);
  const notes=()=>walk(root).filter(n=>n['aria-label']==='base history'||/answers the edited conversation/.test(n.textContent||''));
  assert.equal(notes().length,0);
  // What is sent: each model's own replies as the history it answers.
  walk(root).find(n=>n.placeholder==='message').value='again';button(root,'Send').click();await tick();
  assert.deepEqual(held.body.messages.map(m=>m.content),['hi','yo','more','ok','again']);
  assert.deepEqual(held.body.base_messages.map(m=>m.content),['hi','hey','more','fine','again']);
  root=await run(state);
  assert.equal(notes().length,0);                                   // nor on the live base box
  held.finish({current:'x',base:'y',chat_template:false,generation,variants:{current:'complete',base:'complete'}});await settle();
});

test('an interrupted chat reply carries its steered label; a stopped one says what the edited part ran with',async()=>{
  const state=app(),held=server('/api/chat');
  state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','0.5'])};
  const steering={variant:'current',resolved:true,entries:[{addr:'L1N2',mode:'add',dose:0.5,unit:'resid'}]};
  let root=await run(state);
  walk(root).find(n=>n.placeholder==='message').value='hi';button(root,'Send').click();await tick();
  held.finish({interrupted:true,event:{kind:'runtime_lost',message:'lost'},current:'Hal',base:'',variants:{current:'interrupted',base:'not_run'},chat_template:false,generation,steering});
  await settle();
  root=await run(state);
  const partial=walk(root).find(n=>n.className==='card'&&/Interrupted — partial output/.test(text(n)));
  assert.match(text(partial),/Edited: Hal.*steered: L1N2 \+0\.5.*Base: not run/);
  // Stopped while the edited reply ran: nothing is kept, and the notice says it was steered.
  walk(root).find(n=>n.placeholder==='message').value='hi';button(root,'Send').click();await tick();
  held.finish({stopped:true,current:'H',base:'',variants:{current:'stopped',base:'not_run'},chat_template:false,generation,steering});await settle();
  assert.match(state.keep.run.chatNotice,/Stopped\. The turn was not added; .*\(The edited reply was steered: L1N2 \+0\.5\.\)/);
  // Stopped during the column reads: nothing was generated, so no steered reply is claimed.
  root=await run(state);
  walk(root).find(n=>n.placeholder==='message').value='hi';button(root,'Send').click();await tick();
  held.finish({stopped:true,current:'',base:'',variants:{current:'not_run',base:'not_run'},chat_template:false,generation,steering:{...steering,resolved:false}});await settle();
  assert.equal(state.keep.run.chatNotice,'Stopped. The turn was not added; your message is back in the box.');
});

test('a steered run still calibrating and a stopped steered chat each start, then settle, the generation hooks',async()=>{
  // The inspector's busy state (app.js) listens on these: a steered request is "running" from the moment it is
  // sent, through the column reads, until its answer arrives however it ends.
  const seen=[];
  const started=(path,body)=>seen.push(['start',path,body.stream,!!body.steer]);
  const settled=(path,body)=>seen.push(['end',path,body.stream,!!body.steer]);
  generationStarted.add(started);generationSettled.add(settled);
  try{
    const state=app();let held=server('/api/run');
    state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','0.75'])};
    button(await run(state),'Run  (Ctrl+Enter)').click();await tick();
    const run1=held.body.stream;
    assert.ok(run1&&held.body.steer);
    // Sent and unanswered: the server is still reading column norms. Started, not settled.
    assert.deepEqual(seen,[['start','/api/run',run1,true]]);
    let root=await run(state);
    button(root,'Stop').click();await tick();
    assert.deepEqual(seen,[['start','/api/run',run1,true]]);          // the stop request itself is not a generation
    held.finish({stopped:true,prompt:'ab',join:'',tokens:['a','b'],generation,steering:{variant:'current',resolved:false,entries:[]},
      current:{text:'',status:'not_run'},base:{text:'',status:'not_run'}});
    await settle();
    assert.deepEqual(seen,[['start','/api/run',run1,true],['end','/api/run',run1,true]]);
    // A steered chat reply, stopped while it generates.
    held=server('/api/chat');
    state.keep.run={...state.keep.run,mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
    root=await run(state);
    walk(root).find(n=>n.placeholder==='message').value='hi';button(root,'Send').click();await tick();
    const chat1=held.body.stream;
    assert.ok(chat1&&chat1!==run1&&held.body.steer);
    assert.equal(seen.length,3);
    assert.deepEqual(seen[2],['start','/api/chat',chat1,true]);
    root=await run(state);
    button(root,'Stop').click();await tick();
    assert.equal(seen.length,3);
    held.finish({stopped:true,current:'H',base:'',variants:{current:'stopped',base:'not_run'},chat_template:false,generation,
      steering:{variant:'current',resolved:true,entries:[{addr:'L1N2',mode:'add',dose:0.75,unit:'resid'}]}});
    await settle();
    assert.deepEqual(seen.slice(2),[['start','/api/chat',chat1,true],['end','/api/chat',chat1,true]]);
    assert.match(state.keep.run.chatNotice,/^Stopped\./);
  }finally{generationStarted.delete(started);generationSettled.delete(settled);}
});

test('a range in the recipe gets a hint for writing the range; a lone dash-number gets both readings',()=>{
  // A number after the dash and a dose of its own: the person wrote a range. The hint does not suggest a push.
  for(const [recipe,range,dose] of [['L3 N10-20 +0.5','blocks.3.mlp.neurons.10-20','+0.5'],['L3N10–20 x2','blocks.3.mlp.neurons.10-20','x2'],
    ['L12 N5-7  -0.25   raw','blocks.12.mlp.neurons.5-7','-0.25 raw']]){
    const r=parseRecipe(recipe);
    assert.equal(r.entries.length,0,recipe);
    const why=r.problems[0].reason;
    assert.match(why,/^ambiguous: a range or a push\?/);
    assert.ok(why.endsWith(`write a range as ${range} ${dose}`),why);
    assert.doesNotMatch(why,/to push/);
    // and the form the hint gives is read as that range with that dose
    const fixed=parseRecipe(`${range} ${dose}`);
    assert.equal(fixed.problems.length,0,JSON.stringify(fixed));
    assert.equal(fixed.entries[0].addr,range);
  }
  // The dash-number is the only number: a push or a range, and the hint says how to write each.
  let why=parseRecipe('L3N10-0.5').problems[0].reason;
  assert.match(why,/to push, put a space before the dose: L3N10 -0\.5; for a range, write blocks\.3\.mlp\.neurons\.10-<last> and a dose/);
  why=parseRecipe('L3N10-20').problems[0].reason;
  assert.match(why,/to push, put a space before the dose: L3N10 -20; for a range, write blocks\.3\.mlp\.neurons\.10-20 and a dose/);
  // No number after the dash (a polarity note), or a head: no range is suggested.
  for(const recipe of ['L19 N7839−','L12 N5-','L11H3-2'])assert.doesNotMatch(parseRecipe(recipe).problems[0].reason,/for a range|blocks\./,recipe);
});

test('F8: a stop between the column reads and the edited reply never says the reply was steered, in Chat or Run',async()=>{
  const steering={variant:'current',resolved:true,entries:[{addr:'L1N2',mode:'add',dose:0.5,unit:'resid'}]};
  // Chat: the reads finished (resolved) but the edited reply never started.
  const state=app();let held=server('/api/chat');
  state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','0.5'])};
  let root=await run(state);
  walk(root).find(n=>n.placeholder==='message').value='hi';button(root,'Send').click();await tick();
  held.finish({stopped:true,current:'',base:'',variants:{current:'not_run',base:'not_run'},chat_template:false,generation,steering});await settle();
  assert.equal(state.keep.run.chatNotice,'Stopped. The turn was not added; your message is back in the box.');
  // Outside the panel's own header (what the panel would send next), nothing says the reply was steered.
  const outside=(root)=>text(root).replace(text(walk(root).find(n=>n['aria-label']==='steering header')||{}),'');
  root=await run(state);
  assert.match(text(walk(root).find(n=>n['aria-label']==='steering header')),/steered: L1N2/);
  assert.doesNotMatch(outside(root),/steered: L1N2|was steered/);
  // Without a variants report at all, nothing is claimed either.
  walk(root).find(n=>n.placeholder==='message').value='hi';button(root,'Send').click();await tick();
  held.finish({stopped:true,current:'',base:'',chat_template:false,generation,steering});await settle();
  assert.equal(state.keep.run.chatNotice,'Stopped. The turn was not added; your message is back in the box.');
  // Run: the same stop, and a stop mid-reply; the notice keeps nothing and claims no steering.
  for(const current of [{text:'',status:'not_run'},{text:'H',status:'stopped'}]){
    const s2=app();held=server('/api/run');
    s2.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','0.5'])};
    button(await run(s2),'Run  (Ctrl+Enter)').click();await tick();
    held.finish({stopped:true,prompt:'ab',join:'',tokens:['a','b'],generation,steering,current,base:{text:'',status:'not_run'}});await settle();
    root=await run(s2);
    const notice=walk(root).find(n=>n.role==='status'&&/Run stopped/.test(text(n)));
    assert.equal(text(notice),'Run stopped; nothing was kept. Edit the prompt and run again.');
    assert.ok(!walk(root).some(n=>n.className==='card'&&/^edited/.test(text(n))));
    assert.doesNotMatch(outside(root),/steered: L1N2|next token \(steered\)/);
  }
});

test('token chips and next-token labels write out control and bidirectional characters',async()=>{
  assert.equal(visible('a\u202Eb\u0001c\u200Bd é→'),'a\\u{202E}b\\u{0001}c\\u{200B}d é→');
  assert.equal(visible('plain ⏎·'),'plain ⏎·');
  const state=app(),held=server('/api/run');
  button(await run(state),'Run  (Ctrl+Enter)').click();await tick();
  held.finish({prompt:'ab',join:'',tokens:['a','\u202Eevil'],generation,current:{text:'z',status:'complete',next:[['\u202Ex',0.9,3]]},base:{text:'z',status:'complete',next:[]}});
  await settle();
  const root=await run(state);
  assert.ok(walk(root).some(n=>n.className==='tok'&&text(n)==='\\u{202E}evil'));
  assert.ok(!text(root).includes('\u202E'));
});

// ---------------------------------------------------------------- Probe
test('FM7, FM8: reads put the edited values on the edited row, shaded red above 0 and blue below',async()=>{
  const a=probeApp('reads'),held=server('/api/probe/reads');
  let root=await probe(a);
  type(root,'reads text','abc');type(root,'reads neurons','L1N3');
  button(root,'Read').click();await tick();
  held.finish({addrs:['L1N3'],token_strs:['a','b','c'],steered:false,values:{current:[[2,-4,0]],base:[[1,1,1]]}});await settle();
  root=await probe(a);
  const rows=walk(root).filter(n=>n.className==='toks');
  const label=(row)=>text(walk(root).find(n=>n.children?.includes(row)).children[0]);
  assert.equal(label(rows[0]),'L1N3 · edited');
  assert.equal(label(rows[1]),'L1N3 · base');
  assert.deepEqual(rows[0].children.map(c=>c.title.split(': ').pop()),['2','-4','0']);
  assert.deepEqual(rows[1].children.map(c=>c.title.split(': ').pop()),['1','1','1']);
  assert.equal(rows[0].children[0].style.background,diverge(0.5));
  assert.equal(rows[0].children[1].style.background,diverge(-1));
  assert.notEqual(diverge(0.5),diverge(-0.5));
});

test('FM9: Fires outlines the token it ranked, not the first',async()=>{
  const a=probeApp('fires'),held=server('/api/probe/fires');
  let root=await probe(a);
  type(root,'fires text','abc');button(root,'Find').click();await tick();
  held.finish({position:2,token:2,token_str:'c',token_strs:['a','b','c'],neurons:[{layer:1,neuron:3,addr:'L1N3',value:2}]});await settle();
  root=await probe(a);
  const chips=walk(root).find(n=>n.className==='toks').children;
  assert.deepEqual(chips.map(c=>!!c.style.outline),[false,false,true]);
  assert.deepEqual(chips.map(c=>c['aria-current']),[undefined,undefined,'true']);
});

test('as a chat message: offered only with a chat template, sent with Chat\'s thinking choice, template tokens marked',async()=>{
  // Without a template the box is disabled and says why.
  let a=probeApp('fires');server('/api/probe/fires');
  let root=await probe(a);await settle();root=await probe(a);
  const box=labelled(root,'fires as a chat message');
  assert.equal(box.disabled,true);
  assert.match(text(root),/as a chat message \(unavailable: this built-in model has no chat template\)/);
  // With one: ticked, the request carries chat and the Run tab's thinking choice.
  a=probeApp('fires');a.keep.run={thinking:'off'};
  const held=server('/api/probe/fires',{chat:{available:true,reason:"the model's own chat template"}});
  root=await probe(a);await settle();root=await probe(a);
  assert.equal(labelled(root,'fires as a chat message').disabled,false);
  labelled(root,'fires as a chat message').events.change({target:{checked:true}});
  root=await probe(a);
  assert.match(text(labelled(root,'fires text bound')),/up to 256 tokens, counted by the server \(as a chat message the template's tokens count too\)/);
  type(root,'fires text','hi');button(root,'Find').click();await tick();
  assert.deepEqual(held.body,{prompt:'hi',position:-1,k:20,sign:'both',chat:true,chat_options:{enable_thinking:false}});
  held.finish({position:3,token:9,token_str:'⏎',token_strs:['<|user|>','hi','<|end|>','<|assistant|>'],chat:{template:[true,false,true,true]},
    neurons:[{layer:0,neuron:1,addr:'L0N1',value:1}]});await settle();
  root=await probe(a);
  assert.match(text(labelled(root,'chat rendering')),/dashed border are the chat template's/);
  const chips=walk(root).find(n=>n.className==='toks').children;
  assert.deepEqual(chips.map(c=>c.style.border||''),['1px dashed currentColor','','1px dashed currentColor','1px dashed currentColor']);
  assert.match(chips[0].title,/^chat template token 0/);
  // Clicking a token asks about the same rendering, even after the box was unticked.
  labelled(root,'fires as a chat message').events.change({target:{checked:false}});
  root=await probe(a);
  walk(root).find(n=>n.className==='toks').children[1].click();await tick();
  assert.deepEqual(held.body,{prompt:'hi',position:1,k:20,sign:'both',chat:true,chat_options:{enable_thinking:false}});
  held.finish({position:1,token:3,token_str:'hi',token_strs:['<|user|>','hi'],chat:{template:null},neurons:[]});await settle();
  assert.match(text(labelled(await probe(a),'chat rendering')),/could not be told apart from your text/);
});

test('as a chat message on Reads, Next token and Contrast sends the same fields',async()=>{
  for (const [view,fill,go,path] of [
    ['reads',(r)=>{type(r,'reads text','hi');type(r,'reads neurons','L0N1');},'Read','/api/probe/reads'],
    ['next token',(r)=>{type(r,'next token prompt','hi');},'Check the next token','/api/probe/next'],
    ['contrast',(r)=>{type(r,'contrast set A','hi');type(r,'contrast set B','yo');},'Contrast','/api/probe/contrast'],
  ]) {
    const a=probeApp(view);
    a.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L0N1','add','0.5'])};
    const held=server(path,{chat:{available:true,reason:'ok'}});
    let root=await probe(a);await settle();root=await probe(a);
    labelled(root,`${view} as a chat message`).events.change({target:{checked:true}});
    root=await probe(a);fill(root);button(root,go).click();await tick();
    assert.equal(held.body.chat,true,view);
    assert.deepEqual(held.body.chat_options,{},view);                   // the model's default thinking
  }
});

test('the typed next-token entry is refused while the panel\'s per-layer p50 is wrong, and says so',async()=>{
  const a=probeApp('next token'),held=server('/api/probe/next');
  a.keep.steer={on:false,recipe:'',p50:'12: 3, 12: 4',entries:[]};
  a.keep.probe.next.source='typed';
  let root=await probe(a);
  type(root,'next token prompt','abc');type(root,'next token entry','L1N3 +0.75');
  button(root,'Check the next token').click();await tick();
  assert.equal(held.bodies.length,0);
  assert.match(text(root),/Steering: per-layer p50 \(the Run tab's panel\): layer 12 is given twice/);
});

test('each probe input shows its bound; characters and counts are checked here, tokens by the server',async()=>{
  server();
  let a=probeApp('reads'),root=await probe(a);
  assert.match(text(labelled(root,'reads text bound')),/^0 of 20,000 characters · up to 256 tokens, counted by the server$/);
  type(root,'reads neurons','L1N3, L2N4');
  assert.match(labelled(root,'reads neurons bound').textContent,/^2 of 16 neurons/);   // the fake DOM keeps the old child
  a=probeApp('fires');root=await probe(a);
  assert.match(text(root),/1 to 200/);
  a=probeApp('next token');root=await probe(a);
  assert.match(text(labelled(root,'next token prompt bound')),/up to 512 tokens, counted by the server/);
  assert.match(text(root),/1 to 50/);
  a=probeApp('contrast');root=await probe(a);
  type(root,'contrast set A','one\ntwo\n\nthree');
  assert.equal(labelled(root,'contrast set A bound').textContent,'3 of 16 texts');
  assert.match(text(root),/Each text: up to 20,000 characters \(checked here\) · up to 256 tokens, counted by the server/);
});

test('probe chips and tables write out control and bidirectional characters',async()=>{
  const a=probeApp('fires'),held=server('/api/probe/fires');
  let root=await probe(a);
  type(root,'fires text','abc');button(root,'Find').click();await tick();
  held.finish({position:1,token:2,token_str:'\u202E',token_strs:['a','\u202E'],neurons:[]});await settle();
  root=await probe(a);
  assert.deepEqual(walk(root).find(n=>n.className==='toks').children.map(text),['a','\\u{202E}']);
  assert.match(text(root),/At token 1 "\\u\{202E\}"/);
  const b=probeApp('next token');b.keep.probe.next.source='typed';
  const held2=server('/api/probe/next');
  root=await probe(b);
  type(root,'next token prompt','abc');type(root,'next token entry','L1N3 +0.75');
  button(root,'Check the next token').click();await tick();
  held2.finish({kl:0.1,top_unsteered:[{id:1,token:'\u2066x',p:0.5}],top_steered:[],rose:[],fell:[]});await settle();
  root=await probe(b);
  assert.match(text(root),/"\\u\{2066\}x"/);
  assert.ok(!text(root).includes('\u2066'));
});
