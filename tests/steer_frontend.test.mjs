// Model-free steering UI: the recipe parser, panel state and storage, request bodies, and the
// steered labels on live cards, results and chat turns.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRecipe, parseP50, parseTarget, entryLabel, steerLabel, steerRequest, loadSteer, saveSteer, steerState,
  steerPanel, onSteeringEvent, addSteerEntry, panelPickers } from '../web/js/steer.js';
import { runView, onGenerationEvent } from '../web/js/run.js';
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
global.document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text})};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const button=(root,label)=>walk(root).find(n=>n.tag==='button'&&text(n)===label);
const labelled=(root,label)=>walk(root).find(n=>n['aria-label']===label);
const response=(data,ok=true)=>({ok,statusText:'Bad Request',headers:{get:()=> 'application/json'},json:async()=>data});
const runtime={generation:{default_output:16,output_limit:64,context_limit:128}};
const generation={requested_output:8,effective_output:8,prompt_tokens:2,context:128};
function memoryStorage(){const m=new Map();return {m,getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,String(v)),removeItem:k=>m.delete(k)};}
function app(name='fixture'){return {sel:'model',tab:'Run',variant:'current',keep:{},state:{open:{name,kind:'wsp'},status:{can_run:true,version:0}},rerender(){},select(){}};}
function server(path){
  const held={bodies:[],calls:[]};
  global.fetch=async(url,o)=>{
    held.calls.push(url);
    if(url==='/api/generation/stop')return response({ok:true,found:true});
    if(url!==path)return response(runtime);
    held.body=JSON.parse(o.body);held.bodies.push(held.body);
    return new Promise(resolve=>{held.finish=(data,ok=true)=>resolve(response(data,ok));});
  };
  return held;
}
async function view(state){const root=new Element('root');await runView(root,state);return root;}
const entries=(...list)=>list.map(([addr,mode,dose,unit='resid',on=true])=>({addr,mode,dose:String(dose),unit,on}));

// ---------------------------------------------------------------- the recipe parser
const READ=[
  ['L34 N3838 +0.75',[{addr:'L34N3838',mode:'add',dose:0.75,unit:'resid'}]],
  ['L10 N7757 +1.3, L34 N3838 -0.5',[{addr:'L10N7757',mode:'add',dose:1.3,unit:'resid'},{addr:'L34N3838',mode:'add',dose:-0.5,unit:'resid'}]],
  ['L34N3838 +0.75',[{addr:'L34N3838',mode:'add',dose:0.75,unit:'resid'}]],
  ['L24 h30 ×2',[{addr:'L24H30',mode:'mul',dose:2}]],
  ['L11H3 x2; L11H3 *0',[{addr:'L11H3',mode:'mul',dose:2},{addr:'L11H3',mode:'mul',dose:0}]],
  ['L5N9 x-1',[{addr:'L5N9',mode:'mul',dose:-1}]],
  ['L5N9×2',[{addr:'L5N9',mode:'mul',dose:2}]],
  ['blocks.12.mlp x0.5',[{addr:'blocks.12.mlp',mode:'mul',dose:0.5}]],
  ['L12 mlp ×0',[{addr:'blocks.12.mlp',mode:'mul',dose:0}]],
  ['blocks.3.attn.head.4 x3',[{addr:'L3H4',mode:'mul',dose:3}]],
  ['blocks.3.mlp.neuron.7 +1',[{addr:'L3N7',mode:'add',dose:1,unit:'resid'}]],
  ['blocks.3.mlp.neurons.1-4 -0.5',[{addr:'blocks.3.mlp.neurons.1-4',mode:'add',dose:-0.5,unit:'resid'}]],
  ['L5N9 mul 2',[{addr:'L5N9',mode:'mul',dose:2}]],
  ['L5N9 add 0.4',[{addr:'L5N9',mode:'add',dose:0.4,unit:'resid'}]],
  ['L5N9 +3 raw',[{addr:'L5N9',mode:'add',dose:3,unit:'raw'}]],
  ['L5N9 -3 RAW',[{addr:'L5N9',mode:'add',dose:-3,unit:'raw'}]],
  ['L5N9: +1',[{addr:'L5N9',mode:'add',dose:1,unit:'resid'}]],
  ['L5N9 −0.5',[{addr:'L5N9',mode:'add',dose:-0.5,unit:'resid'}]],
  ['`L5N9 +1`',[{addr:'L5N9',mode:'add',dose:1,unit:'resid'}]],
  ['- L5N9 +1\n* L6N1 x2\n2. L7N3 -1',[{addr:'L5N9',mode:'add',dose:1,unit:'resid'},{addr:'L6N1',mode:'mul',dose:2},{addr:'L7N3',mode:'add',dose:-1,unit:'resid'}]],
  ['group:fox-1 -0.2',[{addr:'group:fox-1',mode:'add',dose:-0.2,unit:'resid'}]],
  ['l2 n5 +1e-3',[{addr:'L2N5',mode:'add',dose:0.001,unit:'resid'}]],
  ['# the recipe from the notes\nL1N2 +1\n\n  \nL3H4 x0',[{addr:'L1N2',mode:'add',dose:1,unit:'resid'},{addr:'L3H4',mode:'mul',dose:0}]],
  ['',[]],
];
for (const [input,expected] of READ) test(`the recipe ${JSON.stringify(input)} is read`,()=>{
  assert.deepEqual(parseRecipe(input),{entries:expected,problems:[]});
});

const JUNK=[
  ['L11 N6261+',/sign without a dose/],
  ['L5N9 0.75',/"0\.75" has no sign: write \+0\.75 to add or x0\.75 to multiply/],
  ['L11H3 +1',/add pushes neurons only; multiply a head/],
  ['blocks.12.mlp +1',/add pushes neurons only; multiply a whole MLP/],
  ['L5N9 x2 raw',/raw applies to add only/],
  ['L5N9 +2e6',/larger than/],
  ['hello world',/unknown target/],
  ['L5N9',/no dose/],
  ['L5N9 +abc',/could not read the dose "\+abc"/],
  ['L5N9abc +1',/unknown target/],
  ['L5N9 x',/no multiplier/],
  ['N3838 +1',/unknown target/],
];
for (const [input,reason] of JUNK) test(`the recipe ${JSON.stringify(input)} is reported, not read`,()=>{
  const r=parseRecipe(input);
  assert.deepEqual(r.entries,[]);
  assert.equal(r.problems.length,1);
  assert.equal(r.problems[0].item,input);
  assert.match(r.problems[0].reason,reason);
});

test('a recipe with junk keeps every entry it could read and reports the rest',()=>{
  const r=parseRecipe('L1N2 +1, garbage, L3H4 x0\nL11 N6261+; L4N4 -0.5');
  assert.deepEqual(r.entries,[{addr:'L1N2',mode:'add',dose:1,unit:'resid'},{addr:'L3H4',mode:'mul',dose:0},{addr:'L4N4',mode:'add',dose:-0.5,unit:'resid'}]);
  assert.deepEqual(r.problems.map(p=>p.item),['garbage','L11 N6261+']);
});

test('targets are written one way, and unknown ones say what to write',()=>{
  assert.deepEqual(parseTarget(' l3 n17 '),{addr:'L3N17',kind:'neuron'});
  assert.deepEqual(parseTarget('L3H5'),{addr:'L3H5',kind:'head'});
  assert.deepEqual(parseTarget('blocks.2.mlp'),{addr:'blocks.2.mlp',kind:'mlp'});
  assert.deepEqual(parseTarget('group:circuit'),{addr:'group:circuit',kind:'group'});
  assert.match(parseTarget('embed').error,/unknown target "embed": write a neuron L34N3838/);
  assert.match(parseTarget('').error,/no target/);
});

test('the per-layer p50 field takes "layer: value" pairs or JSON and refuses anything else',()=>{
  assert.deepEqual(parseP50(''),{map:null});
  assert.deepEqual(parseP50('12: 3.5, L13=4'),{map:{'12':3.5,'13':4}});
  assert.deepEqual(parseP50('{"12": 3.5}'),{map:{'12':3.5}});
  assert.match(parseP50('12: -1').error,/above 0/);
  assert.match(parseP50('twelve: 3').error,/could not read/);
  assert.match(parseP50('{bad').error,/not valid JSON/);
});

test('labels say what was pushed, as the recipe would',()=>{
  assert.equal(entryLabel({addr:'L34N3838',mode:'add',dose:0.75,unit:'resid'}),'L34N3838 +0.75');
  assert.equal(entryLabel({addr:'L34N3838',mode:'add',dose:-1.5,unit:'raw'}),'L34N3838 -1.5 raw');
  assert.equal(entryLabel({addr:'L11H3',mode:'mul',dose:2}),'L11H3 ×2');
  assert.equal(steerLabel([{addr:'L1N2',mode:'add',dose:1,unit:'resid'},{addr:'blocks.3.mlp',mode:'mul',dose:0}]),'L1N2 +1, blocks.3.mlp ×0');
  // A label parses back to the same entries.
  const spec=[{addr:'L1N2',mode:'add',dose:1,unit:'resid'},{addr:'L3N4',mode:'add',dose:-2,unit:'raw'},{addr:'L5H1',mode:'mul',dose:0.5}];
  assert.deepEqual(parseRecipe(steerLabel(spec)).entries,spec);
});

// ---------------------------------------------------------------- panel state and request fields
test('only switched-on entries of a switched-on panel are sent, with a given p50',()=>{
  const st={on:true,p50:'',entries:entries(['L1N2','add','0.5'],['L3H4','mul','2','resid',false],['l5 n6','add','1','raw'])};
  assert.deepEqual(steerRequest(st),{fields:{steer:[{addr:'L1N2',mode:'add',dose:0.5,unit:'resid'},{addr:'L5N6',mode:'add',dose:1,unit:'raw'}]},label:'L1N2 +0.5, L5N6 +1 raw',problems:[]});
  st.p50='1: 3.5';
  assert.deepEqual(steerRequest(st).fields.p50,{'1':3.5});
  st.on=false;
  assert.deepEqual(steerRequest(st),{fields:null,label:'',problems:[]});
  st.on=true;st.entries=entries(['L3H4','add','1'],['L1N2','add','lots'],['nope','mul','1']);
  const r=steerRequest(st);
  assert.equal(r.fields,null);
  assert.equal(r.problems.length,3);
  assert.match(r.problems[0],/^L3H4: add pushes neurons only/);
  assert.match(r.problems[1],/^L1N2: the dose must be a number/);
  assert.match(r.problems[2],/unknown target "nope"/);
});

test('panel state is kept per workspace in browser storage and the page works without it',()=>{
  const storage=memoryStorage();global.window={localStorage:storage};
  try {
    const a=app('alpha');
    steerState(a).entries.push(...entries(['L1N2','add','0.75']));
    assert.ok(saveSteer('alpha',steerState(a)));
    assert.deepEqual(loadSteer('alpha').entries,entries(['L1N2','add','0.75']));
    assert.equal(loadSteer('beta'),null);
    assert.deepEqual(steerState(app('alpha')).entries,entries(['L1N2','add','0.75']));   // a reload
    assert.deepEqual(steerState(app('beta')).entries,[]);
    addSteerEntry(a,'L3H4');                                                             // heads are multiplied
    assert.deepEqual(loadSteer('alpha').entries.at(-1),{addr:'L3H4',mode:'mul',dose:'0.5',unit:'resid',on:true});
    storage.setItem('wsp.steer.v1.gamma','{not json');
    assert.equal(loadSteer('gamma'),null);
    // Storage that throws (blocked site data): nothing is lost from the page, nothing breaks.
    global.window={get localStorage(){throw new Error('denied');}};
    assert.equal(loadSteer('alpha'),null);
    assert.equal(saveSteer('alpha',steerState(a)),false);
    const root=steerPanel(a);
    assert.ok(button(root,'Check'));
    global.window={localStorage:{getItem(){throw new Error('quota');},setItem(){throw new Error('quota');}}};
    assert.equal(loadSteer('alpha'),null);
    assert.equal(saveSteer('alpha',steerState(a)),false);
  } finally { delete global.window; }
});

test('the panel disables what the server refuses and explains the rest',()=>{
  const a=app();a.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L3H4','mul','2'],['L1N2','add','x'],['blocks.2.mlp','add','1'])};
  const root=steerPanel(a);
  const options=(label)=>labelled(root,label).children;
  assert.equal(options('steer entry 1 mode')[0].disabled,true);                 // add on a head
  assert.equal(labelled(root,'steer entry 1 unit').disabled,true);              // a multiplier has no unit
  assert.equal(options('steer entry 2 mode')[0].disabled,false);
  assert.match(text(root),/the dose must be a number/);
  assert.match(text(root),/add pushes neurons only; multiply a whole MLP instead/);
  // Typing a head as the target switches the entry to multiply.
  const target=labelled(root,'steer entry 2 target');
  target.events.input({target:{value:'L7 h1'}});target.events.change({});
  assert.deepEqual(a.keep.steer.entries[1],{addr:'L7H1',mode:'mul',dose:'x',unit:'resid',on:true});
  // Remove, toggle and the whole-panel switch.
  walk(root).find(n=>n['aria-label']==='remove steer entry 3').click();
  assert.equal(a.keep.steer.entries.length,2);
  labelled(root,'steer entry 1 on').events.change({target:{checked:false}});
  assert.equal(a.keep.steer.entries[0].on,false);
  labelled(root,'steering on').events.change({target:{checked:false}});
  assert.equal(a.keep.steer.on,false);
  assert.match(text(root),/off: the edited model runs unsteered/);
});

test('pasting a recipe adds what it could read and lists what it could not',()=>{
  const a=app(),root=steerPanel(a);
  labelled(root,'steering recipe').events.input({target:{value:'L10 N7757 +1.3, L34 N3838 -0.5\nL11 N6261+\nL24 h30 ×2'}});
  button(root,'add from recipe').click();
  assert.deepEqual(a.keep.steer.entries,entries(['L10N7757','add','1.3'],['L34N3838','add','-0.5'],['L24H30','mul','2']));
  assert.match(text(root),/3 entries added\./);
  assert.match(text(root),/1 not read: L11 N6261\+ — a sign without a dose/);
  assert.match(text(root),/the edited model is steered: L10N7757 \+1\.3, L34N3838 -0\.5, L24H30 ×2/);
});

test('Check shows each constant, column length and p50, and offers to measure when calibration is due',async()=>{
  const bodies=[];let answer;
  global.fetch=async(url,o)=>{bodies.push([url,JSON.parse(o.body)]);return response(answer());};
  const a=app();a.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N3','add','0.75'],['L0H1','mul','0'])};
  const root=steerPanel(a);
  answer=()=>({entries:[{addr:'L1N3',mode:'add',dose:0.75,unit:'resid',kind:'neurons',p50:{'1':null},norm:0.5,value:null,note:'not calibrated yet'},
    {addr:'L0H1',mode:'mul',dose:0,kind:'head',value:0}],calibration:{needed:true,cached:false,ran:false,tokens:90},reads:{reads:1}});
  button(root,'Check').click();await tick();await tick();
  assert.deepEqual(bodies[0],['/api/steer/resolve',{steer:[{addr:'L1N3',mode:'add',dose:0.75,unit:'resid'},{addr:'L0H1',mode:'mul',dose:0}],calibrate:false}]);
  assert.match(text(root),/Calibration will run before the first steered reply: one forward pass of the base model over a reference text, then cached/);
  assert.match(text(root),/not measured yet/);
  answer=()=>({entries:[{addr:'L1N3',mode:'add',dose:0.75,unit:'resid',kind:'neurons',p50:{'1':4},norm:0.5,value:6,source:'measured'},
    {addr:'L0H1',mode:'mul',dose:0,kind:'head',value:0}],calibration:{needed:true,cached:false,ran:true,tokens:90},reads:{reads:1}});
  button(root,'measure now').click();
  assert.match(text(root),/checking… \(waits for a running generation to finish\)/);
  onSteeringEvent({type:'steering',workspace:'fixture',state:'calibrating',tokens:90},a);
  assert.match(text(root),/measuring the base model's residual norms: one forward pass over 90 tokens/);
  await tick();await tick();
  assert.equal(bodies[1][1].calibrate,true);
  const row=walk(root).filter(n=>n.tag==='tr')[1];
  assert.deepEqual(row.children.map(text),['L1N3 +0.75','6','0.5','4','measured']);
  assert.match(text(root),/Measured now on the base model \(90 tokens\); cached\./);
  // Given p50 goes with the check; an error appears where the button is.
  a.keep.steer.p50='1: 3';
  answer=()=>({error:'SteeringError: steer entry 1 (L1N3): layer 1 does not exist'});
  global.fetch=async(url,o)=>{bodies.push([url,JSON.parse(o.body)]);return response(answer(),false);};
  button(root,'Check').click();await tick();await tick();
  assert.deepEqual(bodies[2][1].p50,{'1':3});
  assert.match(text(root),/layer 1 does not exist/);
});

test('a picker registered for the panel is shown and can add entries',()=>{
  const a=app();
  panelPickers.push((app,add)=>{const b=new Element('button');b.textContent='from selection';b.addEventListener('click',()=>add('L2N2'));return b;});
  try {
    const root=steerPanel(a);
    button(root,'from selection').click();
    assert.deepEqual(a.keep.steer.entries,entries(['L2N2','add','0.5']));
  } finally { panelPickers.length=0; }
});

// ---------------------------------------------------------------- Run and Chat
test('a completion without steering sends exactly the old body',async()=>{
  const state=app(),held=server('/api/run');
  state.keep.steer={on:false,recipe:'',p50:'',entries:entries(['L1N2','add','0.5'])};
  const root=await view(state);
  button(root,'Run  (Ctrl+Enter)').click();await tick();
  assert.deepEqual(Object.keys(held.body).sort(),['max_new_tokens','prompt','seed','stream','temperature','top_k']);
});

test('a steered completion says so while it runs and on its result; the base column says unsteered',async()=>{
  const state=app(),held=server('/api/run');
  state.keep.steer={on:true,recipe:'',p50:'1: 3.5',entries:entries(['L1N2','add','0.75'],['L0H1','mul','0'])};
  let root=await view(state);
  button(root,'Run  (Ctrl+Enter)').click();await tick();
  assert.deepEqual(held.body.steer,[{addr:'L1N2',mode:'add',dose:0.75,unit:'resid'},{addr:'L0H1',mode:'mul',dose:0}]);
  assert.deepEqual(held.body.p50,{'1':3.5});
  const id=held.body.stream;
  root=await view(state);
  assert.match(text(root),/steered: L1N2 \+0\.75, L0H1 ×0/);
  onSteeringEvent({type:'steering',workspace:'fixture',stream:id,state:'reading',done:1,total:4},state);
  root=await view(state);
  assert.match(text(root),/reading output weights: 1 of 4 neurons…/);
  onSteeringEvent({type:'steering',workspace:'fixture',stream:'another',state:'calibrating',tokens:9},state);   // another request's
  assert.doesNotMatch(text(await view(state)),/measuring the base model/);
  assert.match(text(await view(state)),/reading output weights: 1 of 4/);
  onGenerationEvent({type:'generation',stream:id,variant:'current',state:'started',keep:0,text:'xy',tokens:2},state);
  assert.match(text(await view(state)),/generating…/);
  const steering={variant:'current',resolved:true,entries:[{addr:'L1N2',mode:'add',dose:0.75,unit:'resid',value:5.25,norm:0.5,p50:{'1':3.5},source:'given'},
    {addr:'L0H1',mode:'mul',dose:0,value:0}],calibration:{needed:false,given:[1]}};
  held.finish({prompt:'ab',join:'',tokens:['a','b'],generation,steering,current:{text:'zz',status:'complete',steered:true,next:[['z',0.9,3]]},base:{text:'cd',status:'complete',next:[['c',0.8,2]]}});
  await tick();await tick();
  root=await view(state);
  const cards=walk(root).filter(n=>n.className==='card'&&walk(n).some(c=>c.tag==='h3'));
  const base=cards.find(c=>text(c).startsWith('base')),edited=cards.find(c=>text(c).startsWith('edited'));
  assert.match(text(edited),/steered: L1N2 \+0\.75, L0H1 ×0/);
  assert.match(text(edited),/next token \(steered\)/);
  assert.match(text(base),/unsteered/);
  assert.doesNotMatch(text(base),/steered:/);
});

test('a steering entry the server would refuse blocks the send and says why',async()=>{
  const state=app(),held=server('/api/run');
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L3H4','add','1'])};
  const root=await view(state);
  button(root,'Run  (Ctrl+Enter)').click();await tick();
  assert.equal(held.body,undefined);
  assert.match(text(await view(state)),/Steering: L3H4: add pushes neurons only/);
});

test('each chat turn records the steering it ran with',async()=>{
  const state=app(),held=server('/api/chat');
  state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','-0.5','raw'])};
  let root=await view(state);
  const input=()=>walk(root).find(n=>n.placeholder==='message');
  input().value='hi';button(root,'Send').click();await tick();
  assert.deepEqual(held.body.steer,[{addr:'L1N2',mode:'add',dose:-0.5,unit:'raw'}]);
  root=await view(state);
  assert.match(text(walk(root).filter(n=>n.className==='chat-msg bot')[1]),/steered: L1N2 -0\.5 raw/);
  held.finish({current:'yo',base:'hey',chat_template:false,generation,variants:{current:'complete',base:'complete'},
    steering:{variant:'current',entries:[{addr:'L1N2',mode:'add',dose:-0.5,unit:'raw',value:-0.5}],calibration:{needed:false}}});
  await tick();await tick();
  // The next turn runs unsteered and its record has no steer field.
  state.keep.steer.on=false;
  root=await view(state);
  input().value='again';button(root,'Send').click();await tick();
  assert.equal(held.body.steer,undefined);
  held.finish({current:'ok',base:'ok too',chat_template:false,generation,variants:{current:'complete',base:'complete'}});
  await tick();await tick();
  assert.deepEqual(state.keep.run.chat,[{role:'user',content:'hi'},{role:'assistant',content:'yo',steer:'L1N2 -0.5 raw'},
    {role:'user',content:'again'},{role:'assistant',content:'ok'}]);
  assert.deepEqual(state.keep.run.baseChat,[{role:'user',content:'hi'},{role:'assistant',content:'hey'},
    {role:'user',content:'again'},{role:'assistant',content:'ok too'}]);
  root=await view(state);
  const bots=walk(root).filter(n=>n.className==='chat-msg bot');
  assert.match(text(bots[1]),/edited.*steered: L1N2 -0\.5 raw.*yo/);
  assert.doesNotMatch(text(bots[0]),/steered/);
  assert.doesNotMatch(text(bots[3]),/steered/);
});
