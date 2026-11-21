// Model-free seams between steering and saved selections: saving Probe results as selections, the panel's
// selection picker and neuron counts, the neuron bound checked before sending, and the inspector's steer controls.
import test from 'node:test';
import assert from 'node:assert/strict';
import { probeView } from '../web/js/probe.js';
import { steerPanel, steerRequest, steerState, steerControl, parseTarget, entryReach, skippedText, MAX_NEURONS } from '../web/js/steer.js';
const tick=()=>new Promise(r=>setImmediate(r));
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  click(){if(!this.disabled)this.events.click?.({target:this});}
  focus(){}
}
global.document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text}),documentElement:{dataset:{theme:'light'}}};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const button=(root,label)=>walk(root).find(n=>n.tag==='button'&&text(n)===label);
const labelled=(root,label)=>walk(root).find(n=>n['aria-label']===label);
const type=(root,label,value)=>labelled(root,label).events.input({target:{value}});
const response=(data,ok=true)=>({ok,statusText:'Bad Request',headers:{get:()=> 'application/json'},json:async()=>data});
const settle=async()=>{await tick();await tick();await tick();};
function app(view,groups){
  const a={sel:'model',tab:'Probe',variant:'current',keep:{},state:{open:{name:'fixture',kind:'wsp'},status:{can_run:true,version:0},groups},selected:[],rerender(){},select(addr,tab){a.selected.push([addr,tab]);}};
  if(view)a.keep.probe={view,reads:{text:'',neurons:''},fires:{text:'',k:20,sign:'both'},next:{prompt:'',source:'panel',typed:'',top:10},contrast:{a:'',b:'',k:20}};
  return a;
}
// POST requests are held until answered; /api/groups and /api/steer/groups answer at once.
function server({groups=[],max=MAX_NEURONS}={}){
  const held={requests:[],saved:[]};
  global.fetch=async(url,o)=>{
    const body=o&&o.body?JSON.parse(o.body):undefined;
    held.requests.push({url,body});
    if(url==='/api/steer/groups')return response({groups,max_neurons:max});
    if(url==='/api/groups'){held.saved.push(body);return response({groups:{[body.name]:{members:[],note:''}},name:body.name,summary:{neurons:body.neurons.length}});}
    return new Promise(resolve=>{held.answer=(data,ok=true)=>resolve(response(data,ok));});
  };
  return held;
}
async function view(state){const root=new Element('root');await probeView(root,state);return root;}
function asking(){const asked=[];global.prompt=(m,d)=>{asked.push(d);return d;};return asked;}

const FIRES={position:4,token:1,token_str:'b',token_strs:['a','b','c','a','b'],
  neurons:[{layer:1,neuron:3,addr:'L1N3',value:2.5},{layer:0,neuron:7,addr:'L0N7',value:-1.25},{layer:2,neuron:9,addr:'L2N9',value:1}]};

test('fires: save every row, or the ticked rows, as a selection under a new suggested name',async()=>{
  const a=app('fires',{'fires at token 4':{members:['L5N5'],note:''}}),held=server();
  const asked=asking();
  let root=await view(a);
  type(root,'fires text','abcab');button(root,'Find').click();
  held.answer(FIRES);await settle();
  root=await view(a);
  assert.equal(button(root,'save ticked as selection').disabled,true);          // nothing ticked yet
  button(root,'save all as selection').click();await settle();
  assert.deepEqual(held.saved[0],{name:'fires at token 4 2',neurons:[[1,3],[0,7],[2,9]],mode:'replace'});
  assert.equal(asked[0],'fires at token 4 2');                                     // never the existing selection
  assert.match(text(await view(a)),/Saved as selection "fires at token 4 2": 3 neurons\./);
  root=await view(a);
  labelled(root,'tick L2N9').events.change({target:{checked:true}});
  labelled(root,'tick L1N3').events.change({target:{checked:true}});
  root=await view(a);                                                              // ticks survive a redraw
  assert.equal(labelled(root,'tick L2N9').checked,true);
  assert.ok(!button(root,'save ticked as selection').disabled);
  button(root,'save ticked as selection').click();await settle();
  assert.deepEqual(held.saved[1].neurons,[[1,3],[2,9]]);
  assert.ok(held.requests.filter(r=>r.url==='/api/groups').every(r=>r.body.neurons));  // the one save path
  // a new result starts with nothing ticked
  type(root,'fires text','xy');button(root,'Find').click();held.answer({...FIRES,position:1});await settle();
  root=await view(a);
  assert.ok(!labelled(root,'tick L2N9').checked);
});

test('contrast: each list saves its own rows',async()=>{
  const a=app('contrast',{}),held=server();
  asking();
  let root=await view(a);
  type(root,'contrast set A','x');type(root,'contrast set B','y');button(root,'Contrast').click();
  held.answer({a:1,b:1,measure:'',up:[{layer:0,neuron:1,addr:'L0N1',diff:1,a:1,b:0}],down:[{layer:1,neuron:2,addr:'L1N2',diff:-1,a:0,b:1},{layer:1,neuron:4,addr:'L1N4',diff:-1,a:0,b:1}]});
  await settle();
  root=await view(a);
  const saves=walk(root).filter(n=>n.tag==='button'&&text(n)==='save all as selection');
  assert.equal(saves.length,2);
  saves[1].click();await settle();
  assert.deepEqual(held.saved[0],{name:'fires on B not A',neurons:[[1,2],[1,4]],mode:'replace'});
});

test('the panel lists saved selections with their neuron counts and adds group entries',async()=>{
  const groups=[{name:'probe picks',addr:'group:probe picks',neurons:3,layers:[0,1],skipped:{}},
    {name:'mixed',addr:'group:mixed',neurons:4,layers:[0,1],skipped:{unresolved:1,heads:1,slices:0}},
    {name:'heads only',addr:'group:heads only',neurons:0,layers:[],skipped:{heads:2}},
    {name:'huge',addr:'group:huge',neurons:MAX_NEURONS+1,layers:[3],skipped:{}}];
  const a=app(null,{'probe picks':{},mixed:{},'heads only':{},huge:{}});a.tab='Run';
  const held=server({groups});
  let box=steerPanel(a);
  assert.match(text(box),/saved selections \(counting…\)/);
  await settle();
  assert.equal(held.requests.filter(r=>r.url==='/api/steer/groups').length,1);
  const pick=labelled(box,'steer a saved selection');
  const opts=walk(pick).filter(n=>n.tag==='option');
  assert.deepEqual(opts.slice(1).map(text),['probe picks (3 neurons)','mixed (4 neurons)','heads only (0 neurons: no neurons to push)',`huge (131,073 neurons: more than 131,072 neurons)`]);
  assert.deepEqual(opts.slice(1).map(o=>o.disabled),[false,false,true,true]);
  pick.value='group:probe picks';button(box,'add selection').click();
  const st=steerState(a);
  assert.deepEqual(st.entries.map(e=>[e.addr,e.mode]),[['group:probe picks','add']]);
  assert.equal(text(labelled(box,'steer entry 1 reach')),'pushes 3 neurons');
  assert.deepEqual(steerRequest(st).fields.steer,[{addr:'group:probe picks',mode:'add',dose:0.5,unit:'resid'}]);
  // what a group leaves out is said beside it
  st.entries.push({addr:'group:mixed',mode:'add',dose:'0.5',unit:'resid',on:true});st.paint();
  assert.equal(text(labelled(box,'steer entry 2 reach')),'pushes 4 neurons; leaves out 1 member that no longer resolves, 1 head');
  // above the bound, unknown, or nothing to push: refused before sending, shown in the panel
  st.entries.push({addr:'group:huge',mode:'mul',dose:'2',unit:'resid',on:true},{addr:'group:gone',mode:'add',dose:'1',unit:'resid',on:true},
    {addr:'group:heads only',mode:'add',dose:'1',unit:'resid',on:true});st.paint();
  const r=steerRequest(st);
  assert.equal(r.fields,null);
  assert.deepEqual(r.problems,['group:huge: pushes 131,073 neurons; at most 131,072 can be pushed at once',
    'group:gone: there is no saved selection named "gone"','group:heads only: the selection has no neurons to push (it holds 2 heads)']);
  assert.match(text(box),/at most 131,072 can be pushed at once/);
  // the selections change: counted again
  a.state.groups={...a.state.groups,another:{}};
  steerPanel(a);await settle();
  assert.equal(held.requests.filter(r=>r.url==='/api/steer/groups').length,2);
});

test('the bound counts every entry together, ranges included',()=>{
  const st={on:true,p50:'',groups:{list:[{name:'g',addr:'group:g',neurons:100000,skipped:{}}],max:MAX_NEURONS},
    entries:[{addr:'group:g',mode:'add',dose:'1',unit:'resid',on:true},{addr:'blocks.2.mlp.neurons.0-40000',mode:'add',dose:'1',unit:'resid',on:true}]};
  assert.deepEqual(steerRequest(st).problems,['the entries push 140,001 neurons in all; at most 131,072 can be pushed at once']);
  st.entries[1].on=false;
  assert.ok(steerRequest(st).fields);
  assert.equal(entryReach({addr:'blocks.2.mlp.neurons.5-9'}).text,'5 neurons');
  assert.equal(entryReach({addr:'L2H1'}).text,'1 head');
  assert.equal(entryReach({addr:'group:g'}).text,'counting…');                    // not counted yet: the server checks
  // without the selections known (no list), a group entry is sent and the server decides
  assert.ok(steerRequest({on:true,p50:'',entries:[{addr:'group:x',mode:'add',dose:'1',unit:'resid',on:true}]}).fields);
  assert.equal(skippedText({unresolved:2,heads:0,slices:3}),'2 members that no longer resolve, 3 tensor slices');
});

test('a saved name with spaces is a target as saved',()=>{
  assert.deepEqual(parseTarget('group:fires at token 4 2'),{addr:'group:fires at token 4 2',kind:'group'});
});

test('the inspector: steer this selection, or add a neuron, head or MLP to steering',()=>{
  const a=app(null,{});
  const status=new Element('div');
  const sel=steerControl(a,{addr:'group:picks',selection:{neurons:12}},status);
  assert.equal(text(sel),'steer this selection');
  sel.click();
  assert.deepEqual(a.keep.steer.entries.at(-1),{addr:'group:picks',mode:'add',dose:'0.5',unit:'resid',on:true});
  assert.match(status.textContent,/group:picks \+0\.5 added to the Run tab's steering panel/);
  const huge=steerControl(a,{addr:'group:huge',selection:{neurons:MAX_NEURONS+5}},status);
  assert.equal(huge.disabled,true);
  assert.match(huge.title,/at most 131,072 can be pushed/);
  assert.equal(steerControl(a,{addr:'group:none',selection:{neurons:0,heads:2}},status).disabled,true);
  for(const [addr,want] of [['blocks.3.mlp.neuron.7',{addr:'L3N7',mode:'add'}],['blocks.2.attn.head.1',{addr:'L2H1',mode:'mul'}],
    ['blocks.4.mlp',{addr:'blocks.4.mlp',mode:'mul'}],['blocks.4.mlp.neurons.2-5',{addr:'blocks.4.mlp.neurons.2-5',mode:'add'}]]){
    const b=steerControl(a,{addr},status);
    assert.equal(text(b),'+ steering');
    b.click();
    const e=a.keep.steer.entries.at(-1);
    assert.deepEqual({addr:e.addr,mode:e.mode},want,addr);
  }
  assert.equal(steerControl(a,{addr:'blocks.4.mlp.W_out'},status),null);        // a tensor: nothing to push
  assert.equal(steerControl(a,{addr:'model'},status),null);
});

test('Check says what a group entry leaves out',async()=>{
  const a=app(null,{g:{}});a.tab='Run';
  const held=server({groups:[{name:'g',addr:'group:g',neurons:2,layers:[1],skipped:{heads:1}}]});
  const box=steerPanel(a);await settle();
  steerState(a).entries.push({addr:'group:g',mode:'add',dose:'0.5',unit:'raw',on:true});
  steerState(a).paint();
  button(box,'Check').click();
  held.answer({entries:[{index:0,addr:'group:g',mode:'add',dose:0.5,unit:'raw',kind:'neurons',neurons:2,skipped:{unresolved:0,heads:1,slices:0},values:{min:0.5,max:0.5}}],calibration:{}});
  await settle();
  assert.match(text(box),/2 neurons/);
  assert.match(text(box),/not pushed: 1 head/);
});
