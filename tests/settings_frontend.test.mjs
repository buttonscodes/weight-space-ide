// Model-free DOM/network contracts: node --test tests/settings_frontend.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { settingsView, memoryImpact } from '../web/js/settings.js';
import { runView } from '../web/js/run.js';
const tick=()=>new Promise(r=>setImmediate(r));
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  click(){this.events.click?.({target:this});}
}
global.document={createElement:t=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text})};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const button=(root,label)=>walk(root).find(n=>n.tag==='button'&&text(n)===label);
const response=(data)=>({ok:true,headers:{get:()=> 'application/json'},json:async()=>data});
const values={backend:'cpu',context:64,new_tokens:4,resident_seconds:900,threads:4,floor_gib:null,preferred_gib:null,budget_gib:null,ceiling_gib:null,reservation_gib:null,swap_growth_mib:0,compression_growth_mib:0};
const preview={machine:{total:100*2**30,available:70*2**30},incremental:20*2**30,required:20*2**30,cold_remaining:50*2**30,
  forecast:null,policy:null,warnings:[],errors:[],backend:{selected:'cpu'},context_policy_limit:32768,kernel_token_rows:2048,prefill_chunk:128,effective_context:64,qualification:'Advisory estimate'};
function payload(){return {revision:'initial',csrf:'token',config:'/tmp/fake/config.json',target:'profile test',managed:true,editable:Object.keys(values),
  saved:{values:{...values}},active:{values:{...values},sources:Object.fromEntries(Object.keys(values).map(k=>[k,'default']))},
  overridden:[],changed:[],restart_required:false,note:'Saved settings require restart.',interactive_defaults:{context:8192,new_tokens:2048,resident_seconds:14400,budget_gib:null,ceiling_gib:null,reservation_gib:null},preview:structuredClone(preview)};}
function app(){return {sel:'model',tab:'Run',variant:'current',keep:{},state:{open:{name:'fixture',kind:'hf'},status:{can_run:true,version:0}},rerender(){}};}

test('Settings saves only edited fields, protects repeated clicks, and keeps controls named',async()=>{
  const data=payload(),pending=deferred(),calls=[];
  global.fetch=async(path,options)=>{calls.push([path,options]);return path==='/api/settings'&&!options?.method?response(data):pending.promise;};
  const root=new Element('root');await settingsView(root,app());
  const input=walk(root).find(n=>n['aria-label']==='Context tokens');assert.ok(input);
  input.value='512';input.events.input({target:input});
  const save=button(root,'Save settings');save.click();save.click();save.click();await tick();
  const writes=calls.filter(([,o])=>o?.method==='POST');assert.equal(writes.length,1);
  assert.deepEqual(JSON.parse(writes[0][1].body).changes,{context:512});assert.equal(writes[0][1].headers['X-WSP-Settings'],'token');
  assert.equal(save.disabled,true); assert.equal(input.disabled,true);
  assert.equal(button(root,'Use capacity-aware defaults').disabled,true);
  assert.equal(button(root,'Reload saved settings').disabled,true);
  const saved=payload();saved.saved.values.context=512;saved.restart_required=true;saved.changed=['context'];
  pending.resolve(response(saved));await tick();assert.equal(save.disabled,false);
});

test('capacity preset clears obsolete caps and preserves floor and preferred headroom',async()=>{
  const data=payload(),calls=[];Object.assign(data.saved.values,{budget_gib:24,ceiling_gib:26,reservation_gib:16,floor_gib:20,preferred_gib:25});global.fetch=async(path,o)=>{calls.push([path,o]);return response(data);};
  const root=new Element('root');await settingsView(root,app());
  button(root,'Use capacity-aware defaults').click();button(root,'Save settings').click();await tick();
  const post=calls.find(([,o])=>o?.method==='POST');
  assert.deepEqual(JSON.parse(post[1].body).changes,{context:8192,new_tokens:2048,resident_seconds:14400,budget_gib:null,ceiling_gib:null,reservation_gib:null});
});

test('stale Settings response cannot append to a new view and unknown memory stays explicit',async()=>{
  const wait=deferred();global.fetch=()=>wait.promise;const root=new Element('root');
  const pending=settingsView(root,app(),()=>false);wait.resolve(response(payload()));await pending;assert.equal(root.children.length,0);
  memoryImpact(root,{machine:{error:'unavailable'},warnings:[],errors:[],qualification:'Advisory'});
  assert.match(text(root),/Current memory is unknown/);
  memoryImpact(root,{...structuredClone(preview),incremental:null,required:null,cold_remaining:null});
  assert.match(text(root),/Model reservation: unknown/);assert.match(text(root),/Measured available/);
});

const runtime={generation:{default_output:128,output_limit:256,context_limit:512}};
test('Run has no automatic completion, uses active defaults, and joins repeat clicks',async()=>{
  const state=app(),pending=deferred(),calls=[];
  global.fetch=async(path,o)=>{calls.push([path,o]);return path==='/api/runtime'?response(runtime):pending.promise;};
  const root=new Element('root');await runView(root,state);await tick();
  assert.deepEqual(calls.map(([p])=>p),['/api/runtime']);assert.equal(state.keep.run.n,128);
  const run=button(root,'Run  (Ctrl+Enter)');run.click();run.click();await tick();
  assert.equal(calls.filter(([p])=>p==='/api/run').length,1);
  assert.equal(JSON.parse(calls[1][1].body).max_new_tokens,128);
  pending.resolve(response({base:{},current:{},tokens:[]}));await tick();
});

test('Chat uses shared output defaults and prevents duplicate history and sends',async()=>{
  const state=app();state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  const pending=deferred(),calls=[];
  global.fetch=async(path,o)=>{calls.push([path,o]);return path==='/api/runtime'?response(runtime):pending.promise;};
  const root=new Element('root');await runView(root,state);
  const input=walk(root).find(n=>n.placeholder==='message');input.value='hello';
  const send=button(root,'Send');send.click();send.click();await tick();
  assert.equal(state.keep.run.chat.length,0);assert.equal(state.keep.run.chatBusy,true);
  assert.equal(calls.filter(([p])=>p==='/api/chat').length,1);
  assert.equal(JSON.parse(calls[1][1].body).max_new_tokens,128);
  pending.resolve(response({base:'reply',current:'reply',chat_template:true}));await tick();assert.equal(state.keep.run.chatBusy,false);
});

test('clamped thread notice renders as a runtime warning in thread units',async()=>{
  const notice={code:'threads_clamped',action:'warning',message:'Saved thread count 12 exceeds this host limit of 8; using 8 threads.',measured:12,limit:8,recommendations:['Lower the saved thread count.']};
  global.fetch=async()=>response({...runtime,warnings:[notice]});
  const root=new Element('root');await runView(root,app());
  const body=text(root);
  assert.match(body,/Runtime warning: threads_clamped/);assert.match(body,/using 8 threads/);
  assert.match(body,/Measured: 12 threads; limit: 8 threads/);assert.match(body,/Lower the saved thread count/);
});

test('stale runtime settings fetch does not start or append a Run view',async()=>{
  global.fetch=async()=>response(runtime);const root=new Element('root');await runView(root,app(),()=>false);assert.equal(root.children.length,0);
});

for (const mode of ['completion','chat']) test(`${mode} rejects invalid output before fetch/history on click and keyboard`,async()=>{
  const state=app(),calls=[];
  const limits={generation:{default_output:4,output_limit:4,context_limit:64},settings_summary:{profile:'qwen38',saved:{context:8192,new_tokens:2048},next_start:{context:8192,new_tokens:2048},restart_required:true,overridden:[]}};
  if(mode==='chat')state.keep.run={mode:'chat',chat:[],tpl:{},chatDraft:'keep this unsent'};
  global.fetch=async(path,o)=>{calls.push([path,o]);return response(path==='/api/runtime'?limits:{base:'ok',current:'ok',chat_template:true});};
  const root=new Element('root');await runView(root,state);
  const count=walk(root).find(n=>n['aria-label']===(mode==='chat'?'Chat output tokens':'Completion output tokens'));
  const message=mode==='chat'?walk(root).find(n=>n.placeholder==='message'):walk(root).find(n=>n.tag==='textarea');
  for(const value of ['', 'NaN','Infinity','1.5','-1','5']) {
    count.value=value;count.events.input({target:count});button(root,mode==='chat'?'Send':'Run  (Ctrl+Enter)').click();
    message.events.keydown({key:'Enter',ctrlKey:true});await tick();assert.equal(calls.length,1);
    assert.match(text(root),/whole number from/);assert.match(text(root),/saved output value is a default/);
    if(mode==='chat'){assert.equal(state.keep.run.chat.length,0);assert.equal(state.keep.run.chatDraft,'keep this unsent');assert.equal(message.value,'keep this unsent');}
  }
  assert.match(text(root),/Active output limit 4/);assert.match(text(root),/Saved changes are pending restart/);
  count.value='4';count.events.input({target:count});assert.doesNotMatch(text(root),/Enter a whole number/);
  message.events.keydown({key:'Enter',ctrlKey:true});await tick();
  assert.equal(calls.length,2);assert.equal(JSON.parse(calls[1][1].body).max_new_tokens,4);
});

test('resident memory guidance places cold subtraction in hypothetical details',()=>{
  const root=new Element('root');
  memoryImpact(root,{...structuredClone(preview),resident:true,incremental:0,required:23*2**30,active_reservation:23*2**30,
    machine:{total:128*2**30,available:47.12*2**30},cold_remaining:24.12*2**30,
    policy_hypothetical:true,policy:{budget:22.12*2**30,ceiling:103*2**30,preferred:25*2**30,floor:20*2**30},
    active_policy:{budget:24*2**30,ceiling:26*2**30,preferred:25*2**30,floor:20*2**30},
    info:['The resident model is already included.'],warnings:[]});
  assert.equal(walk(root).filter(n=>n.className==='pill warn').length,0);
  assert.match(text(root),/Estimated remaining headroom: 47.12 GiB/);
  assert.ok(walk(root).filter(n=>n.tag==='details').some(n=>text(n).includes('24.12 GiB')));
  assert.match(text(root),/Automatic budget after unload\/restart is unknown/);
});

test('draft preview hides old numbers and ignores stale responses or failed estimates',async()=>{
  const old=deferred(),failed=deferred();let previews=0;
  global.fetch=async(path,o)=>path==='/api/settings'?response(payload()):(++previews===1?old.promise:failed.promise);
  const root=new Element('root');await settingsView(root,app());
  const input=walk(root).find(n=>n['aria-label']==='Context tokens');input.value='512';input.events.input({target:input});
  assert.doesNotMatch(text(root),/Proposed model reservation/);assert.match(text(root),/Unsaved draft/);
  assert.equal(button(root,'Use capacity-aware defaults').disabled,true);
  await new Promise(r=>setTimeout(r,270));input.value='8192';input.events.input({target:input});
  const stale=payload();stale.preview.effective_context=512;old.resolve(response(stale));await tick();
  assert.doesNotMatch(text(root),/Unsaved draft effective context 512/);
  await new Promise(r=>setTimeout(r,270));failed.resolve({...response({error:'invalid draft'}),ok:false});await tick();
  assert.match(text(root),/No current draft memory estimate/);assert.equal(button(root,'Use capacity-aware defaults').disabled,true);
  assert.doesNotMatch(text(root),/Proposed model reservation/);
});

test('server chat errors remain outside history and exact draft retries once',async()=>{
  const state=app();state.keep.run={mode:'chat',chat:[],tpl:{},chatDraft:'preserve complete history'};
  const bodies=[];let failed=true;
  global.fetch=async(path,o)=>{
    if(path==='/api/runtime')return response(runtime);
    bodies.push(JSON.parse(o.body));
    if(failed){failed=false;return {ok:false,status:400,statusText:'Bad Request',headers:{get:()=> 'application/json'},json:async()=>({error:'synthetic refusal.'})};}
    return response({base:'base reply',current:'reply',chat_template:true,generation:{requested_output:128,effective_output:1}});
  };
  const root=new Element('root');await runView(root,state);button(root,'Send').click();await tick();
  assert.deepEqual(state.keep.run.chat,[]);assert.equal(state.keep.run.chatDraft,'preserve complete history');
  assert.match(state.keep.run.chatError,/synthetic refusal/);
  const retry=new Element('root');await runView(retry,state);button(retry,'Send').click();await tick();
  assert.deepEqual(bodies[0].messages,bodies[1].messages);
  assert.deepEqual(state.keep.run.chat.map(m=>m.role),['user','assistant']);
  assert.equal(state.keep.run.chat[1].content,'reply');assert.equal(state.keep.run.chatDraft,'');
});

test('no feasible preset is explained and cannot apply stale defaults',async()=>{
  const data=payload();data.interactive_defaults=null;data.capacity_recommendation={status:'blocked',note:'No usable context fits the known physical bounds.'};
  global.fetch=async()=>response(data);
  const root=new Element('root');await settingsView(root,app());
  assert.equal(button(root,'Use capacity-aware defaults').disabled,true);
  button(root,'Use capacity-aware defaults').click();
  assert.match(text(root),/No usable context fits/);
  assert.equal(walk(root).find(n=>n['aria-label']==='Context tokens').value,64);
});


test('managed chat override exceeds saved default; partial output stays outside history and preserves draft',async()=>{
  const a=app();a.keep.run={mode:'chat',chat:[],tpl:{},chatDraft:'retry this',cmax:128};
  const calls=[];global.fetch=async(path,options)=>{
    calls.push([path,options]);
    if(path==='/api/runtime')return response({generation:{default_output:4,output_limit:null,context_limit:512}});
    return response({base:'partial base',current:'',interrupted:true,event:{code:'runtime_budget',action:'abort',message:'Stopped safely',measured:4*2**30,limit:3*2**30,recommendations:['Free memory and retry.']}});
  };
  let root=new Element('root');await runView(root,a);button(root,'Send').click();await tick();
  assert.equal(JSON.parse(calls.find(([p])=>p==='/api/chat')[1].body).max_new_tokens,128);
  assert.deepEqual(a.keep.run.chat,[]);assert.equal(a.keep.run.chatDraft,'retry this');
  root=new Element('root');await runView(root,a);
  assert.match(text(root),/partial output \(not added to conversation\)/);
  assert.match(text(root),/4.00 GiB/);assert.match(text(root),/Free memory and retry/);
});

test('Largest weight patch is a Settings field that applies once saved',async()=>{
  const data=payload();data.editable.push('patch_limit_gib');data.saved.values.patch_limit_gib=null;data.active.values.patch_limit_gib=null;data.active.sources.patch_limit_gib='default';data.live=['patch_limit_gib'];
  const calls=[];let saved=null;
  global.fetch=async(path,o)=>{calls.push([path,o]);return response(o?.method==='POST'&&path==='/api/settings'?saved:data);};
  const root=new Element('root');await settingsView(root,app());
  const input=walk(root).find(n=>n['aria-label']==='Largest weight patch (GiB)');assert.ok(input);
  assert.equal(input.placeholder,'Auto');
  assert.match(text(root),/never overrides the memory check/);assert.match(text(root),/Applies as soon as it is saved/);
  const row=walk(root).find(n=>n.tag==='tr'&&walk(n).includes(input));assert.equal(text(row.children[2]),'Auto');
  input.value='2';input.events.input({target:input});
  saved=structuredClone(data);saved.saved.values.patch_limit_gib=2;saved.active.values.patch_limit_gib=2;saved.active.sources.patch_limit_gib='config defaults';
  saved.applied={now:['patch_limit_gib'],next_load:[],next_start:[],overridden:[],supervised:true,loaded:null,effect:null};
  button(root,'Save settings').click();await tick();await tick();
  const post=calls.find(([,o])=>o?.method==='POST');assert.deepEqual(JSON.parse(post[1].body).changes,{patch_limit_gib:2});
  assert.match(text(root),/Largest weight patch applies now: 2 GiB\./);
  const cells=row.children;assert.equal(text(cells[2]),'2');               // Active now shows the saved value
});
