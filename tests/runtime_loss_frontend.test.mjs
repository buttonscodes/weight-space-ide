// Model-free Run UI: a model-worker loss reads as a process event, not as memory figures.
import test from 'node:test';
import assert from 'node:assert/strict';
import { runView } from '../web/js/run.js';
const tick=()=>new Promise(r=>setImmediate(r));
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  click(){this.events.click?.({target:this});}
}
global.document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text})};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const button=(root,label)=>walk(root).find(n=>n.tag==='button'&&text(n)===label);
const response=(data)=>({ok:true,headers:{get:()=> 'application/json'},json:async()=>data});
const runtime={generation:{default_output:16,output_limit:64,context_limit:128}};
const message='The model worker was stopped to protect this machine; the IDE remains available.';
const killed={code:'worker_killed',action:'worker_killed',message,measured:{trigger:'memory_pressure',worker_pid:4242},
  limit:{abort_seconds:0.3,unload_seconds:1},recommendations:['Run again to load the model through fresh admission.']};
const exited={code:'worker_exited',action:'worker_killed',message:message+' The worker exited with return code -9.',
  measured:{trigger:'worker_exited',worker_pid:4243,returncode:-9},limit:{abort_seconds:0.3,unload_seconds:1},
  recommendations:['Run again to load the model through fresh admission.']};
function app(){return {sel:'model',tab:'Run',variant:'current',keep:{},state:{open:{name:'fixture',kind:'gguf'},status:{can_run:true,version:0}},rerender(){},select(){}};}

test('a worker_killed runtime event names the pid and the grace periods in their own units',async()=>{
  global.fetch=async()=>response({...runtime,event:killed});
  const root=new Element('root');await runView(root,app());
  const body=text(root);
  assert.match(body,/Runtime worker_killed: worker_killed/);
  assert.match(body,/Measured: trigger: memory_pressure; worker pid: 4242; limit: abort seconds: 0\.3 seconds; unload seconds: 1\.0 seconds/);
  assert.match(body,/fresh admission/);
  assert.doesNotMatch(body,/MiB|GiB/);
});

test('an interrupted completion shows the partial output and the worker exit',async()=>{
  const state=app(),calls=[];
  const result={prompt:'ab',join:'',tokens:['a','b'],interrupted:true,event:exited,
    current:{next:[],status:'interrupted',text:'cde',event:exited},base:{next:[],status:'not_run'},
    generation:{requested_output:8,effective_output:8,prompt_tokens:2,context:128}};
  global.fetch=async(path,o)=>{calls.push(path);return response(path==='/api/runtime'?runtime:result);};
  let root=new Element('root');await runView(root,state);
  button(root,'Run  (Ctrl+Enter)').click();await tick();await tick();
  assert.equal(calls.filter(p=>p==='/api/run').length,1);
  root=new Element('root');await runView(root,state);
  const body=text(root);
  assert.match(body,/Runtime worker_killed: worker_exited/);
  assert.match(body,/worker pid: 4243; returncode: -9/);
  assert.match(body,/Interrupted — partial output/);assert.match(body,/cde/);
  assert.match(body,/Not run after interruption/);
  assert.doesNotMatch(body,/MiB|GiB/);
});

test('an interrupted chat keeps the draft and shows the loss legibly',async()=>{
  const state=app();state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  global.fetch=async(path)=>response(path==='/api/runtime'?runtime:{interrupted:true,event:exited,current:'cd',base:'',chat_template:false,
    variants:{current:'interrupted',base:'not_run'},generation:{requested_output:8,effective_output:8,prompt_tokens:4,context:128}});
  let root=new Element('root');await runView(root,state);
  const input=walk(root).find(n=>n.placeholder==='message');input.value='hello';
  button(root,'Send').click();await tick();await tick();
  assert.equal(state.keep.run.chat.length,0);
  root=new Element('root');await runView(root,state);
  const body=text(root);
  assert.match(body,/Runtime worker_killed: worker_exited/);assert.match(body,/worker pid: 4243/);
  assert.match(body,/Interrupted — partial output \(not added to conversation\)/);assert.match(body,/Edited: cd/);
  assert.match(body,/Base: not run after the interruption/);
  assert.doesNotMatch(body,/MiB|GiB/);
});
