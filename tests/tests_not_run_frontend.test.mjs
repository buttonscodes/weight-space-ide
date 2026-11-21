// Model-free Tests UI: structured refusals remain neutral and explicitly retryable.
import test from 'node:test';
import assert from 'node:assert/strict';
import { testsView } from '../web/js/work.js';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(key,value){this[key]=value;}
  click(){this.events.click?.({target:this});}
}
global.document={createElement:tag=>new Element(tag),createTextNode:text=>({nodeType:3,textContent:text})};
const text=node=>(node.textContent||'')+(node.children||[]).map(text).join('');
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const button=(node,label)=>walk(node).find(n=>n.tag==='button'&&text(n)===label);
const response=(data,ok=true)=>({ok,status:ok?200:409,headers:{get:()=> 'application/json'},json:async()=>data});
const record={code:'forecast_exceeds_budget',action:'refused',message:'Not enough fresh memory.',measured:30,limit:25,recommendations:['Free memory and retry.']};
const suite={name:'demo',description:'synthetic',cases:[{name:'first',type:'next_token',prompt:'one',expect:'yes'},{name:'second',type:'next_token',prompt:'two',expect:'yes'}]};
const complete=()=>({name:'demo',passed:2,total:2,not_run:0,cases:suite.cases.map(c=>({...c,pass:true,got:'yes',value:1}))});
const blocked=()=>({name:'demo',passed:1,total:2,not_run:1,cases:[complete().cases[0],{...suite.cases[1],pass:null,status:'not_run',event:record,value:null}]});
function app(){return {tab:'Tests',keep:{},state:{suites:[{name:'demo',n:2}],task:{suite:'demo'},open:{name:'fixture',kind:'gguf'},status:{can_run:true,version:0,workspace_id:'fixture'}},rerender(){},async refreshStatus(){}};}

test('base refusal shows neutral NOT RUN, skips current, and retry executes once',async()=>{
  const state=app(),calls=[];let refuse=true;
  global.fetch=async(path)=>{calls.push(path);if(path.startsWith('/api/suite?'))return response(suite);
    return refuse?response({status:'not_run',error:record.message,event:record,report:blocked()},false):response(complete());};
  let root=new Element('root');await testsView(root,state);
  assert.equal(calls.filter(p=>p.startsWith('/api/tests?')).length,0);
  const run=button(root,'Run tests');run.click();run.click();await tick();
  assert.equal(calls.filter(p=>p.startsWith('/api/tests?')).length,1);
  root=new Element('root');await testsView(root,state);
  assert.match(text(root),/NOT RUN/);assert.match(text(root),/Free memory and retry/);
  assert.doesNotMatch(text(root),/fixed|broken|✗/);
  const neutral=walk(root).filter(n=>n.tag==='span'&&text(n)==='NOT RUN');
  assert.ok(neutral.length>=3);assert.ok(neutral.every(n=>n.className==='muted'));
  refuse=false;button(root,'Retry tests').click();button(root,'Retry tests').click();await tick();
  assert.equal(calls.filter(p=>p.startsWith('/api/tests?')).length,3);
  root=new Element('root');await testsView(root,state);
  assert.doesNotMatch(text(root),/NOT RUN|Not enough fresh memory/);
  assert.match(text(root),/edited 2\/2 pass/);
});

test('current refusal retains finished base and never treats null as broken',async()=>{
  const state=app();global.fetch=async(path)=>path.startsWith('/api/suite?')?response(suite):path.includes('variant=base')?response(complete()):response({status:'not_run',event:record,error:record.message,report:blocked()},false);
  let root=new Element('root');await testsView(root,state);button(root,'Run tests').click();await tick();
  root=new Element('root');await testsView(root,state);
  assert.equal(state.keep.tests.result.base.passed,2);
  assert.equal(state.keep.tests.result.cur.cases[1].pass,null);
  assert.match(text(root),/edited 1\/2 pass · 1 NOT RUN/);assert.doesNotMatch(text(root),/broken|fixed/);
});

test('cold refusal without case report still shows all cases as not attempted',async()=>{
  const state=app();global.fetch=async(path)=>path.startsWith('/api/suite?')?response(suite):response({status:'not_run',event:record,error:record.message},false);
  let root=new Element('root');await testsView(root,state);button(root,'Run tests').click();await tick();
  root=new Element('root');await testsView(root,state);
  assert.equal(state.keep.tests.result.base.not_run,2);assert.equal(state.keep.tests.result.cur.not_run,2);
  assert.match(text(root),/Retry tests/);assert.doesNotMatch(text(root),/✗|fixed|broken/);
});
