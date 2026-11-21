// Model-free Run UI: the Stop button for completion and chat.
import test from 'node:test';
import assert from 'node:assert/strict';
import { runView, onGenerationEvent, restoredDraft } from '../web/js/run.js';
const tick=()=>new Promise(r=>setImmediate(r));
const later=()=>new Promise(r=>setTimeout(r,0));
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  click(){this.events.click?.({target:this});}
  focus(){global.document.activeElement=this;}
}
global.document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text}),activeElement:null};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const button=(root,label)=>walk(root).find(n=>n.tag==='button'&&text(n)===label);
const input=(root)=>walk(root).find(n=>n.placeholder==='message');
const response=(data,ok=true)=>({ok,statusText:'Bad Request',headers:{get:()=> 'application/json'},json:async()=>data});
const runtime={generation:{default_output:16,output_limit:64,context_limit:128}};
const generation={requested_output:8,effective_output:8,prompt_tokens:2,context:128};
function app(){return {sel:'model',tab:'Run',variant:'current',keep:{},state:{open:{name:'fixture',kind:'gguf'},status:{can_run:true,version:0}},rerender(){},select(){}};}
function chatState(){const state=app();state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};return state;}
// The Run or Chat request stays pending until the test answers it; stop requests answer at once.
function server(path,{stop=()=>response({ok:true,found:true})}={}){
  const held={stops:[]};
  global.fetch=async(url,o)=>{
    if(url==='/api/generation/stop'){held.stops.push(JSON.parse(o.body));return stop();}
    if(url!==path)return response(runtime);
    held.body=JSON.parse(o.body);
    return new Promise(resolve=>{held.finish=data=>resolve(response(data));});
  };
  return held;
}
async function view(state){const root=new Element('root');await runView(root,state);return root;}
const liveCards=(root)=>walk(root).filter(n=>n.className==='chat-msg bot'||(n.className==='card'&&walk(n).some(c=>c.role==='status'&&/generating|waiting|stopping|starting|runs after/.test(text(c)))));
async function sendChat(state,message){
  const root=await view(state);
  input(root).value=message;input(root).oninput?.({target:input(root)});
  button(root,'Send').click();await tick();
}
async function pressStop(state){
  const root=await view(state);
  const stop=button(root,'Stop');
  assert.ok(stop,'a Stop button is shown while the request runs');
  stop.click();await tick();
}

test('stopping a chat turn while the edited reply is generated leaves it out of both conversations',async()=>{
  const state=chatState(),held=server('/api/chat');
  state.keep.run.chat=[{role:'user',content:'earlier'},{role:'assistant',content:'edited before'}];
  state.keep.run.baseChat=[{role:'user',content:'earlier'},{role:'assistant',content:'base before'}];
  await sendChat(state,'helo wrold');
  const id=held.body.stream;
  onGenerationEvent({type:'generation',stream:id,variant:'current',state:'started',keep:0,text:'Hi th',tokens:2},state);
  await pressStop(state);
  assert.deepEqual(held.stops,[{stream:id}]);
  // Immediate feedback, and a second press sends nothing more.
  let root=await view(state);
  const stopping=button(root,'Stopping…');
  assert.ok(stopping&&stopping.disabled);
  assert.match(text(root),/stopping…/);
  stopping.click();await tick();
  assert.equal(held.stops.length,1);
  onGenerationEvent({type:'generation',stream:id,variant:'current',state:'stopped',tokens:3},state);
  held.finish({stopped:true,current:'Hi the',base:'',variants:{current:'stopped',base:'not_run'},tokens:{current:3},chat_template:false,generation});
  await tick();await tick();
  assert.deepEqual(state.keep.run.chat,[{role:'user',content:'earlier'},{role:'assistant',content:'edited before'}]);
  assert.deepEqual(state.keep.run.baseChat,[{role:'user',content:'earlier'},{role:'assistant',content:'base before'}]);
  assert.equal(state.keep.run.live.chat,null);
  root=await view(state);
  assert.equal(liveCards(root).filter(n=>/Hi th/.test(text(n))).length,0);
  assert.ok(!walk(root).some(n=>n.className==='chat-msg user'&&text(n)==='helo wrold'));
  assert.equal(input(root).value,'helo wrold');
  assert.equal(input(root).disabled,false);
  await later();
  assert.equal(global.document.activeElement,input(root));
  assert.match(text(root),/Stopped\. The turn was not added; your message is back in the box\./);
  assert.equal(button(root,'Stop'),undefined);
  assert.ok(button(root,'Send'));
  // The corrected message is sent as a new turn; the notice goes.
  input(root).value='hello world';
  button(root,'Send').click();await tick();
  assert.deepEqual(held.body.messages.map(m=>m.content),['earlier','edited before','hello world']);
  assert.deepEqual(held.body.base_messages.map(m=>m.content),['earlier','base before','hello world']);
  held.finish({current:'Hi!',base:'Hey',chat_template:false,generation,variants:{current:'complete',base:'complete'}});
  await tick();await tick();
  assert.deepEqual(state.keep.run.chat.slice(2),[{role:'user',content:'hello world'},{role:'assistant',content:'Hi!'}]);
  assert.deepEqual(state.keep.run.baseChat.slice(2),[{role:'user',content:'hello world'},{role:'assistant',content:'Hey'}]);
  assert.doesNotMatch(text(await view(state)),/The turn was not added/);
});

test('stopping a chat turn after the edited reply completed, while the base reply runs, also leaves it out',async()=>{
  const state=chatState(),held=server('/api/chat');
  await sendChat(state,'hello');
  const id=held.body.stream,send=fields=>onGenerationEvent({type:'generation',stream:id,...fields},state);
  send({variant:'current',state:'started',keep:0,text:'Full edited reply',tokens:3});
  send({variant:'current',state:'complete',tokens:3});
  send({variant:'base',state:'started',keep:0,text:'Par',tokens:1});
  await pressStop(state);
  send({variant:'base',state:'stopped',tokens:1});
  held.finish({stopped:true,current:'Full edited reply',base:'Par',variants:{current:'complete',base:'stopped'},tokens:{base:1},chat_template:false,generation});
  await tick();await tick();
  assert.deepEqual(state.keep.run.chat,[]);
  const root=await view(state);
  assert.equal(input(root).value,'hello');
  assert.doesNotMatch(text(root),/Full edited reply|Par\b/);
  assert.ok(!walk(root).some(n=>n.className==='chat-msg bot'||n.className==='chat-msg user'));
});

test('a chat stop that loses the race shows the finished turn as usual',async()=>{
  const state=chatState(),held=server('/api/chat',{stop:()=>response({ok:true,found:false})});
  await sendChat(state,'hello');
  await pressStop(state);
  held.finish({current:'done',base:'also done',chat_template:false,generation,variants:{current:'complete',base:'complete'}});
  await tick();await tick();
  assert.deepEqual(state.keep.run.chat,[{role:'user',content:'hello'},{role:'assistant',content:'done'}]);
  assert.deepEqual(state.keep.run.baseChat,[{role:'user',content:'hello'},{role:'assistant',content:'also done'}]);
  const root=await view(state);
  assert.equal(input(root).value,'');
  assert.doesNotMatch(text(root),/Stopped/);
});

test('a stop that fails says so where it was pressed and can be pressed again',async()=>{
  let fail=true;
  const state=chatState(),held=server('/api/chat',{stop:()=>fail?response({error:'ValueError: no'},false):response({ok:true,found:true})});
  await sendChat(state,'hello');
  await pressStop(state);
  let root=await view(state);
  assert.match(text(walk(root).find(n=>n.role==='alert'&&/Stop failed/.test(text(n)))),/Stop failed: ValueError: no\. Try again\./);
  fail=false;
  button(root,'Stop').click();await tick();
  assert.equal(held.stops.length,2);
  root=await view(state);
  assert.ok(button(root,'Stopping…'));
  assert.ok(!walk(root).some(n=>/Stop failed/.test(n.textContent||'')));
  held.finish({stopped:true,current:'',base:'',variants:{current:'not_run',base:'not_run'},chat_template:false,generation});
  await tick();await tick();
  assert.deepEqual(state.keep.run.chat,[]);
});

test('text typed into the box while a turn was pending is kept after the stopped message',()=>{
  assert.equal(restoredDraft('helo','helo'),'helo');
  assert.equal(restoredDraft('helo',''),'helo');
  assert.equal(restoredDraft('helo',undefined),'helo');
  assert.equal(restoredDraft('helo','  '),'helo');
  assert.equal(restoredDraft('helo','helo and more'),'helo and more');
  assert.equal(restoredDraft('helo','something new'),'helo something new');
});

test('stopping a completion keeps nothing, removes the live cards and says the run was stopped',async()=>{
  const state=app(),held=server('/api/run');
  button(await view(state),'Run  (Ctrl+Enter)').click();await tick();
  const id=held.body.stream;
  onGenerationEvent({type:'generation',stream:id,variant:'current',state:'started',keep:0,text:'partial',tokens:1},state);
  let root=await view(state);
  assert.ok(button(root,'Run  (Ctrl+Enter)').disabled);
  await pressStop(state);
  assert.deepEqual(held.stops,[{stream:id}]);
  root=await view(state);
  assert.ok(button(root,'Stopping…').disabled);
  assert.match(text(root),/stopping…/);
  held.finish({stopped:true,prompt:held.body.prompt,join:'',tokens:['a'],base:{next:[],status:'not_run'},current:{next:[],status:'stopped',text:'partial',tokens:1},generation});
  await tick();await tick();
  assert.deepEqual(state.keep.run.result,{stopped:true});
  assert.equal(state.keep.run.live.completion,null);
  root=await view(state);
  const body=text(root);
  assert.match(body,/Run stopped; nothing was kept\. Edit the prompt and run again\./);
  assert.doesNotMatch(body,/partial|running…|stopping…/);
  assert.equal(button(root,'Stop'),undefined);
  assert.equal(button(root,'Run  (Ctrl+Enter)').disabled,false);
  assert.equal(walk(root).find(n=>n.tag==='textarea').value,held.body.prompt);
});

test('a completion prompt edited during the run stays as edited; an emptied box gets the prompt back',async()=>{
  for(const [edited,expected] of [['a better prompt','a better prompt'],['','The capital of France is']]){
    const state=app(),held=server('/api/run');
    button(await view(state),'Run  (Ctrl+Enter)').click();await tick();
    state.keep.run.prompt=edited;
    await pressStop(state);
    held.finish({stopped:true,base:{next:[],status:'not_run'},current:{next:[],status:'not_run'},generation});
    await tick();await tick();
    assert.equal(walk(await view(state)).find(n=>n.tag==='textarea').value,expected);
  }
});

test('a completion stop that loses the race shows the finished result',async()=>{
  const state=app(),held=server('/api/run',{stop:()=>response({ok:true,found:false})});
  button(await view(state),'Run  (Ctrl+Enter)').click();await tick();
  await pressStop(state);
  held.finish({prompt:'ab',join:'',tokens:['a','b'],base:{next:[],text:'base text',status:'complete'},current:{next:[],text:'edited text',status:'complete'},generation});
  await tick();await tick();
  const body=text(await view(state));
  assert.match(body,/edited text/);assert.match(body,/base text/);assert.doesNotMatch(body,/stopped/i);
});

test('no Stop button is shown when nothing is running',async()=>{
  server('/api/run');
  assert.equal(button(await view(app()),'Stop'),undefined);
  assert.equal(button(await view(chatState()),'Stop'),undefined);
});

const revoked={code:'lease_revoked',action:'abort',message:'Lease revoked while the request ran',recommendations:['Retry.']};
test('a stopped chat turn that was also interrupted still shows the runtime event',async()=>{
  const state=chatState(),held=server('/api/chat');
  await sendChat(state,'hello');
  await pressStop(state);
  held.finish({stopped:true,interrupted:true,event:revoked,current:'Hi',base:'',variants:{current:'stopped',base:'not_run'},tokens:{current:1},chat_template:false,generation});
  await tick();await tick();
  assert.deepEqual(state.keep.run.chat,[]);
  const root=await view(state),body=text(root);
  assert.match(body,/Runtime abort: lease_revoked/);assert.match(body,/Lease revoked while the request ran/);
  assert.match(body,/The turn was not added/);
  assert.equal(input(root).value,'hello');
});

test('a stopped completion that was also interrupted shows the runtime event above the stopped line',async()=>{
  const state=app(),held=server('/api/run');
  button(await view(state),'Run  (Ctrl+Enter)').click();await tick();
  await pressStop(state);
  held.finish({stopped:true,interrupted:true,event:revoked,prompt:'p',join:'',tokens:['a'],base:{next:[],status:'not_run'},current:{next:[],status:'stopped',text:'x',tokens:1},generation});
  await tick();await tick();
  const body=text(await view(state));
  assert.match(body,/Runtime abort: lease_revoked[\s\S]*Run stopped; nothing was kept/);
  assert.doesNotMatch(body,/\bx\b/);
});

test('while stopping, only the reply being generated says stopping; the one not started says it runs after',async()=>{
  const state=chatState(),held=server('/api/chat');
  await sendChat(state,'hello');
  const id=held.body.stream;
  await pressStop(state);
  let statuses=walk(await view(state)).filter(n=>n.role==='status').map(text);
  assert.ok(!statuses.includes('stopping…'),JSON.stringify(statuses));   // nothing has started yet
  onGenerationEvent({type:'generation',stream:id,variant:'current',state:'started',keep:0,text:'Hi',tokens:1},state);
  statuses=walk(await view(state)).filter(n=>n.role==='status').map(text);
  assert.ok(statuses.includes('stopping…')&&statuses.includes('runs after the edited model'),JSON.stringify(statuses));
  assert.equal(statuses.filter(s=>s==='stopping…').length,1);
  held.finish({stopped:true,current:'Hi',base:'',variants:{current:'stopped',base:'not_run'},tokens:{current:1},chat_template:false,generation});
  await tick();
});
