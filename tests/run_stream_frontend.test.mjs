// Model-free Run UI: text appears while a Run or Chat request is still pending.
import test from 'node:test';
import assert from 'node:assert/strict';
import { runView, onGenerationEvent } from '../web/js/run.js';
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
const generation={requested_output:8,effective_output:8,prompt_tokens:2,context:128};
function app(){return {sel:'model',tab:'Run',variant:'current',keep:{},state:{open:{name:'fixture',kind:'gguf'},status:{can_run:true,version:0}},rerender(){},select(){}};}
// The request stays pending until the test releases it, as it does while a model generates.
function pending(path){
  const held={};
  global.fetch=async(url,o)=>{
    if(url!==path)return response(runtime);
    held.body=JSON.parse(o.body);
    return new Promise(resolve=>{held.finish=data=>resolve(response(data));});
  };
  return held;
}
async function view(state){const root=new Element('root');await runView(root,state);return root;}
// The two live cards, in screen order: base on the left, edited on the right.
function cards(root,cls){
  const [base,edited]=walk(root).filter(n=>n.className===cls&&walk(n).some(c=>c.role==='status'));
  const read=card=>({text:text(walk(card).find(n=>n.tag==='span'&&n.style.whiteSpace==='pre-wrap')),status:text(walk(card).find(n=>n.role==='status'))});
  return {base:read(base),edited:read(edited)};
}

test('a completion shows the edited text as it is generated, then the base text',async()=>{
  const state=app(),held=pending('/api/run');
  button(await view(state),'Run  (Ctrl+Enter)').click();await tick();
  const id=held.body.stream;
  assert.match(id,/^[A-Za-z0-9_-]{1,64}$/);
  const root=await view(state),send=fields=>onGenerationEvent({type:'generation',stream:id,...fields},state);
  assert.match(text(root),/running… the edited model first, then the base model/);
  assert.deepEqual(cards(root,'card'),{base:{text:'',status:'runs after the edited model'},edited:{text:'',status:'starting…'}});
  send({variant:'current',state:'started',keep:0,text:'',tokens:0});
  assert.equal(cards(root,'card').edited.status,'waiting for the first token…');
  send({variant:'current',state:'started',keep:0,text:'Hel',tokens:1});
  assert.deepEqual(cards(root,'card').edited,{text:'Hel',status:'generating…'});
  send({variant:'current',state:'started',keep:2,text:'y there',tokens:3});      // the last character was replaced
  assert.equal(cards(root,'card').edited.text,'Hey there');
  onGenerationEvent({type:'generation',stream:'another-request',variant:'current',state:'started',keep:0,text:'no',tokens:1},state);
  send({variant:'neither',state:'started',keep:0,text:'no',tokens:1});
  assert.equal(cards(root,'card').edited.text,'Hey there');
  send({variant:'current',state:'complete',tokens:3});
  send({variant:'base',state:'started',keep:0,text:'Hi',tokens:1});
  assert.deepEqual(cards(root,'card'),{base:{text:'Hi',status:'generating…'},edited:{text:'Hey there',status:'complete'}});
  // A view drawn later in the same request starts from the text received so far.
  assert.deepEqual(cards(await view(state),'card'),cards(root,'card'));
  held.finish({prompt:'ab',join:'',tokens:['a','b'],base:{next:[],text:'Hi.'},current:{next:[],text:'Hey there.'},generation});
  await tick();await tick();
  assert.equal(state.keep.run.live.completion,null);
  send({variant:'base',state:'complete',tokens:2});                               // a late event changes nothing
  const body=text(await view(state));
  assert.match(body,/Hey there\./);assert.match(body,/Hi\./);assert.doesNotMatch(body,/running…|generating…/);
});

test('live text that cannot be trusted gives way to the result',async()=>{
  const state=app(),held=pending('/api/run');
  button(await view(state),'Run  (Ctrl+Enter)').click();await tick();
  const root=await view(state),send=fields=>onGenerationEvent({type:'generation',stream:held.body.stream,...fields},state);
  const unavailable='Live text is unavailable; the result appears when the request finishes.';
  send({variant:'current',state:'started',keep:0,text:'ab',tokens:2});
  send({variant:'current',state:'started',keep:5,text:'z',tokens:6});            // an earlier event never arrived
  assert.deepEqual(cards(root,'card').edited,{text:'',status:unavailable});
  send({variant:'current',state:'started',keep:0,text:'abcdefz',tokens:7});
  assert.deepEqual(cards(root,'card').edited,{text:'',status:unavailable});
  send({variant:'base',state:'started',keep:0,text:'ok',tokens:1});
  send({variant:'base',state:'started',lost:true});                              // the server could not render it
  assert.deepEqual(cards(root,'card').base,{text:'',status:unavailable});
  send({variant:'base',state:'failed',tokens:1,lost:true});
  assert.deepEqual(cards(root,'card').base,{text:'',status:unavailable});
  held.finish({prompt:'ab',join:'',tokens:['a','b'],base:{next:[],text:'cd'},current:{next:[],text:'ce'},generation});
  await tick();await tick();
  const body=text(await view(state));
  assert.match(body,/ce/);assert.doesNotMatch(body,/Live text is unavailable/);
});

test('a chat turn shows the message and both replies as they are generated',async()=>{
  const state=app();state.keep.run={mode:'chat',chat:[],tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  const held=pending('/api/chat');
  let root=await view(state);
  walk(root).find(n=>n.placeholder==='message').value='hello';
  button(root,'Send').click();await tick();
  assert.match(held.body.stream,/^[A-Za-z0-9_-]{1,64}$/);
  root=await view(state);
  const send=fields=>onGenerationEvent({type:'generation',stream:held.body.stream,...fields},state);
  assert.ok(walk(root).some(n=>n.className==='chat-msg user'&&text(n)==='hello'));
  assert.deepEqual(cards(root,'chat-msg bot'),{base:{text:'',status:'runs after the edited model'},edited:{text:'',status:'starting…'}});
  send({variant:'current',state:'started',keep:0,text:'Hi 👋',tokens:2});
  send({variant:'current',state:'started',keep:5,text:' there',tokens:3});        // offsets count UTF-16 units
  send({variant:'current',state:'interrupted',tokens:3});
  assert.deepEqual(cards(root,'chat-msg bot').edited,{text:'Hi 👋 there',status:'interrupted'});
  send({variant:'base',state:'started',keep:0,text:'Hel',tokens:1});
  send({variant:'base',state:'failed',tokens:1});
  assert.deepEqual(cards(root,'chat-msg bot').base,{text:'Hel',status:'stopped by an error'});
  held.finish({current:'Hi 👋 there',base:'Hello',chat_template:false,generation});
  await tick();await tick();
  assert.equal(state.keep.run.live.chat,null);
  assert.deepEqual(state.keep.run.chat,[{role:'user',content:'hello'},{role:'assistant',content:'Hi 👋 there'}]);
  assert.deepEqual(state.keep.run.baseChat,[{role:'user',content:'hello'},{role:'assistant',content:'Hello'}]);
  assert.doesNotMatch(text(await view(state)),/generating…|runs after the edited model/);
});

test('a completion started while a chat turn is pending does not take over its live text',async()=>{
  const state=app();state.keep.run={mode:'chat',chat:[],prompt:'ab',n:8,temp:0,result:null,tpl:{user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'}};
  const held={};
  global.fetch=async(url,o)=>{
    if(url==='/api/runtime')return response(runtime);
    const one=held[url]={body:JSON.parse(o.body)};
    return new Promise(resolve=>{one.finish=data=>resolve(response(data));});
  };
  let root=await view(state);
  walk(root).find(n=>n.placeholder==='message').value='hello';
  button(root,'Send').click();await tick();
  state.keep.run.mode='completion';
  button(await view(state),'Run  (Ctrl+Enter)').click();await tick();
  const chat=held['/api/chat'].body.stream,run=held['/api/run'].body.stream;
  assert.notEqual(chat,run);
  onGenerationEvent({type:'generation',stream:chat,variant:'current',state:'started',keep:0,text:'chat reply',tokens:2},state);
  onGenerationEvent({type:'generation',stream:run,variant:'current',state:'started',keep:0,text:'completion',tokens:1},state);
  assert.equal(cards(await view(state),'card').edited.text,'completion');
  held['/api/run'].finish({prompt:'ab',join:'',tokens:['a','b'],base:{next:[],text:'x'},current:{next:[],text:'y'},generation});
  await tick();await tick();
  state.keep.run.mode='chat';
  root=await view(state);
  assert.ok(walk(root).some(n=>n.className==='chat-msg user'&&text(n)==='hello'));
  assert.equal(cards(root,'chat-msg bot').edited.text,'chat reply');
  held['/api/chat'].finish({current:'chat reply.',base:'b',chat_template:false,generation});
  await tick();await tick();
  assert.deepEqual(state.keep.run.live,{chat:null,completion:null});
});

test('a generation event with no request in flight is ignored',()=>{
  const state=app();
  onGenerationEvent({type:'generation',stream:'x',variant:'current',state:'started',keep:0,text:'a',tokens:1},state);
  state.keep.run={mode:'completion'};
  onGenerationEvent({type:'generation',stream:'x',variant:'current',state:'started',keep:0,text:'a',tokens:1},state);
  assert.equal(state.keep.run.live,undefined);
});
