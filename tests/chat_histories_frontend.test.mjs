// Model-free Chat UI: a conversation per model, and one message to both or a separate message to each.
import test from 'node:test';
import assert from 'node:assert/strict';
import { runView, onGenerationEvent } from '../web/js/run.js';
const tick=()=>new Promise(r=>setImmediate(r));
const settle=async()=>{await tick();await tick();await tick();};
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  click(){this.events.click?.({target:this});}
  focus(){global.document.activeElement=this;}
}
global.document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text}),activeElement:null,documentElement:{dataset:{theme:'light'}}};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const buttons=(root,label)=>walk(root).filter(n=>n.tag==='button'&&text(n)===label);
const labelled=(root,label)=>walk(root).find(n=>n['aria-label']===label);
const placeholder=(root,p)=>walk(root).find(n=>n.tag==='input'&&n.placeholder===p);
const response=(data,ok=true)=>({ok,statusText:'Bad Request',headers:{get:()=> 'application/json'},json:async()=>data});
const runtime={generation:{default_output:16,output_limit:64,context_limit:128}};
const generation={requested_output:8,effective_output:8,prompt_tokens:2,context:128};
const TPL={user:'User: ',assistant:'Assistant: ',system:'',sep:'\n'};
function app(run={}){return {sel:'model',tab:'Run',variant:'current',keep:{run:{mode:'chat',chat:[],tpl:TPL,...run}},selected:[],
  state:{open:{name:'fixture',kind:'wsp'},status:{can_run:true,version:0}},rerender(){},select(){}};}
const entries=(...list)=>list.map(([addr,mode,dose,unit='resid',on=true])=>({addr,mode,dose:String(dose),unit,on}));
// The chat request is held until the test answers it; stop requests and GETs answer at once.
function server(){
  const held={bodies:[],stops:[]};
  global.fetch=async(url,o={})=>{
    if(url==='/api/generation/stop'){held.stops.push(JSON.parse(o.body));return response({ok:true,found:true});}
    if(url!=='/api/chat')return response(runtime);
    held.body=JSON.parse(o.body);held.bodies.push(held.body);
    return new Promise(resolve=>{held.finish=(data,ok=true)=>resolve(response(data,ok));});
  };
  return held;
}
async function view(state){const root=new Element('root');await runView(root,state);return root;}
// The "Send:" toggle: a group of two buttons, the mode in force pressed.
const mode=(root)=>labelled(root,'Send one message to both models, or a separate message to each');
const CHOICE={both:'One message to both models',separate:'A separate message to each model'};
const choice=(root,v)=>mode(root).children.find(b=>b.tag==='button'&&text(b)===CHOICE[v]);
const choose=(root,v)=>choice(root,v).click();
const pressed=(root)=>mode(root).children.filter(b=>b['aria-pressed']==='true').map(text);
async function separate(state){const root=await view(state);choose(root,'separate');return view(state);}
const BOX={base:'message to the base model',current:'message to the edited model'};
async function sendTo(state,v,message){
  const root=await view(state),inp=placeholder(root,BOX[v]);
  inp.value=message;inp.events.input({target:inp});
  labelled(root,`Send to the ${v==='base'?'base':'edited'} model`).click();await tick();
  return root;
}
// The column that holds an input: the element whose child row contains it.
const column=(root,inp)=>walk(root).find(n=>(n.children||[]).some(r=>r.className==='row'&&(r.children||[]).includes(inp)));
const conversation=(root,name)=>labelled(root,`${name} conversation`);
const done=(fields)=>({chat_template:false,generation,...fields});

test('one message to both: the second turn sends each model its own earlier reply',async()=>{
  const state=app(),held=server();
  let root=await view(state);
  placeholder(root,'message').value='hi';buttons(root,'Send')[0].click();await tick();
  assert.deepEqual(held.body.messages,[{role:'user',content:'hi'}]);
  assert.deepEqual(held.body.base_messages,[{role:'user',content:'hi'}]);
  assert.equal(held.body.variants,undefined);                      // both models, the default
  held.finish(done({current:'edited one',base:'base one',variants:{current:'complete',base:'complete'}}));await settle();
  root=await view(state);
  placeholder(root,'message').value='more';buttons(root,'Send')[0].click();await tick();
  assert.deepEqual(held.body.messages.map(m=>m.content),['hi','edited one','more']);
  assert.deepEqual(held.body.base_messages.map(m=>m.content),['hi','base one','more']);
  assert.deepEqual(held.body.messages.map(m=>m.role),['user','assistant','user']);
  held.finish(done({current:'edited two',base:'base two',variants:{current:'complete',base:'complete'}}));await settle();
  assert.deepEqual(state.keep.run.chat.map(m=>m.content),['hi','edited one','more','edited two']);
  assert.deepEqual(state.keep.run.baseChat.map(m=>m.content),['hi','base one','more','base two']);
  assert.ok(!/answers the edited conversation/.test(text(await view(state))));
});

test('the choice can be made before the first message, and back again',async()=>{
  const state=app();server();
  let root=await view(state);
  const toggle=mode(root);
  assert.ok(toggle&&toggle.role==='group');
  assert.deepEqual(toggle.children.map(b=>[b.tag,text(b),b['aria-pressed'],b.className,!!b.disabled]),
    [['button','One message to both models','true','on',false],['button','A separate message to each model','false','',false]]);
  assert.ok(placeholder(root,'message')&&!placeholder(root,BOX.base));
  root=await separate(state);
  assert.equal(state.keep.run.chatSplit,true);
  assert.equal(placeholder(root,'message'),undefined);
  assert.ok(placeholder(root,BOX.base)&&placeholder(root,BOX.current));
  assert.equal(buttons(root,'Send').length,2);
  assert.ok(labelled(root,'Send to the base model')&&labelled(root,'Send to the edited model'));
  assert.deepEqual(pressed(root),[CHOICE.separate]);
  choose(root,'both');
  root=await view(state);
  assert.equal(state.keep.run.chatSplit,false);
  assert.ok(placeholder(root,'message'));assert.equal(buttons(root,'Send').length,1);
});

test('a separate base send runs the base model alone, unsteered, even with a steering problem',async()=>{
  const state=app(),held=server();
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L3H4','add','0.5'])};   // add pushes neurons only
  await separate(state);
  await sendTo(state,'base','hello base');
  assert.deepEqual(held.body.variants,['base']);
  assert.deepEqual(held.body.base_messages,[{role:'user',content:'hello base'}]);
  for(const k of ['messages','steer','p50','recalibrate'])assert.equal(held.body[k],undefined,k);
  held.finish(done({current:'',base:'base reply',variants:{base:'complete'},generations:{base:generation}}));await settle();
  assert.deepEqual(state.keep.run.baseChat,[{role:'user',content:'hello base'},{role:'assistant',content:'base reply'}]);
  assert.deepEqual(state.keep.run.chat,[]);
  // The same problem refuses an edited send, under the edited column, and nothing is sent.
  const root=await sendTo(state,'current','hello edited');
  assert.equal(held.bodies.length,1);
  const fresh=await view(state),alert=walk(column(fresh,placeholder(fresh,BOX.current))).find(n=>n.role==='alert');
  assert.match(text(alert),/Steering: L3H4: add pushes neurons only/);
  assert.ok(!walk(column(fresh,placeholder(fresh,BOX.base))).some(n=>n.role==='alert'&&/Steering/.test(text(n))));
  assert.equal(placeholder(root,BOX.current).value,'hello edited');
});

test('a separate edited send runs the edited model alone, with its steering',async()=>{
  const state=app(),held=server();
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L1N2','add','-0.5','raw'])};
  await separate(state);
  await sendTo(state,'current','hello');
  assert.deepEqual(held.body.variants,['current']);
  assert.deepEqual(held.body.messages,[{role:'user',content:'hello'}]);
  assert.equal(held.body.base_messages,undefined);
  assert.deepEqual(held.body.steer,[{addr:'L1N2',mode:'add',dose:-0.5,unit:'raw'}]);
  held.finish(done({current:'yo',base:'',variants:{current:'complete'},steering:{variant:'current',entries:[{addr:'L1N2',mode:'add',dose:-0.5,unit:'raw'}]}}));await settle();
  assert.deepEqual(state.keep.run.chat,[{role:'user',content:'hello'},{role:'assistant',content:'yo',steer:'L1N2 -0.5 raw'}]);
  assert.deepEqual(state.keep.run.baseChat,[]);
  const root=await view(state);
  assert.match(text(conversation(root,'edited')),/steered: L1N2 -0\.5 raw/);
});

test('while one column runs both Sends are disabled and Stop is at that column; Stop restores its draft only',async()=>{
  const state=app({chat:[{role:'user',content:'a'},{role:'assistant',content:'A'}],baseChat:[{role:'user',content:'a'},{role:'assistant',content:'B'}]}),held=server();
  await separate(state);
  let root=await view(state);
  placeholder(root,BOX.current).events.input({target:{value:'edited draft'}});
  await sendTo(state,'base','base message');
  root=await view(state);
  assert.ok(buttons(root,'Send').every(b=>b.disabled)&&buttons(root,'Sending…').length===1);
  assert.ok(mode(root).children.length===2&&mode(root).children.every(b=>b.disabled));
  choose(root,'both');assert.equal(state.keep.run.chatSplit,true);   // pressing it while a request runs changes nothing
  const baseColumn=column(root,placeholder(root,BOX.base)),editedColumn=column(root,placeholder(root,BOX.current));
  assert.ok(buttons(baseColumn,'Stop').length===1&&buttons(editedColumn,'Stop').length===0);
  assert.ok(buttons(baseColumn,'Sending…').length===1);
  // The pending base turn is live in the base column only; base runs alone, so it does not wait for the edited model.
  assert.deepEqual(walk(conversation(root,'base')).filter(n=>n.role==='status').map(text),['starting…']);
  assert.doesNotMatch(text(root),/runs after the edited model/);
  onGenerationEvent({type:'generation',stream:held.body.stream,variant:'base',state:'started',keep:0,text:'Par',tokens:1},state);
  root=await view(state);
  assert.match(text(conversation(root,'base')),/base message.*Par/);
  assert.doesNotMatch(text(conversation(root,'edited')),/base message|Par/);
  assert.doesNotMatch(text(root),/runs after the edited model/);
  buttons(column(root,placeholder(root,BOX.base)),'Stop')[0].click();await tick();
  assert.deepEqual(held.stops,[{stream:held.body.stream}]);
  held.finish(done({stopped:true,current:'',base:'Par',variants:{base:'stopped'},tokens:{base:1}}));await settle();
  assert.deepEqual(state.keep.run.baseChat,[{role:'user',content:'a'},{role:'assistant',content:'B'}]);
  assert.deepEqual(state.keep.run.chat,[{role:'user',content:'a'},{role:'assistant',content:'A'}]);
  root=await view(state);
  assert.equal(placeholder(root,BOX.base).value,'base message');
  assert.equal(placeholder(root,BOX.current).value,'edited draft');
  assert.ok(buttons(root,'Send').every(b=>!b.disabled));
  assert.match(text(column(root,placeholder(root,BOX.base))),/Stopped\. The turn was not added; your message is back in the box\./);
  assert.doesNotMatch(text(column(root,placeholder(root,BOX.current))),/Stopped/);
  await new Promise(r=>setTimeout(r,0));
  assert.equal(global.document.activeElement,placeholder(root,BOX.base));
  // Two different messages cannot become one: switching to one message is refused with the reason, and both stay.
  choose(root,'both');
  root=await view(state);
  assert.equal(state.keep.run.chatSplit,true);
  assert.match(text(root),/The two boxes hold different messages\. Send or clear one of them before switching to one message to both\./);
  assert.equal(placeholder(root,BOX.base).value,'base message');assert.equal(placeholder(root,BOX.current).value,'edited draft');
  assert.deepEqual(pressed(root),[CHOICE.separate]);                 // the toggle shows the mode still in force
  assert.doesNotMatch(text(root),/Stopped\./);
});

test('Enter in a column sends that column; switching the choice carries what is typed to the boxes shown',async()=>{
  const state=app({chatDraft:'shared draft'}),held=server();
  let root=await separate(state);
  // The one message was meant for both models, so it is in both column boxes, and the one box no longer holds it.
  assert.equal(placeholder(root,BOX.base).value,'shared draft');assert.equal(placeholder(root,BOX.current).value,'shared draft');
  assert.equal(state.keep.run.chatDraft,'');
  placeholder(root,BOX.base).events.input({target:{value:'b'}});
  const inp=placeholder(root,BOX.current);inp.value='e';inp.events.input({target:inp});
  inp.events.keydown({key:'Enter'});await tick();
  assert.deepEqual(held.body.variants,['current']);assert.deepEqual(held.body.messages,[{role:'user',content:'e'}]);
  held.finish(done({current:'E',base:'',variants:{current:'complete'}}));await settle();
  root=await view(state);
  assert.equal(placeholder(root,BOX.current).value,'');assert.equal(placeholder(root,BOX.base).value,'b');
  choose(root,'both');
  root=await view(state);
  assert.equal(placeholder(root,'message').value,'b');
  assert.deepEqual(state.keep.run.chatDrafts,{current:'',base:''});
});

test('aligned conversations show each message across; diverged ones show two columns, top to bottom',async()=>{
  const both=[{role:'user',content:'one'}];
  const state=app({chat:[...both,{role:'assistant',content:'E1'}],baseChat:[...both,{role:'assistant',content:'B1'}]});server();
  let root=await view(state);
  const top=(r)=>walk(r).filter(n=>n.className==='chat-msg user');
  assert.equal(top(root).length,1);
  const pair=walk(root).find(n=>n.className==='split'&&n.children.every(c=>c.className==='chat-msg bot'));
  assert.deepEqual(pair.children.map(text),['baseB1','editedE1']);
  assert.equal(conversation(root,'base'),undefined);
  assert.doesNotMatch(text(root),/One message still goes to both/);
  // The base model is asked something the edited one was not: two conversations.
  state.keep.run.baseChat.push({role:'user',content:'only base'},{role:'assistant',content:'B2'});
  root=await view(state);
  assert.deepEqual(walk(conversation(root,'base')).filter(n=>/^chat-msg/.test(n.className||'')).map(n=>text(n)),['one','baseB1','only base','baseB2']);
  assert.deepEqual(walk(conversation(root,'edited')).filter(n=>/^chat-msg/.test(n.className||'')).map(n=>text(n)),['one','editedE1']);
  assert.match(text(root),/One message still goes to both conversations\./);
  // In separate mode the boxes sit under their columns and the one-message note is not shown.
  root=await separate(state);
  assert.doesNotMatch(text(root),/One message still goes to both/);
  const split=walk(root).filter(n=>n.className==='split').at(-1);
  assert.ok(walk(split.children[0]).includes(placeholder(root,BOX.base))&&walk(split.children[1]).includes(placeholder(root,BOX.current)));
});

test('one message to both after they diverged still sends each model its own conversation',async()=>{
  const state=app({chat:[{role:'user',content:'e'},{role:'assistant',content:'E'}],baseChat:[]}),held=server();
  const root=await view(state);
  placeholder(root,'message').value='both now';buttons(root,'Send')[0].click();await tick();
  assert.deepEqual(held.body.messages.map(m=>m.content),['e','E','both now']);
  assert.deepEqual(held.body.base_messages.map(m=>m.content),['both now']);
  // The pending turn shows at the bottom of each column.
  const live=await view(state);
  assert.match(text(conversation(live,'base')),/both now/);assert.match(text(conversation(live,'edited')),/E.*both now/);
  held.finish(done({current:'E2',base:'B2',variants:{current:'complete',base:'complete'}}));await settle();
  assert.deepEqual(state.keep.run.baseChat,[{role:'user',content:'both now'},{role:'assistant',content:'B2'}]);
});

test('clear empties both conversations and the notices',async()=>{
  const state=app({chat:[{role:'user',content:'a'},{role:'assistant',content:'A'}],baseChat:[{role:'user',content:'b'},{role:'assistant',content:'B'}],
    chatNotice:'Stopped. The turn was not added; your message is back in the box.',chatError:'Request failed: no.',chatPartial:{current:'x',base:'',variants:{current:'interrupted',base:'not_run'}},
    chatGenerations:{base:{requested_output:8,effective_output:2}}});server();
  let root=await view(state);
  assert.match(text(root),/Stopped\.|Request failed/);
  buttons(root,'clear')[0].click();
  assert.deepEqual([state.keep.run.chat,state.keep.run.baseChat],[[],[]]);
  root=await view(state);
  assert.doesNotMatch(text(root),/Stopped\.|Request failed|Interrupted|ceiling|\bA\b|\bB\b/);
});

test('the response ceiling is told per model when the two prompts differ, and once when they agree',async()=>{
  const state=app(),held=server();
  let root=await view(state);
  placeholder(root,'message').value='hi';buttons(root,'Send')[0].click();await tick();
  const g=(effective,prompt)=>({requested_output:8,effective_output:effective,prompt_tokens:prompt,context:128});
  held.finish(done({current:'E',base:'B',variants:{current:'complete',base:'complete'},generation:g(3,125),generations:{current:g(3,125),base:g(8,40)}}));await settle();
  root=await view(state);
  assert.match(text(root),/Last edited response ceiling: 3 tokens after fitting the edited model's full history in context\./);
  assert.doesNotMatch(text(root),/Last base response ceiling|Last response ceiling/);
  placeholder(root,'message').value='again';buttons(root,'Send')[0].click();await tick();
  held.finish(done({current:'E',base:'B',variants:{current:'complete',base:'complete'},generation:g(1,127),generations:{current:g(1,127),base:g(1,127)}}));await settle();
  assert.match(text(await view(state)),/Last response ceiling: 1 token after fitting the full history in context\./);
});

test('an interrupted base-only send shows only the base part, and a refused send says so at its column',async()=>{
  const state=app(),held=server();
  await separate(state);
  await sendTo(state,'base','hi');
  held.finish(done({interrupted:true,event:{code:'runtime_budget',action:'abort',message:'Synthetic stop'},current:'',base:'Ha',variants:{base:'interrupted'}}));await settle();
  let root=await view(state);
  const card=walk(root).find(n=>n.className==='card'&&/Interrupted — partial output/.test(text(n)));
  assert.match(text(card),/Base: Ha/);assert.doesNotMatch(text(card),/Edited/);
  assert.deepEqual(state.keep.run.baseChat,[]);
  assert.equal(placeholder(root,BOX.base).value,'hi');
  // A server refusal appears under the column whose Send was pressed, and the message is kept.
  await sendTo(state,'base','hi');
  held.finish({error:"ValueError: The base model's conversation: Prompt/history uses 130 tokens"},false);await settle();
  root=await view(state);
  const alert=walk(column(root,placeholder(root,BOX.base))).find(n=>n.role==='alert');
  assert.match(text(alert),/Request failed: ValueError: The base model's conversation: Prompt\/history uses 130 tokens\. Your message is preserved for retry\./);
  assert.ok(!walk(column(root,placeholder(root,BOX.current))).some(n=>n.role==='alert'&&text(n)));
  assert.equal(placeholder(root,BOX.base).value,'hi');
});

// ------------------------------------------------------------------ review follow-ups
const msgs=(root)=>walk(root).filter(n=>/^chat-msg/.test(n.className||'')).map(text);
const statuses=(root)=>walk(root).filter(n=>n.role==='status').map(text);

test('a column that catches up shows its message once, with a pending reply only for the model answering it',async()=>{
  const state=app(),held=server();
  await separate(state);
  await sendTo(state,'current','same');
  held.finish(done({current:'E',base:'',variants:{current:'complete'},generations:{current:generation}}));await settle();
  await sendTo(state,'base','same');                      // now both conversations hold "same": aligned
  let root=await view(state);
  assert.equal(conversation(root,'base'),undefined);
  assert.deepEqual(msgs(root),['same','basestarting…','editedE']);  // the message once; the base reply pending, the edited one shown
  assert.deepEqual(statuses(root),['starting…']);         // no pending box for the edited model, which is not running
  held.finish(done({current:'',base:'B',variants:{base:'complete'},generations:{base:generation}}));await settle();
  assert.deepEqual(msgs(await view(state)),['same','baseB','editedE']);
});

test('a column one message behind that sends the same message joins the aligned view without repeating it',async()=>{
  const state=app({chat:[{role:'user',content:'a'},{role:'assistant',content:'A'},{role:'user',content:'b'},{role:'assistant',content:'Bx'}],
    baseChat:[{role:'user',content:'a'},{role:'assistant',content:'A0'}]}),held=server();
  await separate(state);
  let root=await view(state);
  assert.ok(conversation(root,'base'));                   // diverged before the send
  await sendTo(state,'base','b');
  root=await view(state);
  assert.deepEqual(msgs(root),['a','baseA0','editedA','b','basestarting…','editedBx']);
  assert.deepEqual(statuses(root),['starting…']);
  held.finish(done({current:'',base:'B0',variants:{base:'complete'},generations:{base:generation}}));await settle();
  assert.deepEqual(msgs(await view(state)),['a','baseA0','editedA','b','baseB0','editedBx']);
});

test('conversations of the same length with different messages are shown as two columns',async()=>{
  const state=app({chat:[{role:'user',content:'to edited'},{role:'assistant',content:'E'}],baseChat:[{role:'user',content:'to base'},{role:'assistant',content:'B'}]});server();
  const root=await view(state);
  assert.deepEqual(msgs(conversation(root,'base')),['to base','baseB']);
  assert.deepEqual(msgs(conversation(root,'edited')),['to edited','editedE']);
  assert.match(text(root),/One message still goes to both conversations\./);
});

test('after a stop or a failure in a column, switching to one message brings the message to the one box and drops the notice',async()=>{
  const state=app(),held=server();
  await separate(state);
  await sendTo(state,'base','lost?');
  let root=await view(state);
  buttons(root,'Stop')[0].click();await tick();
  held.finish(done({stopped:true,current:'',base:'x',variants:{base:'stopped'},tokens:{base:1}}));await settle();
  root=await view(state);
  assert.match(text(column(root,placeholder(root,BOX.base))),/your message is back in the box/);
  choose(root,'both');
  root=await view(state);
  assert.equal(placeholder(root,'message').value,'lost?');
  assert.doesNotMatch(text(root),/Stopped\.|back in the box/);
  // The same for a request that failed: the message reaches the box now shown, and the old error is gone.
  await separate(state);
  assert.equal(placeholder(await view(state),BOX.current).value,'lost?');
  await sendTo(state,'current','retry me');
  held.finish({error:'ValueError: no'},false);await settle();
  root=await view(state);
  assert.match(text(column(root,placeholder(root,BOX.current))),/Your message is preserved for retry/);
  choose(root,'both');
  root=await view(state);
  assert.doesNotMatch(text(root),/preserved for retry|Request failed/);
  // The base box still held "lost?" and the edited box "retry me": the refusal keeps both.
  assert.equal(state.keep.run.chatSplit,true);
  assert.match(text(root),/The two boxes hold different messages/);
});

test('after a stop with one message to both, switching to separate messages puts the message in both column boxes',async()=>{
  const state=app(),held=server();
  let root=await view(state);
  placeholder(root,'message').value='q';buttons(root,'Send')[0].click();await tick();
  root=await view(state);
  buttons(root,'Stop')[0].click();await tick();
  held.finish(done({stopped:true,current:'E',base:'',variants:{current:'stopped',base:'not_run'},tokens:{current:1}}));await settle();
  root=await view(state);
  assert.equal(placeholder(root,'message').value,'q');assert.match(text(root),/back in the box/);
  root=await separate(state);
  assert.equal(placeholder(root,BOX.base).value,'q');assert.equal(placeholder(root,BOX.current).value,'q');
  assert.doesNotMatch(text(root),/back in the box/);
  assert.equal(state.keep.run.chatDraft,'');
});

test('a notice from one column does not appear under the other when the other refuses a send',async()=>{
  const state=app(),held=server();
  state.keep.steer={on:true,recipe:'',p50:'',entries:entries(['L3H4','add','0.5'])};   // refuses edited sends only
  await separate(state);
  await sendTo(state,'base','b');
  let root=await view(state);
  buttons(root,'Stop')[0].click();await tick();
  held.finish(done({stopped:true,current:'',base:'x',variants:{base:'stopped'},tokens:{base:1}}));await settle();
  root=await view(state);
  assert.match(text(column(root,placeholder(root,BOX.base))),/Stopped\./);
  await sendTo(state,'current','e');                      // refused before any request
  assert.equal(held.bodies.length,1);
  root=await view(state);
  assert.match(text(column(root,placeholder(root,BOX.current))),/Steering: L3H4/);
  assert.doesNotMatch(text(root),/Stopped\./);
});

test('a stopped edited send puts the focus back in the edited box',async()=>{
  const state=app(),held=server();
  await separate(state);
  await sendTo(state,'current','edited message');
  let root=await view(state);
  buttons(column(root,placeholder(root,BOX.current)),'Stop')[0].click();await tick();
  held.finish(done({stopped:true,current:'E',base:'',variants:{current:'stopped'},tokens:{current:1}}));await settle();
  root=await view(state);
  await new Promise(r=>setTimeout(r,0));
  assert.equal(placeholder(root,BOX.current).value,'edited message');
  assert.equal(global.document.activeElement,placeholder(root,BOX.current));
  assert.notEqual(global.document.activeElement,placeholder(root,BOX.base));
});
