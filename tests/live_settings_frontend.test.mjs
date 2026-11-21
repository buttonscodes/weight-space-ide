// What the Settings page says after a save, and its unload control: node --test tests/live_settings_frontend.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { settingsView, savedReport } from '../web/js/settings.js';
const tick=()=>new Promise(r=>setImmediate(r));
class Element {
  constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.style={};this.events={};this.disabled=false;this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  click(){return this.events.click?.({target:this});}
}
global.document={createElement:t=>new Element(t),createTextNode:text=>({nodeType:3,textContent:text})};
const walk=(root)=>[root,...(root.children||[]).flatMap(walk)];
const text=(node)=>(node.textContent||'')+(node.children||[]).map(text).join('');
const button=(root,label)=>walk(root).find(n=>n.tag==='button'&&text(n)===label);
const response=(data)=>({ok:true,headers:{get:()=> 'application/json'},json:async()=>data});
const values={backend:'cpu',context:64,new_tokens:4,resident_seconds:900,threads:4,floor_gib:20,preferred_gib:25,budget_gib:null,ceiling_gib:null,reservation_gib:null,swap_growth_mib:64,compression_growth_mib:1024};
const preview={machine:{total:100*2**30,available:70*2**30},incremental:20*2**30,required:20*2**30,cold_remaining:50*2**30,
  forecast:null,policy:null,warnings:[],errors:[],backend:{selected:'cpu'},context_policy_limit:32768,kernel_token_rows:2048,prefill_chunk:128,effective_context:64,qualification:'Advisory estimate'};
const when={backend:'next_load',context:'next_load',threads:'next_load',new_tokens:'saved',resident_seconds:'saved',floor_gib:'saved',preferred_gib:'saved',budget_gib:'saved',ceiling_gib:'saved',reservation_gib:'saved',swap_growth_mib:'saved',compression_growth_mib:'saved'};
const load={context:64,threads:4,backend:'cpu',selected:'cpu'};
function payload(over={}){return {revision:'initial',csrf:'token',config:'/tmp/fake/config.json',target:'defaults',managed:true,editable:Object.keys(values),
  saved:{values:{...values}},active:{values:{...values},sources:Object.fromEntries(Object.keys(values).map(k=>[k,'config defaults']))},
  overridden:[],changed:[],pending:{},restart_required:false,supervised:true,loaded:null,takes_effect:{...when},
  note:'Saving applies settings to this server as the Takes effect column says.',interactive_defaults:null,preview:structuredClone(preview),...over};}
const applied=(over)=>({now:[],next_load:[],next_start:[],overridden:[],supervised:true,running:false,effect:null,loaded:null,warning:null,...over});
function app(){return {sel:'model',tab:'Settings',keep:{},state:{},rerender(){}};}

test('the report names each change under what happened to it', ()=>{
  assert.deepEqual(savedReport(applied({now:['floor_gib','preferred_gib']}),values),['Applied now: Runtime headroom floor (GiB), Admission headroom reserve (GiB).']);
  const now=savedReport(applied({now:['preferred_gib']}),values).join(' ');
  assert.match(now,/^Applied now: /);assert.doesNotMatch(now,/restart/i);
  const later=savedReport(applied({next_load:['context'],loaded:load}),values).join(' ');
  assert.match(later,/Applies at the next model load: Context tokens\./);
  assert.match(later,/A model is loaded now with context 64, 4 threads, cpu backend; it keeps these until it is unloaded\. Unload it and load it again/);
  assert.match(savedReport(applied({next_load:['threads']}),values).join(' '),/No model is loaded; the next load uses the new values\./);
  const unmanaged=savedReport(applied({supervised:false,now:['new_tokens'],next_start:['floor_gib']}),values).join(' ');
  assert.match(unmanaged,/Saved for the next server start \(this server has no memory supervisor to apply them to\)/);
  assert.match(savedReport(applied({overridden:['floor_gib']}),values).join(' '),/launch override .* still wins/);
  const running=applied({now:['preferred_gib'],running:true,effect:{kind:'after_operation',message:'A model operation is running. It keeps the memory limits it was admitted with; the new limits and idle timeout apply when it finishes.'}});
  assert.match(savedReport(running,values).join(' '),/keeps the memory limits it was admitted with/);
  const unload=applied({now:['resident_seconds'],effect:{kind:'unload',message:'With these values the memory watchdog unloads the loaded model within about a second.'}});
  assert.match(savedReport(unload,values).join(' '),/unloads the loaded model/);
  assert.deepEqual(savedReport(applied({}),values),['Saved. No setting changed.']);
});

test('Takes effect column, save button and no restart claim for a supervised server', async()=>{
  global.fetch=async()=>response(payload({loaded:load}));
  const root=new Element('root');await settingsView(root,app());
  assert.ok(button(root,'Save settings'));assert.equal(button(root,'Save for next server start'),undefined);
  const body=text(root);
  assert.doesNotMatch(body,/restart required|require restart|requires restart|after a server restart/i);
  const context=walk(root).find(n=>n['aria-label']==='Context tokens');
  const row=walk(root).find(n=>n.tag==='tr'&&walk(n).includes(context));
  assert.equal(text(row.children[3]),'Next model load');
  const floor=walk(root).find(n=>n.tag==='tr'&&walk(n).includes(walk(root).find(x=>x['aria-label']==='Runtime headroom floor (GiB)')));
  assert.equal(text(floor.children[3]),'When saved');
  assert.match(body,/A model is loaded with context 64, 4 threads, cpu backend; it keeps these until it is unloaded\./);
});

test('an unmanaged server says memory limits have nothing to apply to', async()=>{
  const data=payload({supervised:false,takes_effect:Object.fromEntries(Object.entries(when).map(([k,v])=>[k,k==='new_tokens'?'saved':'next_start'])),note:'This server has no memory supervisor.'});
  global.fetch=async()=>response(data);
  const root=new Element('root');await settingsView(root,app());
  assert.match(text(root),/no memory supervisor, so memory limits and the idle timeout have nothing to apply to here/);
  const floor=walk(root).find(n=>n.tag==='tr'&&walk(n).includes(walk(root).find(x=>x['aria-label']==='Runtime headroom floor (GiB)')));
  assert.equal(text(floor.children[3]),'Next server start');
});

test('a load setting saved while a model is loaded offers the existing unload route', async()=>{
  const calls=[];let unloadResult=response({ok:true});
  const saved=payload({loaded:load});saved.saved.values.context=32;saved.active.values.context=32;
  saved.applied=applied({next_load:['context'],loaded:load});
  global.fetch=async(path,o)=>{calls.push([path,o]);if(path==='/api/runtime/unload')return unloadResult;return response(o?.method==='POST'?saved:payload({loaded:load}));};
  const root=new Element('root');await settingsView(root,app());
  const input=walk(root).find(n=>n['aria-label']==='Context tokens');input.value='32';input.events.input({target:input});
  button(root,'Save settings').click();await tick();await tick();
  assert.match(text(root),/Applies at the next model load: Context tokens\./);
  const row=walk(root).find(n=>n.tag==='tr'&&walk(n).includes(input));
  assert.equal(text(row.children[2]),'32 (loaded model: 64)');
  const unload=button(root,'Unload model');assert.ok(unload);
  await unload.click();await tick();
  assert.equal(calls.filter(([p])=>p==='/api/runtime/unload').length,1);
  assert.equal(calls.find(([p])=>p==='/api/runtime/unload')[1].method,'POST');
  assert.match(text(root),/Model unloaded\. The next load uses the saved settings\./);
});

test('a save that changes only memory limits offers no unload', async()=>{
  const saved=payload({loaded:load});saved.saved.values.floor_gib=15;saved.active.values.floor_gib=15;
  saved.applied=applied({now:['floor_gib'],loaded:load});
  global.fetch=async(path,o)=>response(o?.method==='POST'?saved:payload({loaded:load}));
  const root=new Element('root');await settingsView(root,app());
  const input=walk(root).find(n=>n['aria-label']==='Runtime headroom floor (GiB)');input.value='15';input.events.input({target:input});
  button(root,'Save settings').click();await tick();await tick();
  assert.match(text(root),/Applied now: Runtime headroom floor/);
  assert.equal(button(root,'Unload model'),undefined);
});

// The review's probe: Save clicked before the draft preview ever ran. The page cannot be relied on to have warned,
// so the server refuses a save that would unload the loaded model until it is confirmed for that very candidate.
const unloading='With these values the memory watchdog unloads the loaded model within about a second.';
const asks=(token)=>({ok:false,status:409,headers:{get:()=> 'application/json'},json:async()=>({detail:{message:`Not saved: ${unloading} Nothing was changed; confirm to save these settings.`,
  effect:{kind:'unload',confirm:true,loaded:true,message:unloading},confirm:token}})});
test('Save before any preview: the server asks first and the page confirms the same candidate', async()=>{
  const calls=[];
  const done=payload({loaded:load});done.saved.values.resident_seconds=10;done.active.values.resident_seconds=10;
  done.applied=applied({now:['resident_seconds'],loaded:load,effect:{kind:'unload',confirm:true,message:unloading}});
  global.fetch=async(path,opt={})=>{calls.push([path,opt.body?JSON.parse(opt.body):null]);
    if(path==='/api/settings'&&opt.method==='POST')return JSON.parse(opt.body).confirm==='t1'?response(done):asks('t1');
    if(path==='/api/settings/preview')return response(payload({loaded:load}));
    return response(payload({loaded:load}));};
  const root=new Element('root');await settingsView(root,app());
  const idle=walk(root).find(n=>n['aria-label']==='Model idle timeout (seconds)');
  idle.events.input({target:{value:'10'}});
  await button(root,'Save settings').click();for(let i=0;i<5;i++)await tick();
  assert.deepEqual(calls.map(([p])=>p),['/api/settings','/api/settings'],'the save went before any preview');
  assert.deepEqual(calls[1][1],{revision:'initial',changes:{resident_seconds:10}});
  assert.match(text(root),/Not saved: With these values the memory watchdog unloads the loaded model/);
  const yes=button(root,'Save and unload the model');assert.ok(yes);assert.ok(button(root,'Cancel'));
  await yes.click();for(let i=0;i<5;i++)await tick();
  assert.deepEqual(calls.at(-1)[1],{revision:'initial',changes:{resident_seconds:10},confirm:'t1'});
  assert.match(text(root),/Applied now: Model idle timeout/);assert.match(text(root),/unloads the loaded model/);
  assert.equal(button(root,'Save and unload the model'),undefined);
  await new Promise(r=>setTimeout(r,300));        // the cancelled preview timer never fires
  assert.equal(calls.filter(([p])=>p==='/api/settings/preview').length,0);
});

test('a confirmation is dropped when the draft changes, and Cancel saves nothing', async()=>{
  const calls=[];
  global.fetch=async(path,opt={})=>{calls.push([path,opt.body?JSON.parse(opt.body):null]);
    if(path==='/api/settings'&&opt.method==='POST')return asks('t2');
    return response(payload({loaded:load}));};
  const root=new Element('root');await settingsView(root,app());
  const idle=walk(root).find(n=>n['aria-label']==='Model idle timeout (seconds)');
  idle.events.input({target:{value:'10'}});
  await button(root,'Save settings').click();for(let i=0;i<5;i++)await tick();
  await button(root,'Cancel').click();
  assert.match(text(root),/Not saved\. The draft is unchanged\./);assert.equal(button(root,'Save and unload the model'),undefined);
  await button(root,'Save settings').click();for(let i=0;i<5;i++)await tick();
  assert.ok(button(root,'Save and unload the model'));
  idle.events.input({target:{value:'12'}});                                       // another draft
  assert.equal(button(root,'Save and unload the model'),undefined);
  assert.equal(calls.filter(([,b])=>b&&b.confirm).length,0);
});

test('a page whose server does not know what the memory watchdog runs says so', async()=>{
  global.fetch=async()=>response(payload({out_of_step:'Not saved: it may be running the unsaved values until the next accepted save or a restart.'}));
  const root=new Element('root');await settingsView(root,app());
  const alert=walk(root).find(n=>n.role==='alert');
  assert.equal(alert.style.display,'block');
  assert.match(text(alert),/may be running the unsaved values until the next accepted save or a restart\. Until an accepted save, Active now may not be what the memory watchdog runs\./);
  global.fetch=async()=>response(payload({}));
  const clean=new Element('root');await settingsView(clean,app());
  assert.equal(walk(clean).find(n=>n.role==='alert').style.display,'none');
});

test('a confirmation that meets a changed model shows the new warning and asks again', async()=>{
  const calls=[];
  const changed={ok:false,status:409,headers:{get:()=> 'application/json'},json:async()=>({detail:{message:'Not saved: A model operation is running. When it finishes, the memory watchdog would then unload the loaded model. The model or the running operation changed since you confirmed. Nothing was changed; confirm to save these settings.',
    effect:{kind:'after_operation',unload:true,confirm:true,loaded:true},confirm:'t2'}})};
  const done=payload({loaded:load});done.applied=applied({now:['resident_seconds'],loaded:load});
  global.fetch=async(path,opt={})=>{calls.push([path,opt.body?JSON.parse(opt.body):null]);
    if(path==='/api/settings'&&opt.method==='POST'){const c=JSON.parse(opt.body).confirm;return c==='t2'?response(done):c==='t1'?changed:asks('t1');}
    return response(payload({loaded:load}));};
  const root=new Element('root');await settingsView(root,app());
  walk(root).find(n=>n['aria-label']==='Model idle timeout (seconds)').events.input({target:{value:'10'}});
  await button(root,'Save settings').click();for(let i=0;i<5;i++)await tick();
  await button(root,'Save and unload the model').click();for(let i=0;i<5;i++)await tick();
  assert.match(text(root),/changed since you confirmed/);
  await button(root,'Save and unload the model').click();for(let i=0;i<5;i++)await tick();
  assert.deepEqual(calls.filter(([p,b])=>p==='/api/settings'&&b).map(([,b])=>b.confirm??null),[null,'t1','t2']);
  assert.match(text(root),/Applied now: Model idle timeout/);
});
