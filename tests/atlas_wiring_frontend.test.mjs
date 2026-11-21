// The wiring stage in the Atlas tab: a time estimate instead of a multiply-add count, one line on memory and what
// waits, a continue card after an interruption, and progress while it runs.  Fake DOM/network only, no model.
import test from 'node:test';
import assert from 'node:assert/strict';
import {atlasView, onAtlasEvent, duration, progressText, estimateLabel} from '../web/js/atlas.js';
class Element {
 constructor(tag) { this.tag=tag; this.nodeType=1; this.children=[]; this.style={}; this.events={}; this.disabled=false; }
 append(...kids) { this.children.push(...kids); }
 replaceChildren(...kids) { this.children=kids; }
 addEventListener(e,f) { this.events[e]=f; }
 setAttribute(k,v) { this[k]=v; }
 click() { return this.events.click?.({target:this}); }
}
const byId={};
global.document={createElement:t=>new Element(t),createTextNode:t=>({nodeType:3,textContent:t}),getElementById:id=>byId[id]||null};
const descendants=(el,tag)=>[el,...(el.children||[]).flatMap(c=>typeof c==='object'?descendants(c,tag):[])].filter(e=>e.tag===tag);
const text=(el)=>typeof el==='string'?el:el.nodeType===3?el.textContent:(el.textContent??'')+(el.children||[]).map(text).join('');
const app=()=>({tab:'Atlas',variant:'current',state:{status:{workspace_id:'w',version:0}},keep:{},rerender(){this.renders=(this.renders||0)+1;}});
const response=d=>({ok:true,headers:{get:()=> 'application/json'},json:async()=>d});
const tick=()=>new Promise(r=>setImmediate(r));
const unbuilt={built:false,can_build:true,can_observe:true,wiring_auto:false,wiring_cost:1.7e15,neurons:2,model:'fake'};
const est=(o={})=>({search_s:6600,read_s:null,read_known:false,read_note:'the weights stage has not been built yet',total_s:null,
 memory_bytes:48.5e9,read_vectors_bytes:45e9,dtype:'float32',threads:12,waits:['Run','Chat','edits'],continuing:null,flops:1.7e15,...o});

function serve(status, estimate) {
 const seen={estimates:[],builds:[]};
 global.fetch=async(path,opt)=>{
  if(opt?.body){ seen.builds.push(JSON.parse(opt.body)); return response({running:true}); }
  if(path.startsWith('/api/atlas/estimate')){ seen.estimates.push(path); return response(typeof estimate==='function'?estimate(path):estimate); }
  return response(status);
 };
 return seen;
}

test('the wiring box shows a time estimate and a memory and waiting line, never the multiply-add count',async()=>{
 const a=app(), root=new Element('root'), seen=serve(unbuilt,est());
 await atlasView(root,a); await tick();
 const all=text(root);
 assert.match(all,/Wiring: which neurons read what which other neurons write \(estimate: about 1 h 50 min, plus reading the weights, not yet known: the weights stage has not been built yet\)/);
 assert.match(all,/While it runs it holds about 49 GB of memory, and Run, Chat and edits wait until it finishes\./);
 assert.doesNotMatch(all,/multiply|1\.7e\+?15|e15/);
 assert.equal(seen.estimates.length,1); assert.match(seen.estimates[0],/variant=base/); assert.match(seen.estimates[0],/restart=0/);
 const boxes=descendants(root,'input'); assert.equal(boxes[0].checked,true); assert.ok(!boxes[1].checked);   // default stays weights only
});

test('a known reading time is included; a model that cannot run says only edits wait',async()=>{
 const a=app(), root=new Element('root');
 serve(unbuilt,est({read_known:true,read_s:1200,total_s:7800,waits:['edits'],memory_bytes:3.2e9,read_note:null}));
 await atlasView(root,a); await tick();
 const all=text(root);
 assert.match(all,/\(estimate: about 2 h 10 min, including reading the weights\)/);
 assert.match(all,/holds about 3\.2 GB of memory, and edits wait until it finishes\./);
 assert.doesNotMatch(all,/Run, Chat/);
});

test('an estimate that fails to come back says so instead of a number',async()=>{
 const a=app(), root=new Element('root');
 serve(unbuilt,{detail:'x'});
 await atlasView(root,a); await tick();
 assert.match(text(root),/\(no time estimate: no estimate came back\)/);
 assert.doesNotMatch(text(root),/multiply/);
});

const built={built:true,can_build:true,can_observe:true,variant:'base',stale_layers:[],stages:{weights:{neurons:10,layers:[0,1,2,3]}},
 layers:{stages:[]},validation:null,shift:null,residual:null,wiring_built:false,mech:[],role:[],
 wiring_progress:{usable:true,reason:null,done:23,total:64,next_layer:23,variant:'base',state:'searching',rebuilds_weights:false,drops_observed:false}};

test('an interrupted wiring stage offers to continue from the saved layer, with the estimate of what is left',async()=>{
 const a=app(), root=new Element('root');
 const seen=serve(built,est({search_s:2400,continuing:{done:23,total:64,next_layer:23}}));
 await atlasView(root,a); await tick();
 const all=text(root);
 assert.match(all,/An earlier wiring build stopped after 23 of 64 layers; this continues from layer 23\./);
 assert.match(all,/\(estimate: about 40 min, plus reading the weights/);
 assert.match(seen.estimates[0],/restart=0/);
 const button=descendants(root,'button').find(b=>text(b)==='Continue the wiring stage');
 button.click(); await tick();
 assert.equal(seen.builds.length,1);
 assert.deepEqual(seen.builds[0].stages,['wiring']); assert.equal(seen.builds[0].force,false); assert.equal(seen.builds[0].variant,'base');
 assert.equal(a.renders,1);
});

test('an interrupted edited-weights stage continues with its weights stage; a rebuild asks for the full estimate',async()=>{
 const a=app(), root=new Element('root');
 const st={...built,stages:{...built.stages,observed:{}},wiring_progress:{...built.wiring_progress,variant:'current',rebuilds_weights:true,drops_observed:true}};
 const seen=serve(st,(p)=>p.includes('restart=1')?est():est({continuing:{done:23,total:64,next_layer:23},weights_rebuild:{seconds:900,drops_observed:true}}));
 await atlasView(root,a); await tick();
 const card=text(root);
 assert.match(card,/\(estimate: about 2 h 5 min, including rebuilding the weights stage, plus reading the weights, not yet known/);
 assert.match(card,/It first rebuilds the weights stage for the edited weights \(about 15 min last time\), and the observed stage then has to be built again\./);
 descendants(root,'button').find(b=>text(b)==='Continue the wiring stage').click(); await tick();
 assert.deepEqual(seen.builds[0].stages,['weights','wiring']); assert.equal(seen.builds[0].variant,'current');
 assert.match(seen.estimates[0],/variant=current/);
 descendants(root,'button').find(b=>text(b)==='Rebuild…').click(); await tick();
 assert.match(seen.estimates.at(-1),/restart=1/);
 const rebuild=text(root); assert.match(rebuild,/Rebuild/); assert.doesNotMatch(rebuild.split('Rebuild')[1],/this continues from layer/);
});

test('a running build shows layers done and time left, and events update it',async()=>{
 const a=app(), root=new Element('root');
 const job={running:true,log:['  neuron partners (about 1.7e+15 multiply-adds; estimate: ...)'],
  progress:{stage:'wiring',total:64,phase:'searching',done:24,layer:23,remaining_s:4200,resumed_from:23}};
 serve({...unbuilt,job},est());
 await atlasView(root,a);
 const p=descendants(root,'p').find(x=>x.id==='atlas-progress');
 assert.equal(text(p),'Wiring: 24 of 64 layers done, about 1 h 10 min left (continued from layer 23)');
 byId['atlas-progress']=p;
 onAtlasEvent({workspace_id:'w',version:0,progress:{stage:'wiring',total:64,phase:'searching',done:30,remaining_s:3000,resumed_from:null}},a);
 assert.equal(p.textContent,'Wiring: 30 of 64 layers done, about 50 min left');
 onAtlasEvent({workspace_id:'other',version:0,progress:{stage:'wiring',total:64,phase:'searching',done:31,remaining_s:10}},a);
 assert.equal(p.textContent,'Wiring: 30 of 64 layers done, about 50 min left');
 delete byId['atlas-progress'];
 assert.equal(a.renders||0,0);
});

test('duration and progress text',()=>{
 assert.equal(duration(null),'an unknown time'); assert.equal(duration(0.4),'under a second');
 assert.equal(duration(40),'about 40 s'); assert.equal(duration(720),'about 12 min');
 assert.equal(duration(6600),'about 1 h 50 min'); assert.equal(duration(7200),'about 2 h');
 assert.equal(progressText({stage:'wiring',phase:'reading',layer:5,read:2,read_total:63,total:64,done:0}),'Wiring: reading the weights of layer 5 (2 of 63)');
 assert.equal(progressText({stage:'wiring',phase:'finishing',total:64,done:64}),'Wiring: all 64 layers done; saving');
 assert.equal(progressText({stage:'wiring',phase:'searching',total:64,done:1,remaining_s:null}),'Wiring: 1 of 64 layers done');
 assert.equal(progressText(null),'');
});

test('saved progress the build would not use is described with its reason, not offered as a continuation',async()=>{
 const a=app(), root=new Element('root');
 const st={...built,wiring_progress:{...built.wiring_progress,usable:false,reason:'it was computed in float32 and this build works in bfloat16'}};
 const seen=serve(st,est({saved:{usable:false,reason:'it was computed in float32 and this build works in bfloat16'}}));
 await atlasView(root,a); await tick();
 const all=text(root);
 assert.match(all,/An earlier wiring build was stopped, but its saved progress cannot be used: it was computed in float32 and this build works in bfloat16\. Building the stage again starts over\./);
 assert.doesNotMatch(all,/continues from layer/);
 assert.ok(!descendants(root,'button').some(b=>text(b)==='Continue the wiring stage'));
 descendants(root,'button').find(b=>text(b)==='Build the wiring stage again').click(); await tick();
 assert.deepEqual(seen.builds[0].stages,['wiring']); assert.equal(seen.builds[0].force,false);
});

test('an edited-weights continuation whose weights-stage time is not known says so',async()=>{
 const e=est({read_known:true,read_s:600,total_s:3600,read_note:null});
 assert.equal(estimateLabel(e,{seconds:null,drops_observed:false}),' (estimate: about 1 h, including reading the weights, plus rebuilding the weights stage, time not known)');
 assert.equal(estimateLabel(e,{seconds:300,drops_observed:false}),' (estimate: about 1 h 5 min, including reading the weights and rebuilding the weights stage)');
 assert.equal(estimateLabel(e),' (estimate: about 1 h, including reading the weights)');
});

test('durations round as the server does',()=>{
 assert.equal(duration(89),'about 89 s'); assert.equal(duration(90),'about 2 min'); assert.equal(duration(151),'about 3 min');
 assert.equal(duration(3569),'about 59 min'); assert.equal(duration(3599),'about 1 h');
 assert.equal(duration(6449),'about 1 h 45 min'); assert.equal(duration(6451),'about 1 h 50 min'); assert.equal(duration(4950),'about 1 h 25 min');
});

test('the Build panel estimates the wiring stage alone, whichever weights are chosen: the weights stage is its own box',async()=>{
 const a=app(), root=new Element('root');
 const seen=serve(unbuilt,(p)=>est(p.includes('variant=current')?{weights_rebuild:{seconds:900,drops_observed:false}}:{}));
 await atlasView(root,a); await tick();
 const select=descendants(root,'select')[0];
 select.value='current'; await select.events.change(); await tick();
 assert.match(seen.estimates.at(-1),/variant=current/);
 const all=text(root);
 assert.match(all,/\(estimate: about 1 h 50 min, plus reading the weights, not yet known: the weights stage has not been built yet\)/);
 assert.doesNotMatch(all,/rebuilding the weights stage|It first rebuilds/);
});

const busyMsg='A build of this atlas is already running in this IDE; wait for it to finish, then build again (it continues where it stopped).';

test('a build started elsewhere is shown, with no Build, Continue or Rebuild offered',async()=>{
 for (const [where,who] of [['this IDE','in this IDE, started by an agent or another window'],['another process','in another process \\(for example the command line\\)']]) {
  const a=app(), root=new Element('root');
  const st={...built,build_running:where,job:null,wiring_progress:{...built.wiring_progress,done:1,total:4,next_layer:1,usable:false,reason:`a build of this atlas is running ${where==='this IDE'?'in this IDE':'in another process'}`}};
  const seen=serve(st,est());
  await atlasView(root,a); await tick();
  const all=text(root);
  assert.match(all,new RegExp(`A build of this atlas is running ${who}\\. Its wiring stage has 1 of 4 layers done so far\\. Run, Chat, edits and inspector reads wait until it finishes`));
  assert.doesNotMatch(all,/An earlier wiring build|Continue the wiring stage|Build the wiring stage again/);
  const labels=descendants(root,'button').map(text);
  assert.deepEqual(labels.filter(l=>/Build|Continue|Rebuild/.test(l)),[]);
  assert.equal(seen.estimates.length,0);
  descendants(root,'button').find(b=>text(b)==='Check again').click(); assert.equal(a.renders,1);
 }
 const a=app(), root=new Element('root');
 serve({...unbuilt,build_running:'another process',job:null},est());
 await atlasView(root,a); await tick();
 assert.match(text(root),/A build of this atlas is running in another process/);
 assert.ok(!descendants(root,'button').some(b=>text(b)==='Build atlas'));
});

test('a browser build that failed shows its error on the overview, built or not',async()=>{
 for (const st of [{...built,job:{running:false,error:busyMsg,log:[]}},{...unbuilt,job:{running:false,error:busyMsg,log:[]}}]) {
  const a=app(), root=new Element('root');
  serve(st,est());
  await atlasView(root,a); await tick();
  const all=text(root);
  assert.equal(all.split('The last atlas build failed: '+busyMsg).length,2);         // once, at the top
 }
});

// The observed stage: the box is offered only when the build would accept it.
const buildButton=(root)=>descendants(root,'button').find(b=>/build/i.test(text(b)));
const cannotRun='This model cannot be run here, so the map will be weights-only and firing signs will be predictions.';
const workerNote='This stage loads and runs the model on about 5,000 tokens of probe text, one window at a time.';

test('a model the server says cannot build the observed stage has the box disabled, the reason shown, and never sends it',async()=>{
 const a=app(), root=new Element('root'), seen=serve({...unbuilt,can_observe:true,observed_unavailable:cannotRun},est());
 await atlasView(root,a); await tick();
 const boxes=descendants(root,'input');
 assert.equal(boxes.length,3); assert.equal(boxes[2].disabled,true); assert.ok(!boxes[2].checked);
 assert.equal(text(root).split(cannotRun).length,2);                                   // the server's sentence, once
 boxes[2].events.change({target:{checked:true}});                                      // a tick that slipped through
 await buildButton(root).click(); await tick();
 assert.equal(seen.builds.length,1); assert.deepEqual(seen.builds[0].stages,['weights']);
});

test('a model that runs in the model worker keeps the box, says what the stage runs, and a tick sends it',async()=>{
 const a=app(), root=new Element('root'), seen=serve({...unbuilt,can_observe:true,observed_unavailable:null,observed_note:workerNote},est());
 await atlasView(root,a); await tick();
 const boxes=descendants(root,'input');
 assert.equal(boxes[2].disabled,false); assert.ok(!boxes[2].checked);                 // offered, not ticked for the person
 assert.equal(text(root).split(workerNote).length,2);                                  // the server's sentence, once
 assert.doesNotMatch(text(root),/not available for this model|cannot be run here|batch/);
 const box=descendants(root,'label')[2];
 assert.equal(box.title,'needs a model that can be run');
 boxes[2].events.change({target:{checked:true}});
 await buildButton(root).click(); await tick();
 assert.deepEqual(seen.builds[0].stages,['weights','observed']);
});

test('a model that can build the observed stage keeps the box, and a tick sends it',async()=>{
 const a=app(), root=new Element('root'), seen=serve({...unbuilt,can_observe:true,observed_unavailable:null,observed_note:null},est());
 await atlasView(root,a); await tick();
 const boxes=descendants(root,'input');
 assert.equal(boxes[2].disabled,false);
 assert.doesNotMatch(text(root),/not available for this model|cannot be run here|one window at a time/);
 boxes[2].events.change({target:{checked:true}});
 await buildButton(root).click(); await tick();
 assert.deepEqual(seen.builds[0].stages,['weights','observed']);
});

test('a model that cannot be run at all keeps its sentence when the server gives no reason',async()=>{
 const a=app(), root=new Element('root');
 serve({...unbuilt,can_observe:false},est());
 await atlasView(root,a); await tick();
 assert.equal(descendants(root,'input')[2].disabled,true);
 assert.match(text(root),/This model cannot be run here, so the map will be weights-only and firing signs will be predictions\./);
});
