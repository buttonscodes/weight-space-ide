// Fake DOM/network only, no model.
import test from 'node:test';
import assert from 'node:assert/strict';
import {atlasView, onAtlasEvent} from '../web/js/atlas.js';
import {inspection} from '../web/js/inspection.js';
class Element {
 constructor(tag) { this.tag=tag; this.nodeType=1; this.children=[]; this.style={}; this.events={}; this.disabled=false; }
 append(...kids) { this.children.push(...kids); }
 replaceChildren(...kids) { this.children=kids; }
 addEventListener(e,f) { this.events[e]=f; }
 setAttribute(k,v) { this[k]=v; }
 click() { return this.events.click?.({target:this}); }
}
global.document={createElement:t=>new Element(t),createTextNode:t=>({nodeType:3,textContent:t}),getElementById:()=>null};
const descendants=(el,tag)=>[el,...(el.children||[]).flatMap(c=>descendants(c,tag))].filter(e=>e.tag===tag);
const app=()=>({tab:'Atlas',variant:'current',state:{status:{workspace_id:'w',version:0}},keep:{atlas:{tab:'Attribute',attr:{prompt:'x',target:'y',k:5,scale:0}}},rerender(){this.renders=(this.renders||0)+1;}});
const response=d=>({ok:true,headers:{get:()=> 'application/json'},json:async()=>d});
const tick=()=>new Promise(r=>setImmediate(r));

test('remembered attribution passive; repeat clicks shared; navigation completion refreshes busy button',async()=>{
 const a=app(), root=new Element('root'); let calls=0, resolve;
 global.fetch=async()=>response({built:true,can_observe:true,stale_layers:[]});
 inspection.entries.clear(); inspection.scope=null;
 inspection.send=()=>{calls++;return new Promise(r=>resolve=r);};
 await atlasView(root,a); await tick(); assert.equal(calls,0);
 const button=descendants(root,'button').at(-1);
 button.click(); button.click(); await tick(); assert.equal(calls,1); assert.equal(button.disabled,true);
 const next=new Element('root'); await atlasView(next,a); assert.equal(descendants(next,'button').at(-1).disabled,true);
 resolve({by_layer:{},top:[]}); await tick(); assert.equal(a.renders,1); assert.equal(a.keep.atlas.attr.busy,false);
 onAtlasEvent({workspace_id:'old',version:0,done:true},a); assert.equal(a.renders,1);
});

test('build starts weights-only and blocks rapid repeated clicks',async()=>{
 const a=app(), root=new Element('root'); let body, resolve, calls=0;
 global.fetch=async(path,opt)=>{
  if(!opt?.body) return response({built:false,can_build:true,can_observe:true,wiring_auto:true,neurons:2,model:'fake'});
  calls++; body=JSON.parse(opt.body); await new Promise(r=>resolve=r); return response({running:true});
 };
 await atlasView(root,a);
 const boxes=descendants(root,'input'); assert.equal(boxes[0].checked,true); assert.ok(!boxes[1].checked && !boxes[2].checked);
 const button=descendants(root,'button')[0]; button.click(); button.click(); await tick();
 assert.equal(calls,1); assert.deepEqual(body.stages,['weights']); assert.equal(body.workspace_id,'w');
 resolve(); await tick(); assert.equal(a.renders,1);
});
