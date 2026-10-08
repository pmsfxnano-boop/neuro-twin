import {useMemo,useState} from 'react';
import type * as React from 'react';
import {useRuntime} from '../hooks/useRuntime';
import {Panel} from '../components/Panel';
import {Metric} from '../components/Metric';
import {StatusPill} from '../components/StatusPill';
import {Sparkline} from '../components/Sparkline';
import {MatrixHeatmap} from '../components/MatrixHeatmap';
import {ThreeStateManifold} from '../components/ThreeStateManifold';
import type {StateKey} from '../types';
import '../styles/tokens.css';
import '../styles/app.css';

type View='runtime'|'overview'|'trajectory'|'imaging'|'inference'|'evidence';
const order:StateKey[]=['P','I','N','Q'];
const labels:Record<StateKey,string>={P:'Pathology',I:'Inflammatory',N:'Neuroaxonal',Q:'Functional'};
const fmt=(v:string|null|undefined)=>v?new Date(v).toISOString().replace('T',' ').replace('.000Z','Z'):'—';

const runtimeSchema={
  dataset_id:'REQUIRED',dataset_version:'REQUIRED',source_name:'REQUIRED',source_version:'REQUIRED',access_tier:'public_dataset|controlled_dataset|public_api',subject_id:'REQUIRED',event_id:'LATEST_EVENT_ID',processing_pipeline:'REQUIRED',processing_version:'REQUIRED',retrieval_uri:null,raw_hashes:[],feature_hashes:[],observations:[],
  operator:{name:'EXPLICIT_MODEL_NAME',version:'1.0.0',specs:[{name:'OBSERVATION_FEATURE',weights:[1,0,0,0],bias:0,sigma:1}]},
  initial_state:[0,0,0,0],initial_state_covariance:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],initial_theta:Array(11).fill(0),initial_theta_covariance:null,
  theta_lower:null,theta_upper:null,x0_lower:null,x0_upper:null,oos_fraction:0.8,analysis_mode:'retrospective',pit_as_of:null,model_input_raw_hash:null,pet_kinetic_posterior:null,pet_neuro_binding:null
};

function readTextFile(file:File):Promise<string>{return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result??''));r.onerror=()=>reject(r.error??new Error('file read failed'));r.readAsText(file);});}

export default function App(){
  const [view,setView]=useState<View>('runtime');
  const [origin,setOrigin]=useState('');
  const [payload,setPayload]=useState<Record<string,unknown>|null>(null);
  const [fileName,setFileName]=useState('');
  const [parseError,setParseError]=useState<string|null>(null);
  const [running,setRunning]=useState(false);
  const [runMessage,setRunMessage]=useState<string|null>(null);
  const {runtime,caps,connected,error,manualRefresh,runRuntime}=useRuntime(origin);
  const live=runtime.state_status==='LIVE_RESEARCH';
  const s=runtime.state;
  const ci=runtime.ci;
  const traj=runtime.trajectory?.states;
  const pet=runtime.pet_kinetic_posterior;
  const assim=runtime.pet_to_state_assimilation;
  const src=useMemo(()=>runtime.provenance?`${runtime.provenance.source_name} / ${runtime.provenance.dataset_id}`:'NO LIVE SOURCE',[runtime.provenance]);

  const loadPackage=async(file:File)=>{
    setParseError(null);setRunMessage(null);setFileName(file.name);
    try{
      const text=await readTextFile(file);
      const parsed=JSON.parse(text);
      if(!parsed || typeof parsed!=='object' || Array.isArray(parsed)) throw new Error('Run package must be a JSON object');
      if(!Array.isArray(parsed.observations) || !parsed.operator) throw new Error('Package must include observations[] and operator');
      if(parsed.runtime_class==='synthetic_reference' || parsed.runtime_class==='SYNTHETIC_REFERENCE') throw new Error('Synthetic/reference runtime packages are blocked');
      setPayload(parsed as Record<string,unknown>);
    }catch(e){setPayload(null);setParseError(e instanceof Error?e.message:'Invalid JSON');}
  };

  const execute=async()=>{
    if(!payload)return;
    setRunning(true);setRunMessage(null);setParseError(null);
    try{
      await runRuntime(payload);setRunMessage('REAL SCIENTIFIC RUNTIME PUBLISHED');setView('overview');
    }catch(e){setRunMessage(e instanceof Error?e.message:'Runtime execution failed');}
    finally{setRunning(false);}
  };

  const downloadSchema=()=>{const blob=new Blob([JSON.stringify(runtimeSchema,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='neuro_twin_runtime_package_schema.json';a.click();URL.revokeObjectURL(url);};

  return <div className="shell">
    <aside className="side">
      <div className="brand"><div className="logo">NT</div><div><b>NEURO-TWIN</b><small>FRONTIER RESEARCH CONSOLE</small></div></div>
      <StatusPill label={live?'LIVE RESEARCH':'WAITING FOR DATA'} active={live} danger={!!error}/>
      <nav><span>RUNTIME</span><button onClick={()=>setView('runtime')} className={view==='runtime'?'active':''}><em>00</em>data & run</button><span>OBSERVATORY</span>{(['overview','trajectory','imaging'] as View[]).map((v,i)=><button key={v} onClick={()=>setView(v)} className={view===v?'active':''}><em>0{i+1}</em>{v}</button>)}<span>MODEL</span>{(['inference','evidence'] as View[]).map((v,i)=><button key={v} onClick={()=>setView(v)} className={view===v?'active':''}><em>0{i+4}</em>{v}</button>)}</nav>
      <footer><div>Runtime <b>{runtime.runtime_id?runtime.runtime_id.slice(0,12):'—'}</b></div><div>PIT <b>{runtime.pit?'VALIDATED':'—'}</b></div><div>Synthetic path <b className="red">BLOCKED</b></div></footer>
    </aside>
    <main className="main">
      <header className="top"><div><span>SCIENTIFIC RUNTIME</span><b>{src}</b></div><div className="actions"><input value={origin} onChange={(e:React.ChangeEvent<HTMLInputElement>)=>setOrigin(e.target.value)} placeholder="same-origin bridge"/><StatusPill label={connected?'BRIDGE ONLINE':'BRIDGE OFFLINE'} active={connected} danger={!!error}/><button onClick={()=>void manualRefresh()}>Refresh runtime</button></div></header>
      {error&&<div className="error">{error}</div>}
      <div className="integrity"><div><span>PIT AS-OF</span><b>{fmt(runtime.pit?.as_of)}</b></div><div><span>MODEL</span><b>{runtime.provenance?.model_version??'—'}</b></div><div><span>PROCESSING</span><b>{runtime.provenance?.processing_version??'—'}</b></div><div><span>OOS</span><b className={runtime.oos?.status==='PASS'?'green':''}>{runtime.oos?.status??'NOT REPORTED'}</b></div><div><span>LEAKAGE</span><b className={!runtime.oos||!runtime.oos.temporal_leakage?'green':'red'}>{runtime.oos?String(runtime.oos.temporal_leakage).toUpperCase():'—'}</b></div></div>

      {view==='runtime'&&<div className="grid">
        <Panel title="Data Intake & Runtime Execution" kicker="REAL INPUT → REAL MOTOR" wide>
          <div className="intake">
            <div className="intake-copy"><strong>Load a scientific runtime package</strong><p>Upload canonical observations plus an explicit, versioned observation operator and initial conditions. The backend will run the P-I-N-Q engine, temporal OOS split, Laplace uncertainty and publish the result.</p><div className="row-actions"><input type="file" accept="application/json,.json" onChange={(e: React.ChangeEvent<HTMLInputElement>)=>{const f=e.target.files?.[0];if(f)void loadPackage(f)}}/><button onClick={downloadSchema}>Download schema</button></div>{fileName&&<div className="filetag">{fileName}</div>}{parseError&&<div className="error">{parseError}</div>}{payload&&<div className="ready">Package parsed · {Array.isArray(payload.observations)?payload.observations.length:0} observations · operator present</div>}</div>
            <div className="runbox"><div><span>ENGINE GATE</span><b>{payload?'READY':'WAITING FOR DATA'}</b></div><div><span>PYTHON MOTOR</span><b>{caps?.pinq?'AVAILABLE':'—'}</b></div><div><span>OOS</span><b>STRICT TEMPORAL</b></div><button className="run" disabled={!payload||running} onClick={()=>void execute()}>{running?'RUNNING SCIENTIFIC ENGINE…':'RUN FULL RUNTIME'}</button>{runMessage&&<small className={runMessage.includes('PUBLISHED')?'green':'red'}>{runMessage}</small>}</div>
          </div>
        </Panel>
        <Panel title="Runtime pipeline" kicker="EXECUTION GRAPH" wide><div className="pipeline"><span>INGEST</span><i>→</i><span>SCHEMA/QC</span><i>→</i><span>OBSERVATION MODEL</span><i>→</i><span>P-I-N-Q</span><i>→</i><span>STATE + θ</span><i>→</i><span>UNCERTAINTY</span><i>→</i><span>OOS</span><i>→</i><span>EVIDENCE</span></div></Panel>
      </div>}

      {view!=='runtime'&&<>
        <section className="hero"><div><span>NEURO-TWIN / {live?'LIVE POSTERIOR':'NO LIVE POSTERIOR'}</span><h1>Latent neurodegenerative state, treated as a scientific object.</h1><p>Every number on the console is downstream of the runtime contract. No synthetic state is fabricated in the production UI.</p></div><div className="vector"><span>STATE VECTOR</span><strong>{order.map(k=><code key={k}>{k}={s?.[k]!=null?Number(s[k]).toFixed(3):'—'}</code>)}</strong><small>{runtime.subject_id??'No subject loaded'} {runtime.event_id?`· ${runtime.event_id}`:''}</small></div></section>
        <div className="metrics"><Metric label="OBSERVATIONS" value={runtime.metrics?.observations??'—'} meta="canonical inputs"/><Metric label="PET UNCERTAINTY" value={runtime.metrics?.pet_uncertainty_channels??'—'} meta={pet?.kinetic_model??'no PET posterior'} tone="cyan"/><Metric label="STATE TRACE" value={runtime.metrics?.state_trace!=null?runtime.metrics.state_trace.toExponential(2):'—'} meta="posterior dispersion"/><Metric label="OOS GATE" value={runtime.oos?.status??'—'} meta="temporal validation" tone={runtime.oos?.status==='PASS'?'green':'warn'}/></div>
        {view==='overview'&&<div className="grid"><Panel title="Latent state manifold" kicker="3D / GPU INSPECTION" wide>{live?<ThreeStateManifold runtime={runtime}/>:<div className="waiting"><strong>WAITING FOR REAL RUNTIME</strong><span>Upload data and execute the scientific engine to materialize the 3D posterior.</span></div>}</Panel><Panel title="Posterior state" kicker="P-I-N-Q / 4D"><div className="state-list">{order.map(k=><div className="state" key={k}><b className={`s-${k}`}>{k}</b><div><strong>{labels[k]}</strong><i><u style={{width:s?.[k]!=null?`${Math.min(100,Math.max(3,Number(s[k])*100))}%`:'0%'}}/></i></div><strong>{s?.[k]!=null?Number(s[k]).toFixed(4):'—'}<small> ± {ci?.[k]!=null?Number(ci[k]).toFixed(4):'—'}</small></strong></div>)}</div></Panel><Panel title="Longitudinal state" kicker="STATE-SPACE / OOS"><Sparkline values={traj?.P??[]}/><div className="footline">{traj?.P?.length??0} published points · future observations excluded by PIT</div></Panel><Panel title="Evidence integrity" kicker="PROVENANCE / GATES"><div className="rows"><div>Runtime class <b>{live?'RESEARCH OBSERVATIONAL':'WAITING'}</b></div><div>Raw provenance <b className={runtime.provenance?.raw_hashes?.length?'green':''}>{runtime.provenance?.raw_hashes?.length?'PRESENT':'—'}</b></div><div>Observation hash <b>{runtime.provenance?.observation_hash?.slice(0,16)??'—'}</b></div><div>Result hash <b>{runtime.provenance?.result_hash?.slice(0,16)??'—'}</b></div><div>Synthetic path <b className="red">BLOCKED</b></div></div></Panel></div>}
        {view==='trajectory'&&<div className="grid two"><Panel title="P trajectory" kicker="POSTERIOR"><Sparkline values={traj?.P??[]}/></Panel><Panel title="I trajectory" kicker="POSTERIOR"><Sparkline values={traj?.I??[]}/></Panel><Panel title="N trajectory" kicker="POSTERIOR"><Sparkline values={traj?.N??[]}/></Panel><Panel title="Q trajectory" kicker="POSTERIOR"><Sparkline values={traj?.Q??[]}/></Panel></div>}
        {view==='imaging'&&<div className="grid two"><Panel title="PET kinetic posterior" kicker="FRAME-INTEGRATED / AIF">{pet?<div className="pet">{pet.names.map((n,i)=><div key={n}><span>{n}</span><b>{Number(pet.mean[i]).toPrecision(6)}</b><small>sd {Math.sqrt(Math.max(0,pet.covariance[i]?.[i]??0)).toPrecision(4)}</small></div>)}</div>:<div className="empty">No PET posterior published.</div>}</Panel><Panel title="PET → P/I/N/Q uncertainty" kicker="R_eff = R_kin + HθΣθHθᵀ">{assim?<><MatrixHeatmap matrix={assim.parameter_uncertainty_contribution}/><div className="chips">{assim.assumptions.map(a=><span key={a}>{a}</span>)}</div></>:<div className="empty">No registered PET→Neuro update.</div>}</Panel></div>}
        {view==='inference'&&<div className="grid two"><Panel title="Runtime capabilities" kicker="CONTRACT">{caps?<div className="rows">{Object.entries(caps).map(([k,v])=><div key={k}>{k}<b className={v===true?'green':''}>{String(v).toUpperCase()}</b></div>)}</div>:<div className="empty">Capabilities unavailable.</div>}</Panel><Panel title="Assimilation gain" kicker="LOCAL GAUSSIAN UPDATE"><MatrixHeatmap matrix={assim?.kalman_gain??null}/></Panel></div>}
        {view==='evidence'&&<div className="grid"><Panel title="Immutable provenance chain" kicker="REPRODUCIBILITY" wide>{runtime.provenance?<div className="rows evidence">{Object.entries(runtime.provenance).map(([k,v])=><div key={k}><span>{k}</span><b>{Array.isArray(v)?v.join(' · '):String(v??'—')}</b></div>)}</div>:<div className="empty">No evidence published.</div>}</Panel><Panel title="Point-in-time envelope" kicker="TEMPORAL INTEGRITY"><div className="rows">{runtime.pit?Object.entries(runtime.pit).map(([k,v])=><div key={k}><span>{k}</span><b>{String(v)}</b></div>):<div className="empty">No PIT envelope.</div>}</div></Panel></div>}
      </>}
    </main>
  </div>
}
