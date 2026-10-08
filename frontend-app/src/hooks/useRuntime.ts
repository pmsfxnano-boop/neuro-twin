import {useCallback,useEffect,useMemo,useRef,useState} from 'react';
import {RuntimeApi} from '../lib/api';
import type {CapabilitySet,RuntimeSnapshot} from '../types';

const empty:RuntimeSnapshot={runtime_id:null,state_status:'WAITING_FOR_LIVE_RUNTIME',source:null,subject_id:null,event_id:null,time:null,state:null,ci:null,metrics:null,provenance:null,pit:null,oos:null,pet_kinetic_posterior:null,pet_to_state_assimilation:null,trajectory:null,prediction:null,evidence:null,evidence_id:null};
export function useRuntime(base=''){
  const api=useMemo(()=>new RuntimeApi(base),[base]);
  const [runtime,setRuntime]=useState(empty); const [caps,setCaps]=useState<CapabilitySet|null>(null); const [connected,setConnected]=useState(false); const [error,setError]=useState<string|null>(null); const ws=useRef<WebSocket|null>(null);
  const refresh=useCallback(async()=>{try{const [s,c]=await Promise.all([api.status(),api.capabilities()]);setRuntime(s);setCaps(c);setConnected(true);setError(null);}catch(e){setConnected(false);setError(e instanceof Error?e.message:'Bridge unavailable');}},[api]);
  const open=useCallback(()=>{ws.current?.close();try{const socket=new WebSocket(api.wsUrl());ws.current=socket;socket.onopen=()=>setConnected(true);socket.onclose=()=>setConnected(false);socket.onerror=()=>setError('WebSocket error');socket.onmessage=e=>{try{const p=JSON.parse(e.data);if(p.type!=='heartbeat')setRuntime(p);}catch{setError('Malformed runtime event')}}}catch{setError('WebSocket init failed')}},[api]);
  const runRuntime=useCallback(async(payload:unknown)=>{try{const r=await api.runRuntime(payload);setRuntime(r.summary);setConnected(true);setError(null);return r}catch(e){setError(e instanceof Error?e.message:'Runtime execution failed');throw e}},[api]);
  const manualRefresh=useCallback(async()=>{try{const r=await api.refresh();setRuntime(r.summary);setConnected(true);setError(null);return r.status}catch(e){setError(e instanceof Error?e.message:'Refresh failed');return 'ERROR'}},[api]);
  useEffect(()=>{void refresh().then(open);return()=>ws.current?.close()},[refresh,open]);
  return {runtime,caps,connected,error,manualRefresh,runRuntime};
}
