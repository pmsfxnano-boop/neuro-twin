import {useCallback,useEffect,useMemo,useRef,useState} from 'react';
import {RuntimeApi} from '../lib/api';
import type {CapabilitySet,RuntimeSnapshot} from '../types';

const POLL_MS = 5000;

const empty:RuntimeSnapshot={
  runtime_id:null,state_status:'WAITING_FOR_LIVE_RUNTIME',source:null,subject_id:null,event_id:null,time:null,
  state:null,ci:null,metrics:null,provenance:null,pit:null,oos:null,pet_kinetic_posterior:null,
  pet_to_state_assimilation:null,trajectory:null,prediction:null,evidence:null,evidence_id:null
};

export function useRuntime(base=''){
  const api=useMemo(()=>new RuntimeApi(base),[base]);
  const [runtime,setRuntime]=useState(empty);
  const [caps,setCaps]=useState<CapabilitySet|null>(null);
  const [connected,setConnected]=useState(false);
  const [error,setError]=useState<string|null>(null);
  const ws=useRef<WebSocket|null>(null);

  const refresh=useCallback(async(quiet=false)=>{
    try{
      const [statusResult,capsResult]=await Promise.allSettled([api.status(),api.capabilities()]);
      if(statusResult.status==='rejected'){
        if(!quiet) setError(statusResult.reason instanceof Error ? statusResult.reason.message : 'API unavailable');
        setConnected(false);
        return;
      }
      setRuntime(statusResult.value);
      setConnected(true);
      setError(null);
      if(capsResult.status==='fulfilled') setCaps(capsResult.value);
      else if(!quiet) setError(capsResult.reason instanceof Error ? capsResult.reason.message : 'Capabilities unavailable');
    }catch(e){
      if(!quiet){setConnected(false);setError(e instanceof Error?e.message:'API unavailable');}
    }
  },[api]);

  const open=useCallback(()=>{
    ws.current?.close();
    try{
      const socket=new WebSocket(api.wsUrl());
      ws.current=socket;
      socket.onopen=()=>undefined;
      socket.onclose=()=>undefined;
      socket.onerror=()=>undefined;
      socket.onmessage=e=>{
        try{
          const p=JSON.parse(e.data);
          if(p.type!=='heartbeat') setRuntime(p as RuntimeSnapshot);
        }catch{
          // Optional stream only; HTTP polling remains authoritative.
        }
      };
    }catch{
      // Optional WebSocket stream unavailable; HTTP polling continues.
    }
  },[api]);

  const runRuntime=useCallback(async(payload:unknown)=>{
    try{const r=await api.runRuntime(payload);setRuntime(r.summary);setConnected(true);setError(null);return r;}
    catch(e){setError(e instanceof Error?e.message:'Runtime execution failed');throw e;}
  },[api]);

  const manualRefresh=useCallback(async()=>{
    try{const r=await api.refresh();setRuntime(r.summary);setConnected(true);setError(null);return r.status;}
    catch(e){setError(e instanceof Error?e.message:'Refresh failed');setConnected(false);return 'ERROR';}
  },[api]);

  useEffect(()=>{
    let disposed=false;
    const start=async()=>{await refresh();if(!disposed) open();};
    void start();
    const timer=window.setInterval(()=>{
      if(!disposed && document.visibilityState!=='hidden') void refresh(true);
    },POLL_MS);
    return()=>{disposed=true;window.clearInterval(timer);ws.current?.close();};
  },[refresh,open]);

  return {runtime,caps,connected,error,manualRefresh,runRuntime};
}
