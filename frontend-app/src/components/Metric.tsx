import type {ReactNode} from 'react';
export function Metric({label,value,meta,tone=''}:{label:string;value:ReactNode;meta:string;tone?:string}){return <section className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{meta}</small></section>}
