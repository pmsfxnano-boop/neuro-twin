import type {ReactNode} from 'react';
export function Panel({title,kicker,children,wide=false}:{title:string;kicker:string;children:ReactNode;wide?:boolean}){return <section className={`panel ${wide?'wide':''}`}><header className="panel-head"><div><span>{kicker}</span><h2>{title}</h2></div><em>RUNTIME</em></header>{children}</section>}
