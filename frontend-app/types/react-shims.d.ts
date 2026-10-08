declare module 'react' {
  export type ReactNode = any;
  export type ChangeEvent<T=any> = { target: T };
  export function useState<T>(v: T): [T, (x: T | ((x: T) => T)) => void];
  export function useMemo<T>(f: () => T, d: any[]): T;
  export function useCallback<T extends (...a:any[])=>any>(f:T,d:any[]):T;
  export function useEffect(f: () => void | (() => void), d?: any[]): void;
  export function useRef<T>(v:T): { current:T };
  export const StrictMode: any;
}
declare module 'react-dom/client' { export function createRoot(e:Element):{render(n:any):void}; }
declare module 'react/jsx-runtime' { export const Fragment:any; export function jsx(...a:any[]):any; export function jsxs(...a:any[]):any; }
