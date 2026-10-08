declare interface ImportMetaEnv { readonly [key: string]: string | boolean | undefined }
declare interface ImportMeta { readonly env: ImportMetaEnv }
declare module 'vite/client' {}
