const CACHE='neuro-twin-v5-shell';
const isApiPath=(pathname)=>pathname.startsWith('/v1/')||pathname==='/health'||pathname==='/ws';

self.addEventListener('install',event=>{
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate',event=>{
  event.waitUntil((async()=>{
    const keys=await caches.keys();
    await Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET') return;
  const url=new URL(event.request.url);
  if(url.origin!==self.location.origin||isApiPath(url.pathname)) return;

  event.respondWith((async()=>{
    try{
      const response=await fetch(event.request);
      if(response.ok){
        const cache=await caches.open(CACHE);
        await cache.put(event.request,response.clone());
      }
      return response;
    }catch{
      const cached=await caches.match(event.request);
      if(cached) return cached;
      throw new Error('offline and no cached resource');
    }
  })());
});
