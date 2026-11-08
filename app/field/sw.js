/* ShelfSense Field service worker: makes the app installable and lets it open without a connection.
   Only the app's own files are cached. API answers, videos and evidence images are never cached
   (they are private and must be fresh); the page itself queues uploads while offline. */
const CACHE = "shelfsense-field-v1";
const SHELL = ["/field/", "/field/manifest.webmanifest", "/field/icons/icon-192.png", "/field/icons/icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || !url.pathname.startsWith("/field/")) return;
  // network first (always the newest app when online), the cached copy when offline
  e.respondWith(fetch(e.request).then(r => {
    if (r.ok) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request).then(r => r || caches.match("/field/"))));
});
