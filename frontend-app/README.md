# NEURO-TWIN Frontier Console v5

React + TypeScript + Vite research UI for the executable NEURO-TWIN runtime.

## Important

This is not a static dashboard. The **Data & Run** screen accepts a runtime package containing canonical observations and an explicit versioned observation operator. The browser submits that package to `POST /v1/runtime/run`; the Python scientific engine executes the P-I-N-Q inference and the result returns to the console through REST/WebSocket.

No synthetic/reference execution is exposed in the production UI.

## Development

```bash
npm install
npm run dev
```

## Validation

```bash
npm run check
npm run build
```

The environment used to construct this artifact did not have the React/Vite/Three.js packages installed locally, so the TypeScript source was type-checked with local shims but a production Vite bundle was not claimed as executed.
