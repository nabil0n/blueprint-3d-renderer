import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import '@fontsource/ibm-plex-mono/600.css'
import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/500.css'
import '@fontsource/ibm-plex-sans/600.css'
import './index.css'
import App from './App.tsx'

// Room labels are drawn once onto canvas textures, so their font must be loaded first.
// The fonts are bundled, so this is quick; if it fails the labels fall back to a system font.
const LABEL_FONTS = ['600 24px "IBM Plex Mono"', '400 20px "IBM Plex Mono"']

void Promise.allSettled(LABEL_FONTS.map((font) => document.fonts.load(font))).then(() =>
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  ),
)
