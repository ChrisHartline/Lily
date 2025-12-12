/**
 * Clara Configuration
 *
 * Configure your API endpoints here.
 *
 * For local development:
 *   API_URL: http://localhost:8000
 *   WS_URL: ws://localhost:8000/ws/chat
 *
 * For Modal deployment:
 *   API_URL: https://<username>--clara-api-fastapi-app.modal.run
 *   WS_URL: wss://<username>--clara-api-fastapi-app.modal.run/ws/chat
 */

// Environment detection
const isDevelopment = import.meta.env?.DEV ?? process.env.NODE_ENV === 'development';

// Modal deployment URL (update this after deploying to Modal)
const MODAL_URL = 'https://chrishartline--clara-api-fastapi-app.modal.run';

// Local development URL
const LOCAL_URL = 'http://localhost:8000';

// Select base URL based on environment
const BASE_URL = isDevelopment ? LOCAL_URL : MODAL_URL;

// WebSocket protocol (ws for local, wss for Modal)
const WS_PROTOCOL = isDevelopment ? 'ws' : 'wss';
const HTTP_BASE = isDevelopment ? LOCAL_URL : MODAL_URL;

export const config = {
  // API endpoints
  api: {
    baseUrl: BASE_URL,
    health: `${BASE_URL}/`,
    tools: `${BASE_URL}/api/tools`,
    chat: `${BASE_URL}/api/chat`,
  },

  // WebSocket
  ws: {
    url: isDevelopment
      ? `ws://localhost:8000/ws/chat`
      : `wss://${MODAL_URL.replace('https://', '')}/ws/chat`,
    reconnectDelay: 3000,
    maxReconnectAttempts: 5,
    pingInterval: 30000,
  },

  // Feature flags
  features: {
    tts: true,
    summarization: true,
    webSearch: true,
  },

  // Environment info
  env: {
    isDevelopment,
    isProduction: !isDevelopment,
  },
};

// Export individual values for convenience
export const API_BASE_URL = config.api.baseUrl;
export const WS_URL = config.ws.url;
export const IS_DEV = config.env.isDevelopment;

export default config;
