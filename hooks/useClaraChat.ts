
import { useState, useRef, useEffect, useCallback } from 'react';
import type { Message } from '../types';

// WebSocket configuration
const WS_URL = 'ws://localhost:8000/ws/chat';
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

// Message types from/to server
interface ServerMessage {
  type: 'message' | 'typing' | 'tool_result' | 'session' | 'error' | 'pong';
  content: string;
  session_id?: string;
  metadata?: {
    routing?: {
      brain?: string;
      domain?: string;
      confidence?: number;
    };
    tool_name?: string;
    success?: boolean;
    result?: any;
    error?: string;
    tools?: string[];
    clara_available?: boolean;
  };
}

interface ClientMessage {
  type: 'message' | 'tool_call' | 'ping';
  content?: string;
  session_id?: string;
  tool_name?: string;
  tool_args?: Record<string, any>;
}

interface ToolInfo {
  name: string;
  description: string;
  category: string;
}

interface ClaraChatState {
  messages: Message[];
  isTyping: boolean;
  isConnected: boolean;
  error: string | null;
  sessionId: string | null;
  availableTools: string[];
  claraAvailable: boolean;
}

export function useClaraChat(initialMessages: Message[] = []) {
  const [state, setState] = useState<ClaraChatState>({
    messages: initialMessages,
    isTyping: false,
    isConnected: false,
    error: null,
    sessionId: null,
    availableTools: [],
    claraAvailable: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      console.log('[Clara] Connecting to WebSocket...');
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('[Clara] Connected');
        reconnectAttempts.current = 0;
        setState(prev => ({
          ...prev,
          isConnected: true,
          error: null,
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data: ServerMessage = JSON.parse(event.data);
          handleServerMessage(data);
        } catch (e) {
          console.error('[Clara] Failed to parse message:', e);
        }
      };

      ws.onclose = (event) => {
        console.log('[Clara] Disconnected:', event.code, event.reason);
        setState(prev => ({ ...prev, isConnected: false }));
        wsRef.current = null;

        // Attempt reconnection
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          console.log(`[Clara] Reconnecting (attempt ${reconnectAttempts.current})...`);
          reconnectTimeout.current = setTimeout(connect, RECONNECT_DELAY);
        } else {
          setState(prev => ({
            ...prev,
            error: 'Connection lost. Please refresh the page.',
          }));
        }
      };

      ws.onerror = (error) => {
        console.error('[Clara] WebSocket error:', error);
        setState(prev => ({
          ...prev,
          error: 'Connection error. Retrying...',
        }));
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('[Clara] Failed to create WebSocket:', e);
      setState(prev => ({
        ...prev,
        error: 'Failed to connect to Clara. Is the backend running?',
      }));
    }
  }, []);

  // Handle messages from server
  const handleServerMessage = useCallback((data: ServerMessage) => {
    switch (data.type) {
      case 'session':
        setState(prev => ({
          ...prev,
          sessionId: data.session_id || null,
          availableTools: data.metadata?.tools || [],
          claraAvailable: data.metadata?.clara_available || false,
        }));
        break;

      case 'typing':
        setState(prev => ({ ...prev, isTyping: true }));
        break;

      case 'message':
        setState(prev => {
          const newMessage: Message = {
            id: Date.now().toString(),
            text: data.content,
            sender: 'assistant',
            delivered: true,
          };
          return {
            ...prev,
            messages: [...prev.messages, newMessage],
            isTyping: false,
          };
        });
        break;

      case 'tool_result':
        // Handle tool results - could show in UI or inject into chat
        console.log('[Clara] Tool result:', data.metadata);
        if (data.metadata?.success && data.metadata?.tool_name === 'text_to_speech') {
          // Special handling for TTS - could trigger audio playback
          const result = data.metadata.result;
          if (result?.audio_base64) {
            playAudioBase64(result.audio_base64);
          }
        }
        break;

      case 'error':
        setState(prev => ({
          ...prev,
          error: data.content,
          isTyping: false,
        }));
        // Also add error as a message
        setState(prev => ({
          ...prev,
          messages: [...prev.messages, {
            id: Date.now().toString(),
            text: `Error: ${data.content}`,
            sender: 'assistant',
          }],
        }));
        break;

      case 'pong':
        // Heartbeat response - no action needed
        break;

      default:
        console.log('[Clara] Unknown message type:', data.type);
    }
  }, []);

  // Play audio from base64
  const playAudioBase64 = (base64: string) => {
    try {
      const audio = new Audio(`data:audio/mp3;base64,${base64}`);
      audio.play();
    } catch (e) {
      console.error('[Clara] Failed to play audio:', e);
    }
  };

  // Send a message
  const sendMessage = useCallback((text: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setState(prev => ({
        ...prev,
        error: 'Not connected. Trying to reconnect...',
      }));
      connect();
      return;
    }

    if (!text.trim() || state.isTyping) {
      return;
    }

    // Add user message to state
    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      sender: 'user',
      delivered: true,
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      error: null,
    }));

    // Send to server
    const message: ClientMessage = {
      type: 'message',
      content: text.trim(),
      session_id: state.sessionId || undefined,
    };

    wsRef.current.send(JSON.stringify(message));
  }, [state.isTyping, state.sessionId, connect]);

  // Execute a tool
  const executeTool = useCallback((toolName: string, args: Record<string, any> = {}) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setState(prev => ({
        ...prev,
        error: 'Not connected',
      }));
      return;
    }

    const message: ClientMessage = {
      type: 'tool_call',
      tool_name: toolName,
      tool_args: args,
      session_id: state.sessionId || undefined,
    };

    wsRef.current.send(JSON.stringify(message));
  }, [state.sessionId]);

  // Request TTS for last assistant message
  const speakLastMessage = useCallback(() => {
    const lastAssistantMessage = [...state.messages]
      .reverse()
      .find(m => m.sender === 'assistant');

    if (lastAssistantMessage) {
      executeTool('text_to_speech', { text: lastAssistantMessage.text });
    }
  }, [state.messages, executeTool]);

  // Connect on mount
  useEffect(() => {
    connect();

    // Cleanup on unmount
    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Heartbeat to keep connection alive
  useEffect(() => {
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    return () => clearInterval(pingInterval);
  }, []);

  return {
    messages: state.messages,
    sendMessage,
    isTyping: state.isTyping,
    isConnected: state.isConnected,
    error: state.error,
    sessionId: state.sessionId,
    availableTools: state.availableTools,
    claraAvailable: state.claraAvailable,
    executeTool,
    speakLastMessage,
    reconnect: connect,
  };
}
