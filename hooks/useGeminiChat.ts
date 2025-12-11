
import { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Chat } from '@google/genai';
import type { Message } from '../types';

// Safely access process.env to prevent crashing in browser environments where it's not defined.
const API_KEY = typeof process !== 'undefined' && process.env ? process.env.API_KEY : undefined;

export function useGeminiChat(initialMessages: Message[]) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatRef = useRef<Chat | null>(null);

  useEffect(() => {
    if (!API_KEY) {
      const errorMessage = "API key is not configured. Please set the API_KEY environment variable.";
      console.error(errorMessage);
      setError(errorMessage);
      setMessages(prev => {
        if (prev.some(m => m.id === 'error-init')) return prev;
        return [...prev, { id: 'error-init', text: errorMessage, sender: 'assistant' }];
      });
      return;
    }
    try {
      const ai = new GoogleGenAI({ apiKey: API_KEY, vertexai: true });
      chatRef.current = ai.chats.create({
        model: 'gemini-2.5-flash',
        config: {
          systemInstruction: 'You are Aria, a helpful and friendly AI assistant. Keep your responses concise and helpful.',
        },
      });
    } catch (e) {
      console.error(e);
      const errorMessage = "Failed to initialize the AI model.";
      setError(errorMessage);
      setMessages(prev => {
        if (prev.some(m => m.id === 'error-init-catch')) return prev;
        return [...prev, { id: 'error-init-catch', text: errorMessage, sender: 'assistant' }];
      });
    }
  }, []);

  const sendMessage = useCallback(async (userMessage: string) => {
    if (!chatRef.current) {
        const errorMsg = "Chat is not initialized. The API key may be missing or invalid.";
        if (!messages.some(m => m.text === errorMsg)) {
            console.error(errorMsg);
            setError(errorMsg);
            setMessages(prev => [...prev, { id: 'error-chat-not-init', text: errorMsg, sender: 'assistant' }]);
        }
        return;
    }
    if (isTyping) return;

    setIsTyping(true);
    setError(null);

    const newUserMessage: Message = {
      id: Date.now().toString(),
      text: userMessage,
      sender: 'user',
      delivered: true,
    };
    setMessages(prev => [...prev, newUserMessage]);

    try {
      const stream = await chatRef.current.sendMessageStream({ message: userMessage });
      
      let firstChunk = true;
      let assistantMessageId = '';
      let fullResponse = '';

      for await (const chunk of stream) {
        const chunkText = chunk.text;
        if (chunkText) {
            if (firstChunk) {
                setIsTyping(false);
                assistantMessageId = (Date.now() + 1).toString();
                setMessages(prev => [...prev, { id: assistantMessageId, text: '', sender: 'assistant' }]);
                firstChunk = false;
            }
            fullResponse += chunkText;
            setMessages(prev => prev.map(msg => 
                msg.id === assistantMessageId ? { ...msg, text: fullResponse } : msg
            ));
        }
      }
      if (firstChunk) { // Handle cases where the stream is empty
        setIsTyping(false);
      }

    } catch (e) {
      console.error(e);
      const errorMessage = "Sorry, I encountered an error. Please try again.";
      setError(errorMessage);
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), text: errorMessage, sender: 'assistant' }]);
      setIsTyping(false);
    }
  }, [isTyping, messages]);

  return { messages, sendMessage, isTyping, error };
}
