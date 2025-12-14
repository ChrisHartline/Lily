import { useState, useCallback } from 'react';
import type { Message } from '../types';

const CLARA_API_URL = 'https://chrishartline--clara-api-fastapi-app.modal.run';

export type Personality = 'warmth' | 'playful' | 'encouragement';

export function useClaraChat(initialMessages: Message[]) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [personality, setPersonality] = useState<Personality>('warmth');

  const sendMessage = useCallback(async (userMessage: string) => {
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
      const response = await fetch(`${CLARA_API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: userMessage,
          personality: personality,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        sender: 'assistant',
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (e) {
      console.error(e);
      const errorMessage = "Sorry, I encountered an error connecting to Clara. Please try again.";
      setError(errorMessage);
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        text: errorMessage,
        sender: 'assistant'
      }]);
    } finally {
      setIsTyping(false);
    }
  }, [isTyping, personality]);

  return { messages, sendMessage, isTyping, error, personality, setPersonality };
}
