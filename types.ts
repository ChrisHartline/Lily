
export type Message = {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  delivered?: boolean;
};

export type View = 'chat' | 'files' | 'settings';
