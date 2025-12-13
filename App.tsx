
import React, { useState, useRef, useEffect } from 'react';
import type { Message, View } from './types';
import { useClaraChat } from './hooks/useClaraChat.ts';
import {
  ArrowLeftIcon, SettingsIcon, MicrophoneIcon, SendIcon, ChevronRightIcon, FolderIcon, PdfIcon, XlsIcon,
  PhotoIcon, DocumentIcon, AudioIcon, TtsIcon, SummarizeIcon, CameraIcon, VolumeIcon, ChatBubbleIcon
} from './components/Icons.tsx';

const initialMessages: Message[] = [
  {
    id: '1',
    text: "Hello! I'm Clara, your warm and caring AI assistant. How can I help you today?",
    sender: 'assistant',
  },
];

// --- Helper Components ---

const ChatBackground = () => (
  <div
    className="absolute inset-0 z-0"
    style={{
      backgroundColor: '#0A0E1A',
      backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23101623' fill-opacity='0.4'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
    }}
  />
);

const Header: React.FC<{ title: string; showBackButton: boolean; onBack: () => void; onSettings?: () => void; showSettingsButton?: boolean; }> = ({ title, showBackButton, onBack, onSettings, showSettingsButton = false }) => (
  <div className="relative z-10 flex items-center justify-between p-4 bg-aria-dark/80 backdrop-blur-sm border-b border-white/10 flex-shrink-0">
    {showBackButton ? (
      <button onClick={onBack} className="p-2 -ml-2" aria-label="Go back">
        <ArrowLeftIcon className="w-6 h-6 text-white" />
      </button>
    ) : (
      <div className="w-10"></div> // Placeholder for alignment
    )}
    <div className="text-center">
      <h1 className="text-lg font-semibold text-white">{title}</h1>
      {title === 'Clara' && <div className="flex items-center justify-center space-x-1.5">
        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
        <p className="text-xs text-green-400">Online</p>
      </div>}
    </div>
    {showSettingsButton ? (
      <button onClick={onSettings} className="p-2 -mr-2" aria-label="Open settings">
        <SettingsIcon className="w-6 h-6 text-white" />
      </button>
    ) : <div className="w-10"></div>}
  </div>
);

const ToggleSwitch: React.FC<{ enabled: boolean; setEnabled: (e: boolean) => void; }> = ({ enabled, setEnabled }) => (
  <button
    onClick={() => setEnabled(!enabled)}
    className={`${enabled ? 'bg-aria-blue' : 'bg-aria-light-gray'} relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none`}
    role="switch"
    aria-checked={enabled}
  >
    <span
      className={`${enabled ? 'translate-x-5' : 'translate-x-0'} pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
    />
  </button>
);

// --- View Components ---

const ChatView: React.FC<{ setView: (v: View) => void; avatarUrl: string; wallpaperUrl: string }> = ({ setView, avatarUrl, wallpaperUrl }) => {
  const { messages, sendMessage, isTyping } = useClaraChat(initialMessages);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = () => {
    if (input.trim()) {
      sendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className="flex flex-col h-full bg-aria-dark">
      <Header title="Clara" showBackButton={false} onBack={() => {}} onSettings={() => setView('settings')} showSettingsButton={true} />
      <div className="flex-1 overflow-y-auto p-4 relative">
        {wallpaperUrl ? (
          <div className="absolute inset-0 z-0 bg-cover bg-center" style={{ backgroundImage: `url("${wallpaperUrl}")` }} />
        ) : (
          <ChatBackground />
        )}
        <div className="relative z-10 flex flex-col space-y-4">
          <div className="flex justify-center my-8">
            <img src={avatarUrl} alt="Clara Avatar" className="w-24 h-24 rounded-full border-2 border-white/20 shadow-lg" />
          </div>
          <div className="text-center text-xs text-aria-gray mb-4">Today, 10:23 AM</div>
          {messages.map((msg) => (
            <div key={msg.id} className={`flex items-end gap-2 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.sender === 'assistant' && <img src={avatarUrl} alt="Clara Avatar" className="w-8 h-8 rounded-full flex-shrink-0" />}
              <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${msg.sender === 'user' ? 'bg-aria-blue text-white rounded-br-lg' : 'bg-aria-dark-blue text-gray-200 rounded-bl-lg'}`}>
                <p className="text-sm">{msg.text}</p>
                {msg.sender === 'user' && msg.delivered && <p className="text-right text-xs text-blue-200 mt-1">Delivered</p>}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex items-end gap-2 justify-start">
              <img src={avatarUrl} alt="Clara Avatar" className="w-8 h-8 rounded-full flex-shrink-0" />
              <div className="bg-aria-dark-blue px-4 py-3 rounded-2xl rounded-bl-lg">
                <div className="flex items-center space-x-1">
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-0"></span>
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-200"></span>
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-400"></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <div className="p-4 bg-aria-dark/80 backdrop-blur-sm border-t border-white/10">
        <div className="flex items-center bg-aria-dark-blue rounded-full p-2">
          <button className="p-2 text-aria-gray" aria-label="Voice input options">
            <MicrophoneIcon className="w-6 h-6" />
          </button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Message Clara..."
            className="flex-1 bg-transparent text-white placeholder-aria-gray focus:outline-none px-2"
          />
          <button onClick={handleSend} className="w-10 h-10 rounded-full bg-aria-blue flex items-center justify-center text-white" aria-label="Send message">
            <SendIcon className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

const FilesView: React.FC<{ setView: (v: View) => void }> = ({ setView }) => {
  const recentFiles = [
    { name: 'project_v1.jpg', icon: <FolderIcon className="w-12 h-12 text-yellow-500" />, specialBg: true },
    { name: 'Invoice_Oct.pdf', icon: <PdfIcon className="w-8 h-8 text-red-500" /> },
    { name: 'Q3_Data.xls', icon: <XlsIcon className="w-8 h-8 text-green-500" /> },
    { name: 'demo.mov', icon: <PhotoIcon className="w-8 h-8 text-purple-500" /> },
  ];
  return (
    <div className="flex flex-col h-full bg-aria-dark text-white">
      <Header title="File & Tools Access" showBackButton={true} onBack={() => setView('chat')} />
      <div className="flex-1 overflow-y-auto p-4 space-y-8">
        <div className="bg-aria-dark-blue p-4 rounded-lg flex items-center gap-4">
          <img src="https://i.imgur.com/O5nC3A9.png" alt="Assistant Avatar" className="w-10 h-10 rounded-full" />
          <div>
            <p className="text-xs text-aria-blue font-semibold">ASSISTANT</p>
            <p className="text-sm">I can analyze files or run AI tools for you.</p>
          </div>
        </div>
        <div>
          <h2 className="text-xs font-semibold text-aria-gray mb-3">RECENT FILES</h2>
          <div className="grid grid-cols-4 gap-4">
            {recentFiles.map((file, i) => (
              <div key={i} className="text-center">
                <div 
                  className="aspect-square rounded-lg flex items-center justify-center"
                  style={file.specialBg ? { background: 'linear-gradient(to bottom right, #FBBF24, #F59E0B)' } : { backgroundColor: '#171F30' }}
                >
                  {file.icon}
                </div>
                <p className="text-xs mt-2 truncate">{file.name}</p>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h2 className="text-xs font-semibold text-aria-gray mb-2">FILE TYPES</h2>
          <div className="bg-aria-dark-blue rounded-lg">
            <div className="flex items-center justify-between p-4 border-b border-white/10 cursor-pointer hover:bg-white/5">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-aria-blue rounded-lg flex items-center justify-center"><PhotoIcon className="w-6 h-6" /></div>
                <div>
                  <p className="font-medium">Photos & Videos</p>
                  <p className="text-xs text-aria-gray">.jpg, .png, .webp, .mp4</p>
                </div>
              </div>
              <ChevronRightIcon className="w-5 h-5 text-aria-gray" />
            </div>
            <div className="flex items-center justify-between p-4 border-b border-white/10 cursor-pointer hover:bg-white/5">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-aria-blue rounded-lg flex items-center justify-center"><DocumentIcon className="w-6 h-6" /></div>
                <div>
                  <p className="font-medium">Documents</p>
                  <p className="text-xs text-aria-gray">.pdf, .docx, .csv, .xls, .pptx</p>
                </div>
              </div>
              <ChevronRightIcon className="w-5 h-5 text-aria-gray" />
            </div>
            <div className="flex items-center justify-between p-4 cursor-pointer hover:bg-white/5">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-aria-blue rounded-lg flex items-center justify-center"><AudioIcon className="w-6 h-6" /></div>
                <div>
                  <p className="font-medium">Audio</p>
                  <p className="text-xs text-aria-gray">.mp3, .wav, .m4a</p>
                </div>
              </div>
              <ChevronRightIcon className="w-5 h-5 text-aria-gray" />
            </div>
          </div>
        </div>
        <div>
          <h2 className="text-xs font-semibold text-aria-gray mb-2">AI TOOLS</h2>
          <div className="bg-aria-dark-blue rounded-lg">
            <div className="flex items-center justify-between p-4 border-b border-white/10 cursor-pointer hover:bg-white/5">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-aria-blue rounded-lg flex items-center justify-center"><TtsIcon className="w-6 h-6" /></div>
                <div>
                  <p className="font-medium">Text-to-Speech</p>
                  <p className="text-xs text-aria-gray">Convert text to natural audio</p>
                </div>
              </div>
              <ChevronRightIcon className="w-5 h-5 text-aria-gray" />
            </div>
            <div className="flex items-center justify-between p-4 cursor-pointer hover:bg-white/5">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-aria-blue rounded-lg flex items-center justify-center"><SummarizeIcon className="w-6 h-6" /></div>
                <div>
                  <p className="font-medium">Summarization</p>
                  <p className="text-xs text-aria-gray">Get quick summaries of long text</p>
                </div>
              </div>
              <ChevronRightIcon className="w-5 h-5 text-aria-gray" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

type SettingsViewProps = {
  setView: (v: View) => void;
  avatarUrl: string;
  setAvatarUrl: (url: string) => void;
  wallpaperUrl: string;
  setWallpaperUrl: (url: string) => void;
  defaultAvatar: string;
};

const SettingsView: React.FC<SettingsViewProps> = ({ setView, avatarUrl, setAvatarUrl, wallpaperUrl, setWallpaperUrl, defaultAvatar }) => {
  const [voiceOutput, setVoiceOutput] = useState(true);
  const [hapticFeedback, setHapticFeedback] = useState(true);
  const avatarInputRef = useRef<HTMLInputElement>(null);
  const wallpaperInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>, setter: (url: string) => void) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setter(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col h-full bg-aria-dark text-white">
      <Header title="Settings" showBackButton={true} onBack={() => setView('chat')} />
      <div className="flex-1 overflow-y-auto p-4 space-y-8">
        {/* Hidden file inputs */}
        <input
          type="file"
          ref={avatarInputRef}
          className="hidden"
          accept="image/png,image/jpeg,image/jpg,image/gif"
          onChange={(e) => handleFileSelect(e, setAvatarUrl)}
        />
        <input
          type="file"
          ref={wallpaperInputRef}
          className="hidden"
          accept="image/png,image/jpeg,image/jpg"
          onChange={(e) => handleFileSelect(e, setWallpaperUrl)}
        />

        <div>
          <h2 className="text-xs font-semibold text-aria-gray mb-2">AI PERSONA</h2>
          <div className="bg-aria-dark-blue rounded-lg p-6 text-center">
            <div className="relative inline-block">
              <img src={avatarUrl} alt="Assistant Avatar" className="w-24 h-24 rounded-full mx-auto object-cover" />
              <button
                onClick={() => avatarInputRef.current?.click()}
                className="absolute bottom-0 right-0 w-8 h-8 bg-aria-blue rounded-full flex items-center justify-center border-2 border-aria-dark-blue"
              >
                <CameraIcon className="w-5 h-5 text-white" />
              </button>
            </div>
            <h3 className="text-lg font-semibold mt-4">Assistant Avatar</h3>
            <p className="text-sm text-aria-gray mt-1">Customize how your assistant looks in chat.</p>
            <button
              onClick={() => avatarInputRef.current?.click()}
              className="w-full bg-white/10 hover:bg-white/20 text-white font-semibold py-2.5 rounded-lg mt-4"
            >
              Edit Photo
            </button>
            {avatarUrl !== defaultAvatar && (
              <button
                onClick={() => setAvatarUrl(defaultAvatar)}
                className="w-full text-sm text-aria-gray hover:text-white mt-2"
              >
                Reset to Default
              </button>
            )}
          </div>
        </div>
        <div>
          <h2 className="text-xs font-semibold text-aria-gray mb-2">APPEARANCE</h2>
          <div className="bg-aria-dark-blue rounded-lg p-6">
            <div
              className="h-32 rounded-lg bg-cover bg-center mb-4 relative overflow-hidden"
              style={{
                backgroundImage: wallpaperUrl ? `url("${wallpaperUrl}")` : `url("https://i.imgur.com/8sC2TSy.png")`,
                backgroundColor: '#0A0E1A'
              }}
            >
              <div className="absolute inset-0 bg-black/30 p-2 flex flex-col justify-end space-y-2">
                <div className="flex items-end gap-2 justify-start">
                  <div className="w-6 h-6 rounded-full bg-gray-300 flex-shrink-0"></div>
                  <div className="bg-white text-black text-xs px-3 py-1.5 rounded-xl rounded-bl-sm">Hello! How can I help you today?</div>
                </div>
                <div className="flex items-end gap-2 justify-end">
                  <div className="bg-blue-600 text-white text-xs px-3 py-1.5 rounded-xl rounded-br-sm">Change my background please.</div>
                </div>
              </div>
            </div>
            <h3 className="text-lg font-semibold">Chat Wallpaper</h3>
            <p className="text-sm text-aria-gray mt-1">Upload a .png or .jpg to customize your chat view.</p>
            <div className="flex items-center justify-between mt-4">
              <button
                onClick={() => wallpaperInputRef.current?.click()}
                className="flex-1 bg-aria-blue hover:bg-blue-500 text-white font-semibold py-2.5 rounded-lg"
              >
                Choose Wallpaper
              </button>
              <button
                onClick={() => setWallpaperUrl('')}
                className="text-sm text-aria-gray hover:text-white ml-4"
              >
                Reset
              </button>
            </div>
          </div>
        </div>
        <div>
          <h2 className="text-xs font-semibold text-aria-gray mb-2">GENERAL</h2>
          <div className="bg-aria-dark-blue rounded-lg">
            <div className="flex items-center justify-between p-4 border-b border-white/10">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-aria-blue rounded-lg flex items-center justify-center"><VolumeIcon className="w-6 h-6" /></div>
                <p className="font-medium">Voice Output</p>
              </div>
              <ToggleSwitch enabled={voiceOutput} setEnabled={setVoiceOutput} />
            </div>
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-aria-blue rounded-lg flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                  </svg>
                </div>
                <p className="font-medium">Haptic Feedback</p>
              </div>
              <ToggleSwitch enabled={hapticFeedback} setEnabled={setHapticFeedback} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const BottomNavBar: React.FC<{ currentView: View; setView: (v: View) => void }> = ({ currentView, setView }) => {
  const navItems = [
    { view: 'chat' as View, label: 'Chat', icon: <ChatBubbleIcon className="w-7 h-7" /> },
    { view: 'files' as View, label: 'Files', icon: <FolderIcon className="w-7 h-7" /> },
    { view: 'settings' as View, label: 'Settings', icon: <SettingsIcon className="w-7 h-7" /> },
  ];

  return (
    <div className="bg-aria-dark/90 backdrop-blur-sm border-t border-white/10 flex justify-around items-center py-1">
      {navItems.map(item => (
        <button
          key={item.view}
          onClick={() => setView(item.view)}
          className={`flex flex-col items-center justify-center w-20 h-16 rounded-lg transition-colors duration-200 ${currentView === item.view ? 'text-aria-blue' : 'text-aria-gray hover:text-white'}`}
          aria-current={currentView === item.view}
          aria-label={item.label}
        >
          {item.icon}
          <span className="text-xs mt-1">{item.label}</span>
        </button>
      ))}
    </div>
  );
};


// --- Main App Component ---

// Default images
const DEFAULT_AVATAR = "https://i.imgur.com/s22ZlE8.png";
const DEFAULT_WALLPAPER = "";

export default function App() {
  const [currentView, setCurrentView] = useState<View>('chat');
  const [avatarUrl, setAvatarUrl] = useState<string>(DEFAULT_AVATAR);
  const [wallpaperUrl, setWallpaperUrl] = useState<string>(DEFAULT_WALLPAPER);

  return (
    <div className="bg-gray-900 min-h-screen flex items-center justify-center p-4 font-sans">
      <div className="w-full max-w-sm h-[90vh] max-h-[844px] bg-aria-dark rounded-[40px] shadow-2xl overflow-hidden relative flex flex-col border-4 border-gray-700">
        <div className="flex-1 overflow-hidden">
          {currentView === 'chat' && <ChatView setView={setCurrentView} avatarUrl={avatarUrl} wallpaperUrl={wallpaperUrl} />}
          {currentView === 'files' && <FilesView setView={setCurrentView} />}
          {currentView === 'settings' && <SettingsView setView={setCurrentView} avatarUrl={avatarUrl} setAvatarUrl={setAvatarUrl} wallpaperUrl={wallpaperUrl} setWallpaperUrl={setWallpaperUrl} defaultAvatar={DEFAULT_AVATAR} />}
        </div>
        <BottomNavBar currentView={currentView} setView={setCurrentView} />
      </div>
    </div>
  );
}
