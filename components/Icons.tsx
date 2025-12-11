
import React from 'react';

interface IconProps {
  className?: string;
}

export const ArrowLeftIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
  </svg>
);

export const SettingsIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-1.007 1.11-1.226.55-.22 1.156-.22 1.706 0 .55.22 1.02.684 1.11 1.226l.043.252a2.25 2.25 0 013.484 2.27l.043.252c.09.542.56 1.007 1.11 1.226.55.22 1.156.22 1.706 0 .55-.22 1.02-.684 1.11-1.226l.043-.252a2.25 2.25 0 013.484-2.27l.043-.252c.09-.542.56-1.007 1.11-1.226.55-.22 1.156-.22 1.706 0 .55.22 1.02.684 1.11 1.226l.043.252a2.25 2.25 0 01-3.484 2.27l-.043.252c-.09.542-.56 1.007-1.11 1.226-.55.22-1.156.22-1.706 0-.55-.22-1.02-.684-1.11-1.226l-.043-.252a2.25 2.25 0 01-3.484-2.27l-.043-.252zm-3.484-.252l.043.252c.09.542.56 1.007 1.11 1.226.55.22 1.156.22 1.706 0 .55-.22 1.02-.684 1.11-1.226l.043-.252a2.25 2.25 0 013.484-2.27l.043-.252c.09-.542.56-1.007 1.11-1.226.55-.22 1.156-.22 1.706 0 .55.22 1.02.684 1.11 1.226l.043.252a2.25 2.25 0 01-3.484 2.27l-.043.252c-.09.542-.56 1.007-1.11 1.226-.55.22-1.156.22-1.706 0-.55-.22-1.02-.684-1.11-1.226l-.043-.252a2.25 2.25 0 01-3.484-2.27l-.043-.252zm-3.484-.252a2.25 2.25 0 013.484 2.27l.043.252c.09.542.56 1.007 1.11 1.226.55.22 1.156.22 1.706 0 .55-.22 1.02-.684 1.11-1.226l.043-.252a2.25 2.25 0 013.484-2.27l.043-.252c.09-.542.56-1.007 1.11-1.226.55-.22 1.156-.22 1.706 0 .55.22 1.02.684 1.11 1.226l.043.252a2.25 2.25 0 01-3.484 2.27l-.043.252c-.09.542-.56 1.007-1.11 1.226-.55.22-1.156.22-1.706 0-.55-.22-1.02-.684-1.11-1.226l-.043-.252a2.25 2.25 0 01-3.484-2.27z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 15.75a3.75 3.75 0 100-7.5 3.75 3.75 0 000 7.5z" />
  </svg>
);

export const MicrophoneIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 016 0v8.25a3 3 0 01-3 3z" />
  </svg>
);

export const SendIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
  </svg>
);

export const ChevronRightIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
  </svg>
);

export const FolderIcon: React.FC<IconProps> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
        <path d="M19.5 21a3 3 0 003-3V9a3 3 0 00-3-3h-5.25a3 3 0 01-2.65-1.5L9.9 3.45A3 3 0 007.25 2.25H4.5a3 3 0 00-3 3v12a3 3 0 003 3h15z" />
    </svg>
);

export const PdfIcon: React.FC<IconProps> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
        <path fillRule="evenodd" d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0016.5 9h-1.875a.375.375 0 01-.375-.375V6.75A3.75 3.75 0 009 3H5.625zM15 6.75v2.25a.75.75 0 00.75.75h2.25a.75.75 0 000-1.5h-1.5V6.75a.75.75 0 00-1.5 0z" clipRule="evenodd" />
        <path d="M7.5 15a.75.75 0 01.75-.75h3a.75.75 0 010 1.5h-3a.75.75 0 01-.75-.75zm0 3a.75.75 0 01.75-.75h3a.75.75 0 010 1.5h-3a.75.75 0 01-.75-.75z" />
    </svg>
);

export const XlsIcon: React.FC<IconProps> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
        <path fillRule="evenodd" d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0016.5 9h-1.875a.375.375 0 01-.375-.375V6.75A3.75 3.75 0 009 3H5.625zM15 6.75v2.25a.75.75 0 00.75.75h2.25a.75.75 0 000-1.5h-1.5V6.75a.75.75 0 00-1.5 0z" clipRule="evenodd" />
        <path d="M7.5 15a.75.75 0 01.75-.75h.75v.75a.75.75 0 001.5 0v-.75h.75a.75.75 0 010 1.5h-.75v.75a.75.75 0 001.5 0v-.75h.75a.75.75 0 010 1.5H12v.75a.75.75 0 001.5 0V15a.75.75 0 01.75-.75h.75a.75.75 0 010 1.5h-.75v.75a.75.75 0 001.5 0v-.75h.75a.75.75 0 010 1.5h-3a.75.75 0 01-.75-.75v-.75H9a.75.75 0 01-.75-.75zm0 3a.75.75 0 01.75-.75h9a.75.75 0 010 1.5h-9a.75.75 0 01-.75-.75z" />
    </svg>
);

export const PhotoIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path fillRule="evenodd" d="M1.5 6a2.25 2.25 0 012.25-2.25h16.5A2.25 2.25 0 0122.5 6v12a2.25 2.25 0 01-2.25 2.25H3.75A2.25 2.25 0 011.5 18V6zM3 16.06V18c0 .414.336.75.75.75h16.5A.75.75 0 0021 18v-1.94l-2.69-2.689a1.5 1.5 0 00-2.12 0l-.88.879.97.97a.75.75 0 11-1.06 1.06l-5.16-5.159a1.5 1.5 0 00-2.12 0L3 16.061zm10.125-7.81a1.125 1.125 0 112.25 0 1.125 1.125 0 01-2.25 0z" clipRule="evenodd" />
  </svg>
);

export const DocumentIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path fillRule="evenodd" d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0016.5 9h-1.875a.375.375 0 01-.375-.375V6.75A3.75 3.75 0 009 3H5.625zM10.5 17.25a.75.75 0 000-1.5H7.5a.75.75 0 000 1.5h3zM10.5 14.25a.75.75 0 000-1.5H7.5a.75.75 0 000 1.5h3z" clipRule="evenodd" />
  </svg>
);

export const AudioIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M2.25 9.375c0-1.036.84-1.875 1.875-1.875h15.75c1.035 0 1.875.84 1.875 1.875v5.25c0 1.035-.84 1.875-1.875 1.875H4.125c-1.036 0-1.875-.84-1.875-1.875V9.375z" />
    <path fillRule="evenodd" d="M8.25 12a.75.75 0 01.75-.75h6a.75.75 0 010 1.5h-6a.75.75 0 01-.75-.75z" clipRule="evenodd" />
  </svg>
);

export const TtsIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M8.25 4.5a3.75 3.75 0 117.5 0 3.75 3.75 0 01-7.5 0zM15.75 4.5a.75.75 0 001.5 0 5.25 5.25 0 00-10.5 0 .75.75 0 001.5 0 3.75 3.75 0 017.5 0z" />
    <path fillRule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zM12.625 18a.375.375 0 10-1.25 0v-8.69L9.8 9.935a.75.75 0 01-1.06-1.06l2.25-2.25a.75.75 0 011.06 0l2.25 2.25a.75.75 0 11-1.06 1.06l-1.565-.625v8.69z" clipRule="evenodd" />
  </svg>
);

export const SummarizeIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M4.5 3.75a3 3 0 00-3 3v.75h21v-.75a3 3 0 00-3-3h-15z" />
    <path fillRule="evenodd" d="M22.5 9.75h-21v7.5a3 3 0 003 3h15a3 3 0 003-3v-7.5zm-18 3.75a.75.75 0 01.75-.75h6a.75.75 0 010 1.5h-6a.75.75 0 01-.75-.75zm.75 2.25a.75.75 0 000 1.5h3a.75.75 0 000-1.5h-3z" clipRule="evenodd" />
  </svg>
);

export const CameraIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M12 9a3.75 3.75 0 100 7.5A3.75 3.75 0 0012 9z" />
    <path fillRule="evenodd" d="M9.344 3.071a49.52 49.52 0 015.312 0c.967.052 1.83.585 2.332 1.39l.821 1.317c.24.383.645.643 1.11.71.386.054.77.113 1.152.177 1.432.239 2.429 1.493 2.429 2.909V18a3 3 0 01-3 3h-15a3 3 0 01-3-3V9.574c0-1.416.997-2.67 2.429-2.909.382-.064.766-.123 1.152-.177.465-.067.87-.327 1.11-.71l.82-1.318a2.999 2.999 0 012.332-1.39zM12 18a5.25 5.25 0 100-10.5 5.25 5.25 0 000 10.5z" clipRule="evenodd" />
  </svg>
);

export const VolumeIcon: React.FC<IconProps> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0 001.5 12c0 .898.121 1.768.348 2.595.341 1.24 1.518 1.905 2.66 1.905H6.44l4.5 4.5c.944.945 2.56.276 2.56-1.06V4.06zM18.584 5.106a.75.75 0 011.06 0c3.808 3.807 3.808 9.98 0 13.788a.75.75 0 11-1.06-1.06 8.25 8.25 0 000-11.668.75.75 0 010-1.06z" />
    <path d="M15.932 7.757a.75.75 0 011.06 0 4.5 4.5 0 010 6.364.75.75 0 11-1.06-1.06 3 3 0 000-4.243.75.75 0 010-1.06z" />
  </svg>
);

export const ChatBubbleIcon: React.FC<IconProps> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={className}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193l-3.722.234c-.386.024-.734.22-1.003.49A5.002 5.002 0 0113 20.25h-2a5.002 5.002 0 01-2.519-1.003c-.268-.27-.617-.466-1.003-.49l-3.722-.234A2.096 2.096 0 012.25 15.18V10.894c0-.97.616-1.813 1.5-2.097m0 0A5.25 5.25 0 006.75 5.25h10.5a5.25 5.25 0 00-3-4.04" />
    </svg>
);
