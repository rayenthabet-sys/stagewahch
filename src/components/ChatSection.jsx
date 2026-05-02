import React, { useRef, useEffect } from 'react';
import { Send, Mic, Image as ImageIcon, X, Loader2, FileAudio } from 'lucide-react';

export default function ChatSection({
  messages,
  isProcessing,
  selectedImage,
  selectedAudio,
  audioFile,
  handleSend,
  handleImageChange,
  handleAudioChange,
  removeImage,
  removeAudio
}) {
  const imageInputRef = useRef(null);
  const audioInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const onRemoveImage = () => {
    removeImage();
    if (imageInputRef.current) imageInputRef.current.value = '';
  };

  const onRemoveAudio = () => {
    removeAudio();
    if (audioInputRef.current) audioInputRef.current.value = '';
  };

  return (
    <div className="chat-section">
      <header className="navbar">
        <div className="brand">
          <span className="text-gradient">Falla7</span> AI
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Powered by Groq & Gemini
        </div>
      </header>

      <div className="messages-area">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            {msg.role === 'ai' && <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent)', marginBottom: '6px' }}>Falla7 Agent</div>}
            {msg.role === 'user' && <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.8, marginBottom: '6px', textAlign: 'right' }}>You</div>}
            
            {msg.image && (
              <img src={msg.image} alt="uploaded" style={{ width: '100%', maxWidth: '300px', borderRadius: '8px', marginBottom: '8px' }} />
            )}
            {msg.audioBaseUrl && (
              <audio controls src={msg.audioBaseUrl} style={{ width: '100%', maxWidth: '300px', margin: '8px 0' }} />
            )}
            
            {msg.content && <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>}
            
            {msg.audioBase64 && (
              <audio 
                controls 
                autoPlay 
                src={`data:audio/mp3;base64,${msg.audioBase64}`} 
                style={{ width: '100%', maxWidth: '100%', marginTop: '12px' }} 
              />
            )}
          </div>
        ))}
        {isProcessing && (
          <div className="message ai" style={{ display: 'flex', alignItems: 'center', gap: '8px', opacity: 0.7 }}>
            <Loader2 className="animate-spin" size={16} /> Backend processing your voice query...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', padding: '0 16px' }}>
          {selectedImage && (
            <div className="image-preview" style={{ marginRight: '8px' }}>
              <img src={selectedImage} alt="preview" />
              <button className="remove-img-btn" onClick={onRemoveImage}><X size={12} /></button>
            </div>
          )}
          {selectedAudio && audioFile && (
            <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(255,255,255,0.05)', padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--card-border)'}}>
              <FileAudio size={16} style={{ marginRight: '8px', color: 'var(--accent)'}} />
              <span style={{ fontSize: '0.8rem', marginRight: '16px' }}>{audioFile.name}</span>
              <button onClick={onRemoveAudio} style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer' }}><X size={14} /></button>
            </div>
          )}
        </div>

        <div className="input-wrapper glass-panel" style={{ padding: '8px 16px', borderRadius: '30px' }}>
          <button className="icon-btn" onClick={() => imageInputRef.current?.click()} title="Upload Image">
            <ImageIcon size={20} />
          </button>
          <input 
            type="file" 
            ref={imageInputRef} 
            style={{ display: 'none' }} 
            accept="image/*" 
            onChange={handleImageChange}
          />
          
          <button className="icon-btn" onClick={() => audioInputRef.current?.click()} title="Upload Audio" style={{ marginRight: 'auto' }}>
            <Mic size={20} />
          </button>
          <input 
            type="file" 
            ref={audioInputRef} 
            style={{ display: 'none' }} 
            accept="audio/*" 
            onChange={handleAudioChange}
          />
          
          <div style={{ flex: 1, color: 'var(--text-muted)', fontSize: '0.9rem', textAlign: 'center' }}>
            {audioFile ? "Voice query ready to send." : "Upload an audio query to interact."}
          </div>
          
          <button 
            className="icon-btn primary" 
            onClick={handleSend} 
            disabled={isProcessing || !audioFile}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
