import React from 'react';
import { Loader2, Image as ImageIcon } from 'lucide-react';

export default function VisionSidebar({ isModelLoading, prediction }) {
  return (
    <div className="tm-sidebar glass-panel" style={{ borderRadius: 0, borderTop: 'none', borderBottom: 'none' }}>
      <h2 className="title-glow" style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Vision Analytics</h2>
      {isModelLoading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
          <Loader2 className="animate-spin" size={16} /> Loading Model...
        </div>
      ) : (
        <div style={{ color: 'var(--success)', fontSize: '0.9rem' }}>✓ System Ready</div>
      )}

      {prediction && (
        <div className="classification-result" style={{ marginTop: '20px' }}>
          <div className="diagnosis-header">AI Diagnosis</div>
          {prediction.slice(0, 3).map((p, i) => (
            <div key={i} style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px', color: 'var(--text-main)' }}>
                <span>{p.className}</span>
                <span>{(p.probability * 100).toFixed(1)}%</span>
              </div>
              <div className="class-bar-container">
                <div className="class-bar" style={{ width: `${p.probability * 100}%` }}></div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {!prediction && (
        <div style={{ opacity: 0.5, fontSize: '0.9rem', marginTop: '20px', textAlign: 'center' }}>
          <ImageIcon size={32} style={{ margin: '0 auto 12px' }} />
          <p>Upload an image to run local classification model.</p>
        </div>
      )}
    </div>
  );
}
