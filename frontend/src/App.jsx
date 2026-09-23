import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Car, 
  Users, 
  UploadCloud, 
  AlertCircle, 
  CheckCircle2, 
  ShieldCheck, 
  Zap, 
  RefreshCw, 
  Download, 
  Eye, 
  Activity,
  Layers,
  Sparkles
} from 'lucide-react';
import './App.css';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedImg, setAnnotatedImg] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState('checking');
  const [isDragging, setIsDragging] = useState(false);

  // Check Backend API Health Status on Mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await axios.get('http://127.0.0.1:8000/health');
      if (res.data.status === 'healthy') {
        setApiHealth('online');
      } else {
        setApiHealth('offline');
      }
    } catch {
      setApiHealth('offline');
    }
  };

  const handleFileChange = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnnotatedImg(null);
      setMetrics(null);
      setError(null);
    } else {
      setError('Please select a valid image file (JPG, PNG).');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleUploadAndAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.success) {
        setAnnotatedImg(response.data.image_base64);
        setMetrics(response.data.summary);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process image. Please verify FastAPI backend on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnnotatedImg(null);
    setMetrics(null);
    setError(null);
  };

  return (
    <div className="app-container">
      {/* Header Bar */}
      <header className="glass-panel" style={{ padding: '24px 32px', marginBottom: '28px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ background: 'linear-gradient(135deg, #0284c7, #38bdf8)', padding: '10px', borderRadius: '12px', display: 'flex', boxShadow: '0 0 20px rgba(56, 189, 248, 0.4)' }}>
              <Car size={28} color="#ffffff" />
            </div>
            <div>
              <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.85rem', fontWeight: '700', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
                Traffic Vision AI
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '2px' }}>
                Vehicle Color Classification & Pedestrian Analytics Engine
              </p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(255, 255, 255, 0.05)', padding: '6px 14px', borderRadius: '20px', border: '1px solid var(--border-color)', fontSize: '0.82rem' }}>
            <Sparkles size={14} color="#38bdf8" />
            <span style={{ color: '#cbd5e1' }}>YOLOv8s + CIE-LAB Engine</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: apiHealth === 'online' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)', padding: '6px 14px', borderRadius: '20px', border: `1px solid ${apiHealth === 'online' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`, fontSize: '0.82rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: apiHealth === 'online' ? '#10b981' : '#ef4444', boxShadow: apiHealth === 'online' ? '0 0 10px #10b981' : '0 0 10px #ef4444' }} />
            <span style={{ color: apiHealth === 'online' ? '#34d399' : '#fca5a5', fontWeight: '500' }}>
              {apiHealth === 'online' ? 'API Online' : 'API Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Analytics Metric Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px', marginBottom: '28px' }}>
        <div className="glass-panel metric-card blue-car" style={{ padding: '20px 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: '500' }}>Blue Cars [RED BOX]</span>
            <div style={{ background: 'rgba(239, 68, 68, 0.15)', padding: '8px', borderRadius: '10px' }}>
              <Car size={20} color="#ef4444" />
            </div>
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: '700', fontFamily: 'var(--font-heading)', color: '#f8fafc', marginTop: '10px' }}>
            {metrics ? metrics.blue_cars : '--'}
          </div>
          <span style={{ fontSize: '0.78rem', color: '#ef4444', marginTop: '4px', display: 'block' }}>Red bounding box annotated</span>
        </div>

        <div className="glass-panel metric-card other-car" style={{ padding: '20px 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: '500' }}>Other Cars [BLUE BOX]</span>
            <div style={{ background: 'rgba(59, 130, 246, 0.15)', padding: '8px', borderRadius: '10px' }}>
              <Car size={20} color="#3b82f6" />
            </div>
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: '700', fontFamily: 'var(--font-heading)', color: '#f8fafc', marginTop: '10px' }}>
            {metrics ? metrics.other_cars : '--'}
          </div>
          <span style={{ fontSize: '0.78rem', color: '#3b82f6', marginTop: '4px', display: 'block' }}>Blue bounding box annotated</span>
        </div>

        <div className="glass-panel metric-card pedestrian" style={{ padding: '20px 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: '500' }}>Pedestrians</span>
            <div style={{ background: 'rgba(16, 185, 129, 0.15)', padding: '8px', borderRadius: '10px' }}>
              <Users size={20} color="#10b981" />
            </div>
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: '700', fontFamily: 'var(--font-heading)', color: '#f8fafc', marginTop: '10px' }}>
            {metrics ? metrics.people_count : '--'}
          </div>
          <span style={{ fontSize: '0.78rem', color: '#10b981', marginTop: '4px', display: 'block' }}>Green bounding box annotated</span>
        </div>

        <div className="glass-panel metric-card total-vehicle" style={{ padding: '20px 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: '500' }}>Total Vehicles</span>
            <div style={{ background: 'rgba(168, 85, 247, 0.15)', padding: '8px', borderRadius: '10px' }}>
              <ShieldCheck size={20} color="#a855f7" />
            </div>
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: '700', fontFamily: 'var(--font-heading)', color: '#f8fafc', marginTop: '10px' }}>
            {metrics ? metrics.total_cars : '--'}
          </div>
          <span style={{ fontSize: '0.78rem', color: '#a855f7', marginTop: '4px', display: 'block' }}>Cars, trucks, buses & bikes</span>
        </div>
      </div>

      {/* Upload Zone & Control Action Bar */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '28px' }}>
        <div 
          className={`dropzone ${isDragging ? 'active' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          style={{ padding: '28px', textAlign: 'center', cursor: 'pointer', marginBottom: '20px' }}
          onClick={() => document.getElementById('file-upload-input').click()}
        >
          <input 
            type="file" 
            accept="image/*" 
            onChange={(e) => e.target.files && handleFileChange(e.target.files[0])} 
            id="file-upload-input" 
            style={{ display: 'none' }} 
          />
          <div style={{ background: 'rgba(56, 189, 248, 0.1)', width: '54px', height: '54px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
            <UploadCloud size={28} color="#38bdf8" />
          </div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#f8fafc', marginBottom: '4px' }}>
            {selectedFile ? selectedFile.name : 'Click to select or drag & drop traffic image'}
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.84rem' }}>
            Supports JPG, JPEG, and PNG traffic captures
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {selectedFile && (
              <button 
                onClick={handleReset}
                style={{ background: 'rgba(255, 255, 255, 0.06)', border: '1px solid var(--border-color)', color: '#cbd5e1', padding: '10px 18px', borderRadius: '10px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.88rem' }}
              >
                <RefreshCw size={16} /> Reset
              </button>
            )}
          </div>

          <button
            onClick={handleUploadAndAnalyze}
            disabled={!selectedFile || loading}
            className="interactive-btn"
            style={{
              background: loading ? '#475569' : 'linear-gradient(135deg, #0284c7, #2563eb)',
              color: '#ffffff',
              border: 'none',
              padding: '12px 28px',
              borderRadius: '10px',
              cursor: (!selectedFile || loading) ? 'not-allowed' : 'pointer',
              fontWeight: '600',
              fontSize: '0.95rem',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              boxShadow: loading ? 'none' : '0 4px 14px rgba(37, 99, 235, 0.4)'
            }}
          >
            {loading ? (
              <>
                <RefreshCw size={18} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                <span>Running Detection Pipeline...</span>
              </>
            ) : (
              <>
                <Zap size={18} />
                <span>Run Vision Pipeline</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="glass-panel" style={{ backgroundColor: 'rgba(69, 10, 10, 0.6)', borderColor: 'rgba(239, 68, 68, 0.4)', color: '#fca5a5', padding: '16px 20px', borderRadius: '12px', marginBottom: '28px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <AlertCircle size={22} color="#ef4444" />
          <span style={{ fontSize: '0.92rem' }}>{error}</span>
        </div>
      )}

      {/* Dual Visual Canvas Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px' }}>
        {/* Source Image */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Eye size={20} color="#38bdf8" /> Source Image Preview
            </h2>
            {selectedFile && (
              <span style={{ background: 'rgba(255, 255, 255, 0.06)', padding: '4px 10px', borderRadius: '6px', fontSize: '0.78rem', color: '#94a3b8' }}>
                {(selectedFile.size / 1024).toFixed(1)} KB
              </span>
            )}
          </div>
          <div className="image-canvas-wrapper">
            {previewUrl ? (
              <img src={previewUrl} alt="Source Traffic Input" />
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                <UploadCloud size={44} color="#334155" style={{ marginBottom: '10px' }} />
                <p>Select or drag a traffic image to preview</p>
              </div>
            )}
          </div>
        </div>

        {/* Annotated Output */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={20} color="#10b981" /> Annotated AI Detection Output
            </h2>
            {annotatedImg && (
              <a
                href={annotatedImg}
                download="annotated_traffic_output.jpg"
                style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', color: '#34d399', padding: '6px 14px', borderRadius: '8px', fontSize: '0.82rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '6px', textDecoration: 'none' }}
              >
                <Download size={14} /> Download Output
              </a>
            )}
          </div>
          <div className="image-canvas-wrapper">
            {annotatedImg ? (
              <img src={annotatedImg} alt="Annotated Traffic Output" />
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                <Activity size={44} color="#334155" style={{ marginBottom: '10px' }} />
                <p>Annotated output with bounding boxes will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
