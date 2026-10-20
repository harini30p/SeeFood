import { useState, useEffect, useRef } from 'react'
import './App.css'

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000'
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']

export default function App() {
  // State management
  const [selectedFile, setSelectedFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isDragActive, setIsDragActive] = useState(false)

  // Backend connection status: 'checking' | 'online' | 'offline'
  const [backendStatus, setBackendStatus] = useState('checking')
  const [backendError, setBackendError] = useState(null)

  // Ref to programmatically open the hidden file input
  const fileInputRef = useRef(null)

  /**
   * Ping GET /health on the FastAPI backend
   */
  const checkHealth = async () => {
    setBackendStatus('checking')
    setBackendError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      })
      if (!response.ok) {
        throw new Error(`Health check returned status ${response.status}`)
      }
      const data = await response.json()
      if (data.status === 'healthy' && data.model_loaded) {
        setBackendStatus('online')
      } else {
        setBackendStatus('offline')
        setBackendError('Backend reached, but the model is not loaded.')
      }
    } catch {
      setBackendStatus('offline')
      setBackendError('Cannot connect to FastAPI backend at http://127.0.0.1:8000. Is the server running?')
    }
  }

  // Check backend health on component mount
  useEffect(() => {
    let ignore = false
    async function runCheck() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          method: 'GET',
          headers: { Accept: 'application/json' },
        })
        if (!response.ok) {
          throw new Error(`Health check status ${response.status}`)
        }
        const data = await response.json()
        if (!ignore) {
          if (data.status === 'healthy' && data.model_loaded) {
            setBackendStatus('online')
          } else {
            setBackendStatus('offline')
            setBackendError('Backend reached, but the model is not loaded.')
          }
        }
      } catch {
        if (!ignore) {
          setBackendStatus('offline')
          setBackendError('Cannot connect to FastAPI backend at http://127.0.0.1:8000. Is the server running?')
        }
      }
    }
    runCheck()
    return () => {
      ignore = true
    }
  }, [])

  // Clean up object URL when imagePreview changes to prevent memory leaks
  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview)
      }
    }
  }, [imagePreview])

  /**
   * Validate image file type and extension
   */
  const validateFile = (file) => {
    if (!file) return false

    const fileName = file.name.toLowerCase()
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => fileName.endsWith(ext))
    const hasValidMime = ALLOWED_MIME_TYPES.includes(file.type)

    if (!hasValidExt && !hasValidMime) {
      setError(
        `Unsupported file type "${file.name}". Please select a JPEG, PNG, or WebP image.`
      )
      return false
    }

    return true
  }

  /**
   * Process a valid file for preview and upload
   */
  const handleFileSelect = (file) => {
    setError(null)
    setPrediction(null)

    if (!validateFile(file)) {
      return
    }

    setSelectedFile(file)
    const previewUrl = URL.createObjectURL(file)
    setImagePreview(previewUrl)
  }

  // Handle standard file input change
  const handleInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  // Drag and drop event handlers
  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const file = e.dataTransfer.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  /**
   * Send the image to FastAPI POST /predict
   */
  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select or drop an image first.')
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        // Do NOT manually set 'Content-Type': the browser will automatically
        // provide 'multipart/form-data; boundary=...'
        body: formData,
      })

      if (!response.ok) {
        let errorDetail = `Classification failed with status ${response.status}.`
        try {
          const errorJson = await response.json()
          if (errorJson.detail) {
            errorDetail = errorJson.detail
          }
        } catch {
          // If response body is not JSON, retain default errorDetail
        }
        throw new Error(errorDetail)
      }

      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.message || 'Network error: could not connect to backend.')
    } finally {
      setLoading(false)
    }
  }

  /**
   * Reset everything to start fresh
   */
  const handleReset = () => {
    setSelectedFile(null)
    setImagePreview(null)
    setPrediction(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Safe formatting helpers
  const formatConfidence = (val) => {
    if (typeof val === 'number' && !isNaN(val)) {
      return `${val.toFixed(1)}%`
    }
    return 'N/A'
  }

  const formatProbability = (val) => {
    if (typeof val === 'number' && !isNaN(val)) {
      return `${(val * 100).toFixed(1)}%`
    }
    return 'N/A'
  }

  const labelLower = prediction?.label?.toLowerCase() || ''
  const isHotdog = labelLower === 'hotdog'
  const isUncertain = labelLower === 'uncertain'
  const isNotHotdog = labelLower === 'not hotdog'

  // Validate Grad-CAM data URL safely
  const hasValidGradcam =
    typeof prediction?.gradcam_base64 === 'string' &&
    prediction.gradcam_base64.startsWith('data:image/jpeg;base64,')

  return (
    <main className="app-container">
      {/* Header with Title and Backend Health Status */}
      <header className="header-section">
        <div
          className={`status-badge ${backendStatus}`}
          title={backendError || (backendStatus === 'online' ? 'Backend connected & model ready' : 'Checking status')}
        >
          <span className="status-dot" aria-hidden="true"></span>
          <span>
            {backendStatus === 'online' && 'Backend Connected'}
            {backendStatus === 'offline' && 'Backend Offline'}
            {backendStatus === 'checking' && 'Checking Backend...'}
          </span>
          {backendStatus === 'offline' && (
            <button
              onClick={checkHealth}
              className="browse-btn"
              style={{ padding: '0.1rem 0.4rem', fontSize: '0.75rem', marginLeft: '0.25rem' }}
              aria-label="Retry backend connection"
            >
              Retry
            </button>
          )}
        </div>

        <h1 className="brand-title">
          <span className="brand-emoji" aria-hidden="true">🌭</span>
          SeeFood
        </h1>
        <p className="brand-subtitle">
          The ultimate AI food classification engine. Instant verification: is it a hotdog, or not a hotdog?
        </p>
      </header>

      {/* Main Glass Card */}
      <section className="card" aria-label="Image Classifier Section">
        {/* Upload Zone / Drag & Drop */}
        {!imagePreview ? (
          <div
            className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                fileInputRef.current?.click()
              }
            }}
            aria-label="Upload image zone. Drag and drop an image or click to browse"
          >
            <div className="upload-icon-container" aria-hidden="true">
              📷
            </div>
            <div className="dropzone-text">
              <h3>Drag and drop your food photo here</h3>
              <p>Supports JPEG, PNG, and WebP (up to 10MB)</p>
            </div>
            <button
              type="button"
              className="browse-btn"
              onClick={(e) => {
                e.stopPropagation()
                fileInputRef.current?.click()
              }}
            >
              Browse Image
            </button>
          </div>
        ) : (
          /* Preview Selected Image */
          <div className="preview-container">
            <div className="preview-wrapper">
              <img
                src={imagePreview}
                alt="Selected food preview"
                className="image-preview"
              />
            </div>
            <div className="preview-file-info">
              <span>{selectedFile?.name}</span>
              <span>•</span>
              <span>{(selectedFile?.size / 1024).toFixed(1)} KB</span>
            </div>
          </div>
        )}

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
          onChange={handleInputChange}
          style={{ display: 'none' }}
          aria-hidden="true"
        />

        {/* Action Controls */}
        <div className="button-row">
          <button
            type="button"
            className="btn btn-primary"
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                <span>Analyzing Image & Explainability...</span>
              </>
            ) : (
              <>
                <span>⚡</span>
                <span>Classify Image</span>
              </>
            )}
          </button>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleReset}
            disabled={loading || (!selectedFile && !prediction && !error)}
          >
            Reset
          </button>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="error-banner" role="alert">
            <span className="error-icon" aria-hidden="true">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Offline Backend Notice if offline */}
        {backendStatus === 'offline' && backendError && !error && (
          <div className="error-banner" role="alert">
            <span className="error-icon" aria-hidden="true">🔌</span>
            <span>{backendError}</span>
          </div>
        )}

        {/* Prediction Results & Grad-CAM Visual Explainability */}
        {prediction && (
          <article
            className={`result-card ${
              isHotdog ? 'hotdog' : isUncertain ? 'uncertain' : 'not-hotdog'
            }`}
            aria-live="polite"
          >
            <div className="result-header">
              <span className="result-icon" aria-hidden="true">
                {isHotdog ? '🌭' : isUncertain ? '🤔' : '🚫'}
              </span>
              <div>
                <h2 className="result-title">
                  {prediction.label || 'Unknown'}
                </h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  {isHotdog && 'AI verified! This item is classified as a Hotdog.'}
                  {isNotHotdog && 'AI verified! This item is classified as NOT a Hotdog.'}
                  {isUncertain &&
                    'The model is unsure about this image. Try a clearer photo or review the result cautiously.'}
                </p>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Prediction Confidence</span>
                <span className="metric-value">
                  {isUncertain
                    ? 'Indeterminate'
                    : formatConfidence(prediction.confidence)}
                </span>
                <div className="progress-bar-container">
                  <div
                    className="progress-fill"
                    style={{
                      width: isUncertain
                        ? '0%'
                        : `${Math.min(100, Math.max(0, prediction.confidence || 0))}%`,
                    }}
                  ></div>
                </div>
              </div>

              <div className="metric-item">
                <span className="metric-label">Hotdog Probability</span>
                <span className="metric-value">
                  {formatProbability(prediction.hotdog_probability)}
                </span>
                <div className="progress-bar-container">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, (prediction.hotdog_probability || 0) * 100))}%`,
                    }}
                  ></div>
                </div>
              </div>
            </div>

            <p className="provisional-note">
              Decision thresholds are provisional until calibrated on validation data.
            </p>

            {/* Grad-CAM Explainability Section */}
            <div className="explain-section">
              <div className="explain-header">
                <h3>
                  <span aria-hidden="true">🔍</span>
                  Where the AI Looked
                </h3>
                <p className="explain-disclaimer">
                  This Grad-CAM heatmap highlights the key visual regions and patterns that most heavily
                  influenced the neural network's prediction. It highlights model focus areas, but is not an
                  absolute guarantee of classification accuracy.
                </p>
              </div>

              {hasValidGradcam ? (
                <div className="comparison-grid">
                  <div className="comparison-column">
                    <span className="comparison-label">Original Photo</span>
                    <div className="comparison-media-wrapper">
                      <img
                        src={imagePreview}
                        alt="Original food photo uploaded by user"
                        className="comparison-image"
                      />
                    </div>
                  </div>

                  <div className="comparison-column">
                    <span className="comparison-label">Grad-CAM Heatmap Focus</span>
                    <div className="comparison-media-wrapper">
                      <img
                        src={prediction.gradcam_base64}
                        alt="Grad-CAM visual heatmap overlay showing areas that influenced the prediction"
                        className="comparison-image"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="explain-unavailable">
                  Visual explanation is currently unavailable for this prediction.
                </div>
              )}
            </div>
          </article>
        )}
      </section>
    </main>
  )
}
